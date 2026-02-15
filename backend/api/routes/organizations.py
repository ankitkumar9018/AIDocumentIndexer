"""
AIDocumentIndexer - Organization Management API Routes
=======================================================

API endpoints for superadmin organization management:
- CRUD operations for organizations
- Member management
- Feature flag management
- Organization settings
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from backend.db.database import get_async_session
from backend.db.models import Organization, User, OrganizationMember, OrganizationRole, Document
from backend.api.middleware.auth import get_user_context, UserContext

logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class OrganizationCreate(BaseModel):
    """Create organization request."""
    name: str = Field(..., min_length=1, max_length=200)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r'^[a-z0-9-]+$')
    plan: str = Field(default="free", description="Plan: free, pro, enterprise")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_users: int = Field(default=5, ge=1)
    max_storage_gb: int = Field(default=10, ge=1)


class OrganizationUpdate(BaseModel):
    """Update organization request."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    plan: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    max_users: Optional[int] = Field(None, ge=1)
    max_storage_gb: Optional[int] = Field(None, ge=1)
    is_active: Optional[bool] = None


class OrganizationResponse(BaseModel):
    """Organization response."""
    id: str
    name: str
    slug: str
    plan: str
    settings: Dict[str, Any]
    max_users: int
    max_storage_gb: int
    is_active: bool
    created_at: str
    updated_at: str
    member_count: int = 0
    document_count: int = 0
    storage_used_gb: float = 0.0


class OrganizationListResponse(BaseModel):
    """List organizations response."""
    organizations: List[OrganizationResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class MemberCreate(BaseModel):
    """Add member to organization."""
    user_id: str
    role: str = Field(default="member", description="Role: owner, admin, member")


class MemberUpdate(BaseModel):
    """Update member role."""
    role: str = Field(..., description="Role: owner, admin, member")


class MemberResponse(BaseModel):
    """Member response."""
    id: str
    user_id: str
    email: str
    name: Optional[str]
    role: str
    joined_at: str


class FeatureFlagUpdate(BaseModel):
    """Update feature flags for organization."""
    flags: Dict[str, bool]


class OrganizationStatsResponse(BaseModel):
    """Organization statistics."""
    total_organizations: int
    active_organizations: int
    total_users: int
    total_documents: int
    total_storage_gb: float
    organizations_by_plan: Dict[str, int]


# =============================================================================
# Helper Functions
# =============================================================================


async def check_superadmin(user: UserContext, db: AsyncSession) -> User:
    """
    Check if user is a superadmin and return the user object.

    Raises HTTPException if not a superadmin.
    """
    if not user.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Query the user to check superadmin status
    query = select(User).where(User.id == user.user_id)
    result = await db.execute(query)
    db_user = result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    if not db_user.is_superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin access required"
        )

    return db_user


async def check_org_access(
    user: UserContext,
    db: AsyncSession,
    org_id: str,
    require_admin: bool = False
) -> tuple[User, Organization]:
    """
    Check if user has access to an organization.

    Superadmins have access to all organizations.
    Regular users must be members of the organization.

    Returns (user, organization) tuple.
    """
    if not user.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Query the user
    user_query = select(User).where(User.id == user.user_id)
    user_result = await db.execute(user_query)
    db_user = user_result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Query the organization
    org_query = select(Organization).where(Organization.id == org_id)
    org_result = await db.execute(org_query)
    organization = org_result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Superadmins have full access
    if db_user.is_superadmin:
        return db_user, organization

    # Check membership
    member_query = select(OrganizationMember).where(
        OrganizationMember.user_id == db_user.id,
        OrganizationMember.organization_id == org_id
    )
    member_result = await db.execute(member_query)
    membership = member_result.scalar_one_or_none()

    if not membership:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this organization"
        )

    if require_admin and membership.role not in [OrganizationRole.OWNER, OrganizationRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for this operation"
        )

    return db_user, organization


# =============================================================================
# Organization CRUD Endpoints
# =============================================================================


@router.get("/stats", response_model=OrganizationStatsResponse)
async def get_organization_stats(
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Get organization statistics (superadmin only)."""
    await check_superadmin(user, db)

    # Get total and active organization counts
    total_query = select(func.count(Organization.id))
    total_result = await db.execute(total_query)
    total_organizations = total_result.scalar() or 0

    active_query = select(func.count(Organization.id)).where(Organization.is_active == True)
    active_result = await db.execute(active_query)
    active_organizations = active_result.scalar() or 0

    # Get total users
    users_query = select(func.count(User.id))
    users_result = await db.execute(users_query)
    total_users = users_result.scalar() or 0

    # Get organizations by plan
    plan_query = select(
        Organization.plan,
        func.count(Organization.id)
    ).group_by(Organization.plan)
    plan_result = await db.execute(plan_query)
    organizations_by_plan = {row[0]: row[1] for row in plan_result.fetchall()}

    # Get total documents count
    docs_query = select(func.count(Document.id))
    docs_result = await db.execute(docs_query)
    total_documents = docs_result.scalar() or 0

    # Debug: Get raw SQL query result to compare
    from sqlalchemy import text
    raw_result = await db.execute(text("SELECT COUNT(*) FROM documents"))
    raw_count = raw_result.scalar()
    logger.info("Stats: total_documents query result", orm_count=total_documents, raw_count=raw_count)

    # Get total storage used (sum of file sizes in bytes, convert to GB)
    storage_query = select(func.coalesce(func.sum(Document.file_size), 0))
    storage_result = await db.execute(storage_query)
    total_storage_bytes = storage_result.scalar() or 0
    total_storage_gb = round(total_storage_bytes / (1024 * 1024 * 1024), 2)
    logger.info("Stats: storage query result", bytes=total_storage_bytes, gb=total_storage_gb)

    return OrganizationStatsResponse(
        total_organizations=total_organizations,
        active_organizations=active_organizations,
        total_users=total_users,
        total_documents=total_documents,
        total_storage_gb=total_storage_gb,
        organizations_by_plan=organizations_by_plan,
    )


@router.get("", response_model=OrganizationListResponse)
async def list_organizations(
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: Optional[str] = Query(default=None, description="Search in name/slug"),
    plan: Optional[str] = Query(default=None, description="Filter by plan"),
    is_active: Optional[bool] = Query(default=None, description="Filter by active status"),
):
    """List all organizations (superadmin only)."""
    await check_superadmin(user, db)

    # Base query
    query = select(Organization)
    count_query = select(func.count(Organization.id))

    # Apply filters
    if search:
        safe = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        search_filter = or_(
            Organization.name.ilike(f"%{safe}%"),
            Organization.slug.ilike(f"%{safe}%"),
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    if plan:
        query = query.where(Organization.plan == plan)
        count_query = count_query.where(Organization.plan == plan)

    if is_active is not None:
        query = query.where(Organization.is_active == is_active)
        count_query = count_query.where(Organization.is_active == is_active)

    # Get total count
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.order_by(Organization.created_at.desc()).offset(offset).limit(page_size)

    # Execute query
    result = await db.execute(query)
    organizations = result.scalars().all()

    # Build response
    org_responses = []
    for org in organizations:
        # Get member count
        member_count_query = select(func.count(OrganizationMember.id)).where(
            OrganizationMember.organization_id == org.id
        )
        member_count_result = await db.execute(member_count_query)
        member_count = member_count_result.scalar() or 0

        # Get document count for this organization
        doc_count_query = select(func.count(Document.id)).where(
            Document.organization_id == org.id
        )
        doc_count_result = await db.execute(doc_count_query)
        document_count = doc_count_result.scalar() or 0

        # Get storage used for this organization (in GB)
        storage_query = select(func.coalesce(func.sum(Document.file_size), 0)).where(
            Document.organization_id == org.id
        )
        storage_result = await db.execute(storage_query)
        storage_bytes = storage_result.scalar() or 0
        storage_used_gb = round(storage_bytes / (1024 * 1024 * 1024), 2)

        org_responses.append(OrganizationResponse(
            id=str(org.id),
            name=org.name,
            slug=org.slug,
            plan=org.plan or "free",
            settings=org.settings or {},
            max_users=org.max_users or 5,
            max_storage_gb=org.max_storage_gb or 10,
            is_active=org.is_active if org.is_active is not None else True,
            created_at=org.created_at.isoformat() if org.created_at else datetime.utcnow().isoformat(),
            updated_at=org.updated_at.isoformat() if org.updated_at else datetime.utcnow().isoformat(),
            member_count=member_count,
            document_count=document_count,
            storage_used_gb=storage_used_gb,
        ))

    return OrganizationListResponse(
        organizations=org_responses,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.post("", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    request: OrganizationCreate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Create a new organization (superadmin only)."""
    await check_superadmin(user, db)

    # Check if slug is unique
    existing_query = select(Organization).where(Organization.slug == request.slug)
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Organization with slug '{request.slug}' already exists"
        )

    # Create organization
    org = Organization(
        id=uuid.uuid4(),
        name=request.name,
        slug=request.slug,
        plan=request.plan,
        settings=request.settings or {},
        max_users=request.max_users,
        max_storage_gb=request.max_storage_gb,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(org)
    await db.commit()
    await db.refresh(org)

    logger.info("Created organization", org_id=str(org.id), name=org.name)

    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        plan=org.plan or "free",
        settings=org.settings or {},
        max_users=org.max_users or 5,
        max_storage_gb=org.max_storage_gb or 10,
        is_active=org.is_active if org.is_active is not None else True,
        created_at=org.created_at.isoformat(),
        updated_at=org.updated_at.isoformat(),
        member_count=0,
    )


# =============================================================================
# Organization Switching Endpoints (MUST be before /{org_id} routes)
# =============================================================================


class SwitchOrganizationRequest(BaseModel):
    """Request to switch current organization."""
    organization_id: str


class UserOrganizationResponse(BaseModel):
    """Response for user's organization context."""
    user_id: str
    email: str
    name: Optional[str]
    is_superadmin: bool
    current_organization_id: Optional[str]
    current_organization: Optional[OrganizationResponse]
    available_organizations: List[OrganizationResponse]


@router.get("/me/context", response_model=UserOrganizationResponse)
async def get_my_organization_context(
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get the current user's organization context.

    Returns the user's superadmin status, current organization, and list of
    available organizations (all orgs for superadmins, memberships for regular users).
    """
    if not user.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Query the user - convert string user_id to UUID for comparison
    from uuid import UUID as PyUUID
    try:
        user_uuid = PyUUID(user.user_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID format"
        )

    user_query = select(User).where(User.id == user_uuid)
    user_result = await db.execute(user_query)
    db_user = user_result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Get current organization if set
    current_org_response = None
    if db_user.current_organization_id:
        org_query = select(Organization).where(Organization.id == db_user.current_organization_id)
        org_result = await db.execute(org_query)
        current_org = org_result.scalar_one_or_none()
        if current_org:
            current_org_response = OrganizationResponse(
                id=str(current_org.id),
                name=current_org.name,
                slug=current_org.slug,
                plan=current_org.plan or "free",
                settings=current_org.settings or {},
                max_users=current_org.max_users or 5,
                max_storage_gb=current_org.max_storage_gb or 10,
                is_active=current_org.is_active if current_org.is_active is not None else True,
                created_at=current_org.created_at.isoformat() if current_org.created_at else datetime.utcnow().isoformat(),
                updated_at=current_org.updated_at.isoformat() if current_org.updated_at else datetime.utcnow().isoformat(),
                member_count=0,
            )

    # Get available organizations
    available_orgs = []
    if db_user.is_superadmin:
        # Superadmins can access all active organizations
        orgs_query = select(Organization).where(Organization.is_active == True).order_by(Organization.name)
        orgs_result = await db.execute(orgs_query)
        orgs = orgs_result.scalars().all()
        for org in orgs:
            available_orgs.append(OrganizationResponse(
                id=str(org.id),
                name=org.name,
                slug=org.slug,
                plan=org.plan or "free",
                settings=org.settings or {},
                max_users=org.max_users or 5,
                max_storage_gb=org.max_storage_gb or 10,
                is_active=True,
                created_at=org.created_at.isoformat() if org.created_at else datetime.utcnow().isoformat(),
                updated_at=org.updated_at.isoformat() if org.updated_at else datetime.utcnow().isoformat(),
                member_count=0,
            ))
    else:
        # Regular users can only access organizations they're members of
        memberships_query = select(OrganizationMember, Organization).join(
            Organization, OrganizationMember.organization_id == Organization.id
        ).where(
            OrganizationMember.user_id == db_user.id,
            Organization.is_active == True,
        ).order_by(Organization.name)
        memberships_result = await db.execute(memberships_query)
        memberships = memberships_result.fetchall()
        for membership, org in memberships:
            available_orgs.append(OrganizationResponse(
                id=str(org.id),
                name=org.name,
                slug=org.slug,
                plan=org.plan or "free",
                settings=org.settings or {},
                max_users=org.max_users or 5,
                max_storage_gb=org.max_storage_gb or 10,
                is_active=True,
                created_at=org.created_at.isoformat() if org.created_at else datetime.utcnow().isoformat(),
                updated_at=org.updated_at.isoformat() if org.updated_at else datetime.utcnow().isoformat(),
                member_count=0,
            ))

    return UserOrganizationResponse(
        user_id=str(db_user.id),
        email=db_user.email,
        name=db_user.name,
        is_superadmin=db_user.is_superadmin,
        current_organization_id=str(db_user.current_organization_id) if db_user.current_organization_id else None,
        current_organization=current_org_response,
        available_organizations=available_orgs,
    )


@router.post("/me/switch", response_model=UserOrganizationResponse)
async def switch_organization(
    request: SwitchOrganizationRequest,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Switch the current user's active organization context.

    Superadmins can switch to any organization.
    Regular users can only switch to organizations they're members of.
    """
    if not user.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Query the user
    user_query = select(User).where(User.id == user.user_id)
    user_result = await db.execute(user_query)
    db_user = user_result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Query the target organization
    org_query = select(Organization).where(
        Organization.id == uuid.UUID(request.organization_id),
        Organization.is_active == True,
    )
    org_result = await db.execute(org_query)
    target_org = org_result.scalar_one_or_none()

    if not target_org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Check access
    if not db_user.is_superadmin:
        # Regular users must be members
        membership_query = select(OrganizationMember).where(
            OrganizationMember.user_id == db_user.id,
            OrganizationMember.organization_id == target_org.id,
        )
        membership_result = await db.execute(membership_query)
        if not membership_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this organization"
            )

    # Update current organization
    db_user.current_organization_id = target_org.id
    await db.commit()
    await db.refresh(db_user)

    logger.info("User switched organization", user_id=str(db_user.id), org_id=str(target_org.id))

    # Return updated context
    return await get_my_organization_context(user, db)


# =============================================================================
# Organization CRUD by ID Endpoints
# =============================================================================


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Get organization details (superadmin only)."""
    await check_superadmin(user, db)

    query = select(Organization).where(Organization.id == org_id)
    result = await db.execute(query)
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Get member count
    member_count_query = select(func.count(OrganizationMember.id)).where(
        OrganizationMember.organization_id == org.id
    )
    member_count_result = await db.execute(member_count_query)
    member_count = member_count_result.scalar() or 0

    # Get document count
    doc_count_query = select(func.count(Document.id)).where(
        Document.organization_id == org.id
    )
    doc_count_result = await db.execute(doc_count_query)
    document_count = doc_count_result.scalar() or 0

    # Get storage used (in GB)
    storage_query = select(func.coalesce(func.sum(Document.file_size), 0)).where(
        Document.organization_id == org.id
    )
    storage_result = await db.execute(storage_query)
    storage_bytes = storage_result.scalar() or 0
    storage_used_gb = round(storage_bytes / (1024 * 1024 * 1024), 2)

    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        plan=org.plan or "free",
        settings=org.settings or {},
        max_users=org.max_users or 5,
        max_storage_gb=org.max_storage_gb or 10,
        is_active=org.is_active if org.is_active is not None else True,
        created_at=org.created_at.isoformat() if org.created_at else datetime.utcnow().isoformat(),
        updated_at=org.updated_at.isoformat() if org.updated_at else datetime.utcnow().isoformat(),
        member_count=member_count,
        document_count=document_count,
        storage_used_gb=storage_used_gb,
    )


@router.patch("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: uuid.UUID,
    request: OrganizationUpdate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Update organization (superadmin only)."""
    await check_superadmin(user, db)

    query = select(Organization).where(Organization.id == org_id)
    result = await db.execute(query)
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Update fields
    if request.name is not None:
        org.name = request.name
    if request.plan is not None:
        org.plan = request.plan
    if request.settings is not None:
        org.settings = request.settings
    if request.max_users is not None:
        org.max_users = request.max_users
    if request.max_storage_gb is not None:
        org.max_storage_gb = request.max_storage_gb
    if request.is_active is not None:
        org.is_active = request.is_active

    org.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(org)

    logger.info("Updated organization", org_id=str(org.id))

    # Get member count
    member_count_query = select(func.count(OrganizationMember.id)).where(
        OrganizationMember.organization_id == org.id
    )
    member_count_result = await db.execute(member_count_query)
    member_count = member_count_result.scalar() or 0

    # Get document count
    doc_count_query = select(func.count(Document.id)).where(
        Document.organization_id == org.id
    )
    doc_count_result = await db.execute(doc_count_query)
    document_count = doc_count_result.scalar() or 0

    # Get storage used (in GB)
    storage_query = select(func.coalesce(func.sum(Document.file_size), 0)).where(
        Document.organization_id == org.id
    )
    storage_result = await db.execute(storage_query)
    storage_bytes = storage_result.scalar() or 0
    storage_used_gb = round(storage_bytes / (1024 * 1024 * 1024), 2)

    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        plan=org.plan or "free",
        settings=org.settings or {},
        max_users=org.max_users or 5,
        max_storage_gb=org.max_storage_gb or 10,
        is_active=org.is_active if org.is_active is not None else True,
        created_at=org.created_at.isoformat() if org.created_at else datetime.utcnow().isoformat(),
        updated_at=org.updated_at.isoformat() if org.updated_at else datetime.utcnow().isoformat(),
        member_count=member_count,
        document_count=document_count,
        storage_used_gb=storage_used_gb,
    )


@router.delete("/{org_id}")
async def delete_organization(
    org_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Delete organization (superadmin only)."""
    await check_superadmin(user, db)

    query = select(Organization).where(Organization.id == org_id)
    result = await db.execute(query)
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Soft delete by setting is_active to False
    org.is_active = False
    org.updated_at = datetime.utcnow()

    await db.commit()

    logger.info("Deleted organization", org_id=str(org.id))

    return {"message": "Organization deleted", "id": str(org_id)}


# =============================================================================
# Member Management Endpoints
# =============================================================================


@router.get("/{org_id}/members", response_model=List[MemberResponse])
async def list_organization_members(
    org_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """List organization members (superadmin only)."""
    await check_superadmin(user, db)

    # Check organization exists
    org_query = select(Organization).where(Organization.id == org_id)
    org_result = await db.execute(org_query)
    if not org_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Get members with user details
    query = select(OrganizationMember, User).join(
        User, OrganizationMember.user_id == User.id
    ).where(OrganizationMember.organization_id == org_id)

    result = await db.execute(query)
    members = result.fetchall()

    return [
        MemberResponse(
            id=str(member.id),
            user_id=str(member.user_id),
            email=user_obj.email,
            name=user_obj.name,
            role=member.role.value if member.role else "member",
            joined_at=member.joined_at.isoformat() if member.joined_at else datetime.utcnow().isoformat(),
        )
        for member, user_obj in members
    ]


@router.post("/{org_id}/members", response_model=MemberResponse, status_code=status.HTTP_201_CREATED)
async def add_organization_member(
    org_id: uuid.UUID,
    request: MemberCreate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Add member to organization (superadmin only)."""
    await check_superadmin(user, db)

    # Check organization exists
    org_query = select(Organization).where(Organization.id == org_id)
    org_result = await db.execute(org_query)
    org = org_result.scalar_one_or_none()
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Check user exists
    user_query = select(User).where(User.id == uuid.UUID(request.user_id))
    user_result = await db.execute(user_query)
    target_user = user_result.scalar_one_or_none()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if already a member
    existing_query = select(OrganizationMember).where(
        OrganizationMember.organization_id == org_id,
        OrganizationMember.user_id == uuid.UUID(request.user_id),
    )
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a member of this organization"
        )

    # Create member
    try:
        role = OrganizationRole(request.role.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}. Must be one of: owner, admin, member")
    member = OrganizationMember(
        id=uuid.uuid4(),
        organization_id=org_id,
        user_id=uuid.UUID(request.user_id),
        role=role,
        joined_at=datetime.utcnow(),
    )

    db.add(member)
    await db.commit()
    await db.refresh(member)

    logger.info("Added member to organization", org_id=str(org_id), user_id=request.user_id)

    return MemberResponse(
        id=str(member.id),
        user_id=str(member.user_id),
        email=target_user.email,
        name=target_user.name,
        role=member.role.value if member.role else "member",
        joined_at=member.joined_at.isoformat(),
    )


@router.patch("/{org_id}/members/{member_id}", response_model=MemberResponse)
async def update_member_role(
    org_id: uuid.UUID,
    member_id: uuid.UUID,
    request: MemberUpdate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Update member role (superadmin only)."""
    await check_superadmin(user, db)

    query = select(OrganizationMember, User).join(
        User, OrganizationMember.user_id == User.id
    ).where(
        OrganizationMember.id == member_id,
        OrganizationMember.organization_id == org_id,
    )

    result = await db.execute(query)
    row = result.first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )

    member, target_user = row

    # Update role
    try:
        role = OrganizationRole(request.role.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}. Must be one of: owner, admin, member")
    member.role = role

    await db.commit()
    await db.refresh(member)

    logger.info("Updated member role", org_id=str(org_id), member_id=str(member_id), role=request.role)

    return MemberResponse(
        id=str(member.id),
        user_id=str(member.user_id),
        email=target_user.email,
        name=target_user.name,
        role=member.role.value if member.role else "member",
        joined_at=member.joined_at.isoformat() if member.joined_at else datetime.utcnow().isoformat(),
    )


@router.delete("/{org_id}/members/{member_id}")
async def remove_organization_member(
    org_id: uuid.UUID,
    member_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Remove member from organization (superadmin only)."""
    await check_superadmin(user, db)

    query = select(OrganizationMember).where(
        OrganizationMember.id == member_id,
        OrganizationMember.organization_id == org_id,
    )

    result = await db.execute(query)
    member = result.scalar_one_or_none()

    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )

    await db.delete(member)
    await db.commit()

    logger.info("Removed member from organization", org_id=str(org_id), member_id=str(member_id))

    return {"message": "Member removed", "id": str(member_id)}


# =============================================================================
# Feature Flags Endpoints
# =============================================================================


@router.get("/{org_id}/features")
async def get_organization_features(
    org_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Get organization feature flags (superadmin only)."""
    await check_superadmin(user, db)

    query = select(Organization).where(Organization.id == org_id)
    result = await db.execute(query)
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Feature flags are stored in org.settings under 'features' key
    features = (org.settings or {}).get("features", {})

    # Default features with their status
    default_features = {
        "workflows": True,
        "audio_overviews": True,
        "connectors": True,
        "document_generation": True,
        "knowledge_graph": True,
        "collaboration": True,
        "web_scraper": True,
        "image_generation": False,
        "custom_llm_models": False,
        "api_access": False,
    }

    # Merge with stored features
    merged_features = {**default_features, **features}

    return {
        "organization_id": str(org_id),
        "features": merged_features,
    }


@router.patch("/{org_id}/features")
async def update_organization_features(
    org_id: uuid.UUID,
    request: FeatureFlagUpdate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Update organization feature flags (superadmin only)."""
    await check_superadmin(user, db)

    query = select(Organization).where(Organization.id == org_id)
    result = await db.execute(query)
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Update features in settings
    settings = org.settings or {}
    settings["features"] = {**settings.get("features", {}), **request.flags}
    org.settings = settings
    org.updated_at = datetime.utcnow()

    await db.commit()

    logger.info("Updated organization features", org_id=str(org_id), flags=request.flags)

    return {
        "organization_id": str(org_id),
        "features": settings["features"],
    }
