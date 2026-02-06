"""
AIDocumentIndexer - Admin API Routes
=====================================

Endpoints for admin operations:
- User management
- Access tier management
- Audit log viewing
- System configuration
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from backend.core.config import settings
from backend.db.database import get_async_session
from backend.db.models import User, AccessTier, AuditLog, FolderPermission, Folder
from backend.api.middleware.auth import (
    get_user_context,
    require_admin,
    AdminUser,
)
from backend.services.permissions import UserContext
from backend.services.audit import (
    AuditService,
    AuditAction,
    AuditEntry,
    LogSeverity,
    get_audit_service,
)
from backend.services.settings import (
    SettingsService,
    SettingCategory,
    get_settings_service,
)
from backend.services.database_manager import (
    DatabaseManager,
    DatabaseConnectionService,
    get_database_manager,
)
from backend.services.llm_provider import (
    LLMProviderService,
    PROVIDER_TYPES,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models - Access Tiers
# =============================================================================

class AccessTierBase(BaseModel):
    """Base access tier model."""
    name: str = Field(..., min_length=1, max_length=100)
    level: int = Field(..., ge=0, le=100)  # 0 = Public/Everyone
    description: Optional[str] = None
    color: str = Field(default="#6B7280", pattern="^#[0-9A-Fa-f]{6}$")


class AccessTierCreate(AccessTierBase):
    """Access tier creation request."""
    pass


class AccessTierUpdate(BaseModel):
    """Access tier update request."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    level: Optional[int] = Field(None, ge=0, le=100)  # 0 = Public/Everyone
    description: Optional[str] = None
    color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")


class AccessTierResponse(AccessTierBase):
    """Access tier response model."""
    id: UUID
    user_count: int = 0
    document_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AccessTierListResponse(BaseModel):
    """Paginated access tier list response."""
    tiers: List[AccessTierResponse]
    total: int


# =============================================================================
# Pydantic Models - Users
# =============================================================================

class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    name: Optional[str] = None


class InitialFolderPermission(BaseModel):
    """Initial folder permission for user creation."""
    folder_id: UUID
    permission_level: str = Field(default="view", pattern="^(view|edit|manage)$")
    inherit_to_children: bool = True


class UserCreate(UserBase):
    """User creation request."""
    password: str = Field(..., min_length=8)
    access_tier_id: UUID
    use_folder_permissions_only: bool = False
    initial_folder_permissions: Optional[List[InitialFolderPermission]] = None


class UserUpdate(BaseModel):
    """User update request."""
    name: Optional[str] = None
    access_tier_id: Optional[UUID] = None
    is_active: Optional[bool] = None
    use_folder_permissions_only: Optional[bool] = None


class UserResponse(BaseModel):
    """User response model."""
    id: UUID
    email: str
    name: Optional[str]
    is_active: bool
    access_tier_id: UUID
    access_tier_name: str
    access_tier_level: int
    use_folder_permissions_only: bool = False
    created_at: datetime
    last_login_at: Optional[datetime]

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Paginated user list response."""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


# =============================================================================
# Pydantic Models - Audit Logs
# =============================================================================

class AuditLogResponse(BaseModel):
    """Audit log entry response."""
    id: str
    action: str
    severity: str = "info"  # Phase 52: Log severity level
    user_id: Optional[str]
    user_email: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Optional[dict]
    ip_address: Optional[str]
    created_at: datetime


class AuditLogListResponse(BaseModel):
    """Paginated audit log list response."""
    logs: List[AuditLogResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class SeverityCountResponse(BaseModel):
    """Log severity count response (Phase 52)."""
    debug: int = 0
    info: int = 0
    notice: int = 0
    warning: int = 0
    error: int = 0
    critical: int = 0


# =============================================================================
# Helper Functions
# =============================================================================

def get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


# =============================================================================
# Access Tier Endpoints
# =============================================================================

@router.get("/tiers", response_model=AccessTierListResponse)
async def list_access_tiers(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all access tiers.

    Admin only endpoint.
    """
    logger.info("Listing access tiers", admin_id=admin.user_id)

    # Get all tiers with counts
    query = select(AccessTier).order_by(AccessTier.level)
    result = await db.execute(query)
    tiers = result.scalars().all()

    # Get user counts per tier
    user_count_query = (
        select(User.access_tier_id, func.count(User.id))
        .group_by(User.access_tier_id)
    )
    user_counts_result = await db.execute(user_count_query)
    user_counts = {row[0]: row[1] for row in user_counts_result.all()}

    # Get document counts per tier
    from backend.db.models import Document
    doc_count_query = (
        select(Document.access_tier_id, func.count(Document.id))
        .group_by(Document.access_tier_id)
    )
    doc_counts_result = await db.execute(doc_count_query)
    doc_counts = {row[0]: row[1] for row in doc_counts_result.all()}

    # Build response
    tier_responses = [
        AccessTierResponse(
            id=tier.id,
            name=tier.name,
            level=tier.level,
            description=tier.description,
            color=tier.color,
            user_count=user_counts.get(tier.id, 0),
            document_count=doc_counts.get(tier.id, 0),
            created_at=tier.created_at,
            updated_at=tier.updated_at,
        )
        for tier in tiers
    ]

    return AccessTierListResponse(
        tiers=tier_responses,
        total=len(tier_responses),
    )


@router.post("/tiers", response_model=AccessTierResponse, status_code=status.HTTP_201_CREATED)
async def create_access_tier(
    tier: AccessTierCreate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new access tier.

    Admin only endpoint.
    """
    logger.info("Creating access tier", admin_id=admin.user_id, tier_name=tier.name)

    # Check for duplicate name
    existing_query = select(AccessTier).where(AccessTier.name == tier.name)
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Access tier with name '{tier.name}' already exists",
        )

    # Check for duplicate level
    level_query = select(AccessTier).where(AccessTier.level == tier.level)
    level_result = await db.execute(level_query)
    if level_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Access tier with level {tier.level} already exists",
        )

    # Create tier
    new_tier = AccessTier(
        name=tier.name,
        level=tier.level,
        description=tier.description,
        color=tier.color,
    )
    db.add(new_tier)
    await db.commit()
    await db.refresh(new_tier)

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.TIER_CREATE,
        admin_user_id=admin.user_id,
        target_resource_type="access_tier",
        target_resource_id=str(new_tier.id),
        changes={"name": tier.name, "level": tier.level},
        ip_address=get_client_ip(request),
        session=db,
    )

    return AccessTierResponse(
        id=new_tier.id,
        name=new_tier.name,
        level=new_tier.level,
        description=new_tier.description,
        color=new_tier.color,
        user_count=0,
        document_count=0,
        created_at=new_tier.created_at,
        updated_at=new_tier.updated_at,
    )


@router.patch("/tiers/{tier_id}", response_model=AccessTierResponse)
async def update_access_tier(
    tier_id: UUID,
    update: AccessTierUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update an access tier.

    Admin only endpoint.
    """
    logger.info("Updating access tier", admin_id=admin.user_id, tier_id=str(tier_id))

    # Get tier
    query = select(AccessTier).where(AccessTier.id == tier_id)
    result = await db.execute(query)
    tier = result.scalar_one_or_none()

    if not tier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access tier not found",
        )

    # Track changes for audit
    changes = {}

    # Check for duplicate name if updating
    if update.name and update.name != tier.name:
        existing_query = select(AccessTier).where(
            and_(AccessTier.name == update.name, AccessTier.id != tier_id)
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Access tier with name '{update.name}' already exists",
            )
        changes["name"] = {"old": tier.name, "new": update.name}
        tier.name = update.name

    # Check for duplicate level if updating
    if update.level and update.level != tier.level:
        level_query = select(AccessTier).where(
            and_(AccessTier.level == update.level, AccessTier.id != tier_id)
        )
        level_result = await db.execute(level_query)
        if level_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Access tier with level {update.level} already exists",
            )
        changes["level"] = {"old": tier.level, "new": update.level}
        tier.level = update.level

    if update.description is not None:
        changes["description"] = {"old": tier.description, "new": update.description}
        tier.description = update.description

    if update.color:
        changes["color"] = {"old": tier.color, "new": update.color}
        tier.color = update.color

    await db.commit()
    await db.refresh(tier)

    # Log the action
    if changes:
        audit_service = get_audit_service()
        await audit_service.log_admin_action(
            action=AuditAction.TIER_UPDATE,
            admin_user_id=admin.user_id,
            target_resource_type="access_tier",
            target_resource_id=str(tier_id),
            changes=changes,
            ip_address=get_client_ip(request),
            session=db,
        )

    # Get user count
    user_count_query = select(func.count(User.id)).where(User.access_tier_id == tier_id)
    user_count_result = await db.execute(user_count_query)
    user_count = user_count_result.scalar() or 0

    return AccessTierResponse(
        id=tier.id,
        name=tier.name,
        level=tier.level,
        description=tier.description,
        color=tier.color,
        user_count=user_count,
        document_count=0,
        created_at=tier.created_at,
        updated_at=tier.updated_at,
    )


@router.delete("/tiers/{tier_id}")
async def delete_access_tier(
    tier_id: UUID,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete an access tier.

    Cannot delete a tier that has users assigned to it.
    Admin only endpoint.
    """
    logger.info("Deleting access tier", admin_id=admin.user_id, tier_id=str(tier_id))

    # Get tier
    query = select(AccessTier).where(AccessTier.id == tier_id)
    result = await db.execute(query)
    tier = result.scalar_one_or_none()

    if not tier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access tier not found",
        )

    # Check if tier has users
    user_count_query = select(func.count(User.id)).where(User.access_tier_id == tier_id)
    user_count_result = await db.execute(user_count_query)
    user_count = user_count_result.scalar() or 0

    if user_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete tier with {user_count} users. Reassign users first.",
        )

    tier_name = tier.name
    tier_level = tier.level

    # Delete tier
    await db.delete(tier)
    await db.commit()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.TIER_DELETE,
        admin_user_id=admin.user_id,
        target_resource_type="access_tier",
        target_resource_id=str(tier_id),
        changes={"name": tier_name, "level": tier_level},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Access tier deleted successfully", "tier_id": str(tier_id)}


# =============================================================================
# User Management Endpoints
# =============================================================================

@router.get("/users", response_model=UserListResponse)
async def list_users(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    tier_id: Optional[UUID] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
):
    """
    List all users with filtering.

    Admin only endpoint.
    """
    logger.info("Listing users", admin_id=admin.user_id, page=page)

    # Build query
    query = select(User).options(selectinload(User.access_tier))

    conditions = []
    if tier_id:
        conditions.append(User.access_tier_id == tier_id)
    if is_active is not None:
        conditions.append(User.is_active == is_active)
    if search:
        conditions.append(
            User.email.ilike(f"%{search}%") | User.name.ilike(f"%{search}%")
        )

    if conditions:
        query = query.where(and_(*conditions))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.order_by(User.created_at.desc()).offset(offset).limit(page_size)

    # Execute query
    result = await db.execute(query)
    users = result.scalars().all()

    # Build response
    user_responses = [
        UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            access_tier_id=user.access_tier_id,
            access_tier_name=user.access_tier.name if user.access_tier else "Unknown",
            access_tier_level=user.access_tier.level if user.access_tier else 0,
            use_folder_permissions_only=user.use_folder_permissions_only,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )
        for user in users
    ]

    return UserListResponse(
        users=user_responses,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(users)) < total,
    )


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new user.

    Admin only endpoint.
    Admins can only create users with tiers at or below their own level.
    """
    logger.info("Creating user", admin_id=admin.user_id, email=user_data.email)

    # Check if email already exists
    existing_query = select(User).where(User.email == user_data.email)
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{user_data.email}' already exists",
        )

    # Verify access tier exists and admin can assign it
    tier_query = select(AccessTier).where(AccessTier.id == user_data.access_tier_id)
    tier_result = await db.execute(tier_query)
    tier = tier_result.scalar_one_or_none()

    if not tier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access tier not found",
        )

    # Admin can only assign tiers at or below their own level
    if tier.level > admin.access_tier_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cannot assign tier {tier.level} (your tier: {admin.access_tier_level})",
        )

    # Hash password
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed_password = pwd_context.hash(user_data.password)

    # Create user
    new_user = User(
        email=user_data.email,
        name=user_data.name,
        password_hash=hashed_password,
        access_tier_id=user_data.access_tier_id,
        is_active=True,
        use_folder_permissions_only=user_data.use_folder_permissions_only,
    )
    db.add(new_user)
    await db.flush()  # Get the user ID without committing

    # Add initial folder permissions if provided
    if user_data.initial_folder_permissions:
        for perm in user_data.initial_folder_permissions:
            # Verify folder exists
            folder_query = select(Folder).where(Folder.id == perm.folder_id)
            folder_result = await db.execute(folder_query)
            folder = folder_result.scalar_one_or_none()
            if not folder:
                await db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Folder {perm.folder_id} not found",
                )

            folder_permission = FolderPermission(
                folder_id=perm.folder_id,
                user_id=new_user.id,
                permission_level=perm.permission_level,
                inherit_to_children=perm.inherit_to_children,
                granted_by_id=admin.user_id,
            )
            db.add(folder_permission)

    await db.commit()
    await db.refresh(new_user)

    # Reload with access tier
    user_query = select(User).options(selectinload(User.access_tier)).where(User.id == new_user.id)
    user_result = await db.execute(user_query)
    new_user = user_result.scalar_one()

    # Log action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.USER_CREATE,
        admin_user_id=admin.user_id,
        target_user_id=str(new_user.id),
        target_resource_type="user",
        changes={
            "email": new_user.email,
            "name": new_user.name,
            "access_tier": tier.name,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        name=new_user.name,
        is_active=new_user.is_active,
        access_tier_id=new_user.access_tier_id,
        access_tier_name=new_user.access_tier.name if new_user.access_tier else "Unknown",
        access_tier_level=new_user.access_tier.level if new_user.access_tier else 0,
        use_folder_permissions_only=new_user.use_folder_permissions_only,
        created_at=new_user.created_at,
        last_login_at=new_user.last_login_at,
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific user by ID.

    Admin only endpoint.
    """
    query = (
        select(User)
        .where(User.id == user_id)
        .options(selectinload(User.access_tier))
    )
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        is_active=user.is_active,
        access_tier_id=user.access_tier_id,
        access_tier_name=user.access_tier.name if user.access_tier else "Unknown",
        access_tier_level=user.access_tier.level if user.access_tier else 0,
        use_folder_permissions_only=user.use_folder_permissions_only,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    update: UserUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a user.

    Admin only endpoint.
    """
    logger.info("Updating user", admin_id=admin.user_id, target_user_id=str(user_id))

    # Get user
    query = (
        select(User)
        .where(User.id == user_id)
        .options(selectinload(User.access_tier))
    )
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Track changes for audit
    changes = {}

    if update.name is not None:
        changes["name"] = {"old": user.name, "new": update.name}
        user.name = update.name

    if update.is_active is not None:
        changes["is_active"] = {"old": user.is_active, "new": update.is_active}
        user.is_active = update.is_active

    if update.access_tier_id:
        # Verify tier exists
        tier_query = select(AccessTier).where(AccessTier.id == update.access_tier_id)
        tier_result = await db.execute(tier_query)
        new_tier = tier_result.scalar_one_or_none()

        if not new_tier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Access tier not found",
            )

        # Check admin can assign this tier (must be at or below admin's tier)
        if new_tier.level > admin.access_tier_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cannot assign tier {new_tier.level}. Your tier: {admin.access_tier_level}",
            )

        old_tier_name = user.access_tier.name if user.access_tier else "Unknown"
        changes["access_tier"] = {"old": old_tier_name, "new": new_tier.name}
        user.access_tier_id = update.access_tier_id

    if update.use_folder_permissions_only is not None:
        changes["use_folder_permissions_only"] = {
            "old": user.use_folder_permissions_only,
            "new": update.use_folder_permissions_only,
        }
        user.use_folder_permissions_only = update.use_folder_permissions_only

    await db.commit()
    await db.refresh(user)

    # Log the action
    if changes:
        audit_service = get_audit_service()

        # Special handling for tier change
        if "access_tier" in changes:
            await audit_service.log_admin_action(
                action=AuditAction.USER_TIER_CHANGE,
                admin_user_id=admin.user_id,
                target_user_id=str(user_id),
                changes=changes,
                ip_address=get_client_ip(request),
                session=db,
            )
        else:
            await audit_service.log_admin_action(
                action=AuditAction.USER_UPDATE,
                admin_user_id=admin.user_id,
                target_user_id=str(user_id),
                changes=changes,
                ip_address=get_client_ip(request),
                session=db,
            )

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        is_active=user.is_active,
        access_tier_id=user.access_tier_id,
        access_tier_name=user.access_tier.name if user.access_tier else "Unknown",
        access_tier_level=user.access_tier.level if user.access_tier else 0,
        use_folder_permissions_only=user.use_folder_permissions_only,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


# =============================================================================
# User Folder Permissions Endpoints
# =============================================================================

class UserFolderPermissionResponse(BaseModel):
    """Response model for user folder permission."""
    id: str
    folder_id: str
    folder_name: str
    folder_path: str
    permission_level: str
    inherit_to_children: bool
    created_at: Optional[str] = None


@router.get("/users/{user_id}/folder-permissions", response_model=List[UserFolderPermissionResponse])
async def get_user_folder_permissions(
    user_id: UUID,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get all folder permissions for a user.

    Admin only endpoint.
    """
    from backend.services.folder_service import get_folder_service

    logger.info("Getting folder permissions for user", admin_id=admin.user_id, target_user_id=str(user_id))

    # Verify user exists
    user_query = select(User).where(User.id == user_id)
    result = await db.execute(user_query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    folder_service = get_folder_service()
    permissions = await folder_service.get_user_folder_permissions(str(user_id))

    return [
        UserFolderPermissionResponse(**p)
        for p in permissions
    ]


# =============================================================================
# Audit Log Endpoints
# =============================================================================

@router.get("/audit-logs", response_model=AuditLogListResponse)
async def list_audit_logs(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    action: Optional[str] = None,
    user_id: Optional[UUID] = None,
    resource_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    severity: Optional[str] = Query(
        default=None,
        description="Filter by exact severity (debug, info, notice, warning, error, critical)"
    ),
    min_severity: Optional[str] = Query(
        default=None,
        description="Filter by minimum severity (includes all at or above this level)"
    ),
):
    """
    List audit logs with filtering.

    Admin only endpoint.

    Severity levels (in order of increasing importance):
    - debug: Debug information, very verbose
    - info: Informational messages, normal operations
    - notice: Normal but significant conditions
    - warning: Warning conditions, potential issues
    - error: Error conditions, operation failed
    - critical: Critical conditions, needs immediate attention
    """
    logger.info("Listing audit logs", admin_id=admin.user_id, page=page, severity=severity)

    audit_service = get_audit_service()

    # Convert action string to enum if provided
    action_enum = None
    if action:
        try:
            action_enum = AuditAction(action)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {action}",
            )

    # Phase 52: Convert severity strings to enums
    severity_enum = None
    min_severity_enum = None

    if severity:
        try:
            severity_enum = LogSeverity(severity.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity: {severity}. Valid values: debug, info, notice, warning, error, critical",
            )

    if min_severity:
        try:
            min_severity_enum = LogSeverity(min_severity.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid min_severity: {min_severity}. Valid values: debug, info, notice, warning, error, critical",
            )

    entries, total = await audit_service.get_logs(
        action=action_enum,
        user_id=str(user_id) if user_id else None,
        resource_type=resource_type,
        start_date=start_date,
        end_date=end_date,
        severity=severity_enum,
        min_severity=min_severity_enum,
        page=page,
        page_size=page_size,
        session=db,
    )

    log_responses = [
        AuditLogResponse(
            id=entry.id,
            action=entry.action,
            severity=entry.severity,
            user_id=entry.user_id,
            user_email=entry.user_email,
            resource_type=entry.resource_type,
            resource_id=entry.resource_id,
            details=entry.details,
            ip_address=entry.ip_address,
            created_at=entry.created_at,
        )
        for entry in entries
    ]

    offset = (page - 1) * page_size

    return AuditLogListResponse(
        logs=log_responses,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(entries)) < total,
    )


@router.get("/audit-logs/actions")
async def list_audit_actions(admin: AdminUser):
    """
    List all available audit action types.

    Admin only endpoint.
    """
    return {
        "actions": [
            {"value": action.value, "name": action.name}
            for action in AuditAction
        ]
    }


@router.get("/audit-logs/severities")
async def list_audit_severities(admin: AdminUser):
    """
    List all available log severity levels (Phase 52).

    Admin only endpoint.
    """
    return {
        "severities": [
            {"value": s.value, "name": s.name, "order": i}
            for i, s in enumerate(LogSeverity)
        ],
        "description": {
            "debug": "Debug information, very verbose",
            "info": "Informational messages, normal operations",
            "notice": "Normal but significant conditions",
            "warning": "Warning conditions, potential issues",
            "error": "Error conditions, operation failed",
            "critical": "Critical conditions, needs immediate attention",
        }
    }


@router.get("/audit-logs/severity-counts", response_model=SeverityCountResponse)
async def get_audit_severity_counts(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    hours: int = Query(default=24, ge=1, le=720),
):
    """
    Get count of logs by severity level (Phase 52).

    Useful for dashboard statistics and alerting.

    Admin only endpoint.
    """
    audit_service = get_audit_service()
    counts = await audit_service.get_severity_counts(hours=hours, session=db)

    return SeverityCountResponse(
        debug=counts.get("debug", 0),
        info=counts.get("info", 0),
        notice=counts.get("notice", 0),
        warning=counts.get("warning", 0),
        error=counts.get("error", 0),
        critical=counts.get("critical", 0),
    )


@router.get("/audit-logs/warnings-and-errors")
async def get_warnings_and_errors(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    hours: int = Query(default=24, ge=1, le=168),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
):
    """
    Get logs with severity WARNING or higher (Phase 52).

    Useful for monitoring and alerting on issues.

    Admin only endpoint.
    """
    audit_service = get_audit_service()

    entries, total = await audit_service.get_logs_by_severity(
        min_severity=LogSeverity.WARNING,
        hours=hours,
        page=page,
        page_size=page_size,
        session=db,
    )

    log_responses = [
        AuditLogResponse(
            id=entry.id,
            action=entry.action,
            severity=entry.severity,
            user_id=entry.user_id,
            user_email=entry.user_email,
            resource_type=entry.resource_type,
            resource_id=entry.resource_id,
            details=entry.details,
            ip_address=entry.ip_address,
            created_at=entry.created_at,
        )
        for entry in entries
    ]

    offset = (page - 1) * page_size

    return {
        "hours": hours,
        "logs": log_responses,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": (offset + len(entries)) < total,
    }


@router.get("/audit-logs/user/{user_id}")
async def get_user_audit_logs(
    user_id: UUID,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    days: int = Query(default=30, ge=1, le=365),
):
    """
    Get audit logs for a specific user.

    Admin only endpoint.
    """
    audit_service = get_audit_service()

    entries = await audit_service.get_user_activity(
        user_id=str(user_id),
        days=days,
        session=db,
    )

    return {
        "user_id": str(user_id),
        "days": days,
        "entries": [
            AuditLogResponse(
                id=entry.id,
                action=entry.action,
                severity=entry.severity,
                user_id=entry.user_id,
                user_email=entry.user_email,
                resource_type=entry.resource_type,
                resource_id=entry.resource_id,
                details=entry.details,
                ip_address=entry.ip_address,
                created_at=entry.created_at,
            )
            for entry in entries
        ],
    }


@router.get("/audit-logs/security")
async def get_security_events(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    hours: int = Query(default=24, ge=1, le=168),
):
    """
    Get security-relevant audit events (failed logins, access denials).

    Admin only endpoint.
    """
    audit_service = get_audit_service()

    failed_logins = await audit_service.get_failed_logins(hours=hours, session=db)
    access_denials = await audit_service.get_access_denials(hours=hours, session=db)

    return {
        "hours": hours,
        "failed_logins": {
            "count": len(failed_logins),
            "entries": [
                AuditLogResponse(
                    id=entry.id,
                    action=entry.action,
                    severity=entry.severity,
                    user_id=entry.user_id,
                    user_email=entry.user_email,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    details=entry.details,
                    ip_address=entry.ip_address,
                    created_at=entry.created_at,
                )
                for entry in failed_logins
            ],
        },
        "access_denials": {
            "count": len(access_denials),
            "entries": [
                AuditLogResponse(
                    id=entry.id,
                    action=entry.action,
                    severity=entry.severity,
                    user_id=entry.user_id,
                    user_email=entry.user_email,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    details=entry.details,
                    ip_address=entry.ip_address,
                    created_at=entry.created_at,
                )
                for entry in access_denials
            ],
        },
    }


# =============================================================================
# System Stats Endpoint
# =============================================================================

@router.get("/stats")
async def get_system_stats(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get system statistics for the admin dashboard.

    Admin only endpoint.
    """
    from backend.db.models import Document, Chunk

    # Get user count
    user_count_query = select(func.count(User.id))
    user_count_result = await db.execute(user_count_query)
    user_count = user_count_result.scalar() or 0

    # Get active user count
    active_user_query = select(func.count(User.id)).where(User.is_active == True)
    active_user_result = await db.execute(active_user_query)
    active_user_count = active_user_result.scalar() or 0

    # Get document count
    doc_count_query = select(func.count(Document.id))
    doc_count_result = await db.execute(doc_count_query)
    doc_count = doc_count_result.scalar() or 0

    # Get chunk count
    chunk_count_query = select(func.count(Chunk.id))
    chunk_count_result = await db.execute(chunk_count_query)
    chunk_count = chunk_count_result.scalar() or 0

    # Get tier count
    tier_count_query = select(func.count(AccessTier.id))
    tier_count_result = await db.execute(tier_count_query)
    tier_count = tier_count_result.scalar() or 0

    return {
        "users": {
            "total": user_count,
            "active": active_user_count,
            "inactive": user_count - active_user_count,
        },
        "documents": {
            "total": doc_count,
        },
        "chunks": {
            "total": chunk_count,
        },
        "access_tiers": {
            "total": tier_count,
        },
    }


# =============================================================================
# System Settings Endpoints
# =============================================================================

class SettingsUpdateRequest(BaseModel):
    """Request to update system settings."""
    settings: Dict[str, Any]


class SettingsResponse(BaseModel):
    """System settings response."""
    settings: Dict[str, Any]
    definitions: List[Dict[str, Any]]


@router.get("/settings", response_model=SettingsResponse)
async def get_settings(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get all system settings.

    Returns current values and definitions for all settings.
    Admin only endpoint.
    """
    logger.info("Getting system settings", admin_id=admin.user_id)

    settings_service = get_settings_service()
    settings = await settings_service.get_all_settings(db)
    definitions = settings_service.get_setting_definitions()

    return SettingsResponse(
        settings=settings,
        definitions=definitions,
    )


@router.patch("/settings", response_model=SettingsResponse)
async def update_settings(
    update: SettingsUpdateRequest,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update system settings.

    Admin only endpoint.
    """
    logger.info("Updating system settings", admin_id=admin.user_id, keys=list(update.settings.keys()))

    settings_service = get_settings_service()

    # Get old values for audit
    old_settings = await settings_service.get_all_settings(db)

    # Update settings
    new_settings = await settings_service.update_settings(update.settings, db)

    # Log changes
    changes = {}
    for key, new_value in update.settings.items():
        old_value = old_settings.get(key)
        if old_value != new_value:
            changes[key] = {"old": old_value, "new": new_value}

    if changes:
        audit_service = get_audit_service()
        await audit_service.log_admin_action(
            action=AuditAction.SYSTEM_CONFIG_CHANGE,
            admin_user_id=admin.user_id,
            target_resource_type="system_settings",
            changes=changes,
            ip_address=get_client_ip(request),
            session=db,
        )

    definitions = settings_service.get_setting_definitions()

    return SettingsResponse(
        settings=new_settings,
        definitions=definitions,
    )


@router.post("/settings/reset")
async def reset_settings(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Reset all settings to defaults.

    Admin only endpoint.
    """
    logger.info("Resetting system settings to defaults", admin_id=admin.user_id)

    settings_service = get_settings_service()
    settings = await settings_service.reset_to_defaults(db)

    # Log the reset
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="system_settings",
        changes={"action": "reset_to_defaults"},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Settings reset to defaults", "settings": settings}


# =============================================================================
# Settings Presets
# =============================================================================

# Pre-defined presets for quick configuration
SETTINGS_PRESETS = {
    "speed": {
        "name": "Speed",
        "description": "Optimized for fast responses - reduces quality checks and features",
        "settings": {
            "rag.top_k": 5,
            "rag.rerank_results": False,
            "rag.query_expansion_count": 0,
            "rag.verification_enabled": False,
            "rag.graphrag_enabled": False,
            "rag.agentic_enabled": False,
            "rag.hyde_enabled": False,
            "generation.include_images": False,
            "generation.auto_charts": False,
        },
    },
    "quality": {
        "name": "Quality",
        "description": "Optimized for best results - enables all quality features",
        "settings": {
            "rag.top_k": 15,
            "rag.rerank_results": True,
            "rag.query_expansion_count": 3,
            "rag.verification_enabled": True,
            "rag.verification_level": "thorough",
            "rag.graphrag_enabled": True,
            "rag.agentic_enabled": True,
            "rag.hyde_enabled": True,
            "generation.include_images": True,
            "generation.include_sources": True,
        },
    },
    "balanced": {
        "name": "Balanced",
        "description": "Default balanced configuration - good quality with reasonable speed",
        "settings": {
            "rag.top_k": 10,
            "rag.rerank_results": True,
            "rag.query_expansion_count": 2,
            "rag.verification_enabled": True,
            "rag.verification_level": "quick",
            "rag.graphrag_enabled": True,
            "rag.agentic_enabled": False,
            "rag.hyde_enabled": False,
            "generation.include_images": True,
            "generation.include_sources": True,
        },
    },
    "offline": {
        "name": "Offline/Local",
        "description": "Optimized for offline/local-only operation using Ollama",
        "settings": {
            "rag.top_k": 8,
            "rag.rerank_results": False,  # Reranking often needs cloud API
            "rag.query_expansion_count": 1,
            "rag.verification_enabled": False,
            "rag.graphrag_enabled": True,
            "rag.agentic_enabled": False,
            "rag.hyde_enabled": False,
            "generation.include_images": True,
            "generation.image_backend": "picsum",  # Doesn't need API key
        },
    },
}


class PresetInfo(BaseModel):
    """Information about a settings preset."""
    id: str
    name: str
    description: str
    settings: Dict[str, Any]


class PresetsListResponse(BaseModel):
    """List of available presets."""
    presets: List[PresetInfo]


class ApplyPresetResponse(BaseModel):
    """Response after applying a preset."""
    message: str
    preset_id: str
    preset_name: str
    applied_settings: Dict[str, Any]


@router.get("/settings/presets", response_model=PresetsListResponse)
async def list_settings_presets(
    admin: AdminUser,
):
    """
    List all available settings presets.

    Presets are pre-configured bundles of settings optimized for different use cases:
    - speed: Fast responses, minimal processing
    - quality: Best results, all features enabled
    - balanced: Good quality with reasonable speed (default)
    - offline: Optimized for local-only operation

    Admin only endpoint.
    """
    logger.info("Listing settings presets", admin_id=admin.user_id)

    presets = [
        PresetInfo(
            id=preset_id,
            name=preset_data["name"],
            description=preset_data["description"],
            settings=preset_data["settings"],
        )
        for preset_id, preset_data in SETTINGS_PRESETS.items()
    ]

    return PresetsListResponse(presets=presets)


@router.post("/settings/presets/{preset_id}", response_model=ApplyPresetResponse)
async def apply_settings_preset(
    preset_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Apply a settings preset.

    This updates multiple settings at once to match the preset configuration.

    Available presets:
    - speed: Optimized for fast responses
    - quality: Optimized for best results
    - balanced: Default balanced configuration
    - offline: Optimized for local/offline operation

    Admin only endpoint.
    """
    if preset_id not in SETTINGS_PRESETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown preset: {preset_id}. Available: {list(SETTINGS_PRESETS.keys())}",
        )

    preset = SETTINGS_PRESETS[preset_id]
    logger.info(
        "Applying settings preset",
        admin_id=admin.user_id,
        preset_id=preset_id,
        preset_name=preset["name"],
    )

    settings_service = get_settings_service()

    # Get old values for audit
    old_settings = await settings_service.get_all_settings(db)

    # Apply preset settings
    await settings_service.update_settings(preset["settings"], db)

    # Log changes for audit
    changes = {}
    for key, new_value in preset["settings"].items():
        old_value = old_settings.get(key)
        if old_value != new_value:
            changes[key] = {"old": old_value, "new": new_value}

    if changes:
        audit_service = get_audit_service()
        await audit_service.log_admin_action(
            action=AuditAction.SYSTEM_CONFIG_CHANGE,
            admin_user_id=admin.user_id,
            target_resource_type="system_settings",
            changes={"preset_applied": preset_id, "settings_changed": changes},
            ip_address=get_client_ip(request),
            session=db,
        )

    return ApplyPresetResponse(
        message=f"Applied '{preset['name']}' preset successfully",
        preset_id=preset_id,
        preset_name=preset["name"],
        applied_settings=preset["settings"],
    )


# =============================================================================
# OCR Settings Endpoints
# =============================================================================

class OCRSettingsResponse(BaseModel):
    """OCR settings and model information."""
    settings: Dict[str, Any]
    models: Dict[str, Any]


class OCRModelDownloadRequest(BaseModel):
    """OCR model download request."""
    languages: List[str] = Field(..., min_items=1, description="List of language codes")
    variant: str = Field(default="server", pattern="^(server|mobile)$", description="Model variant")


@router.get("/ocr/settings", response_model=OCRSettingsResponse)
async def get_ocr_settings(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get OCR configuration and model information.

    Returns:
        - OCR settings (provider, languages, variants, etc.)
        - Downloaded model information
    """
    logger.info("Fetching OCR settings", admin_id=admin.user_id)

    settings_service = get_settings_service()
    ocr_settings = await settings_service.get_settings_by_category(SettingCategory.OCR)

    # Import OCR manager (lazy import to avoid circular dependencies)
    from backend.services.ocr_manager import OCRManager

    ocr_manager = OCRManager(settings_service)
    model_info = await ocr_manager.get_model_info()

    return OCRSettingsResponse(
        settings=ocr_settings,
        models=model_info,
    )


@router.patch("/ocr/settings")
async def update_ocr_settings(
    updates: Dict[str, Any],
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update OCR settings.

    If languages change, triggers model download automatically.
    """
    logger.info("Updating OCR settings", admin_id=admin.user_id, updates=updates)

    settings_service = get_settings_service()

    # Update each setting
    for key, value in updates.items():
        if not key.startswith("ocr."):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid OCR setting key: {key}. Must start with 'ocr.'"
            )
        await settings_service.update_setting(key, value)

    # If languages changed, trigger model download
    if "ocr.paddle.languages" in updates or "ocr.paddle.variant" in updates:
        from backend.services.ocr_manager import OCRManager

        ocr_manager = OCRManager(settings_service)
        download_result = await ocr_manager.download_models()

        logger.info("Model download triggered", result=download_result)

    # Log the update
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="ocr_settings",
        changes=updates,
        ip_address=get_client_ip(request),
        session=db,
    )

    # Return updated settings
    ocr_settings = await settings_service.get_settings_by_category(SettingCategory.OCR)

    return {"status": "updated", "settings": ocr_settings}


@router.post("/ocr/models/download")
async def download_ocr_models(
    download_request: OCRModelDownloadRequest,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Manually trigger OCR model download for specified languages.

    This endpoint allows downloading models without changing settings.
    """
    logger.info(
        "Downloading OCR models",
        admin_id=admin.user_id,
        languages=download_request.languages,
        variant=download_request.variant,
    )

    from backend.services.ocr_manager import OCRManager

    settings_service = get_settings_service()
    ocr_manager = OCRManager(settings_service)

    result = await ocr_manager.download_models(
        languages=download_request.languages,
        variant=download_request.variant,
    )

    # Log the download
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="ocr_models",
        changes={
            "action": "download",
            "languages": download_request.languages,
            "variant": download_request.variant,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return result


@router.get("/ocr/models/info")
async def get_ocr_model_info(
    admin: AdminUser,
):
    """
    Get information about available and downloaded OCR models.

    Returns model sizes, download status, and supported languages.
    """
    logger.info("Fetching OCR model info", admin_id=admin.user_id)

    from backend.services.ocr_manager import OCRManager

    settings_service = get_settings_service()
    ocr_manager = OCRManager(settings_service)

    model_info = await ocr_manager.get_model_info()

    return model_info


class OCRBatchDownloadRequest(BaseModel):
    """OCR batch model download request."""
    language_batches: List[List[str]] = Field(
        ...,
        min_items=1,
        description="List of language code batches (e.g., [['en', 'de'], ['fr', 'es']])"
    )
    variant: str = Field(default="server", pattern="^(server|mobile)$", description="Model variant")


@router.post("/ocr/models/download-batch")
async def download_ocr_models_batch(
    download_request: OCRBatchDownloadRequest,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Download OCR models in batches with progress tracking.

    This endpoint allows downloading many languages without blocking for too long.
    Useful for initial setup or bulk language additions.
    """
    logger.info(
        "Downloading OCR models in batches",
        admin_id=admin.user_id,
        batches=download_request.language_batches,
        variant=download_request.variant,
    )

    from backend.services.ocr_manager import OCRManager

    settings_service = get_settings_service()
    ocr_manager = OCRManager(settings_service)

    result = await ocr_manager.download_models_batch(
        language_batches=download_request.language_batches,
        variant=download_request.variant,
    )

    # Log the batch download
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="ocr_models",
        changes={
            "action": "batch_download",
            "batches": download_request.language_batches,
            "variant": download_request.variant,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return result


@router.get("/ocr/models/check-updates")
async def check_ocr_model_updates(
    admin: AdminUser,
):
    """
    Check if newer PaddleOCR models are available.

    Returns:
        - Current installed version
        - Latest available version
        - Update availability status
        - Release information
    """
    logger.info("Checking for OCR model updates", admin_id=admin.user_id)

    from backend.services.ocr_manager import OCRManager

    settings_service = get_settings_service()
    ocr_manager = OCRManager(settings_service)

    update_info = await ocr_manager.check_model_updates()

    return update_info


@router.get("/ocr/models/installed")
async def get_installed_ocr_models(
    admin: AdminUser,
):
    """
    Get detailed information about installed OCR models.

    Returns comprehensive metadata including:
        - Model names and paths
        - File sizes
        - Last modification dates
        - Model versions (where available)
    """
    logger.info("Fetching installed OCR models info", admin_id=admin.user_id)

    from backend.services.ocr_manager import OCRManager

    settings_service = get_settings_service()
    ocr_manager = OCRManager(settings_service)

    installed_info = await ocr_manager.get_installed_models_info()

    return installed_info


# =============================================================================
# OCR Metrics Endpoints
# =============================================================================

@router.get("/ocr/metrics/summary")
async def get_ocr_metrics_summary(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
):
    """
    Get aggregated OCR performance metrics.

    Returns summary statistics including:
    - Total operations
    - Success rate
    - Average processing time
    - Total characters processed
    - Cost metrics
    - Fallback usage
    """
    logger.info("Fetching OCR metrics summary", admin_id=admin.user_id, days=days, provider=provider)

    from datetime import timedelta
    from backend.services.ocr_metrics import OCRMetricsService

    metrics_service = OCRMetricsService(db)

    start_date = datetime.utcnow() - timedelta(days=days)
    summary = await metrics_service.get_metrics_summary(
        start_date=start_date,
        provider=provider,
    )

    return summary


@router.get("/ocr/metrics/by-provider")
async def get_ocr_metrics_by_provider(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
):
    """
    Get OCR metrics grouped by provider.

    Returns performance statistics for each OCR provider:
    - PaddleOCR
    - Tesseract
    - Auto (fallback mode)
    """
    logger.info("Fetching OCR metrics by provider", admin_id=admin.user_id, days=days)

    from datetime import timedelta
    from backend.services.ocr_metrics import OCRMetricsService

    metrics_service = OCRMetricsService(db)

    start_date = datetime.utcnow() - timedelta(days=days)
    metrics = await metrics_service.get_metrics_by_provider(
        start_date=start_date,
    )

    return {"providers": metrics}


@router.get("/ocr/metrics/by-language")
async def get_ocr_metrics_by_language(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
):
    """
    Get OCR metrics grouped by language.

    Returns usage statistics for each language:
    - Total operations
    - Average processing time
    """
    logger.info("Fetching OCR metrics by language", admin_id=admin.user_id, days=days)

    from datetime import timedelta
    from backend.services.ocr_metrics import OCRMetricsService

    metrics_service = OCRMetricsService(db)

    start_date = datetime.utcnow() - timedelta(days=days)
    metrics = await metrics_service.get_metrics_by_language(
        start_date=start_date,
    )

    return {"languages": metrics}


@router.get("/ocr/metrics/trend")
async def get_ocr_metrics_trend(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
):
    """
    Get OCR performance trend over time.

    Returns daily performance metrics:
    - Total operations per day
    - Success rate per day
    - Average processing time per day
    """
    logger.info("Fetching OCR metrics trend", admin_id=admin.user_id, days=days, provider=provider)

    from backend.services.ocr_metrics import OCRMetricsService

    metrics_service = OCRMetricsService(db)

    trend = await metrics_service.get_performance_trend(
        days=days,
        provider=provider,
    )

    return {"trend": trend}


@router.get("/ocr/metrics/recent")
async def get_recent_ocr_operations(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to retrieve"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
):
    """
    Get recent OCR operations.

    Returns list of recent OCR operations with full details:
    - Provider and variant used
    - Processing time
    - Success status
    - Error messages (if any)
    - Document context
    """
    logger.info("Fetching recent OCR operations", admin_id=admin.user_id, limit=limit, provider=provider)

    from backend.services.ocr_metrics import OCRMetricsService

    metrics_service = OCRMetricsService(db)

    operations = await metrics_service.get_recent_metrics(
        limit=limit,
        provider=provider,
    )

    # Convert to dict for JSON serialization
    operations_data = []
    for op in operations:
        operations_data.append({
            "id": str(op.id),
            "provider": op.provider,
            "variant": op.variant,
            "language": op.language,
            "processing_time_ms": op.processing_time_ms,
            "page_count": op.page_count,
            "character_count": op.character_count,
            "confidence_score": op.confidence_score,
            "success": op.success,
            "error_message": op.error_message,
            "fallback_used": op.fallback_used,
            "cost_usd": op.cost_usd,
            "created_at": op.created_at.isoformat() if op.created_at else None,
            "document_id": str(op.document_id) if op.document_id else None,
            "user_id": str(op.user_id) if op.user_id else None,
        })

    return {"operations": operations_data}


# =============================================================================
# System Health Endpoint
# =============================================================================

@router.get("/health")
async def get_system_health(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get system health status.

    Checks database connectivity and service availability.
    Admin only endpoint.
    """
    import os
    from datetime import datetime

    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # Check API Server (always online if we reach this point)
    health_status["services"]["api_server"] = {
        "status": "online",
        "message": "API server is running"
    }

    # Check Database
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        db_type = os.getenv("DATABASE_TYPE", "sqlite")
        health_status["services"]["database"] = {
            "status": "connected",
            "type": db_type,
            "message": f"{db_type.upper()} connection successful"
        }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "error",
            "message": str(e)
        }

    # Check Vector Store (ChromaDB)
    try:
        from backend.services.vectorstore_local import ChromaVectorStore
        vs = ChromaVectorStore()
        # Just check if it initializes
        health_status["services"]["vector_store"] = {
            "status": "connected",
            "type": "chromadb",
            "message": "ChromaDB is available"
        }
    except Exception as e:
        health_status["services"]["vector_store"] = {
            "status": "unavailable",
            "message": str(e)
        }

    # Check LLM Configuration - query database for active providers
    try:
        from backend.db.models import LLMProvider
        from sqlalchemy import select

        result = await db.execute(
            select(LLMProvider).where(LLMProvider.is_active == True)
        )
        active_providers = result.scalars().all()

        if active_providers:
            provider_names = [p.name for p in active_providers]
            provider_types = list(set(p.provider_type for p in active_providers))
            health_status["services"]["llm"] = {
                "status": "configured",
                "providers": provider_types,
                "message": f"Configured: {', '.join(provider_names)}"
            }
        else:
            # Fallback to env vars for backwards compatibility
            openai_key = os.getenv("OPENAI_API_KEY", "")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
            llm_status = []
            if openai_key and openai_key != "your-openai-api-key":
                llm_status.append("OpenAI")
            if anthropic_key and anthropic_key != "your-anthropic-api-key":
                llm_status.append("Anthropic")

            if llm_status:
                health_status["services"]["llm"] = {
                    "status": "configured",
                    "providers": llm_status,
                    "message": f"Configured providers: {', '.join(llm_status)}"
                }
            else:
                health_status["services"]["llm"] = {
                    "status": "not_configured",
                    "providers": [],
                    "message": "No LLM providers configured"
                }
    except Exception as e:
        health_status["services"]["llm"] = {
            "status": "error",
            "providers": [],
            "message": f"Error checking LLM config: {str(e)}"
        }

    return health_status


# =============================================================================
# Phase 51: Ray/Distributed Processor Health Endpoint
# =============================================================================

@router.get("/health/ray")
async def get_ray_health(
    admin: AdminUser,
):
    """
    Get Ray distributed computing health status.

    Returns Ray cluster status, worker availability, and actor pool information.
    Admin only endpoint.
    """
    from datetime import datetime

    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "ray_status": "unavailable",
        "ray_initialized": False,
        "nodes": 0,
        "cpus": {"total": 0, "available": 0},
        "gpus": {"total": 0, "available": 0},
        "actor_pools": [],
        "message": "",
    }

    try:
        # Check Ray availability
        from backend.services.distributed_processor import get_distributed_processor

        processor = await get_distributed_processor()
        health = await processor.health_check()

        # Extract Ray-specific info
        ray_info = health.get("backends", {}).get("ray", {})

        health_status.update({
            "ray_status": ray_info.get("status", "unknown"),
            "ray_initialized": ray_info.get("available", False),
            "nodes": ray_info.get("nodes", 0),
            "cpus": ray_info.get("cpus", {"total": 0, "available": 0}),
            "gpus": ray_info.get("gpus", {"total": 0, "available": 0}),
            "actor_pools": health.get("pools", []),
            "message": "Ray cluster connected" if ray_info.get("available") else "Ray not available",
        })

    except Exception as e:
        health_status["message"] = f"Error checking Ray health: {str(e)}"
        logger.warning("Failed to check Ray health", error=str(e))

    return health_status


@router.get("/health/distributed")
async def get_distributed_processor_health(
    admin: AdminUser,
):
    """
    Get distributed processor health status.

    Returns full health information including Ray, Celery, and fallback status.
    Admin only endpoint.
    """
    from datetime import datetime

    try:
        from backend.services.distributed_processor import get_distributed_processor

        processor = await get_distributed_processor()
        health = await processor.health_check()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            **health,
        }

    except Exception as e:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "error",
            "error": str(e),
            "backends": {
                "ray": {"status": "error", "available": False},
                "celery": {"status": "unknown", "available": False},
                "local": {"status": "available", "available": True},
            },
        }


# =============================================================================
# Database Management Endpoints
# =============================================================================

class DatabaseTestRequest(BaseModel):
    """Request to test a database connection."""
    database_url: str = Field(..., description="Database connection URL to test")


class DatabaseSetupRequest(BaseModel):
    """Request to setup PostgreSQL database."""
    database_url: str = Field(..., description="PostgreSQL connection URL")


class DatabaseImportRequest(BaseModel):
    """Request to import data."""
    clear_existing: bool = Field(default=False, description="Clear existing data before import")


class DatabaseInfoResponse(BaseModel):
    """Database information response."""
    type: str
    url_masked: str
    is_connected: bool
    vector_store: str
    documents_count: int
    chunks_count: int
    users_count: int


@router.get("/database/info", response_model=DatabaseInfoResponse)
async def get_database_info(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get current database configuration and statistics.

    Admin only endpoint.
    """
    logger.info("Getting database info", admin_id=admin.user_id)

    manager = get_database_manager(db)
    info = await manager.get_info()

    return DatabaseInfoResponse(**info)


@router.get("/database/index-stats")
async def get_index_stats(
    admin: AdminUser,
):
    """
    Get vector index statistics and pgvector configuration.

    Returns information about HNSW indexes, pgvector version,
    and current optimization settings.

    Admin only endpoint (Phase 57 - Index Optimization).
    """
    from backend.services.vectorstore import VectorStore
    from backend.core.config import get_settings

    logger.info("Getting index stats", admin_id=admin.user_id)

    settings = get_settings()
    vectorstore = VectorStore()
    stats = await vectorstore.get_index_stats()

    # Add configuration info
    stats["config"] = {
        "hnsw_ef_search": settings.HNSW_EF_SEARCH,
        "hnsw_ef_search_high_precision": settings.HNSW_EF_SEARCH_HIGH_PRECISION,
        "pgvector_iterative_scan": settings.PGVECTOR_ITERATIVE_SCAN,
        "index_build_maintenance_work_mem": settings.INDEX_BUILD_MAINTENANCE_WORK_MEM,
        "index_build_parallel_workers": settings.INDEX_BUILD_PARALLEL_WORKERS,
    }

    return stats


@router.post("/database/reindex/{index_name}")
async def reindex_vector_index(
    index_name: str,
    admin: AdminUser,
    concurrent: bool = True,
    request: Request = None,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Rebuild a vector index with optimized settings.

    This applies maintenance_work_mem and parallel workers before reindexing
    for faster index builds. Uses REINDEX CONCURRENTLY by default to avoid
    blocking queries.

    Admin only endpoint (Phase 57 - Index Optimization).

    Args:
        index_name: Name of the index to rebuild (e.g., idx_chunks_embedding_hnsw)
        concurrent: If True, use REINDEX CONCURRENTLY (non-blocking, default)
    """
    from backend.services.vectorstore import VectorStore

    logger.info(
        "Reindexing vector index",
        admin_id=admin.user_id,
        index_name=index_name,
        concurrent=concurrent
    )

    # Validate index name to prevent SQL injection
    allowed_indexes = [
        "idx_chunks_embedding_hnsw",
        "idx_scraped_content_embedding_hnsw",
        "idx_entities_embedding_hnsw",
        "idx_response_cache_embedding_hnsw",
    ]

    if index_name not in allowed_indexes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid index name. Allowed: {', '.join(allowed_indexes)}"
        )

    vectorstore = VectorStore()
    success = await vectorstore.reindex_with_optimization(
        index_name=index_name,
        concurrent=concurrent
    )

    # Log the reindex action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="index",
        changes={
            "action": "reindex",
            "index_name": index_name,
            "concurrent": concurrent,
            "success": success
        },
        ip_address=get_client_ip(request) if request else None,
        session=db,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex {index_name}. Check logs for details."
        )

    return {
        "status": "success",
        "message": f"Successfully reindexed {index_name}",
        "index_name": index_name,
        "concurrent": concurrent
    }


@router.post("/database/test")
async def test_database_connection(
    request: DatabaseTestRequest,
    admin: AdminUser,
):
    """
    Test a database connection string.

    Admin only endpoint.
    """
    logger.info("Testing database connection", admin_id=admin.user_id)

    manager = get_database_manager()
    result = await manager.test_connection(request.database_url)

    return result


@router.post("/database/export")
async def export_database(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Export all data to JSON format.

    Admin only endpoint.
    """
    from fastapi.responses import JSONResponse

    logger.info("Exporting database", admin_id=admin.user_id)

    manager = get_database_manager(db)
    data = await manager.export_data()

    # Log the export action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="database",
        changes={"action": "export", "records": sum(len(v) for k, v in data.items() if isinstance(v, list))},
        ip_address=get_client_ip(request),
        session=db,
    )

    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f"attachment; filename=aidocindexer_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )


@router.post("/database/import")
async def import_database(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
    clear_existing: bool = Query(default=False, description="Clear existing data before import"),
):
    """
    Import data from JSON export.

    Send JSON data in the request body.
    Admin only endpoint.
    """
    logger.info("Importing database", admin_id=admin.user_id, clear_existing=clear_existing)

    # Get JSON data from request body
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON data: {str(e)}"
        )

    manager = get_database_manager(db)
    result = await manager.import_data(data, clear_existing=clear_existing)

    # Log the import action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="database",
        changes={
            "action": "import",
            "success": result["success"],
            "imported": result["imported"],
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return result


@router.post("/database/setup")
async def setup_postgresql(
    request_data: DatabaseSetupRequest,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Setup PostgreSQL database with pgvector extension.

    Creates the pgvector extension and all required tables.
    Admin only endpoint.
    """
    logger.info("Setting up PostgreSQL database", admin_id=admin.user_id)

    manager = get_database_manager()
    result = await manager.setup_postgresql(request_data.database_url)

    # Log the setup action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="database",
        changes={
            "action": "postgresql_setup",
            "success": result["success"],
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return result


@router.get("/database/migration-instructions")
async def get_migration_instructions(
    admin: AdminUser,
    from_type: str = Query(..., description="Current database type (sqlite, postgresql)"),
    to_type: str = Query(..., description="Target database type (sqlite, postgresql)"),
):
    """
    Get step-by-step migration instructions.

    Admin only endpoint.
    """
    manager = get_database_manager()
    instructions = manager.get_migration_instructions(from_type, to_type)

    return instructions


# =============================================================================
# LLM Provider Management Endpoints
# =============================================================================

class LLMProviderCreate(BaseModel):
    """Request to create an LLM provider."""
    name: str = Field(..., min_length=1, max_length=100)
    provider_type: str = Field(..., description="Provider type (openai, anthropic, ollama, etc.)")
    api_key: Optional[str] = Field(None, description="API key (will be encrypted)")
    api_base_url: Optional[str] = Field(None, description="Custom API base URL")
    organization_id: Optional[str] = Field(None, description="Organization ID (for OpenAI)")
    default_chat_model: Optional[str] = Field(None, description="Default chat model")
    default_embedding_model: Optional[str] = Field(None, description="Default embedding model")
    is_default: bool = Field(False, description="Set as default provider")
    settings: Optional[Dict[str, Any]] = Field(None, description="Additional settings")


class LLMProviderUpdate(BaseModel):
    """Request to update an LLM provider."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    api_key: Optional[str] = Field(None, description="API key (will be encrypted)")
    api_base_url: Optional[str] = Field(None, description="Custom API base URL")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    default_chat_model: Optional[str] = Field(None, description="Default chat model")
    default_embedding_model: Optional[str] = Field(None, description="Default embedding model")
    is_active: Optional[bool] = Field(None, description="Active status")
    settings: Optional[Dict[str, Any]] = Field(None, description="Additional settings")


@router.get("/llm/providers")
async def list_llm_providers(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all configured LLM providers.

    Admin only endpoint.
    """
    logger.info("Listing LLM providers", admin_id=admin.user_id)

    providers = await LLMProviderService.list_providers(db)
    return {
        "providers": [
            LLMProviderService.format_provider_response(p) for p in providers
        ],
        "total": len(providers),
    }


@router.get("/llm/provider-types")
async def get_llm_provider_types(admin: AdminUser):
    """
    Get all supported LLM provider types and their configurations.

    Admin only endpoint.
    """
    return {
        "provider_types": LLMProviderService.get_provider_types(),
    }


@router.get("/llm/ollama-context-length")
async def get_ollama_context_length(
    admin: AdminUser,
    model_name: Optional[str] = Query(None, description="Model name (defaults to configured chat model)"),
    base_url: str = Query("http://localhost:11434", description="Ollama API base URL"),
):
    """
    Get the maximum context length supported by an Ollama model.

    Returns the model's native context_length from its metadata.
    Useful for knowing the max value for the llm.context_window setting.
    """
    from backend.services.llm import get_ollama_model_context_length

    result = await get_ollama_model_context_length(model_name, base_url)
    return result


@router.get("/llm/model-context-recommendations")
async def get_model_context_recommendations(
    admin: AdminUser,
    base_url: str = Query("http://localhost:11434", description="Ollama API base URL"),
):
    """
    Get context window recommendations for all installed Ollama models.

    For each installed model, returns the research-backed recommended context window,
    max context, VRAM estimate, user override (if any), and the effective value
    that would be used at inference time.
    """
    from backend.services.llm import (
        list_ollama_models as get_ollama_models,
        get_recommended_context_window,
        MODEL_CONTEXT_RECOMMENDATIONS,
    )
    from backend.services.settings import get_settings_service

    settings_svc = get_settings_service()

    overrides = await settings_svc.get_setting("llm.model_context_overrides") or {}
    if not isinstance(overrides, dict):
        overrides = {}
    global_ctx = await settings_svc.get_setting("llm.context_window") or 4096

    # Get installed models from Ollama
    models_result = await get_ollama_models(base_url)
    chat_models = models_result.get("chat_models", [])

    models_info = []
    for m in chat_models:
        model_name = m.get("name", "")
        rec = get_recommended_context_window(model_name)

        # Determine effective context window (same resolution as llm.py)
        override_value = overrides.get(model_name)
        if override_value is not None:
            effective = int(override_value)
            source = "override"
        elif rec:
            effective = rec["recommended"]
            source = "recommendation"
        else:
            effective = int(global_ctx)
            source = "global"

        models_info.append({
            "model_name": model_name,
            "parameter_size": m.get("parameter_size", ""),
            "family": m.get("family", ""),
            "recommended": rec["recommended"] if rec else None,
            "max": rec["max"] if rec else None,
            "vram": rec["vram"] if rec else None,
            "override": int(override_value) if override_value is not None else None,
            "effective": effective,
            "source": source,
        })

    return {
        "success": True,
        "models": models_info,
        "global_context_window": int(global_ctx),
        "overrides": overrides,
    }


@router.get("/llm/ollama-models")
async def list_ollama_models(
    admin: AdminUser,
    base_url: str = Query("http://localhost:11434", description="Ollama API base URL"),
):
    """
    List available models from a local Ollama instance.

    This endpoint allows fetching models before creating a provider.
    Admin only endpoint.
    """
    from backend.services.llm import list_ollama_models as get_ollama_models

    result = await get_ollama_models(base_url)
    return result


class OllamaModelRequest(BaseModel):
    """Request body for Ollama model operations."""
    model_name: str = Field(..., description="Model name to pull (e.g., 'qwen2.5vl', 'llava:7b')")
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")


@router.post("/llm/ollama-models/pull")
async def pull_ollama_model(
    request: OllamaModelRequest,
    admin: AdminUser,
):
    """
    Pull (download) an Ollama model.

    This starts the model download. For large models, this may take several minutes.
    Admin only endpoint.
    """
    from backend.services.llm import pull_ollama_model as do_pull

    logger.info(
        "Admin pulling Ollama model",
        admin_id=admin.user_id,
        model_name=request.model_name,
        base_url=request.base_url,
    )

    result = await do_pull(request.model_name, request.base_url)
    return result


@router.delete("/llm/ollama-models/{model_name}")
async def delete_ollama_model(
    model_name: str,
    admin: AdminUser,
    base_url: str = Query("http://localhost:11434", description="Ollama API base URL"),
):
    """
    Delete a local Ollama model.

    Admin only endpoint.
    """
    from backend.services.llm import delete_ollama_model as do_delete

    logger.info(
        "Admin deleting Ollama model",
        admin_id=admin.user_id,
        model_name=model_name,
        base_url=base_url,
    )

    result = await do_delete(model_name, base_url)
    return result


@router.get("/llm/providers/{provider_id}")
async def get_llm_provider(
    provider_id: str,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific LLM provider by ID.

    Admin only endpoint.
    """
    provider = await LLMProviderService.get_provider(db, provider_id)

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found",
        )

    return LLMProviderService.format_provider_response(provider)


@router.post("/llm/providers", status_code=status.HTTP_201_CREATED)
async def create_llm_provider(
    provider_data: LLMProviderCreate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new LLM provider configuration.

    Admin only endpoint.
    """
    logger.info("Creating LLM provider", admin_id=admin.user_id, provider_type=provider_data.provider_type)

    # Check for duplicate name
    existing = await LLMProviderService.get_provider_by_name(db, provider_data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider with name '{provider_data.name}' already exists",
        )

    try:
        provider = await LLMProviderService.create_provider(
            session=db,
            name=provider_data.name,
            provider_type=provider_data.provider_type,
            api_key=provider_data.api_key,
            api_base_url=provider_data.api_base_url,
            organization_id=provider_data.organization_id,
            default_chat_model=provider_data.default_chat_model,
            default_embedding_model=provider_data.default_embedding_model,
            is_default=provider_data.is_default,
            settings=provider_data.settings,
        )

        # Log the action
        audit_service = get_audit_service()
        await audit_service.log_admin_action(
            action=AuditAction.SYSTEM_CONFIG_CHANGE,
            admin_user_id=admin.user_id,
            target_resource_type="llm_provider",
            target_resource_id=str(provider.id),
            changes={"action": "create", "name": provider_data.name, "type": provider_data.provider_type},
            ip_address=get_client_ip(request),
            session=db,
        )

        return LLMProviderService.format_provider_response(provider)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.patch("/llm/providers/{provider_id}")
async def update_llm_provider(
    provider_id: str,
    update_data: LLMProviderUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update an LLM provider configuration.

    Admin only endpoint.
    """
    logger.info("Updating LLM provider", admin_id=admin.user_id, provider_id=provider_id)

    # Check name uniqueness if updating
    if update_data.name:
        existing = await LLMProviderService.get_provider_by_name(db, update_data.name)
        if existing and str(existing.id) != provider_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider with name '{update_data.name}' already exists",
            )

    provider = await LLMProviderService.update_provider(
        session=db,
        provider_id=provider_id,
        name=update_data.name,
        api_key=update_data.api_key,
        api_base_url=update_data.api_base_url,
        organization_id=update_data.organization_id,
        default_chat_model=update_data.default_chat_model,
        default_embedding_model=update_data.default_embedding_model,
        is_active=update_data.is_active,
        settings=update_data.settings,
    )

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="llm_provider",
        target_resource_id=provider_id,
        changes={"action": "update"},
        ip_address=get_client_ip(request),
        session=db,
    )

    return LLMProviderService.format_provider_response(provider)


@router.delete("/llm/providers/{provider_id}")
async def delete_llm_provider(
    provider_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete an LLM provider.

    Admin only endpoint.
    """
    logger.info("Deleting LLM provider", admin_id=admin.user_id, provider_id=provider_id)

    # Get provider for audit log
    provider = await LLMProviderService.get_provider(db, provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found",
        )

    if provider.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the default provider. Set another provider as default first.",
        )

    provider_name = provider.name
    deleted = await LLMProviderService.delete_provider(db, provider_id)

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="llm_provider",
        target_resource_id=provider_id,
        changes={"action": "delete", "name": provider_name},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "LLM provider deleted successfully", "provider_id": provider_id}


@router.post("/llm/providers/{provider_id}/test")
async def test_llm_provider(
    provider_id: str,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Test connection to an LLM provider.

    Admin only endpoint.
    """
    logger.info("Testing LLM provider", admin_id=admin.user_id, provider_id=provider_id)

    result = await LLMProviderService.test_provider(db, provider_id)
    return result


@router.post("/llm/providers/{provider_id}/default")
async def set_default_llm_provider(
    provider_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Set an LLM provider as the default.

    Admin only endpoint.
    """
    logger.info("Setting default LLM provider", admin_id=admin.user_id, provider_id=provider_id)

    provider = await LLMProviderService.set_default_provider(db, provider_id)

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM provider not found",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="llm_provider",
        target_resource_id=provider_id,
        changes={"action": "set_default", "name": provider.name},
        ip_address=get_client_ip(request),
        session=db,
    )

    return LLMProviderService.format_provider_response(provider)


@router.get("/llm/providers/{provider_id}/models")
async def list_llm_provider_models(
    provider_id: str,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List available models for a provider.

    Admin only endpoint.
    """
    result = await LLMProviderService.list_available_models(db, provider_id)
    return result


# =============================================================================
# Database Connection Management Endpoints
# =============================================================================

class DatabaseConnectionCreate(BaseModel):
    """Request to create a database connection."""
    name: str = Field(..., min_length=1, max_length=100)
    db_type: str = Field(..., description="Database type (sqlite, postgresql, mysql)")
    database: str = Field(..., description="Database name or path")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password (will be encrypted)")
    vector_store: str = Field("auto", description="Vector store type (auto, pgvector, chromadb)")
    is_active: bool = Field(False, description="Set as active connection")
    connection_options: Optional[Dict[str, Any]] = Field(None, description="Additional options")


class DatabaseConnectionUpdate(BaseModel):
    """Request to update a database connection."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    database: Optional[str] = Field(None, description="Database name or path")
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    vector_store: Optional[str] = Field(None, description="Vector store type")
    connection_options: Optional[Dict[str, Any]] = Field(None, description="Additional options")


@router.get("/database/connections")
async def list_database_connections(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all saved database connections.

    Admin only endpoint.
    """
    logger.info("Listing database connections", admin_id=admin.user_id)

    connections = await DatabaseConnectionService.list_connections(db)
    return {
        "connections": [
            DatabaseConnectionService.format_connection_response(c) for c in connections
        ],
        "total": len(connections),
    }


@router.get("/database/connection-types")
async def get_database_connection_types(admin: AdminUser):
    """
    Get all supported database types and their configurations.

    Admin only endpoint.
    """
    return {
        "database_types": DatabaseConnectionService.get_database_types(),
    }


@router.get("/database/connections/{connection_id}")
async def get_database_connection(
    connection_id: str,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific database connection by ID.

    Admin only endpoint.
    """
    connection = await DatabaseConnectionService.get_connection(db, connection_id)

    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    return DatabaseConnectionService.format_connection_response(connection)


@router.post("/database/connections", status_code=status.HTTP_201_CREATED)
async def create_database_connection(
    connection_data: DatabaseConnectionCreate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new database connection configuration.

    Admin only endpoint.
    """
    logger.info("Creating database connection", admin_id=admin.user_id, db_type=connection_data.db_type)

    # Check for duplicate name
    existing = await DatabaseConnectionService.get_connection_by_name(db, connection_data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connection with name '{connection_data.name}' already exists",
        )

    try:
        connection = await DatabaseConnectionService.create_connection(
            session=db,
            name=connection_data.name,
            db_type=connection_data.db_type,
            database=connection_data.database,
            host=connection_data.host,
            port=connection_data.port,
            username=connection_data.username,
            password=connection_data.password,
            vector_store=connection_data.vector_store,
            is_active=connection_data.is_active,
            connection_options=connection_data.connection_options,
        )

        # Log the action
        audit_service = get_audit_service()
        await audit_service.log_admin_action(
            action=AuditAction.SYSTEM_CONFIG_CHANGE,
            admin_user_id=admin.user_id,
            target_resource_type="database_connection",
            target_resource_id=str(connection.id),
            changes={"action": "create", "name": connection_data.name, "type": connection_data.db_type},
            ip_address=get_client_ip(request),
            session=db,
        )

        return DatabaseConnectionService.format_connection_response(connection)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.patch("/database/connections/{connection_id}")
async def update_database_connection(
    connection_id: str,
    update_data: DatabaseConnectionUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a database connection configuration.

    Admin only endpoint.
    """
    logger.info("Updating database connection", admin_id=admin.user_id, connection_id=connection_id)

    # Check name uniqueness if updating
    if update_data.name:
        existing = await DatabaseConnectionService.get_connection_by_name(db, update_data.name)
        if existing and str(existing.id) != connection_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Connection with name '{update_data.name}' already exists",
            )

    connection = await DatabaseConnectionService.update_connection(
        session=db,
        connection_id=connection_id,
        name=update_data.name,
        host=update_data.host,
        port=update_data.port,
        database=update_data.database,
        username=update_data.username,
        password=update_data.password,
        vector_store=update_data.vector_store,
        connection_options=update_data.connection_options,
    )

    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="database_connection",
        target_resource_id=connection_id,
        changes={"action": "update"},
        ip_address=get_client_ip(request),
        session=db,
    )

    return DatabaseConnectionService.format_connection_response(connection)


@router.delete("/database/connections/{connection_id}")
async def delete_database_connection(
    connection_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a database connection.

    Cannot delete the active connection.
    Admin only endpoint.
    """
    logger.info("Deleting database connection", admin_id=admin.user_id, connection_id=connection_id)

    # Get connection for audit log
    connection = await DatabaseConnectionService.get_connection(db, connection_id)
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    connection_name = connection.name

    try:
        deleted = await DatabaseConnectionService.delete_connection(db, connection_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="database_connection",
        target_resource_id=connection_id,
        changes={"action": "delete", "name": connection_name},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Database connection deleted successfully", "connection_id": connection_id}


@router.post("/database/connections/{connection_id}/test")
async def test_saved_database_connection(
    connection_id: str,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Test a saved database connection.

    Admin only endpoint.
    """
    logger.info("Testing database connection", admin_id=admin.user_id, connection_id=connection_id)

    result = await DatabaseConnectionService.test_saved_connection(db, connection_id)
    return result


@router.post("/database/connections/{connection_id}/activate")
async def activate_database_connection(
    connection_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Set a database connection as active.

    Note: This only marks the connection as active in the database.
    To actually switch databases, you need to restart the application
    with the new DATABASE_URL environment variable.

    Admin only endpoint.
    """
    logger.info("Activating database connection", admin_id=admin.user_id, connection_id=connection_id)

    connection = await DatabaseConnectionService.set_active_connection(db, connection_id)

    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="database_connection",
        target_resource_id=connection_id,
        changes={"action": "activate", "name": connection.name},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {
        "message": "Database connection activated. Restart the application to use this connection.",
        "connection": DatabaseConnectionService.format_connection_response(connection),
    }


# =============================================================================
# LLM Operation Configuration Endpoints
# =============================================================================

class LLMOperationConfigCreate(BaseModel):
    """Request to set operation-level LLM configuration."""
    operation_type: str = Field(..., description="Operation type (chat, embeddings, document_processing, rag)")
    provider_id: Optional[str] = Field(None, description="Provider ID to use for this operation")
    model_override: Optional[str] = Field(None, description="Model to use (overrides provider default)")
    temperature_override: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature override")
    max_tokens_override: Optional[int] = Field(None, ge=1, description="Max tokens override")
    fallback_provider_id: Optional[str] = Field(None, description="Fallback provider ID")


class LLMOperationConfigResponse(BaseModel):
    """Operation config response."""
    id: str
    operation_type: str
    provider_id: Optional[str]
    provider_name: Optional[str]
    model_override: Optional[str]
    temperature_override: Optional[float]
    max_tokens_override: Optional[int]
    fallback_provider_id: Optional[str]
    fallback_provider_name: Optional[str]
    created_at: datetime
    updated_at: datetime


VALID_OPERATIONS = [
    "chat",
    "embeddings",
    "document_processing",
    "rag",
    "summarization",
    "document_enhancement",
    "auto_tagging",
    "content_generation",
    "collaboration",
    "web_scraping",
    "agent_planning",
    "agent_execution",
    "audio_script",
    "knowledge_graph",
]


@router.get("/llm/operations")
async def list_llm_operations(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all operation-level LLM configurations.

    Admin only endpoint.
    """
    from backend.db.models import LLMOperationConfig, LLMProvider

    logger.info("Listing LLM operation configs", admin_id=admin.user_id)

    query = (
        select(LLMOperationConfig)
        .options(
            selectinload(LLMOperationConfig.provider),
            selectinload(LLMOperationConfig.fallback_provider),
        )
        .order_by(LLMOperationConfig.operation_type)
    )
    result = await db.execute(query)
    configs = result.scalars().all()

    return {
        "operations": [
            LLMOperationConfigResponse(
                id=str(c.id),
                operation_type=c.operation_type,
                provider_id=str(c.provider_id) if c.provider_id else None,
                provider_name=c.provider.name if c.provider else None,
                model_override=c.model_override,
                temperature_override=c.temperature_override,
                max_tokens_override=c.max_tokens_override,
                fallback_provider_id=str(c.fallback_provider_id) if c.fallback_provider_id else None,
                fallback_provider_name=c.fallback_provider.name if c.fallback_provider else None,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in configs
        ],
        "valid_operations": VALID_OPERATIONS,
        "total": len(configs),
    }


@router.put("/llm/operations/{operation_type}")
async def set_llm_operation_config(
    operation_type: str,
    config_data: LLMOperationConfigCreate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Set LLM configuration for a specific operation.

    Admin only endpoint.
    """
    from backend.db.models import LLMOperationConfig, LLMProvider
    from backend.services.llm import LLMConfigManager

    if operation_type not in VALID_OPERATIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid operation type. Valid types: {VALID_OPERATIONS}",
        )

    logger.info("Setting LLM operation config", admin_id=admin.user_id, operation=operation_type)

    # Verify provider exists if provided
    if config_data.provider_id:
        provider_query = select(LLMProvider).where(LLMProvider.id == config_data.provider_id)
        provider_result = await db.execute(provider_query)
        if not provider_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provider not found",
            )

    # Verify fallback provider if provided
    if config_data.fallback_provider_id:
        fallback_query = select(LLMProvider).where(LLMProvider.id == config_data.fallback_provider_id)
        fallback_result = await db.execute(fallback_query)
        if not fallback_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Fallback provider not found",
            )

    # Check if config exists
    existing_query = select(LLMOperationConfig).where(LLMOperationConfig.operation_type == operation_type)
    existing_result = await db.execute(existing_query)
    existing = existing_result.scalar_one_or_none()

    if existing:
        # Update existing
        existing.provider_id = config_data.provider_id
        existing.model_override = config_data.model_override
        existing.temperature_override = config_data.temperature_override
        existing.max_tokens_override = config_data.max_tokens_override
        existing.fallback_provider_id = config_data.fallback_provider_id
        await db.commit()
        await db.refresh(existing)
        config = existing
    else:
        # Create new
        config = LLMOperationConfig(
            operation_type=operation_type,
            provider_id=config_data.provider_id,
            model_override=config_data.model_override,
            temperature_override=config_data.temperature_override,
            max_tokens_override=config_data.max_tokens_override,
            fallback_provider_id=config_data.fallback_provider_id,
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)

    # Invalidate cache
    await LLMConfigManager.invalidate_cache(f"operation:{operation_type}")

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="llm_operation_config",
        target_resource_id=operation_type,
        changes={
            "action": "set",
            "operation": operation_type,
            "provider_id": config_data.provider_id,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    # Reload with relationships
    query = (
        select(LLMOperationConfig)
        .where(LLMOperationConfig.id == config.id)
        .options(
            selectinload(LLMOperationConfig.provider),
            selectinload(LLMOperationConfig.fallback_provider),
        )
    )
    result = await db.execute(query)
    config = result.scalar_one()

    return LLMOperationConfigResponse(
        id=str(config.id),
        operation_type=config.operation_type,
        provider_id=str(config.provider_id) if config.provider_id else None,
        provider_name=config.provider.name if config.provider else None,
        model_override=config.model_override,
        temperature_override=config.temperature_override,
        max_tokens_override=config.max_tokens_override,
        fallback_provider_id=str(config.fallback_provider_id) if config.fallback_provider_id else None,
        fallback_provider_name=config.fallback_provider.name if config.fallback_provider else None,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@router.delete("/llm/operations/{operation_type}")
async def delete_llm_operation_config(
    operation_type: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete operation-level LLM configuration (reset to default).

    Admin only endpoint.
    """
    from backend.db.models import LLMOperationConfig
    from backend.services.llm import LLMConfigManager

    if operation_type not in VALID_OPERATIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid operation type. Valid types: {VALID_OPERATIONS}",
        )

    logger.info("Deleting LLM operation config", admin_id=admin.user_id, operation=operation_type)

    query = select(LLMOperationConfig).where(LLMOperationConfig.operation_type == operation_type)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Operation config not found",
        )

    await db.delete(config)
    await db.commit()

    # Invalidate cache
    await LLMConfigManager.invalidate_cache(f"operation:{operation_type}")

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="llm_operation_config",
        target_resource_id=operation_type,
        changes={"action": "delete", "operation": operation_type},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": f"Operation config for '{operation_type}' deleted. Will use default provider."}


# =============================================================================
# LLM Usage Analytics Endpoints
# =============================================================================

class UsageSummaryResponse(BaseModel):
    """Usage summary response."""
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    avg_duration_ms: float


class UsageByProviderResponse(BaseModel):
    """Usage grouped by provider."""
    provider_id: Optional[str]
    provider_type: str
    provider_name: Optional[str]
    request_count: int
    total_tokens: int
    total_cost_usd: float


class UsageByOperationResponse(BaseModel):
    """Usage grouped by operation."""
    operation_type: str
    request_count: int
    total_tokens: int
    total_cost_usd: float


@router.get("/llm/usage", response_model=UsageSummaryResponse)
async def get_llm_usage_summary(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    provider_id: Optional[str] = Query(None, description="Filter by provider"),
    operation_type: Optional[str] = Query(None, description="Filter by operation"),
):
    """
    Get overall LLM usage summary.

    Admin only endpoint.
    """
    from backend.services.llm import LLMUsageTracker

    logger.info("Getting LLM usage summary", admin_id=admin.user_id)

    summary = await LLMUsageTracker.get_usage_summary(
        provider_id=provider_id,
        operation_type=operation_type,
        start_date=start_date,
        end_date=end_date,
    )

    return UsageSummaryResponse(**summary)


@router.get("/llm/usage/by-provider")
async def get_llm_usage_by_provider(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
):
    """
    Get LLM usage grouped by provider.

    Admin only endpoint.
    """
    from backend.db.models import LLMUsageLog, LLMProvider

    logger.info("Getting LLM usage by provider", admin_id=admin.user_id)

    query = (
        select(
            LLMUsageLog.provider_id,
            LLMUsageLog.provider_type,
            func.count(LLMUsageLog.id).label("request_count"),
            func.sum(LLMUsageLog.total_tokens).label("total_tokens"),
            func.sum(LLMUsageLog.total_cost_usd).label("total_cost_usd"),
        )
        .group_by(LLMUsageLog.provider_id, LLMUsageLog.provider_type)
    )

    if start_date:
        query = query.where(LLMUsageLog.created_at >= start_date)
    if end_date:
        query = query.where(LLMUsageLog.created_at <= end_date)

    result = await db.execute(query)
    rows = result.all()

    # Get provider names
    provider_ids = [r.provider_id for r in rows if r.provider_id]
    provider_names = {}
    if provider_ids:
        providers_query = select(LLMProvider.id, LLMProvider.name).where(LLMProvider.id.in_(provider_ids))
        providers_result = await db.execute(providers_query)
        provider_names = {str(p.id): p.name for p in providers_result.all()}

    return {
        "usage_by_provider": [
            UsageByProviderResponse(
                provider_id=str(r.provider_id) if r.provider_id else None,
                provider_type=r.provider_type,
                provider_name=provider_names.get(str(r.provider_id)) if r.provider_id else None,
                request_count=r.request_count or 0,
                total_tokens=r.total_tokens or 0,
                total_cost_usd=float(r.total_cost_usd or 0),
            )
            for r in rows
        ],
    }


@router.get("/llm/usage/by-operation")
async def get_llm_usage_by_operation(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
):
    """
    Get LLM usage grouped by operation type.

    Admin only endpoint.
    """
    from backend.db.models import LLMUsageLog

    logger.info("Getting LLM usage by operation", admin_id=admin.user_id)

    query = (
        select(
            LLMUsageLog.operation_type,
            func.count(LLMUsageLog.id).label("request_count"),
            func.sum(LLMUsageLog.total_tokens).label("total_tokens"),
            func.sum(LLMUsageLog.total_cost_usd).label("total_cost_usd"),
        )
        .group_by(LLMUsageLog.operation_type)
    )

    if start_date:
        query = query.where(LLMUsageLog.created_at >= start_date)
    if end_date:
        query = query.where(LLMUsageLog.created_at <= end_date)

    result = await db.execute(query)
    rows = result.all()

    return {
        "usage_by_operation": [
            UsageByOperationResponse(
                operation_type=r.operation_type,
                request_count=r.request_count or 0,
                total_tokens=r.total_tokens or 0,
                total_cost_usd=float(r.total_cost_usd or 0),
            )
            for r in rows
        ],
    }


@router.get("/llm/usage/recent")
async def get_recent_llm_usage(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    limit: int = Query(50, ge=1, le=500, description="Number of records to return"),
):
    """
    Get recent LLM usage logs.

    Admin only endpoint.
    """
    from backend.db.models import LLMUsageLog, LLMProvider, User

    logger.info("Getting recent LLM usage", admin_id=admin.user_id, limit=limit)

    query = (
        select(LLMUsageLog)
        .options(
            selectinload(LLMUsageLog.provider),
            selectinload(LLMUsageLog.user),
        )
        .order_by(desc(LLMUsageLog.created_at))
        .limit(limit)
    )

    result = await db.execute(query)
    logs = result.scalars().all()

    return {
        "logs": [
            {
                "id": str(log.id),
                "provider_type": log.provider_type,
                "provider_name": log.provider.name if log.provider else None,
                "model": log.model,
                "operation_type": log.operation_type,
                "user_email": log.user.email if log.user else None,
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "total_tokens": log.total_tokens,
                "total_cost_usd": log.total_cost_usd,
                "request_duration_ms": log.request_duration_ms,
                "success": log.success,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
        "total": len(logs),
    }


@router.post("/llm/cache/invalidate")
async def invalidate_llm_cache(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Invalidate LLM configuration cache.

    Use this after making configuration changes to force immediate effect.
    Admin only endpoint.
    """
    from backend.services.llm import LLMConfigManager, LLMFactory

    logger.info("Invalidating LLM cache", admin_id=admin.user_id)

    # Clear both config cache and model instance cache
    await LLMConfigManager.invalidate_cache()
    LLMFactory.clear_cache()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="llm_cache",
        changes={"action": "invalidate"},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "LLM cache invalidated successfully"}


# =============================================================================
# Rate Limit Management Endpoints
# =============================================================================

class RateLimitConfigCreate(BaseModel):
    """Request to set rate limit configuration for a tier."""
    requests_per_minute: int = Field(60, ge=1, description="Requests per minute")
    requests_per_hour: int = Field(1000, ge=1, description="Requests per hour")
    requests_per_day: int = Field(10000, ge=1, description="Requests per day")
    tokens_per_minute: int = Field(100000, ge=1, description="Tokens per minute")
    tokens_per_day: int = Field(1000000, ge=1, description="Tokens per day")
    operation_limits: Optional[Dict[str, int]] = Field(None, description="Per-operation limits")


class RateLimitConfigResponse(BaseModel):
    """Rate limit configuration response."""
    id: str
    tier_id: str
    tier_name: Optional[str]
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    tokens_per_minute: int
    tokens_per_day: int
    operation_limits: Optional[Dict[str, int]]
    created_at: datetime
    updated_at: datetime


class RateLimitUsageResponse(BaseModel):
    """Rate limit usage response."""
    user_id: str
    user_email: str
    tier_name: str
    requests: Dict[str, Dict[str, int]]
    tokens: Dict[str, Dict[str, int]]


@router.get("/rate-limits")
async def list_rate_limit_configs(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all rate limit configurations by tier.

    Admin only endpoint.
    """
    from backend.db.models import RateLimitConfig

    logger.info("Listing rate limit configs", admin_id=admin.user_id)

    query = (
        select(RateLimitConfig)
        .options(selectinload(RateLimitConfig.tier))
        .order_by(RateLimitConfig.created_at)
    )
    result = await db.execute(query)
    configs = result.scalars().all()

    return {
        "configs": [
            RateLimitConfigResponse(
                id=str(c.id),
                tier_id=str(c.tier_id),
                tier_name=c.tier.name if c.tier else None,
                requests_per_minute=c.requests_per_minute,
                requests_per_hour=c.requests_per_hour,
                requests_per_day=c.requests_per_day,
                tokens_per_minute=c.tokens_per_minute,
                tokens_per_day=c.tokens_per_day,
                operation_limits=c.operation_limits,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in configs
        ],
        "total": len(configs),
    }


@router.put("/rate-limits/{tier_id}")
async def set_rate_limit_config(
    tier_id: str,
    config_data: RateLimitConfigCreate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Set rate limit configuration for an access tier.

    Admin only endpoint.
    """
    from backend.db.models import RateLimitConfig
    from backend.api.middleware.rate_limit import clear_tier_cache

    logger.info("Setting rate limit config", admin_id=admin.user_id, tier_id=tier_id)

    # Verify tier exists
    tier_query = select(AccessTier).where(AccessTier.id == tier_id)
    tier_result = await db.execute(tier_query)
    tier = tier_result.scalar_one_or_none()

    if not tier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access tier not found",
        )

    # Check if config exists
    existing_query = select(RateLimitConfig).where(RateLimitConfig.tier_id == tier_id)
    existing_result = await db.execute(existing_query)
    existing = existing_result.scalar_one_or_none()

    if existing:
        existing.requests_per_minute = config_data.requests_per_minute
        existing.requests_per_hour = config_data.requests_per_hour
        existing.requests_per_day = config_data.requests_per_day
        existing.tokens_per_minute = config_data.tokens_per_minute
        existing.tokens_per_day = config_data.tokens_per_day
        existing.operation_limits = config_data.operation_limits
        await db.commit()
        await db.refresh(existing)
        config = existing
    else:
        config = RateLimitConfig(
            tier_id=tier_id,
            requests_per_minute=config_data.requests_per_minute,
            requests_per_hour=config_data.requests_per_hour,
            requests_per_day=config_data.requests_per_day,
            tokens_per_minute=config_data.tokens_per_minute,
            tokens_per_day=config_data.tokens_per_day,
            operation_limits=config_data.operation_limits,
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)

    # Clear cache
    await clear_tier_cache(tier_id)

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="rate_limit_config",
        target_resource_id=tier_id,
        changes={"action": "set", "tier": tier.name},
        ip_address=get_client_ip(request),
        session=db,
    )

    return RateLimitConfigResponse(
        id=str(config.id),
        tier_id=str(config.tier_id),
        tier_name=tier.name,
        requests_per_minute=config.requests_per_minute,
        requests_per_hour=config.requests_per_hour,
        requests_per_day=config.requests_per_day,
        tokens_per_minute=config.tokens_per_minute,
        tokens_per_day=config.tokens_per_day,
        operation_limits=config.operation_limits,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@router.get("/rate-limits/usage")
async def get_rate_limit_usage(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
):
    """
    Get current rate limit usage for users.

    Admin only endpoint.
    """
    from backend.api.middleware.rate_limit import get_user_rate_limit_usage

    logger.info("Getting rate limit usage", admin_id=admin.user_id)

    # Get users to check usage for
    query = select(User).options(selectinload(User.access_tier))
    if user_id:
        query = query.where(User.id == user_id)
    query = query.limit(100)  # Limit to avoid performance issues

    result = await db.execute(query)
    users = result.scalars().all()

    usage_list = []
    for user in users:
        try:
            usage = await get_user_rate_limit_usage(
                user_id=str(user.id),
                tier_id=str(user.access_tier_id),
                tier_level=user.access_tier.level if user.access_tier else 0,
                db=db
            )
            usage_list.append(RateLimitUsageResponse(
                user_id=str(user.id),
                user_email=user.email,
                tier_name=user.access_tier.name if user.access_tier else "Unknown",
                requests=usage['requests'],
                tokens=usage['tokens'],
            ))
        except Exception as e:
            logger.warning("Failed to get rate limit usage", user_id=str(user.id), error=str(e))

    return {
        "usage": [u.model_dump() for u in usage_list],
        "total": len(usage_list),
    }


@router.post("/rate-limits/reset/{user_id}")
async def reset_user_rate_limits(
    user_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Reset rate limits for a specific user.

    Admin only endpoint.
    """
    from backend.api.middleware.rate_limit import reset_user_rate_limits as reset_limits

    logger.info("Resetting rate limits", admin_id=admin.user_id, target_user_id=user_id)

    # Verify user exists
    user_query = select(User).where(User.id == user_id)
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    await reset_limits(user_id)

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="rate_limit",
        target_user_id=user_id,
        changes={"action": "reset", "user_email": user.email},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": f"Rate limits reset for user {user.email}"}


# =============================================================================
# Cost Limit Management Endpoints
# =============================================================================

class CostLimitConfigCreate(BaseModel):
    """Request to set cost limit configuration for a user."""
    daily_limit_usd: float = Field(10.0, ge=0, description="Daily cost limit in USD")
    monthly_limit_usd: float = Field(100.0, ge=0, description="Monthly cost limit in USD")
    enforce_hard_limit: bool = Field(True, description="Enforce hard limit (block vs warn)")
    alert_thresholds: List[int] = Field([50, 80, 100], description="Alert threshold percentages")


class CostLimitResponse(BaseModel):
    """Cost limit response."""
    user_id: str
    user_email: str
    daily: Dict[str, Any]
    monthly: Dict[str, Any]
    enforce_hard_limit: bool
    alert_thresholds: List[int]


class CostAlertResponse(BaseModel):
    """Cost alert response."""
    id: str
    user_email: str
    alert_type: str
    threshold_percent: int
    usage_at_alert_usd: float
    notified: bool
    acknowledged: bool
    created_at: datetime


@router.get("/cost-limits")
async def list_cost_limits(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
):
    """
    List user cost limits.

    Admin only endpoint.
    """
    from backend.db.models import UserCostLimit

    logger.info("Listing cost limits", admin_id=admin.user_id)

    query = (
        select(UserCostLimit)
        .options(selectinload(UserCostLimit.user))
    )
    if user_id:
        query = query.where(UserCostLimit.user_id == user_id)
    query = query.order_by(UserCostLimit.created_at.desc())

    result = await db.execute(query)
    limits = result.scalars().all()

    return {
        "limits": [
            {
                "id": str(l.id),
                "user_id": str(l.user_id),
                "user_email": l.user.email if l.user else "Unknown",
                "daily_limit_usd": l.daily_limit_usd,
                "monthly_limit_usd": l.monthly_limit_usd,
                "current_daily_usage_usd": l.current_daily_usage_usd,
                "current_monthly_usage_usd": l.current_monthly_usage_usd,
                "enforce_hard_limit": l.enforce_hard_limit,
                "alert_thresholds": l.alert_thresholds,
                "last_daily_reset": l.last_daily_reset.isoformat() if l.last_daily_reset else None,
                "last_monthly_reset": l.last_monthly_reset.isoformat() if l.last_monthly_reset else None,
            }
            for l in limits
        ],
        "total": len(limits),
    }


@router.put("/cost-limits/{user_id}")
async def set_cost_limit(
    user_id: str,
    config_data: CostLimitConfigCreate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Set cost limits for a user.

    Admin only endpoint.
    """
    from backend.api.middleware.cost_limit import get_cost_limit_checker

    logger.info("Setting cost limit", admin_id=admin.user_id, target_user_id=user_id)

    # Verify user exists
    user_query = select(User).options(selectinload(User.access_tier)).where(User.id == user_id)
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    checker = get_cost_limit_checker()
    await checker.update_user_cost_limits(
        db=db,
        user_id=user_id,
        tier_level=user.access_tier.level if user.access_tier else 0,
        daily_limit=config_data.daily_limit_usd,
        monthly_limit=config_data.monthly_limit_usd,
        enforce_hard_limit=config_data.enforce_hard_limit,
        alert_thresholds=config_data.alert_thresholds,
    )
    await db.commit()

    # Get current status
    status_data = await checker.get_user_cost_status(
        db=db,
        user_id=user_id,
        tier_level=user.access_tier.level if user.access_tier else 0,
    )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="cost_limit",
        target_user_id=user_id,
        changes={
            "action": "set",
            "user_email": user.email,
            "daily_limit": config_data.daily_limit_usd,
            "monthly_limit": config_data.monthly_limit_usd,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return CostLimitResponse(
        user_id=user_id,
        user_email=user.email,
        daily=status_data['daily'],
        monthly=status_data['monthly'],
        enforce_hard_limit=status_data['enforce_hard_limit'],
        alert_thresholds=status_data['alert_thresholds'],
    )


@router.get("/cost-limits/{user_id}/status")
async def get_user_cost_status(
    user_id: str,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get cost limit status for a specific user.

    Admin only endpoint.
    """
    from backend.api.middleware.cost_limit import get_cost_limit_checker

    logger.info("Getting cost limit status", admin_id=admin.user_id, target_user_id=user_id)

    # Verify user exists
    user_query = select(User).options(selectinload(User.access_tier)).where(User.id == user_id)
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    checker = get_cost_limit_checker()
    status_data = await checker.get_user_cost_status(
        db=db,
        user_id=user_id,
        tier_level=user.access_tier.level if user.access_tier else 0,
    )

    return CostLimitResponse(
        user_id=user_id,
        user_email=user.email,
        daily=status_data['daily'],
        monthly=status_data['monthly'],
        enforce_hard_limit=status_data['enforce_hard_limit'],
        alert_thresholds=status_data['alert_thresholds'],
    )


@router.get("/cost-alerts")
async def list_cost_alerts(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    unacknowledged_only: bool = Query(True, description="Only show unacknowledged alerts"),
):
    """
    List cost alerts.

    Admin only endpoint.
    """
    from backend.api.middleware.cost_limit import get_cost_limit_checker

    logger.info("Listing cost alerts", admin_id=admin.user_id)

    checker = get_cost_limit_checker()
    alerts = await checker.get_pending_alerts(
        db=db,
        unacknowledged_only=unacknowledged_only,
    )

    # Get user emails for alerts
    from backend.db.models import UserCostLimit
    alert_responses = []

    for alert in alerts:
        # Get user email through cost limit
        limit_query = (
            select(UserCostLimit)
            .options(selectinload(UserCostLimit.user))
            .where(UserCostLimit.id == alert.cost_limit_id)
        )
        limit_result = await db.execute(limit_query)
        limit = limit_result.scalar_one_or_none()

        alert_responses.append(CostAlertResponse(
            id=str(alert.id),
            user_email=limit.user.email if limit and limit.user else "Unknown",
            alert_type=alert.alert_type,
            threshold_percent=alert.threshold_percent,
            usage_at_alert_usd=alert.usage_at_alert_usd,
            notified=alert.notified,
            acknowledged=alert.acknowledged,
            created_at=alert.created_at,
        ))

    return {
        "alerts": [a.model_dump() for a in alert_responses],
        "total": len(alert_responses),
    }


@router.post("/cost-alerts/{alert_id}/acknowledge")
async def acknowledge_cost_alert(
    alert_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Acknowledge a cost alert.

    Admin only endpoint.
    """
    from backend.api.middleware.cost_limit import get_cost_limit_checker

    logger.info("Acknowledging cost alert", admin_id=admin.user_id, alert_id=alert_id)

    checker = get_cost_limit_checker()
    alert = await checker.acknowledge_alert(db, alert_id)
    await db.commit()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="cost_alert",
        target_resource_id=alert_id,
        changes={"action": "acknowledge"},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Alert acknowledged", "alert_id": alert_id}


@router.post("/cost-limits/{user_id}/reset")
async def reset_user_cost_tracking(
    user_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
    reset_daily: bool = Query(True, description="Reset daily usage"),
    reset_monthly: bool = Query(False, description="Reset monthly usage"),
):
    """
    Reset cost tracking for a user.

    Admin only endpoint.
    """
    from backend.api.middleware.cost_limit import reset_user_cost_tracking as reset_cost

    logger.info("Resetting cost tracking", admin_id=admin.user_id, target_user_id=user_id)

    # Verify user exists
    user_query = select(User).options(selectinload(User.access_tier)).where(User.id == user_id)
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    await reset_cost(
        db=db,
        user_id=user_id,
        tier_level=user.access_tier.level if user.access_tier else 0,
        reset_daily=reset_daily,
        reset_monthly=reset_monthly,
    )
    await db.commit()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="cost_limit",
        target_user_id=user_id,
        changes={
            "action": "reset",
            "user_email": user.email,
            "reset_daily": reset_daily,
            "reset_monthly": reset_monthly,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": f"Cost tracking reset for user {user.email}"}


# =============================================================================
# Provider Health Endpoints
# =============================================================================

class ProviderHealthResponse(BaseModel):
    """Provider health status response."""
    provider_id: str
    provider_name: str
    provider_type: str
    is_healthy: bool
    latency_ms: Optional[int]
    error_message: Optional[str]
    consecutive_failures: int
    circuit_open: bool
    last_check: Optional[datetime]


class HealthCheckHistoryResponse(BaseModel):
    """Health check history entry."""
    id: str
    is_healthy: bool
    latency_ms: Optional[int]
    error_message: Optional[str]
    consecutive_failures: int
    checked_at: datetime


@router.get("/provider-health")
async def get_provider_health(
    admin: AdminUser,
):
    """
    Get health status of all LLM providers.

    Admin only endpoint.
    """
    from backend.services.provider_health import ProviderHealthChecker

    logger.info("Getting provider health", admin_id=admin.user_id)

    health_statuses = await ProviderHealthChecker.get_all_provider_health()

    return {
        "providers": [
            ProviderHealthResponse(
                provider_id=str(h.get("provider_id", "")),
                provider_name=h.get("provider_name", ""),
                provider_type=h.get("provider_type", ""),
                is_healthy=h.get("is_healthy", False),
                latency_ms=h.get("last_latency_ms"),
                error_message=h.get("status") if not h.get("is_healthy") else None,
                consecutive_failures=h.get("consecutive_failures", 0),
                circuit_open=h.get("circuit_open", False),
                last_check=datetime.fromisoformat(h["last_check_at"]) if h.get("last_check_at") else None,
            ).model_dump()
            for h in health_statuses
        ],
        "total": len(health_statuses),
    }


@router.post("/provider-health/check")
async def trigger_health_check(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
    provider_id: Optional[str] = Query(None, description="Check specific provider"),
):
    """
    Trigger immediate health check for providers.

    Admin only endpoint.
    """
    from backend.services.provider_health import ProviderHealthChecker

    logger.info("Triggering health check", admin_id=admin.user_id, provider_id=provider_id)

    if provider_id:
        # Check specific provider
        result = await ProviderHealthChecker.check_provider_health(provider_id)
        results = [result] if result else []
    else:
        # Check all providers
        results = await ProviderHealthChecker.run_all_health_checks()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="provider_health",
        changes={
            "action": "health_check",
            "provider_id": provider_id,
            "results_count": len(results),
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return {
        "checked": len(results),
        "results": [
            ProviderHealthResponse(
                provider_id=str(r.provider_id),
                provider_name=r.provider_name,
                provider_type=r.provider_type,
                is_healthy=r.is_healthy,
                latency_ms=r.latency_ms,
                error_message=r.error_message,
                consecutive_failures=0,  # Not available in HealthCheckResult
                circuit_open=False,  # Not available in HealthCheckResult
                last_check=r.checked_at,
            ).model_dump()
            for r in results
        ],
    }


@router.get("/provider-health/{provider_id}/history")
async def get_provider_health_history(
    provider_id: str,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    limit: int = Query(50, ge=1, le=500, description="Number of records"),
):
    """
    Get health check history for a provider.

    Admin only endpoint.
    """
    from backend.db.models import ProviderHealthLog

    logger.info("Getting provider health history", admin_id=admin.user_id, provider_id=provider_id)

    query = (
        select(ProviderHealthLog)
        .where(ProviderHealthLog.provider_id == provider_id)
        .order_by(desc(ProviderHealthLog.created_at))
        .limit(limit)
    )
    result = await db.execute(query)
    logs = result.scalars().all()

    return {
        "provider_id": provider_id,
        "history": [
            HealthCheckHistoryResponse(
                id=str(log.id),
                is_healthy=log.is_healthy,
                latency_ms=log.latency_ms,
                error_message=log.error_message,
                consecutive_failures=log.consecutive_failures,
                checked_at=log.created_at,
            ).model_dump()
            for log in logs
        ],
        "total": len(logs),
    }


@router.post("/provider-health/{provider_id}/reset-circuit")
async def reset_provider_circuit(
    provider_id: str,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Reset circuit breaker for a provider.

    Admin only endpoint.
    """
    from backend.db.models import ProviderHealthCache

    logger.info("Resetting provider circuit", admin_id=admin.user_id, provider_id=provider_id)

    # Reset circuit in cache
    query = select(ProviderHealthCache).where(ProviderHealthCache.provider_id == provider_id)
    result = await db.execute(query)
    cache = result.scalar_one_or_none()

    if cache:
        cache.circuit_open = False
        cache.circuit_open_until = None
        cache.consecutive_failures = 0
        await db.commit()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="provider_health",
        target_resource_id=provider_id,
        changes={"action": "reset_circuit"},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Circuit breaker reset", "provider_id": provider_id}


# =============================================================================
# Response Cache Management
# =============================================================================

class CacheStatsResponse(BaseModel):
    """Response cache statistics."""
    total_entries: int
    total_hits: int
    total_size_bytes: int
    hit_rate: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    estimated_savings_usd: float


class CacheSettingsUpdate(BaseModel):
    """Cache settings update request."""
    is_enabled: Optional[bool] = None
    default_ttl_seconds: Optional[int] = Field(None, ge=60, le=604800)  # 1 min to 7 days
    max_cache_size_mb: Optional[int] = Field(None, ge=10, le=10000)
    cache_temperature_threshold: Optional[float] = Field(None, ge=0, le=2)
    model_settings: Optional[dict] = None
    excluded_operations: Optional[List[str]] = None


class CacheClearRequest(BaseModel):
    """Cache clear request."""
    model_id: Optional[str] = None
    provider_id: Optional[str] = None
    older_than_days: Optional[int] = None


@router.get("/cache/stats")
async def get_cache_stats(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
) -> CacheStatsResponse:
    """
    Get response cache statistics.

    Admin only endpoint.
    """
    from backend.services.response_cache import get_response_cache_service

    cache_service = get_response_cache_service()
    stats = await cache_service.get_cache_stats(db)

    return CacheStatsResponse(
        total_entries=stats.total_entries,
        total_hits=stats.total_hits,
        total_size_bytes=stats.total_size_bytes,
        hit_rate=stats.hit_rate,
        oldest_entry=stats.oldest_entry,
        newest_entry=stats.newest_entry,
        estimated_savings_usd=stats.estimated_savings_usd,
    )


@router.get("/cache/settings")
async def get_cache_settings(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get current cache settings.

    Admin only endpoint.
    """
    from backend.services.response_cache import get_response_cache_service

    cache_service = get_response_cache_service()
    return await cache_service.get_cache_settings(db)


@router.put("/cache/settings")
async def update_cache_settings(
    settings: CacheSettingsUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update cache settings.

    Admin only endpoint.
    """
    from backend.services.response_cache import get_response_cache_service

    logger.info("Updating cache settings", admin_id=admin.user_id)

    cache_service = get_response_cache_service()
    success = await cache_service.update_settings(
        db,
        is_enabled=settings.is_enabled,
        default_ttl_seconds=settings.default_ttl_seconds,
        max_cache_size_mb=settings.max_cache_size_mb,
        cache_temperature_threshold=settings.cache_temperature_threshold,
        model_settings=settings.model_settings,
        excluded_operations=settings.excluded_operations,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update cache settings",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="cache_settings",
        changes=settings.model_dump(exclude_none=True),
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Cache settings updated"}


@router.post("/cache/clear")
async def clear_cache(
    clear_request: CacheClearRequest,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Clear response cache.

    Admin only endpoint.
    """
    from backend.services.response_cache import get_response_cache_service
    from datetime import timedelta

    logger.info(
        "Clearing cache",
        admin_id=admin.user_id,
        model_id=clear_request.model_id,
        provider_id=clear_request.provider_id,
    )

    cache_service = get_response_cache_service()

    older_than = None
    if clear_request.older_than_days:
        older_than = datetime.utcnow() - timedelta(days=clear_request.older_than_days)

    deleted_count = await cache_service.clear_cache(
        db,
        model_id=clear_request.model_id,
        provider_id=clear_request.provider_id,
        older_than=older_than,
    )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="cache",
        changes={
            "action": "clear",
            "deleted_count": deleted_count,
            **clear_request.model_dump(exclude_none=True),
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": f"Cleared {deleted_count} cache entries", "deleted_count": deleted_count}


@router.post("/cache/cleanup")
async def cleanup_expired_cache(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Clean up expired cache entries.

    Admin only endpoint.
    """
    from backend.services.response_cache import get_response_cache_service

    logger.info("Cleaning up expired cache", admin_id=admin.user_id)

    cache_service = get_response_cache_service()
    deleted_count = await cache_service.cleanup_expired(db)

    return {"message": f"Cleaned up {deleted_count} expired entries", "deleted_count": deleted_count}


# =============================================================================
# Smart Routing Management
# =============================================================================

class RoutingStatsResponse(BaseModel):
    """Routing statistics response."""
    total_providers: int
    healthy_providers: int
    unhealthy_providers: int
    average_latency_ms: Optional[float]
    providers: List[dict]


class ProviderPriorityUpdate(BaseModel):
    """Provider priority update request."""
    priority: int = Field(..., ge=0, le=100)


class ProviderOrderUpdate(BaseModel):
    """Provider order update request."""
    provider_order: List[str] = Field(..., min_length=1)


@router.get("/routing/stats")
async def get_routing_stats(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
) -> RoutingStatsResponse:
    """
    Get smart routing statistics.

    Admin only endpoint.
    """
    from backend.services.smart_router import get_smart_router

    router_service = get_smart_router()
    stats = await router_service.get_routing_stats(db)

    return RoutingStatsResponse(**stats)


@router.put("/routing/providers/{provider_id}/priority")
async def update_provider_priority(
    provider_id: str,
    priority_update: ProviderPriorityUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a provider's routing priority.

    Lower priority values are preferred.
    Admin only endpoint.
    """
    from backend.services.smart_router import get_smart_router

    logger.info(
        "Updating provider priority",
        admin_id=admin.user_id,
        provider_id=provider_id,
        priority=priority_update.priority,
    )

    router_service = get_smart_router()
    success = await router_service.update_provider_priority(
        db, provider_id, priority_update.priority
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update provider priority",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="provider_routing",
        target_resource_id=provider_id,
        changes={"priority": priority_update.priority},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Provider priority updated", "provider_id": provider_id}


@router.put("/routing/reorder")
async def reorder_providers(
    order_update: ProviderOrderUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Reorder all providers by priority.

    Providers are ordered by their position in the list (first = highest priority).
    Admin only endpoint.
    """
    from backend.services.smart_router import get_smart_router

    logger.info(
        "Reordering providers",
        admin_id=admin.user_id,
        count=len(order_update.provider_order),
    )

    router_service = get_smart_router()
    success = await router_service.reorder_providers(db, order_update.provider_order)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reorder providers",
        )

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="provider_routing",
        changes={"action": "reorder", "provider_order": order_update.provider_order},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {"message": "Providers reordered successfully"}


# =============================================================================
# Document Enhancement Endpoints
# =============================================================================

class EnhanceDocumentsRequest(BaseModel):
    """Request to enhance documents with LLM-extracted metadata."""
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to enhance (None = all unprocessed)")
    collection: Optional[str] = Field(None, description="Filter by collection")
    force: bool = Field(False, description="Re-enhance already enhanced documents")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum documents to process")
    auto_tag: bool = Field(False, description="Also generate tags using AutoTagger after enhancement")


class EnhancementStatusResponse(BaseModel):
    """Response for enhancement status."""
    document_count: int
    estimated_tokens: int
    estimated_cost_usd: float
    model: str
    avg_tokens_per_doc: int


class EnhancementResultResponse(BaseModel):
    """Response for enhancement operation."""
    total: int
    successful: int
    failed: int
    total_tokens: int
    estimated_cost_usd: float
    message: str


@router.post("/enhance-documents/estimate", response_model=EnhancementStatusResponse)
async def estimate_enhancement_cost(
    request: EnhanceDocumentsRequest,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Estimate the cost of enhancing documents.

    Returns token and cost estimates before running enhancement.
    Admin only.
    """
    from backend.services.document_enhancer import get_document_enhancer

    logger.info(
        "Estimating enhancement cost",
        document_ids=len(request.document_ids) if request.document_ids else "all",
        collection=request.collection,
    )

    enhancer = get_document_enhancer()
    estimate = await enhancer.estimate_cost(
        document_ids=request.document_ids,
        collection=request.collection,
    )

    return EnhancementStatusResponse(**estimate)


@router.post("/enhance-documents", response_model=EnhancementResultResponse)
async def enhance_documents(
    request: EnhanceDocumentsRequest,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Enhance documents with LLM-extracted metadata.

    Extracts summaries, keywords, topics, entities, and hypothetical questions
    from documents to improve RAG search quality.

    Optionally also generates tags using AutoTagger if auto_tag=True.

    Admin only.
    """
    from backend.services.document_enhancer import get_document_enhancer
    from backend.services.auto_tagger import AutoTaggerService

    logger.info(
        "Starting document enhancement",
        document_ids=len(request.document_ids) if request.document_ids else "all",
        collection=request.collection,
        force=request.force,
        auto_tag=request.auto_tag,
    )

    enhancer = get_document_enhancer()

    if request.document_ids:
        # Enhance specific documents
        result = await enhancer.enhance_batch(
            document_ids=request.document_ids,
            force=request.force,
        )
    else:
        # Enhance all unprocessed documents
        result = await enhancer.enhance_all_unprocessed(
            collection=request.collection,
            limit=request.limit,
        )

    # If auto_tag is enabled, run auto-tagging on successfully enhanced documents
    tags_generated = 0
    if request.auto_tag and result.successful > 0:
        logger.info("Running auto-tagging on enhanced documents")
        auto_tagger = AutoTaggerService()

        # Get the document IDs that were successfully enhanced
        enhanced_doc_ids = result.enhanced_document_ids if hasattr(result, 'enhanced_document_ids') else []

        # If we don't have the list, get documents with enhanced metadata
        if not enhanced_doc_ids:
            from backend.db.models import Document
            from sqlalchemy import select

            query = select(Document.id).where(Document.enhanced_metadata.isnot(None))
            if request.document_ids:
                query = query.where(Document.id.in_([UUID(did) for did in request.document_ids]))
            doc_result = await db.execute(query)
            enhanced_doc_ids = [str(row[0]) for row in doc_result.fetchall()]

        # Get existing collections for context
        from backend.db.models import Document, AccessTier
        from sqlalchemy import select, and_

        collections_query = (
            select(Document.tags)
            .where(Document.tags.isnot(None))
        )
        collections_result = await db.execute(collections_query)
        all_tags = collections_result.scalars().all()
        existing_collections = list(set(
            tag for tags in all_tags if tags for tag in tags
        ))

        # Auto-tag each enhanced document
        for doc_id in enhanced_doc_ids:  # Process all documents
            try:
                # Get document with chunks
                from sqlalchemy.orm import selectinload
                doc_query = (
                    select(Document)
                    .options(selectinload(Document.chunks))
                    .where(Document.id == UUID(doc_id))
                )
                doc_result = await db.execute(doc_query)
                document = doc_result.scalar_one_or_none()

                if document and document.chunks:
                    chunks = sorted(document.chunks, key=lambda c: c.chunk_index)[:3]
                    content_sample = "\n".join([chunk.content for chunk in chunks])

                    if content_sample:
                        tags = await auto_tagger.generate_tags(
                            document_name=document.original_filename or document.filename,
                            content_sample=content_sample,
                            existing_collections=existing_collections,
                            max_tags=5,
                        )
                        if tags:
                            # Merge auto-generated tags with existing user tags (preserve user tags)
                            existing_tags = document.tags or []
                            merged_tags = list(dict.fromkeys(existing_tags + tags))
                            document.tags = merged_tags
                            tags_generated += 1
                            # Add new tags to existing collections for next iterations
                            for tag in merged_tags:
                                if tag not in existing_collections:
                                    existing_collections.append(tag)
            except Exception as e:
                logger.warning(f"Auto-tagging failed for document {doc_id}: {e}")
                continue

        await db.commit()
        logger.info(f"Auto-tagging completed: {tags_generated} documents tagged")

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="document_enhancement",
        changes={
            "total": result.total,
            "successful": result.successful,
            "failed": result.failed,
            "tokens_used": result.total_tokens,
            "auto_tag": request.auto_tag,
            "tags_generated": tags_generated,
        },
        session=db,
    )

    message = f"Enhanced {result.successful}/{result.total} documents"
    if request.auto_tag:
        message += f", tagged {tags_generated} documents"

    return EnhancementResultResponse(
        total=result.total,
        successful=result.successful,
        failed=result.failed,
        total_tokens=result.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        message=message,
    )


@router.post("/enhance-documents/{document_id}")
async def enhance_single_document(
    document_id: str,
    admin: AdminUser,
    force: bool = Query(False, description="Re-enhance if already enhanced"),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Enhance a single document with LLM-extracted metadata.

    Admin only.
    """
    from backend.services.document_enhancer import get_document_enhancer

    logger.info("Enhancing single document", document_id=document_id, force=force)

    enhancer = get_document_enhancer()
    result = await enhancer.enhance_document(document_id, force=force)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Enhancement failed",
        )

    return {
        "document_id": result.document_id,
        "success": result.success,
        "tokens_used": result.tokens_used,
        "metadata": result.metadata.model_dump() if result.metadata else None,
    }


@router.post("/backfill-hypothetical-chunks")
async def backfill_hypothetical_chunks(
    admin: AdminUser,
    limit: Optional[int] = Query(None, description="Max documents to process"),
):
    """
    Backfill synthetic chunks from hypothetical questions.

    Creates synthetic question chunks (chunk_level=2, is_summary=True) for
    documents that have enhanced_metadata but are missing these chunks.
    This improves vector search by enabling question-based retrieval.

    Admin only.
    """
    from backend.services.document_enhancer import get_document_enhancer

    logger.info("Starting hypothetical chunk backfill", limit=limit)

    try:
        enhancer = get_document_enhancer()
        result = await enhancer.backfill_hypothetical_chunks(limit=limit)

        return {
            "success": True,
            "message": (
                f"Backfill complete: {result['documents_processed']} documents processed, "
                f"{result['total_chunks_created']} chunks created, "
                f"{result['documents_skipped']} skipped"
            ),
            **result,
        }
    except Exception as e:
        logger.error("Hypothetical chunk backfill failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backfill failed: {str(e)}",
        )


# =============================================================================
# Provider Health Check Endpoints
# =============================================================================

class ProviderHealthCheckResponse(BaseModel):
    """Response for provider health check."""
    provider_id: str
    provider_type: str
    provider_name: str
    is_healthy: bool
    status: str
    latency_ms: Optional[int] = None
    error_message: Optional[str] = None
    checked_at: datetime


class AllProvidersHealthResponse(BaseModel):
    """Response for all providers health check."""
    providers: List[ProviderHealthCheckResponse]
    healthy_count: int
    unhealthy_count: int
    degraded_count: int
    checked_at: datetime


@router.get("/health/providers", response_model=AllProvidersHealthResponse)
async def check_all_providers_health(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Check health status of all configured LLM providers.

    Admin only endpoint. Returns health status for each provider
    including latency and any error messages.
    """
    from backend.services.provider_health import ProviderHealthChecker
    from backend.db.models import LLMProvider

    logger.info("Checking health of all providers", admin_id=admin.user_id)

    # Get all active providers
    result = await db.execute(
        select(LLMProvider).where(LLMProvider.is_active == True)
    )
    providers = result.scalars().all()

    health_results = []
    for provider in providers:
        try:
            check_result = await ProviderHealthChecker.check_provider_health(
                provider_id=str(provider.id),
                check_type="ping",
            )
            health_results.append(ProviderHealthCheckResponse(
                provider_id=str(provider.id),
                provider_type=provider.provider_type,
                provider_name=provider.name,
                is_healthy=check_result.is_healthy,
                status=check_result.status,
                latency_ms=check_result.latency_ms,
                error_message=check_result.error_message,
                checked_at=check_result.checked_at,
            ))
        except Exception as e:
            logger.error("Health check failed", provider_id=str(provider.id), error=str(e))
            health_results.append(ProviderHealthCheckResponse(
                provider_id=str(provider.id),
                provider_type=provider.provider_type,
                provider_name=provider.name,
                is_healthy=False,
                status="error",
                error_message=str(e),
                checked_at=datetime.utcnow(),
            ))

    # Calculate counts
    healthy_count = sum(1 for r in health_results if r.is_healthy and r.status == "healthy")
    unhealthy_count = sum(1 for r in health_results if not r.is_healthy)
    degraded_count = sum(1 for r in health_results if r.status == "degraded")

    return AllProvidersHealthResponse(
        providers=health_results,
        healthy_count=healthy_count,
        unhealthy_count=unhealthy_count,
        degraded_count=degraded_count,
        checked_at=datetime.utcnow(),
    )


@router.get("/health/providers/{provider_id}", response_model=ProviderHealthCheckResponse)
async def check_provider_health(
    provider_id: UUID,
    admin: AdminUser,
    check_type: str = Query("ping", regex="^(ping|completion|embedding)$"),
):
    """
    Check health status of a specific LLM provider.

    Admin only endpoint.

    Args:
        provider_id: UUID of the provider to check
        check_type: Type of health check (ping, completion, or embedding)
    """
    from backend.services.provider_health import ProviderHealthChecker

    logger.info(
        "Checking provider health",
        provider_id=str(provider_id),
        check_type=check_type,
        admin_id=admin.user_id,
    )

    check_result = await ProviderHealthChecker.check_provider_health(
        provider_id=str(provider_id),
        check_type=check_type,
    )

    return ProviderHealthCheckResponse(
        provider_id=check_result.provider_id,
        provider_type=check_result.provider_type,
        provider_name=check_result.provider_name,
        is_healthy=check_result.is_healthy,
        status=check_result.status,
        latency_ms=check_result.latency_ms,
        error_message=check_result.error_message,
        checked_at=check_result.checked_at,
    )


# =============================================================================
# Redis/Celery Status
# =============================================================================


class RedisStatusResponse(BaseModel):
    """Response model for Redis connection status."""
    connected: bool
    enabled: bool
    url: Optional[str] = None
    reason: Optional[str] = None


class CeleryStatusResponse(BaseModel):
    """Response model for Celery status."""
    enabled: bool
    available: bool
    workers: List[str] = []
    worker_count: int = 0
    active_tasks: int = 0
    concurrency: int = 4  # Current configured concurrency from settings
    message: Optional[str] = None


@router.get("/redis/status", response_model=RedisStatusResponse)
async def get_redis_status(
    admin: AdminUser,
):
    """
    Check Redis connection status.

    Admin only endpoint.
    """
    from backend.services.redis_client import check_redis_connection

    status = await check_redis_connection()
    return RedisStatusResponse(**status)


@router.get("/celery/status", response_model=CeleryStatusResponse)
async def get_celery_status(
    admin: AdminUser,
):
    """
    Check Celery worker status.

    Admin only endpoint.
    """
    try:
        from backend.services.task_queue import (
            is_celery_available,
            get_worker_stats,
            get_active_tasks,
        )

        # Get worker info
        worker_stats = get_worker_stats()
        is_available = is_celery_available()

        # Count active tasks
        active_count = 0
        if worker_stats.get("enabled", False):
            try:
                active_tasks = get_active_tasks()
                for worker_tasks in active_tasks.get("active", {}).values():
                    active_count += len(worker_tasks)
            except Exception as e:
                logger.debug("Failed to get active Celery tasks", error=str(e))

        # Get configured concurrency from settings
        try:
            settings_service = get_settings_service()
            concurrency = await settings_service.get_setting("queue.max_workers") or 4
            concurrency = int(concurrency)
        except Exception:
            concurrency = 4

        return CeleryStatusResponse(
            enabled=worker_stats.get("enabled", False),
            available=is_available,
            workers=worker_stats.get("workers", []),
            worker_count=worker_stats.get("count", 0),
            active_tasks=active_count,
            concurrency=concurrency,
            message=worker_stats.get("message"),
        )
    except ImportError:
        # Celery is not installed
        return CeleryStatusResponse(
            enabled=False,
            available=False,
            workers=[],
            worker_count=0,
            active_tasks=0,
            message="Celery is not installed. Install with: pip install celery[redis]",
        )


# Global to track Celery worker process
_celery_worker_process: Optional[subprocess.Popen] = None
_celery_worker_pid: Optional[int] = None


class CeleryStartResponse(BaseModel):
    """Response model for Celery start."""
    success: bool
    message: str
    pid: Optional[int] = None


class CeleryStopResponse(BaseModel):
    """Response model for Celery stop."""
    success: bool
    message: str


@router.post("/celery/start", response_model=CeleryStartResponse)
async def start_celery_worker(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Start a Celery worker process.

    Admin only endpoint. Spawns a Celery worker as a subprocess.
    """
    import subprocess
    import sys
    import os
    global _celery_worker_process, _celery_worker_pid

    # Check if worker already running
    if _celery_worker_process is not None:
        if _celery_worker_process.poll() is None:
            return CeleryStartResponse(
                success=False,
                message=f"Celery worker already running (PID: {_celery_worker_pid})",
                pid=_celery_worker_pid,
            )
        else:
            # Process ended, clear refs
            _celery_worker_process = None
            _celery_worker_pid = None

    # Check if Redis is enabled/reachable
    from backend.services.redis_client import check_redis_connection
    redis_status = await check_redis_connection()
    if not redis_status.get("connected"):
        return CeleryStartResponse(
            success=False,
            message=f"Cannot start Celery: Redis not available. {redis_status.get('reason', '')}",
        )

    # Get project root for working directory
    project_root = Path(__file__).resolve().parents[3]
    backend_dir = project_root / "backend"

    # Build the celery command
    # Use uv run if available, otherwise direct celery
    celery_cmd = [
        sys.executable, "-m", "celery",
        "-A", "backend.services.task_queue:celery_app",
        "worker",
        "--loglevel=info",
        "--concurrency=2",
    ]

    # Create logs directory
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    celery_log = log_dir / "celery_worker.log"

    try:
        # Set environment for subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)

        # Start Celery worker as subprocess
        with open(celery_log, "a") as log_file:
            _celery_worker_process = subprocess.Popen(
                celery_cmd,
                cwd=str(project_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )
        _celery_worker_pid = _celery_worker_process.pid

        # Log the action
        audit_service = get_audit_service()
        await audit_service.log_admin_action(
            action=AuditAction.SYSTEM_CONFIG_CHANGE,
            admin_user_id=admin.user_id,
            target_resource_type="celery_worker",
            changes={"action": "start", "pid": _celery_worker_pid},
            ip_address=get_client_ip(request),
            session=db,
        )

        logger.info("Celery worker started", pid=_celery_worker_pid, admin_id=admin.user_id)

        return CeleryStartResponse(
            success=True,
            message=f"Celery worker started successfully (PID: {_celery_worker_pid}). Logs at: logs/celery_worker.log",
            pid=_celery_worker_pid,
        )

    except Exception as e:
        logger.error("Failed to start Celery worker", error=str(e))
        return CeleryStartResponse(
            success=False,
            message=f"Failed to start Celery worker: {str(e)}",
        )


@router.post("/celery/stop", response_model=CeleryStopResponse)
async def stop_celery_worker(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Stop the Celery worker process.

    Admin only endpoint. Gracefully terminates the worker.
    """
    import signal
    import subprocess
    global _celery_worker_process, _celery_worker_pid

    stopped_pids = []

    # Stop our tracked process if running
    if _celery_worker_process is not None:
        if _celery_worker_process.poll() is None:
            try:
                _celery_worker_process.terminate()
                _celery_worker_process.wait(timeout=10)
                stopped_pids.append(_celery_worker_pid)
            except subprocess.TimeoutExpired:
                _celery_worker_process.kill()
                stopped_pids.append(_celery_worker_pid)
            except Exception as e:
                logger.warning("Error stopping tracked Celery process", error=str(e))

        _celery_worker_process = None
        _celery_worker_pid = None

    # Also try to kill any celery workers by pattern (catches externally started workers)
    try:
        import platform
        if platform.system() != "Windows":
            # Kill celery workers by pattern
            subprocess.run(["pkill", "-f", "celery.*worker"], check=False)
    except Exception as e:
        logger.debug("pkill not available", error=str(e))

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="celery_worker",
        changes={"action": "stop", "pids": stopped_pids},
        ip_address=get_client_ip(request),
        session=db,
    )

    logger.info("Celery worker stopped", admin_id=admin.user_id)

    return CeleryStopResponse(
        success=True,
        message="Celery worker stop signal sent. Workers will terminate gracefully.",
    )


class CeleryRestartResponse(BaseModel):
    """Response for Celery worker restart."""
    success: bool
    message: str
    concurrency: Optional[int] = None


@router.post("/celery/restart", response_model=CeleryRestartResponse)
async def restart_celery_workers(admin: AdminUser):
    """
    Restart Celery workers with updated settings.

    Reads queue.max_workers from database settings and restarts workers
    with the new concurrency. Use this after changing queue settings.

    Admin only endpoint.
    """
    from backend.services.celery_manager import restart_celery_worker, get_worker_status

    try:
        # Restart the worker (will read new settings from DB)
        success = await restart_celery_worker()

        # Get the new status including concurrency
        status = await get_worker_status()
        concurrency = status.get("concurrency", 4)

        if success:
            logger.info("Celery workers restarted with new settings",
                       admin_id=admin.user_id, concurrency=concurrency)
            return CeleryRestartResponse(
                success=True,
                message=f"Celery workers restarted with concurrency={concurrency}",
                concurrency=concurrency,
            )
        else:
            return CeleryRestartResponse(
                success=False,
                message="Failed to restart Celery workers. Check logs for details.",
                concurrency=concurrency,
            )

    except Exception as e:
        logger.error("Error restarting Celery workers", error=str(e), admin_id=admin.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart workers: {str(e)}"
        )


@router.get("/celery/running")
async def is_celery_worker_running(admin: AdminUser):
    """
    Check if a Celery worker is running (either tracked or external).

    Admin only endpoint.
    """
    global _celery_worker_process, _celery_worker_pid

    # Check our tracked process
    tracked_running = False
    if _celery_worker_process is not None:
        if _celery_worker_process.poll() is None:
            tracked_running = True
        else:
            _celery_worker_process = None
            _celery_worker_pid = None

    # Also check via Celery ping
    from backend.services.task_queue import is_celery_available, get_worker_stats
    celery_available = is_celery_available()
    worker_stats = get_worker_stats()

    return {
        "running": tracked_running or celery_available,
        "tracked_pid": _celery_worker_pid if tracked_running else None,
        "worker_count": worker_stats.get("count", 0),
        "workers": worker_stats.get("workers", []),
    }


@router.post("/redis/invalidate-cache")
async def invalidate_redis_settings_cache(
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Invalidate Redis/Celery settings cache.

    Call this after changing queue.* or cache.* settings to apply changes.
    Admin only endpoint.
    """
    from backend.services.redis_client import invalidate_redis_cache
    from backend.services.embedding_cache import invalidate_cache_settings

    invalidate_redis_cache()
    invalidate_cache_settings()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="redis_settings",
        changes={"action": "invalidate_cache"},
        ip_address=get_client_ip(request),
        session=db,
    )

    logger.info("Redis settings cache invalidated", admin_id=admin.user_id)

    return {"message": "Redis settings cache invalidated. Changes will take effect on next request."}


# =============================================================================
# Document Organization Management (Superadmin)
# =============================================================================


class DocumentOrgInfo(BaseModel):
    """Document organization info response."""
    id: str
    filename: str
    original_filename: Optional[str] = None
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    created_at: Optional[datetime] = None


class DocumentOrgUpdate(BaseModel):
    """Update document organization_id request."""
    organization_id: str = Field(..., description="Organization ID to assign")


class BulkDocumentOrgUpdate(BaseModel):
    """Bulk update documents organization_id."""
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to update. If not provided, updates all documents without org_id")
    organization_id: str = Field(..., description="Organization ID to assign")


@router.get("/documents/organization-status")
async def get_documents_organization_status(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    filter_missing_org: bool = Query(False, description="Only show documents missing organization_id"),
):
    """
    Get all documents with their organization_id status.

    Superadmin only endpoint.
    """
    from backend.db.models import Document, Organization
    from sqlalchemy import or_

    logger.info("Getting documents organization status", admin_id=admin.user_id)

    # Build query
    query = select(Document).order_by(desc(Document.created_at))

    if filter_missing_org:
        query = query.where(
            or_(
                Document.organization_id.is_(None),
                Document.organization_id == "",
            )
        )

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    result = await db.execute(query)
    documents = result.scalars().all()

    # Get organization names
    org_ids = {str(d.organization_id) for d in documents if d.organization_id}
    org_names = {}
    if org_ids:
        org_result = await db.execute(
            select(Organization).where(Organization.id.in_([UUID(oid) for oid in org_ids]))
        )
        for org in org_result.scalars().all():
            org_names[str(org.id)] = org.name

    return {
        "documents": [
            DocumentOrgInfo(
                id=str(doc.id),
                filename=doc.filename,
                original_filename=doc.original_filename,
                organization_id=str(doc.organization_id) if doc.organization_id else None,
                organization_name=org_names.get(str(doc.organization_id)) if doc.organization_id else None,
                created_at=doc.created_at,
            )
            for doc in documents
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "missing_org_count": await _count_docs_missing_org(db),
    }


async def _count_docs_missing_org(db: AsyncSession) -> int:
    """Count documents missing organization_id."""
    from backend.db.models import Document
    from sqlalchemy import or_

    result = await db.execute(
        select(func.count(Document.id)).where(
            or_(
                Document.organization_id.is_(None),
                Document.organization_id == "",
            )
        )
    )
    return result.scalar() or 0


@router.put("/documents/{document_id}/organization")
async def update_document_organization(
    document_id: str,
    update_data: DocumentOrgUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a document's organization_id.

    Superadmin only endpoint.
    """
    from backend.db.models import Document, Organization

    logger.info(
        "Updating document organization",
        admin_id=admin.user_id,
        document_id=document_id,
        new_org_id=update_data.organization_id,
    )

    # Verify organization exists
    org_result = await db.execute(
        select(Organization).where(Organization.id == UUID(update_data.organization_id))
    )
    org = org_result.scalar_one_or_none()
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # Get document
    doc_result = await db.execute(
        select(Document).where(Document.id == UUID(document_id))
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    old_org_id = str(doc.organization_id) if doc.organization_id else None
    doc.organization_id = UUID(update_data.organization_id)
    await db.commit()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="document",
        target_resource_id=document_id,
        changes={
            "action": "update_organization",
            "old_organization_id": old_org_id,
            "new_organization_id": update_data.organization_id,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return {
        "message": "Document organization updated",
        "document_id": document_id,
        "organization_id": update_data.organization_id,
        "organization_name": org.name,
    }


@router.post("/documents/bulk-update-organization")
async def bulk_update_documents_organization(
    update_data: BulkDocumentOrgUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Bulk update organization_id for multiple documents.

    If document_ids is not provided, updates all documents without organization_id.

    Superadmin only endpoint.
    """
    from backend.db.models import Document, Organization, Chunk
    from sqlalchemy import or_, update

    logger.info(
        "Bulk updating documents organization",
        admin_id=admin.user_id,
        new_org_id=update_data.organization_id,
        specific_docs=len(update_data.document_ids) if update_data.document_ids else "all_missing",
    )

    # Verify organization exists
    org_result = await db.execute(
        select(Organization).where(Organization.id == UUID(update_data.organization_id))
    )
    org = org_result.scalar_one_or_none()
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    org_uuid = UUID(update_data.organization_id)

    if update_data.document_ids:
        # Update specific documents
        doc_uuids = [UUID(did) for did in update_data.document_ids]

        # Update documents
        doc_update_stmt = (
            update(Document)
            .where(Document.id.in_(doc_uuids))
            .values(organization_id=org_uuid)
        )
        doc_result = await db.execute(doc_update_stmt)
        docs_updated = doc_result.rowcount

        # Update chunks for these documents
        chunk_update_stmt = (
            update(Chunk)
            .where(Chunk.document_id.in_(doc_uuids))
            .values(organization_id=org_uuid)
        )
        chunk_result = await db.execute(chunk_update_stmt)
        chunks_updated = chunk_result.rowcount
    else:
        # Update all documents without organization_id
        doc_update_stmt = (
            update(Document)
            .where(
                or_(
                    Document.organization_id.is_(None),
                    Document.organization_id == "",
                )
            )
            .values(organization_id=org_uuid)
        )
        doc_result = await db.execute(doc_update_stmt)
        docs_updated = doc_result.rowcount

        # Update all chunks without organization_id
        chunk_update_stmt = (
            update(Chunk)
            .where(
                or_(
                    Chunk.organization_id.is_(None),
                    Chunk.organization_id == "",
                )
            )
            .values(organization_id=org_uuid)
        )
        chunk_result = await db.execute(chunk_update_stmt)
        chunks_updated = chunk_result.rowcount

    await db.commit()

    # Log the action
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="documents_bulk",
        changes={
            "action": "bulk_update_organization",
            "organization_id": update_data.organization_id,
            "documents_updated": docs_updated,
            "chunks_updated": chunks_updated,
            "specific_ids": update_data.document_ids,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return {
        "message": "Documents organization updated",
        "documents_updated": docs_updated,
        "chunks_updated": chunks_updated,
        "organization_id": update_data.organization_id,
        "organization_name": org.name,
    }


@router.get("/organizations/list")
async def list_organizations_for_admin(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all organizations.

    Superadmin only endpoint.
    """
    from backend.db.models import Organization

    result = await db.execute(
        select(Organization).order_by(Organization.name)
    )
    orgs = result.scalars().all()

    return {
        "organizations": [
            {
                "id": str(org.id),
                "name": org.name,
                "description": org.description,
                "created_at": org.created_at,
            }
            for org in orgs
        ]
    }


# =============================================================================
# Feature Management (Phase 55 - Superadmin Feature Toggles)
# =============================================================================

class FeatureToggle(BaseModel):
    """Feature toggle model."""
    id: str
    name: str
    description: str
    category: str
    enabled: bool
    requires_api_key: Optional[str] = None
    dependencies: List[str] = []
    status: str = "available"  # available, unavailable, degraded


class FeatureToggleUpdate(BaseModel):
    """Feature toggle update request."""
    enabled: bool


class FeatureToggleListResponse(BaseModel):
    """Feature toggles list response."""
    features: List[FeatureToggle]
    categories: List[str]


# Feature definitions with dependencies and requirements
SYSTEM_FEATURES = {
    # Vision & Document Processing
    "vlm": {
        "name": "Vision Language Model (VLM)",
        "description": "Use AI vision models for charts, tables, and infographics",
        "category": "Document Processing",
        "config_key": "ENABLE_VLM",
        "requires_api_key": "ANTHROPIC_API_KEY",
        "dependencies": [],
    },
    "ocr_surya": {
        "name": "Surya OCR",
        "description": "High-quality document OCR with layout detection",
        "category": "Document Processing",
        "config_key": "USE_SURYA_OCR",
        "requires_api_key": None,
        "dependencies": [],
    },
    "knowledge_graph": {
        "name": "Knowledge Graph Extraction",
        "description": "Extract entities and relationships from documents",
        "category": "Document Processing",
        "config_key": "ENABLE_KNOWLEDGE_GRAPH",
        "requires_api_key": None,
        "dependencies": [],
    },

    # Advanced RAG Features
    "rlm": {
        "name": "Recursive Language Model (RLM)",
        "description": "Process 10M+ token contexts with O(log N) complexity",
        "category": "Advanced RAG",
        "config_key": "ENABLE_RLM",
        "requires_api_key": "ANTHROPIC_API_KEY",
        "dependencies": [],
    },
    "colbert": {
        "name": "ColBERT Retrieval",
        "description": "Token-level retrieval for precise matching",
        "category": "Advanced RAG",
        "config_key": "ENABLE_COLBERT",
        "requires_api_key": None,
        "dependencies": [],
    },
    "warp": {
        "name": "WARP Engine",
        "description": "3x faster retrieval than ColBERT PLAID",
        "category": "Advanced RAG",
        "config_key": "ENABLE_WARP",
        "requires_api_key": None,
        "dependencies": ["colbert"],
    },
    "colpali": {
        "name": "ColPali Visual Retrieval",
        "description": "Visual document retrieval without OCR",
        "category": "Advanced RAG",
        "config_key": "ENABLE_COLPALI",
        "requires_api_key": None,
        "dependencies": [],
    },
    "hybrid_search": {
        "name": "Hybrid Search",
        "description": "Combine vector, keyword, and graph search",
        "category": "Advanced RAG",
        "config_key": "ENABLE_HYBRID_SEARCH",
        "requires_api_key": None,
        "dependencies": [],
    },
    "reranking": {
        "name": "Reranking Pipeline",
        "description": "Multi-stage result reranking for accuracy",
        "category": "Advanced RAG",
        "config_key": "ENABLE_RERANKING",
        "requires_api_key": None,
        "dependencies": [],
    },

    # Distributed Processing
    "ray_embeddings": {
        "name": "Ray Distributed Embeddings",
        "description": "Use Ray cluster for embedding generation",
        "category": "Distributed Processing",
        "config_key": "USE_RAY_FOR_EMBEDDINGS",
        "requires_api_key": None,
        "dependencies": ["ray_cluster"],
    },
    "ray_kg": {
        "name": "Ray Distributed KG Extraction",
        "description": "Use Ray cluster for knowledge graph extraction",
        "category": "Distributed Processing",
        "config_key": "USE_RAY_FOR_KG",
        "requires_api_key": None,
        "dependencies": ["ray_cluster", "knowledge_graph"],
    },
    "ray_vlm": {
        "name": "Ray Distributed VLM",
        "description": "Use Ray cluster for VLM processing",
        "category": "Distributed Processing",
        "config_key": "USE_RAY_FOR_VLM",
        "requires_api_key": None,
        "dependencies": ["ray_cluster", "vlm"],
    },
    "ray_cluster": {
        "name": "Ray Cluster",
        "description": "Enable Ray distributed computing",
        "category": "Distributed Processing",
        "config_key": "RAY_ADDRESS",
        "requires_api_key": None,
        "dependencies": [],
    },

    # Audio Features
    "audio_overviews": {
        "name": "Audio Overviews",
        "description": "Generate audio summaries of documents",
        "category": "Audio & Voice",
        "config_key": "ENABLE_AUDIO_OVERVIEWS",
        "requires_api_key": None,
        "dependencies": ["tts"],
    },
    "tts": {
        "name": "Text-to-Speech",
        "description": "Convert text to speech using various providers",
        "category": "Audio & Voice",
        "config_key": "ENABLE_TTS",
        "requires_api_key": None,
        "dependencies": [],
    },
    "voice_agents": {
        "name": "Voice Agents",
        "description": "Create voice-enabled AI agents",
        "category": "Audio & Voice",
        "config_key": "ENABLE_VOICE_AGENTS",
        "requires_api_key": None,
        "dependencies": ["tts"],
    },

    # Workflow & Agents
    "workflow_engine": {
        "name": "Workflow Engine",
        "description": "Create and run automated workflows",
        "category": "Workflow & Agents",
        "config_key": "ENABLE_WORKFLOW_ENGINE",
        "requires_api_key": None,
        "dependencies": [],
    },
    "chat_agents": {
        "name": "Chat Agents",
        "description": "Create custom chat agents with knowledge bases",
        "category": "Workflow & Agents",
        "config_key": "ENABLE_CHAT_AGENTS",
        "requires_api_key": None,
        "dependencies": [],
    },
    "agent_memory": {
        "name": "Agent Memory System",
        "description": "Persistent memory for agents across sessions",
        "category": "Workflow & Agents",
        "config_key": "ENABLE_AGENT_MEMORY",
        "requires_api_key": None,
        "dependencies": [],
    },

    # Integrations
    "google_drive": {
        "name": "Google Drive Sync",
        "description": "Sync documents from Google Drive",
        "category": "Integrations",
        "config_key": "ENABLE_GOOGLE_DRIVE",
        "requires_api_key": "GOOGLE_CLIENT_ID",
        "dependencies": [],
    },
    "web_scraping": {
        "name": "Web Scraping",
        "description": "Scrape and index web content",
        "category": "Integrations",
        "config_key": "ENABLE_WEB_SCRAPING",
        "requires_api_key": "FIRECRAWL_API_KEY",
        "dependencies": [],
    },
    "connectors": {
        "name": "External Connectors",
        "description": "Connect to external data sources",
        "category": "Integrations",
        "config_key": "ENABLE_CONNECTORS",
        "requires_api_key": None,
        "dependencies": [],
    },
}


def _get_feature_status(feature_id: str, feature_def: dict) -> tuple[bool, str]:
    """Get current feature status from settings."""
    from backend.core.config import settings

    config_key = feature_def.get("config_key", "")

    # Check if feature is enabled
    enabled = getattr(settings, config_key, False) if config_key else False

    # For string config keys (like RAY_ADDRESS), check if it's set
    if isinstance(enabled, str):
        enabled = bool(enabled and enabled not in ["", "None", "null"])

    # Check API key requirement
    requires_key = feature_def.get("requires_api_key")
    if requires_key:
        api_key = getattr(settings, requires_key, None)
        if not api_key or api_key.startswith("your-") or api_key.startswith("sk-your"):
            return enabled, "unavailable"

    # Check dependencies
    for dep in feature_def.get("dependencies", []):
        if dep in SYSTEM_FEATURES:
            dep_enabled, dep_status = _get_feature_status(dep, SYSTEM_FEATURES[dep])
            if not dep_enabled or dep_status == "unavailable":
                return enabled, "degraded"

    return enabled, "available"


@router.get("/features", response_model=FeatureToggleListResponse)
async def list_system_features(
    admin: AdminUser,
    category: Optional[str] = None,
):
    """
    List all system features with their current status.

    Superadmin only endpoint (Phase 55).

    Returns:
        List of features with enabled status, dependencies, and requirements
    """
    features = []
    categories = set()

    for feature_id, feature_def in SYSTEM_FEATURES.items():
        cat = feature_def["category"]
        categories.add(cat)

        if category and cat != category:
            continue

        enabled, status = _get_feature_status(feature_id, feature_def)

        features.append(FeatureToggle(
            id=feature_id,
            name=feature_def["name"],
            description=feature_def["description"],
            category=cat,
            enabled=enabled,
            requires_api_key=feature_def.get("requires_api_key"),
            dependencies=feature_def.get("dependencies", []),
            status=status,
        ))

    return FeatureToggleListResponse(
        features=features,
        categories=sorted(categories),
    )


@router.get("/features/{feature_id}", response_model=FeatureToggle)
async def get_feature_status(
    feature_id: str,
    admin: AdminUser,
):
    """
    Get status of a specific feature.

    Superadmin only endpoint (Phase 55).
    """
    if feature_id not in SYSTEM_FEATURES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature '{feature_id}' not found",
        )

    feature_def = SYSTEM_FEATURES[feature_id]
    enabled, feature_status = _get_feature_status(feature_id, feature_def)

    return FeatureToggle(
        id=feature_id,
        name=feature_def["name"],
        description=feature_def["description"],
        category=feature_def["category"],
        enabled=enabled,
        requires_api_key=feature_def.get("requires_api_key"),
        dependencies=feature_def.get("dependencies", []),
        status=feature_status,
    )


@router.put("/features/{feature_id}")
async def toggle_feature(
    feature_id: str,
    toggle: FeatureToggleUpdate,
    admin: AdminUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Enable or disable a system feature.

    Superadmin only endpoint (Phase 55).

    Note: Some features require server restart to take effect.
    """
    if feature_id not in SYSTEM_FEATURES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature '{feature_id}' not found",
        )

    feature_def = SYSTEM_FEATURES[feature_id]
    config_key = feature_def.get("config_key")

    if not config_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature '{feature_id}' cannot be toggled dynamically",
        )

    # Check dependencies when enabling
    if toggle.enabled:
        for dep in feature_def.get("dependencies", []):
            if dep in SYSTEM_FEATURES:
                dep_enabled, _ = _get_feature_status(dep, SYSTEM_FEATURES[dep])
                if not dep_enabled:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Cannot enable '{feature_id}': dependency '{dep}' is not enabled",
                    )

        # Check API key requirement
        requires_key = feature_def.get("requires_api_key")
        if requires_key:
            from backend.core.config import settings
            api_key = getattr(settings, requires_key, None)
            if not api_key or api_key.startswith("your-") or api_key.startswith("sk-your"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot enable '{feature_id}': {requires_key} is not configured",
                )

    # Update the setting
    settings_service = get_settings_service()
    old_value, _ = _get_feature_status(feature_id, feature_def)

    await settings_service.update_setting(
        key=config_key,
        value=toggle.enabled,
        user_id=admin.user_id,
        session=db,
    )

    # Log the change
    audit_service = get_audit_service()
    await audit_service.log_admin_action(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        admin_user_id=admin.user_id,
        target_resource_type="feature",
        target_resource_id=feature_id,
        changes={
            "feature": feature_id,
            "feature_name": feature_def["name"],
            "old_value": old_value,
            "new_value": toggle.enabled,
            "config_key": config_key,
        },
        ip_address=get_client_ip(request),
        session=db,
    )

    return {
        "feature_id": feature_id,
        "feature_name": feature_def["name"],
        "enabled": toggle.enabled,
        "requires_restart": config_key in [
            "RAY_ADDRESS", "ENABLE_COLBERT", "ENABLE_WARP",
        ],
        "message": f"Feature '{feature_def['name']}' {'enabled' if toggle.enabled else 'disabled'}",
    }


@router.get("/features/health/check")
async def check_features_health(
    admin: AdminUser,
):
    """
    Check health status of all features.

    Superadmin only endpoint (Phase 55).

    Returns counts of available, unavailable, and degraded features.
    """
    counts = {"available": 0, "unavailable": 0, "degraded": 0, "enabled": 0, "disabled": 0}
    issues = []

    for feature_id, feature_def in SYSTEM_FEATURES.items():
        enabled, status = _get_feature_status(feature_id, feature_def)

        counts[status] += 1
        counts["enabled" if enabled else "disabled"] += 1

        if status == "unavailable":
            issues.append({
                "feature_id": feature_id,
                "feature_name": feature_def["name"],
                "issue": f"Missing API key: {feature_def.get('requires_api_key')}",
            })
        elif status == "degraded":
            issues.append({
                "feature_id": feature_id,
                "feature_name": feature_def["name"],
                "issue": f"Missing dependencies: {feature_def.get('dependencies')}",
            })

    return {
        "counts": counts,
        "issues": issues,
        "total_features": len(SYSTEM_FEATURES),
    }


# =============================================================================
# VLM Configuration Endpoints (Phase 54)
# =============================================================================

class VLMConfigResponse(BaseModel):
    """VLM configuration response."""
    enabled: bool
    provider: str
    model: str
    max_images_per_request: int
    auto_process_visual_docs: bool
    extract_tables: bool
    extract_charts: bool
    ocr_fallback: bool


class VLMConfigUpdate(BaseModel):
    """VLM configuration update request."""
    enabled: Optional[bool] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    max_images_per_request: Optional[int] = Field(None, ge=1, le=20)
    auto_process_visual_docs: Optional[bool] = None
    extract_tables: Optional[bool] = None
    extract_charts: Optional[bool] = None
    ocr_fallback: Optional[bool] = None


class VLMTestRequest(BaseModel):
    """VLM test request."""
    image_url: Optional[str] = None
    test_text: Optional[str] = "Describe what you see in this test image."


@router.get("/vlm/config", response_model=VLMConfigResponse)
async def get_vlm_config(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get VLM configuration.

    Reads from database settings (rag.vlm_*) with fallback to environment variables.
    """
    from backend.services.settings import get_settings_service
    settings_service = get_settings_service()

    # Get settings from database
    db_settings = await settings_service.get_settings_batch([
        "rag.vlm_provider",
        "rag.vlm_model",
        "rag.ollama_vision_model",
        "processing.max_image_captioning_concurrency",
        "rag.extract_tables",
        "rag.extract_charts",
    ], session=db)

    # Determine enabled state (env var or has provider configured)
    enable_multimodal = os.getenv("ENABLE_MULTIMODAL", "false").lower() == "true"
    enabled = settings.ENABLE_VLM or enable_multimodal or bool(db_settings.get("rag.vlm_provider"))

    # Get provider (database > env > default)
    provider = db_settings.get("rag.vlm_provider", "auto")
    if provider == "auto":
        # Auto-detect provider
        if os.getenv("USE_OLLAMA") or os.getenv("OLLAMA_BASE_URL"):
            provider = "ollama"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "ollama"  # Default to ollama

    # Get model (database > env > default based on provider)
    model = db_settings.get("rag.vlm_model") or db_settings.get("rag.ollama_vision_model")
    if not model:
        if provider == "ollama":
            model = os.getenv("OLLAMA_VISION_MODEL", "llava")
        elif provider == "openai":
            model = "gpt-4o"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        else:
            model = "llava"

    # Get values with proper None handling
    max_images = db_settings.get("processing.max_image_captioning_concurrency")
    extract_tables = db_settings.get("rag.extract_tables")
    extract_charts = db_settings.get("rag.extract_charts")

    return VLMConfigResponse(
        enabled=enabled,
        provider=provider,
        model=model,
        max_images_per_request=max_images if max_images is not None else 10,
        auto_process_visual_docs=enabled,  # Same as enabled
        extract_tables=extract_tables if extract_tables is not None else True,
        extract_charts=extract_charts if extract_charts is not None else True,
        ocr_fallback=True,  # Always enabled as fallback
    )


@router.put("/vlm/config")
async def update_vlm_config(
    config: VLMConfigUpdate,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update VLM configuration.

    Saves to database settings (rag.vlm_*) for use by multimodal processing.
    """
    logger.info("Updating VLM config", admin_id=admin.user_id)

    # Map to database setting keys
    updates = {}
    if config.provider is not None:
        updates["rag.vlm_provider"] = config.provider
    if config.model is not None:
        updates["rag.vlm_model"] = config.model
        # Also update ollama-specific setting for backwards compatibility
        if config.provider == "ollama" or not config.provider:
            updates["rag.ollama_vision_model"] = config.model
    if config.max_images_per_request is not None:
        updates["processing.max_image_captioning_concurrency"] = config.max_images_per_request
    if config.extract_tables is not None:
        updates["rag.extract_tables"] = config.extract_tables
    if config.extract_charts is not None:
        updates["rag.extract_charts"] = config.extract_charts

    # Update settings in database using settings service
    from backend.services.settings import get_settings_service
    settings_service = get_settings_service()

    for key, value in updates.items():
        await settings_service.set_setting(key, value, session=db)

    await audit_service.log_admin_action(
        db=db,
        user_id=admin.user_id,
        action=AuditAction.UPDATE,
        target_resource_type="vlm_config",
        details={"updates": updates},
    )

    return {"message": "VLM configuration updated", "updates": updates}


@router.post("/vlm/test")
async def test_vlm(
    request: VLMTestRequest,
    admin: AdminUser,
):
    """
    Test VLM connection and processing.

    Superadmin only endpoint (Phase 54).
    """
    if not settings.ENABLE_VLM:
        return {
            "success": False,
            "error": "VLM is disabled",
            "message": "Enable VLM in settings first",
        }

    try:
        from backend.services.vlm_processor import get_vlm_processor

        processor = await get_vlm_processor()
        health = await processor.health_check()

        return {
            "success": health.get("healthy", False),
            "provider": health.get("provider", "unknown"),
            "model": health.get("model", "unknown"),
            "latency_ms": health.get("latency_ms"),
            "message": "VLM connection successful" if health.get("healthy") else "VLM connection failed",
        }

    except Exception as e:
        logger.error("VLM test failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "VLM test failed",
        }


# =============================================================================
# RLM Configuration Endpoints (Phase 54)
# =============================================================================

class RLMConfigResponse(BaseModel):
    """RLM configuration response."""
    enabled: bool
    provider: str
    context_threshold: int
    max_context_tokens: int
    sandbox_type: str
    enable_self_refinement: bool


class RLMConfigUpdate(BaseModel):
    """RLM configuration update request."""
    enabled: Optional[bool] = None
    provider: Optional[str] = None
    context_threshold: Optional[int] = Field(None, ge=10000)
    max_context_tokens: Optional[int] = Field(None, ge=100000)
    sandbox_type: Optional[str] = None
    enable_self_refinement: Optional[bool] = None


@router.get("/rlm/config", response_model=RLMConfigResponse)
async def get_rlm_config(
    admin: AdminUser,
):
    """
    Get RLM configuration.

    Superadmin only endpoint (Phase 54).
    """
    return RLMConfigResponse(
        enabled=settings.ENABLE_RLM,
        provider=settings.RLM_PROVIDER,
        context_threshold=settings.RLM_THRESHOLD,
        max_context_tokens=settings.RLM_MAX_CONTEXT,
        sandbox_type=settings.RLM_SANDBOX,
        enable_self_refinement=settings.RLM_SELF_REFINE,
    )


@router.put("/rlm/config")
async def update_rlm_config(
    config: RLMConfigUpdate,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update RLM configuration.

    Superadmin only endpoint (Phase 54).
    """
    logger.info("Updating RLM config", admin_id=admin.user_id)

    updates = {}
    if config.enabled is not None:
        updates["ENABLE_RLM"] = config.enabled
    if config.provider is not None:
        updates["RLM_PROVIDER"] = config.provider
    if config.context_threshold is not None:
        updates["RLM_THRESHOLD"] = config.context_threshold
    if config.max_context_tokens is not None:
        updates["RLM_MAX_CONTEXT"] = config.max_context_tokens
    if config.sandbox_type is not None:
        updates["RLM_SANDBOX"] = config.sandbox_type
    if config.enable_self_refinement is not None:
        updates["RLM_SELF_REFINE"] = config.enable_self_refinement

    # Update settings in database
    for key, value in updates.items():
        await _update_setting(db, key, value, admin.user_id)

    await audit_service.log_admin_action(
        db=db,
        user_id=admin.user_id,
        action=AuditAction.UPDATE,
        target_resource_type="rlm_config",
        details={"updates": updates},
    )

    return {"message": "RLM configuration updated", "updates": updates}


# =============================================================================
# Ray Status Endpoint (Phase 54)
# =============================================================================

class RayStatusResponse(BaseModel):
    """Ray cluster status response."""
    connected: bool
    address: str
    nodes: int
    cpus_total: int
    cpus_available: int
    gpus_total: int
    gpus_available: int
    memory_total_gb: float
    memory_available_gb: float
    tasks_pending: int
    actors_alive: int
    use_for_embeddings: bool
    use_for_kg: bool
    use_for_vlm: bool


@router.get("/ray/status", response_model=RayStatusResponse)
async def get_ray_status(
    admin: AdminUser,
):
    """
    Get Ray cluster status.

    Superadmin only endpoint (Phase 54).
    """
    try:
        from backend.services.distributed_processor import get_distributed_processor

        processor = await get_distributed_processor()
        health = await processor.health_check()

        ray_info = health.get("backends", {}).get("ray", {})

        return RayStatusResponse(
            connected=ray_info.get("available", False),
            address=getattr(settings, "RAY_ADDRESS", "auto"),
            nodes=ray_info.get("nodes", 0),
            cpus_total=ray_info.get("cpus", {}).get("total", 0),
            cpus_available=ray_info.get("cpus", {}).get("available", 0),
            gpus_total=ray_info.get("gpus", {}).get("total", 0),
            gpus_available=ray_info.get("gpus", {}).get("available", 0),
            memory_total_gb=ray_info.get("memory", {}).get("total_gb", 0),
            memory_available_gb=ray_info.get("memory", {}).get("available_gb", 0),
            tasks_pending=ray_info.get("tasks_pending", 0),
            actors_alive=ray_info.get("actors_alive", 0),
            use_for_embeddings=settings.USE_RAY_FOR_EMBEDDINGS,
            use_for_kg=settings.USE_RAY_FOR_KG,
            use_for_vlm=settings.USE_RAY_FOR_VLM,
        )

    except Exception as e:
        logger.warning("Failed to get Ray status", error=str(e))
        return RayStatusResponse(
            connected=False,
            address=getattr(settings, "RAY_ADDRESS", "auto"),
            nodes=0,
            cpus_total=0,
            cpus_available=0,
            gpus_total=0,
            gpus_available=0,
            memory_total_gb=0,
            memory_available_gb=0,
            tasks_pending=0,
            actors_alive=0,
            use_for_embeddings=settings.USE_RAY_FOR_EMBEDDINGS,
            use_for_kg=settings.USE_RAY_FOR_KG,
            use_for_vlm=settings.USE_RAY_FOR_VLM,
        )


@router.put("/ray/config")
async def update_ray_config(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
    use_for_embeddings: Optional[bool] = None,
    use_for_kg: Optional[bool] = None,
    use_for_vlm: Optional[bool] = None,
):
    """
    Update Ray configuration.

    Superadmin only endpoint (Phase 54).
    """
    logger.info("Updating Ray config", admin_id=admin.user_id)

    updates = {}
    if use_for_embeddings is not None:
        updates["USE_RAY_FOR_EMBEDDINGS"] = use_for_embeddings
    if use_for_kg is not None:
        updates["USE_RAY_FOR_KG"] = use_for_kg
    if use_for_vlm is not None:
        updates["USE_RAY_FOR_VLM"] = use_for_vlm

    # Update settings in database
    for key, value in updates.items():
        await _update_setting(db, key, value, admin.user_id)

    await audit_service.log_admin_action(
        db=db,
        user_id=admin.user_id,
        action=AuditAction.UPDATE,
        target_resource_type="ray_config",
        details={"updates": updates},
    )

    return {"message": "Ray configuration updated", "updates": updates}


# =============================================================================
# Analytics Endpoints (Phase 59: Wire analytics service)
# =============================================================================

@router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    admin: AdminUser,
    period: str = "day",
):
    """
    Get comprehensive analytics dashboard data.

    Phase 59: Integrates the previously unused analytics service.

    Returns query metrics, document metrics, user metrics, and system health.
    """
    from backend.services.analytics import get_analytics_service, MetricPeriod

    try:
        service = get_analytics_service()

        # Convert period string to enum
        period_map = {
            "hour": MetricPeriod.HOUR,
            "day": MetricPeriod.DAY,
            "week": MetricPeriod.WEEK,
            "month": MetricPeriod.MONTH,
        }
        metric_period = period_map.get(period, MetricPeriod.DAY)

        # Get dashboard data
        data = await service.get_dashboard_data(period=metric_period)

        return {
            "query_metrics": {
                "total_queries": data.query_metrics.total_queries,
                "successful_queries": data.query_metrics.successful_queries,
                "failed_queries": data.query_metrics.failed_queries,
                "avg_latency_ms": data.query_metrics.avg_latency_ms,
                "p50_latency_ms": data.query_metrics.p50_latency_ms,
                "p95_latency_ms": data.query_metrics.p95_latency_ms,
                "p99_latency_ms": data.query_metrics.p99_latency_ms,
                "cache_hit_rate": data.query_metrics.cache_hit_rate,
                "by_intent": data.query_metrics.by_intent,
                "by_complexity": data.query_metrics.by_complexity,
            },
            "document_metrics": {
                "total_documents": data.document_metrics.total_documents,
                "total_chunks": data.document_metrics.total_chunks,
                "by_type": data.document_metrics.by_type,
                "by_status": data.document_metrics.by_status,
                "documents_added_today": data.document_metrics.documents_added_today,
            },
            "system_health": {
                "status": data.system_health.status,
                "uptime_seconds": data.system_health.uptime_seconds,
                "error_rate": data.system_health.error_rate,
                "avg_response_time_ms": data.system_health.avg_response_time_ms,
            },
            "period": period,
        }

    except Exception as e:
        logger.error("Failed to get analytics dashboard", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analytics error: {str(e)}")


@router.get("/analytics/queries")
async def get_query_analytics(
    admin: AdminUser,
    period: str = "day",
    limit: int = 100,
):
    """
    Get detailed query analytics and recent queries.

    Phase 59: Provides query performance metrics and logs.
    """
    from backend.services.analytics import get_analytics_service, MetricPeriod

    try:
        service = get_analytics_service()

        period_map = {
            "hour": MetricPeriod.HOUR,
            "day": MetricPeriod.DAY,
            "week": MetricPeriod.WEEK,
            "month": MetricPeriod.MONTH,
        }
        metric_period = period_map.get(period, MetricPeriod.DAY)

        # Get query metrics
        metrics = await service.get_query_metrics(period=metric_period)

        # Get recent queries
        recent = service.get_recent_queries(limit=limit)

        return {
            "metrics": {
                "total_queries": metrics.total_queries,
                "successful_queries": metrics.successful_queries,
                "failed_queries": metrics.failed_queries,
                "avg_latency_ms": metrics.avg_latency_ms,
                "p50_latency_ms": metrics.p50_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "p99_latency_ms": metrics.p99_latency_ms,
                "avg_sources_per_query": metrics.avg_sources_per_query,
                "cache_hit_rate": metrics.cache_hit_rate,
                "by_intent": metrics.by_intent,
                "by_complexity": metrics.by_complexity,
                "by_strategy": metrics.by_strategy,
                "queries_per_hour": metrics.queries_per_hour,
            },
            "recent_queries": [q.to_dict() for q in recent],
            "period": period,
        }

    except Exception as e:
        logger.error("Failed to get query analytics", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analytics error: {str(e)}")


@router.get("/analytics/documents")
async def get_document_analytics(
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get document and indexing analytics.

    Phase 59: Provides document metrics and index health.
    """
    from backend.services.analytics import get_analytics_service

    try:
        service = get_analytics_service()
        metrics = await service.get_document_metrics(session=db)

        return {
            "total_documents": metrics.total_documents,
            "total_chunks": metrics.total_chunks,
            "total_embeddings": metrics.total_embeddings,
            "by_type": metrics.by_type,
            "by_status": metrics.by_status,
            "avg_chunk_size": metrics.avg_chunk_size,
            "avg_chunks_per_doc": metrics.avg_chunks_per_doc,
            "index_last_updated": metrics.index_last_updated.isoformat() if metrics.index_last_updated else None,
            "documents_added_today": metrics.documents_added_today,
            "documents_added_week": metrics.documents_added_week,
        }

    except Exception as e:
        logger.error("Failed to get document analytics", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analytics error: {str(e)}")


@router.post("/analytics/log-query")
async def log_query_for_analytics(
    admin: AdminUser,
    query_id: str,
    query: str,
    user_id: str,
    latency_ms: float,
    source_count: int = 0,
    intent: str = "general",
    complexity: str = "simple",
    strategy: str = "standard",
    cache_hit: bool = False,
    success: bool = True,
    error: Optional[str] = None,
):
    """
    Manually log a query for analytics tracking.

    Phase 59: Allows manual query logging for testing/admin purposes.
    """
    from backend.services.analytics import get_analytics_service

    try:
        service = get_analytics_service()
        await service.log_query(
            query_id=query_id,
            query=query,
            user_id=user_id,
            latency_ms=latency_ms,
            source_count=source_count,
            intent=intent,
            complexity=complexity,
            strategy=strategy,
            cache_hit=cache_hit,
            success=success,
            error=error,
        )

        return {"message": "Query logged successfully", "query_id": query_id}

    except Exception as e:
        logger.error("Failed to log query", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Logging error: {str(e)}")


# =============================================================================
# Phase 70: Circuit Breaker Monitoring
# =============================================================================

class CircuitBreakerStats(BaseModel):
    """Circuit breaker statistics."""
    name: str
    state: str
    failure_count: int
    success_count: int
    last_failure_time: Optional[float] = None
    config: Dict[str, Any]


class CircuitBreakerStatsResponse(BaseModel):
    """Response for circuit breaker stats."""
    llm_circuits: Dict[str, CircuitBreakerStats]
    total_circuits: int
    open_circuits: List[str]
    healthy: bool


@router.get("/resilience/circuit-breakers", response_model=CircuitBreakerStatsResponse)
async def get_circuit_breaker_stats(
    _user: AdminUser,
):
    """
    Get circuit breaker statistics (admin only).

    Returns health and state of all LLM circuit breakers.
    Open circuits indicate services that are failing and being protected.
    """
    try:
        from backend.services.rag import get_llm_circuit_breaker_stats

        llm_stats = get_llm_circuit_breaker_stats()

        # Transform to response format
        circuits = {}
        open_circuits = []
        for name, stats in llm_stats.items():
            circuits[name] = CircuitBreakerStats(
                name=stats["name"],
                state=stats["state"],
                failure_count=stats["failure_count"],
                success_count=stats["success_count"],
                last_failure_time=stats.get("last_failure_time"),
                config=stats.get("config", {}),
            )
            if stats["state"] == "open":
                open_circuits.append(name)

        return CircuitBreakerStatsResponse(
            llm_circuits=circuits,
            total_circuits=len(circuits),
            open_circuits=open_circuits,
            healthy=len(open_circuits) == 0,
        )

    except Exception as e:
        logger.error("Failed to get circuit breaker stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get circuit breaker stats: {str(e)}"
        )


@router.post("/resilience/circuit-breakers/{circuit_name}/reset")
async def reset_circuit_breaker(
    circuit_name: str,
    _user: AdminUser,
):
    """
    Manually reset a circuit breaker (admin only).

    Use this to force a circuit breaker back to closed state
    after the underlying issue has been resolved.
    """
    try:
        from backend.services.resilience import CircuitBreaker

        cb = CircuitBreaker.get(circuit_name)
        if not cb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Circuit breaker '{circuit_name}' not found"
            )

        old_state = cb.state.value
        cb.reset()

        logger.info(
            "Circuit breaker manually reset",
            name=circuit_name,
            old_state=old_state,
        )

        return {
            "message": f"Circuit breaker '{circuit_name}' reset successfully",
            "old_state": old_state,
            "new_state": "closed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reset circuit breaker", name=circuit_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset circuit breaker: {str(e)}"
        )


# =============================================================================
# Performance Optimization Status (Phase 3-4)
# =============================================================================

@router.get(
    "/performance",
    tags=["admin"],
    summary="Get performance optimization status",
    response_model=Dict[str, Any],
)
async def get_performance_status(
    admin: AdminUser,
) -> Dict[str, Any]:
    """
    Get status of all performance optimizations.

    Returns information about:
    - Cython extensions (compiled vs fallback)
    - GPU acceleration (device, memory)
    - MinHash deduplicator (method, complexity)
    """
    try:
        from backend.services.performance_init import get_performance_status
        return get_performance_status()
    except ImportError:
        return {
            "error": "Performance module not available",
            "cython": {"status": "not_installed"},
            "gpu": {"status": "not_installed"},
            "minhash": {"status": "not_installed"},
        }
    except Exception as e:
        logger.error("Failed to get performance status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance status: {str(e)}"
        )


# =============================================================================
# Garbage Chunk Cleanup (Phase 98)
# =============================================================================

class GarbageCleanupRequest(BaseModel):
    """Request model for garbage chunk cleanup."""
    dry_run: bool = Field(
        default=True,
        description="If True, only count garbage chunks without deleting"
    )


class GarbageCleanupResponse(BaseModel):
    """Response model for garbage chunk cleanup."""
    dry_run: bool
    total_garbage_chunks: int
    by_document: Dict[str, int]
    deleted: bool
    message: str


@router.post(
    "/cleanup-garbage-chunks",
    tags=["admin"],
    summary="Clean up garbage chunks (image credits, copyright notices)",
    response_model=GarbageCleanupResponse,
)
async def cleanup_garbage_chunks(
    request: GarbageCleanupRequest,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
) -> GarbageCleanupResponse:
    """
    Scan and optionally delete garbage chunks from the database.

    Garbage chunks are those containing:
    - Image credits (Getty Images, Shutterstock, etc.)
    - Stock photo attributions
    - Copyright notices that don't add value to RAG retrieval

    Use dry_run=true (default) to see what would be deleted first.
    """
    import re
    from backend.db.models import Chunk as ChunkModel, Document as DocumentModel

    # Garbage detection patterns (same as text_preprocessor.py)
    garbage_patterns = [
        r'getty\s*images',
        r'shutterstock',
        r'adobe\s*stock',
        r'123rf',
        r'istock(?:photo)?',
        r'alamy',
        r'dreamstime',
        r'depositphotos',
        r'(?:image|photo|picture)\s*(?:credit|courtesy|source|by)\s*:',
        r'stock\s+(?:photo|image|picture)',
        r'bildnachweis',
        r'fotocredit',
    ]
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in garbage_patterns]

    try:
        # Query all chunks
        result = await db.execute(
            select(ChunkModel, DocumentModel.title)
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
        )
        rows = result.all()

        garbage_chunks = []
        by_document: Dict[str, int] = {}

        for chunk, doc_title in rows:
            content_lower = (chunk.content or "").lower()

            # Check if chunk matches garbage patterns
            match_count = sum(1 for p in compiled_patterns if p.search(content_lower))

            if match_count >= 3:  # Multiple matches = likely garbage
                garbage_chunks.append(chunk)
                doc_name = doc_title or "Unknown"
                by_document[doc_name] = by_document.get(doc_name, 0) + 1

        total_garbage = len(garbage_chunks)

        if request.dry_run:
            return GarbageCleanupResponse(
                dry_run=True,
                total_garbage_chunks=total_garbage,
                by_document=by_document,
                deleted=False,
                message=f"Found {total_garbage} garbage chunks. Set dry_run=false to delete.",
            )

        # Actually delete the garbage chunks
        if garbage_chunks:
            garbage_ids = [str(c.id) for c in garbage_chunks]

            # Delete from SQLite
            for chunk in garbage_chunks:
                await db.delete(chunk)
            await db.commit()

            # Delete from ChromaDB
            try:
                from backend.services.vectorstore_local import LocalVectorStore
                vectorstore = LocalVectorStore()
                vectorstore._collection.delete(ids=garbage_ids)
                logger.info("Deleted garbage chunks from ChromaDB", count=len(garbage_ids))
            except Exception as chroma_err:
                logger.warning(
                    "Failed to delete from ChromaDB (may not exist there)",
                    error=str(chroma_err),
                )

            logger.info(
                "Garbage chunks cleaned up",
                admin_user=admin.user_id,
                total_deleted=total_garbage,
            )

        return GarbageCleanupResponse(
            dry_run=False,
            total_garbage_chunks=total_garbage,
            by_document=by_document,
            deleted=True,
            message=f"Deleted {total_garbage} garbage chunks from {len(by_document)} documents.",
        )

    except Exception as e:
        logger.error("Failed to cleanup garbage chunks", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup garbage chunks: {str(e)}"
        )


# =============================================================================
# Phase 98: Feature Audit Endpoint
# =============================================================================

class FeatureStatus(BaseModel):
    """Status of a single feature."""
    config_default: bool
    runtime_value: bool
    functional: bool
    reason: Optional[str] = None


class FeatureAuditResponse(BaseModel):
    """Response for feature audit endpoint."""
    features: Dict[str, FeatureStatus]
    summary: Dict[str, int]  # counts by status


@router.post("/feature-audit", response_model=FeatureAuditResponse)
async def audit_features(
    admin: AdminUser,
):
    """
    Audit all advanced RAG features and their current status.

    Phase 98: Returns the status of all advanced features, showing:
    - config_default: What the feature defaults to in config
    - runtime_value: What it's currently set to at runtime
    - functional: Whether the feature can actually work (dependencies met)
    - reason: Why a feature might not be functional

    This helps diagnose why features might not be working as expected.
    """
    from backend.core.config import settings

    features = {}

    # Tiered Reranking
    try:
        tiered_functional = True
        tiered_reason = None
        try:
            from backend.services.tiered_reranking import get_tiered_reranker
            # Just check if module imports
        except ImportError as e:
            tiered_functional = False
            tiered_reason = f"Import error: {str(e)}"
        features["tiered_reranking"] = FeatureStatus(
            config_default=True,  # Default in config.py
            runtime_value=settings.ENABLE_TIERED_RERANKING,
            functional=tiered_functional,
            reason=tiered_reason,
        )
    except Exception:
        features["tiered_reranking"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # LightRAG
    try:
        lightrag_functional = True
        lightrag_reason = None
        try:
            from backend.services.lightrag_service import get_lightrag_service
        except ImportError as e:
            lightrag_functional = False
            lightrag_reason = f"Import error: {str(e)}"
        features["lightrag"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_LIGHTRAG,
            functional=lightrag_functional,
            reason=lightrag_reason,
        )
    except Exception:
        features["lightrag"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # RAPTOR
    try:
        raptor_functional = True
        raptor_reason = None
        try:
            from backend.services.raptor_retriever import RaptorRetriever
        except ImportError as e:
            raptor_functional = False
            raptor_reason = f"Import error: {str(e)}"
        features["raptor"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_RAPTOR,
            functional=raptor_functional,
            reason=raptor_reason,
        )
    except Exception:
        features["raptor"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # Self-RAG
    try:
        features["self_rag"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_SELF_RAG,
            functional=True,
            reason=None,
        )
    except Exception:
        features["self_rag"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # Adaptive Routing
    try:
        features["adaptive_routing"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_ADAPTIVE_ROUTING,
            functional=True,
            reason=None,
        )
    except Exception:
        features["adaptive_routing"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # RAG Fusion
    try:
        features["rag_fusion"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_RAG_FUSION,
            functional=True,
            reason=None,
        )
    except Exception:
        features["rag_fusion"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # Context Compression
    try:
        features["context_compression"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_CONTEXT_COMPRESSION,
            functional=True,
            reason=None,
        )
    except Exception:
        features["context_compression"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # Knowledge Graph
    try:
        kg_functional = True
        kg_reason = None
        try:
            from backend.services.knowledge_graph import get_kg_service
        except ImportError as e:
            kg_functional = False
            kg_reason = f"Import error: {str(e)}"
        features["knowledge_graph"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.KG_ENABLED,
            functional=kg_functional,
            reason=kg_reason,
        )
    except Exception:
        features["knowledge_graph"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # LazyGraphRAG
    try:
        features["lazy_graphrag"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_LAZY_GRAPHRAG,
            functional=True,
            reason=None,
        )
    except Exception:
        features["lazy_graphrag"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # User Personalization
    try:
        features["user_personalization"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_USER_PERSONALIZATION,
            functional=True,
            reason=None,
        )
    except Exception:
        features["user_personalization"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # Agent Memory
    try:
        agent_mem_functional = True
        agent_mem_reason = None
        try:
            from backend.services.mem0_memory import get_memory_service
        except ImportError as e:
            agent_mem_functional = False
            agent_mem_reason = f"Import error: {str(e)}"
        features["agent_memory"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_AGENT_MEMORY,
            functional=agent_mem_functional,
            reason=agent_mem_reason,
        )
    except Exception:
        features["agent_memory"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # RLM (Recursive Language Model)
    try:
        rlm_functional = True
        rlm_reason = None
        try:
            from backend.services.rlm_service import RecursiveLMService
        except ImportError as e:
            rlm_functional = False
            rlm_reason = f"Import error: {str(e)}"
        features["rlm"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_RLM,
            functional=rlm_functional,
            reason=rlm_reason,
        )
    except Exception:
        features["rlm"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # VLM (Vision Language Model)
    try:
        vlm_functional = True
        vlm_reason = None
        try:
            from backend.services.vlm_processor import get_vlm_processor
        except ImportError as e:
            vlm_functional = False
            vlm_reason = f"Import error: {str(e)}"
        features["vlm"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_VLM,
            functional=vlm_functional,
            reason=vlm_reason,
        )
    except Exception:
        features["vlm"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # Ultra-fast TTS
    try:
        tts_functional = True
        tts_reason = None
        try:
            from backend.services.audio.ultra_fast_tts import get_ultra_fast_tts
        except ImportError as e:
            tts_functional = False
            tts_reason = f"Import error: {str(e)}"
        features["ultra_fast_tts"] = FeatureStatus(
            config_default=True,
            runtime_value=settings.ENABLE_ULTRA_FAST_TTS,
            functional=tts_functional,
            reason=tts_reason,
        )
    except Exception:
        features["ultra_fast_tts"] = FeatureStatus(
            config_default=True,
            runtime_value=False,
            functional=False,
            reason="Failed to check status",
        )

    # Calculate summary
    enabled_count = sum(1 for f in features.values() if f.runtime_value)
    functional_count = sum(1 for f in features.values() if f.functional and f.runtime_value)
    disabled_count = sum(1 for f in features.values() if not f.runtime_value)

    summary = {
        "total": len(features),
        "enabled": enabled_count,
        "functional": functional_count,
        "disabled": disabled_count,
        "enabled_but_not_functional": enabled_count - functional_count,
    }

    logger.info(
        "Feature audit completed",
        admin_user=admin.user_id,
        enabled=enabled_count,
        functional=functional_count,
    )

    return FeatureAuditResponse(
        features=features,
        summary=summary,
    )


# =============================================================================
# Phase 98: Tag Backfill Endpoint
# =============================================================================

class TagBackfillRequest(BaseModel):
    """Request for tag backfill."""
    dry_run: bool = Field(default=True, description="If true, only count documents needing backfill")
    reembed: bool = Field(default=False, description="Also re-embed chunks with tag prefix (slow)")
    document_ids: Optional[List[str]] = Field(default=None, description="Specific document IDs to backfill (null = all)")


class TagBackfillResponse(BaseModel):
    """Response for tag backfill."""
    dry_run: bool
    total_documents: int
    documents_with_tags: int
    chunks_updated: int
    reembedding_queued: int
    kg_entities_updated: int = 0
    message: str


@router.post("/backfill-tags", response_model=TagBackfillResponse)
async def backfill_document_tags(
    request: TagBackfillRequest,
    admin: AdminUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Backfill document tags to chunk metadata and ChromaDB.

    Phase 98: For documents uploaded before tag-augmented embeddings were added,
    this endpoint propagates existing document tags to:
    1. Chunk metadata (document_tags field)
    2. ChromaDB metadata (for filtering)
    3. Optionally re-embeds chunks with tag prefix (slow, requires Celery)

    Use dry_run=true (default) to see what would be updated first.
    """
    from backend.db.models import Document, Chunk
    from backend.services.vectorstore_local import get_chroma_vector_store as get_local_vectorstore

    try:
        # Query documents with tags
        query = select(Document).where(Document.tags.isnot(None))
        if request.document_ids:
            from uuid import UUID as PyUUID
            doc_uuids = [PyUUID(did) for did in request.document_ids]
            query = query.where(Document.id.in_(doc_uuids))

        result = await db.execute(query)
        documents = result.scalars().all()

        total_documents = len(documents)
        documents_with_tags = sum(1 for d in documents if d.tags and len(d.tags) > 0)
        chunks_updated = 0
        reembedding_queued = 0
        kg_entities_updated = 0

        if request.dry_run:
            # Count chunks that would be updated
            for doc in documents:
                if doc.tags:
                    chunk_result = await db.execute(
                        select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
                    )
                    chunks_updated += chunk_result.scalar() or 0

            return TagBackfillResponse(
                dry_run=True,
                total_documents=total_documents,
                documents_with_tags=documents_with_tags,
                chunks_updated=chunks_updated,
                reembedding_queued=0,
                message=f"Would update {chunks_updated} chunks across {documents_with_tags} documents. Set dry_run=false to apply.",
            )

        # Actually backfill
        vectorstore = get_local_vectorstore()

        for doc in documents:
            if not doc.tags:
                continue

            tags = doc.tags

            # Update chunk metadata in DB
            chunks_result = await db.execute(
                select(Chunk).where(Chunk.document_id == doc.id)
            )
            chunks = chunks_result.scalars().all()

            for chunk in chunks:
                chunk_metadata = chunk.chunk_metadata or {}
                chunk_metadata["document_tags"] = tags
                chunk.chunk_metadata = chunk_metadata
                chunks_updated += 1

            # Update ChromaDB metadata
            if vectorstore:
                try:
                    await vectorstore.update_document_tags(
                        document_id=str(doc.id),
                        tags=tags,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to update ChromaDB tags",
                        document_id=str(doc.id),
                        error=str(e),
                    )

            # Queue re-embedding if requested
            if request.reembed:
                try:
                    from backend.tasks.document_tasks import reembed_document_chunks
                    reembed_document_chunks.delay(str(doc.id), tags)
                    reembedding_queued += 1
                except Exception as e:
                    logger.warning(
                        "Failed to queue re-embedding",
                        document_id=str(doc.id),
                        error=str(e),
                    )

            # Sync tags to KG entities linked to this document
            try:
                from backend.db.models import Entity, EntityMention
                mentions_result = await db.execute(
                    select(EntityMention.entity_id)
                    .where(EntityMention.document_id == doc.id)
                    .distinct()
                )
                entity_ids = mentions_result.scalars().all()
                for entity_id in entity_ids:
                    entity_result = await db.execute(
                        select(Entity).where(Entity.id == entity_id)
                    )
                    entity = entity_result.scalar_one_or_none()
                    if entity:
                        props = entity.properties or {}
                        source_tags = props.get("source_document_tags", {})
                        source_tags[str(doc.id)] = tags
                        props["source_document_tags"] = source_tags
                        entity.properties = props
                        kg_entities_updated += 1
            except Exception as e:
                logger.warning(
                    "Failed to sync tags to KG entities",
                    document_id=str(doc.id),
                    error=str(e),
                )

        await db.commit()

        logger.info(
            "Tag backfill completed",
            admin_user=admin.user_id,
            documents=documents_with_tags,
            chunks=chunks_updated,
            reembedding_queued=reembedding_queued,
        )

        return TagBackfillResponse(
            dry_run=False,
            total_documents=total_documents,
            documents_with_tags=documents_with_tags,
            chunks_updated=chunks_updated,
            reembedding_queued=reembedding_queued,
            kg_entities_updated=kg_entities_updated,
            message=f"Updated {chunks_updated} chunks across {documents_with_tags} documents."
            + (f" Queued {reembedding_queued} documents for re-embedding." if reembedding_queued else "")
            + (f" Synced tags to {kg_entities_updated} KG entities." if kg_entities_updated else ""),
        )

    except Exception as e:
        logger.error("Tag backfill failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tag backfill failed: {str(e)}"
        )


# ── Backfill ChromaDB chunk metadata with document filenames ────────────
@router.post("/backfill-chunk-metadata")
async def backfill_chunk_metadata(
    dry_run: bool = True,
    admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Retroactively update ChromaDB chunk metadata with document_filename
    for all chunks. Fixes 'unknown' source names in chat sources.
    """
    from backend.db.models import Document, Chunk
    from backend.services.vectorstore_local import get_chroma_vector_store as get_local_vectorstore

    # Get all documents with their filenames
    result = await db.execute(
        select(Document.id, Document.filename).where(Document.filename.isnot(None))
    )
    documents = result.all()

    if not documents:
        return {"message": "No documents found", "total": 0, "updated": 0}

    doc_name_map = {str(doc.id): doc.filename for doc in documents}

    if dry_run:
        return {
            "dry_run": True,
            "total_documents": len(documents),
            "message": f"Would update chunk metadata for {len(documents)} documents. Run with dry_run=false to apply.",
        }

    # Get ChromaDB collection
    vectorstore = get_local_vectorstore()
    collection = vectorstore._collection

    chunks_updated = 0
    docs_processed = 0
    errors = []

    for doc_id, filename in doc_name_map.items():
        try:
            # Get all chunks for this document from DB
            chunk_result = await db.execute(
                select(Chunk.id).where(Chunk.document_id == doc_id)
            )
            chunk_ids = [str(c.id) for c in chunk_result.scalars().all()]

            if not chunk_ids:
                continue

            # Update ChromaDB metadata for each chunk
            for chunk_id in chunk_ids:
                try:
                    existing = collection.get(ids=[chunk_id], include=["metadatas"])
                    if existing and existing["metadatas"]:
                        meta = existing["metadatas"][0]
                        if not meta.get("document_name") or meta.get("document_name") == "unknown":
                            meta["document_name"] = filename
                            meta["document_filename"] = filename
                            collection.update(ids=[chunk_id], metadatas=[meta])
                            chunks_updated += 1
                except Exception:
                    pass  # Skip individual chunk errors

            docs_processed += 1
        except Exception as e:
            errors.append(f"Doc {doc_id[:8]}: {str(e)[:100]}")

    logger.info(
        "Chunk metadata backfill completed",
        admin_user=admin.user_id,
        docs_processed=docs_processed,
        chunks_updated=chunks_updated,
    )

    return {
        "dry_run": False,
        "total_documents": len(documents),
        "docs_processed": docs_processed,
        "chunks_updated": chunks_updated,
        "errors": errors[:10] if errors else [],
        "message": f"Updated {chunks_updated} chunks across {docs_processed} documents.",
    }


# =============================================================================
# Storage Stats
# =============================================================================

@router.get("/storage/stats")
async def get_storage_stats(
    admin: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """Get document storage breakdown — local vs external, by source type."""
    from sqlalchemy import case, literal_column

    # Query storage breakdown
    results = await db.execute(
        select(
            Document.source_type,
            func.coalesce(Document.is_stored_locally, True).label("is_local"),
            func.count().label("doc_count"),
            func.coalesce(func.sum(Document.file_size), 0).label("total_size"),
        )
        .group_by(Document.source_type, "is_local")
    )
    rows = results.all()

    local_count = 0
    local_size = 0
    external_count = 0
    external_size = 0
    by_source_type: dict = {}

    for row in rows:
        source_type = row.source_type or "local_upload"
        is_local = row.is_local
        count = row.doc_count
        size = row.total_size or 0

        if is_local:
            local_count += count
            local_size += size
        else:
            external_count += count
            external_size += size

        if source_type not in by_source_type:
            by_source_type[source_type] = {"count": 0, "size": 0, "external_count": 0}

        by_source_type[source_type]["count"] += count
        by_source_type[source_type]["size"] += size
        if not is_local:
            by_source_type[source_type]["external_count"] += count

    return {
        "local_count": local_count,
        "external_count": external_count,
        "local_size_bytes": local_size,
        "external_size_bytes": external_size,
        "total_count": local_count + external_count,
        "total_size_bytes": local_size + external_size,
        "by_source_type": by_source_type,
    }
