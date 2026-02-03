"""
AIDocumentIndexer - Link Groups API Routes
==========================================

Endpoints for managing link groups and saved links for web scraping.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
import hashlib

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from backend.db.database import get_async_session
from backend.db.models import (
    LinkGroup,
    SavedLink,
    ScrapedContentHistory,
)
from backend.api.middleware.auth import AuthenticatedUser


def _to_uuid(user_id: str) -> UUID:
    """Convert user_id string to UUID safely."""
    return UUID(user_id) if isinstance(user_id, str) else user_id


logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class LinkGroupCreate(BaseModel):
    """Request to create a link group."""
    name: str = Field(..., min_length=1, max_length=255, description="Group name")
    description: Optional[str] = Field(None, description="Group description")
    color: Optional[str] = Field(None, max_length=20, description="Hex color code")
    icon: Optional[str] = Field(None, max_length=50, description="Icon name")
    is_shared: bool = Field(default=False, description="Share with organization")


class LinkGroupUpdate(BaseModel):
    """Request to update a link group."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    color: Optional[str] = Field(None, max_length=20)
    icon: Optional[str] = Field(None, max_length=50)
    is_shared: Optional[bool] = None
    sort_order: Optional[int] = None


class LinkGroupResponse(BaseModel):
    """Response model for a link group."""
    id: str
    name: str
    description: Optional[str]
    color: Optional[str]
    icon: Optional[str]
    is_shared: bool
    sort_order: int
    link_count: int
    created_at: datetime
    updated_at: datetime


class SavedLinkCreate(BaseModel):
    """Request to create a saved link."""
    url: str = Field(..., description="URL to save")
    title: Optional[str] = Field(None, max_length=500, description="Link title")
    description: Optional[str] = Field(None, description="Link description")
    group_id: str = Field(..., description="Group ID to add link to")
    tags: Optional[List[str]] = Field(None, description="Tags for the link")
    auto_scrape: bool = Field(default=False, description="Include in auto scraping")
    scrape_frequency: Optional[str] = Field(None, description="Scrape frequency: daily, weekly, monthly")


class SavedLinkUpdate(BaseModel):
    """Request to update a saved link."""
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    group_id: Optional[str] = None
    tags: Optional[List[str]] = None
    auto_scrape: Optional[bool] = None
    scrape_frequency: Optional[str] = None
    sort_order: Optional[int] = None


class SavedLinkResponse(BaseModel):
    """Response model for a saved link."""
    id: str
    url: str
    title: Optional[str]
    description: Optional[str]
    favicon_url: Optional[str]
    group_id: str
    group_name: Optional[str]
    tags: Optional[List[str]]
    auto_scrape: bool
    scrape_frequency: Optional[str]
    last_scraped_at: Optional[datetime]
    last_scrape_status: Optional[str]
    scrape_count: int
    cached_word_count: Optional[int]
    cached_content_preview: Optional[str]
    sort_order: int
    created_at: datetime
    updated_at: datetime


class BulkLinkCreate(BaseModel):
    """Request to create multiple links at once."""
    urls: List[str] = Field(..., min_length=1, max_length=100, description="URLs to save")
    group_id: str = Field(..., description="Group ID to add links to")
    auto_scrape: bool = Field(default=False, description="Include in auto scraping")


class ScrapedContentResponse(BaseModel):
    """Response model for scraped content history."""
    id: str
    saved_link_id: str
    url: str
    title: Optional[str]
    content: Optional[str]
    word_count: int
    scraped_at: datetime
    status: str
    error_message: Optional[str]
    links_found: int
    images_found: int
    metadata: Optional[dict]
    indexed_to_rag: bool


class LinkGroupListResponse(BaseModel):
    """Response for listing link groups."""
    groups: List[LinkGroupResponse]
    total: int


class SavedLinkListResponse(BaseModel):
    """Response for listing saved links."""
    links: List[SavedLinkResponse]
    total: int


class ScrapedContentListResponse(BaseModel):
    """Response for listing scraped content history."""
    history: List[ScrapedContentResponse]
    total: int


# =============================================================================
# Link Group Endpoints
# =============================================================================

@router.post("/groups", response_model=LinkGroupResponse, status_code=status.HTTP_201_CREATED)
async def create_link_group(
    request: LinkGroupCreate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new link group for organizing saved URLs.
    """
    user_uuid = _to_uuid(user.user_id)

    # Check if group with same name already exists for this user
    existing = await db.execute(
        select(LinkGroup).where(
            LinkGroup.user_id == user_uuid,
            LinkGroup.name == request.name,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Link group '{request.name}' already exists",
        )

    # Get max sort order for user
    max_order = await db.execute(
        select(func.max(LinkGroup.sort_order)).where(LinkGroup.user_id == user_uuid)
    )
    next_order = (max_order.scalar() or 0) + 1

    group = LinkGroup(
        name=request.name,
        description=request.description,
        color=request.color,
        icon=request.icon,
        is_shared=request.is_shared,
        user_id=user_uuid,
        sort_order=next_order,
    )

    db.add(group)
    await db.commit()
    await db.refresh(group)

    logger.info("Created link group", group_id=str(group.id), name=request.name)

    return LinkGroupResponse(
        id=str(group.id),
        name=group.name,
        description=group.description,
        color=group.color,
        icon=group.icon,
        is_shared=group.is_shared,
        sort_order=group.sort_order,
        link_count=0,
        created_at=group.created_at,
        updated_at=group.updated_at,
    )


@router.get("/groups", response_model=LinkGroupListResponse)
async def list_link_groups(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all link groups for the current user.
    """
    user_uuid = _to_uuid(user.user_id)

    # Get groups with link counts
    result = await db.execute(
        select(LinkGroup)
        .options(selectinload(LinkGroup.links))
        .where(LinkGroup.user_id == user_uuid)
        .order_by(LinkGroup.sort_order)
    )
    groups = result.scalars().all()

    return LinkGroupListResponse(
        groups=[
            LinkGroupResponse(
                id=str(g.id),
                name=g.name,
                description=g.description,
                color=g.color,
                icon=g.icon,
                is_shared=g.is_shared,
                sort_order=g.sort_order,
                link_count=len(g.links) if g.links else 0,
                created_at=g.created_at,
                updated_at=g.updated_at,
            )
            for g in groups
        ],
        total=len(groups),
    )


@router.get("/groups/{group_id}", response_model=LinkGroupResponse)
async def get_link_group(
    group_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific link group by ID.
    """
    user_uuid = _to_uuid(user.user_id)
    group_uuid = _to_uuid(group_id)

    result = await db.execute(
        select(LinkGroup)
        .options(selectinload(LinkGroup.links))
        .where(
            LinkGroup.id == group_uuid,
            LinkGroup.user_id == user_uuid,
        )
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link group not found: {group_id}",
        )

    return LinkGroupResponse(
        id=str(group.id),
        name=group.name,
        description=group.description,
        color=group.color,
        icon=group.icon,
        is_shared=group.is_shared,
        sort_order=group.sort_order,
        link_count=len(group.links) if group.links else 0,
        created_at=group.created_at,
        updated_at=group.updated_at,
    )


@router.put("/groups/{group_id}", response_model=LinkGroupResponse)
async def update_link_group(
    group_id: str,
    request: LinkGroupUpdate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a link group.
    """
    user_uuid = _to_uuid(user.user_id)
    group_uuid = _to_uuid(group_id)

    result = await db.execute(
        select(LinkGroup)
        .options(selectinload(LinkGroup.links))
        .where(
            LinkGroup.id == group_uuid,
            LinkGroup.user_id == user_uuid,
        )
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link group not found: {group_id}",
        )

    # Update fields
    if request.name is not None:
        group.name = request.name
    if request.description is not None:
        group.description = request.description
    if request.color is not None:
        group.color = request.color
    if request.icon is not None:
        group.icon = request.icon
    if request.is_shared is not None:
        group.is_shared = request.is_shared
    if request.sort_order is not None:
        group.sort_order = request.sort_order

    group.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(group)

    return LinkGroupResponse(
        id=str(group.id),
        name=group.name,
        description=group.description,
        color=group.color,
        icon=group.icon,
        is_shared=group.is_shared,
        sort_order=group.sort_order,
        link_count=len(group.links) if group.links else 0,
        created_at=group.created_at,
        updated_at=group.updated_at,
    )


@router.delete("/groups/{group_id}")
async def delete_link_group(
    group_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a link group and all its links.
    """
    user_uuid = _to_uuid(user.user_id)
    group_uuid = _to_uuid(group_id)

    result = await db.execute(
        select(LinkGroup).where(
            LinkGroup.id == group_uuid,
            LinkGroup.user_id == user_uuid,
        )
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link group not found: {group_id}",
        )

    group_name = group.name
    await db.delete(group)
    await db.commit()

    logger.info("Deleted link group", group_id=group_id, name=group_name)

    return {"message": f"Link group '{group_name}' deleted successfully"}


# =============================================================================
# Saved Link Endpoints
# =============================================================================

@router.post("/links", response_model=SavedLinkResponse, status_code=status.HTTP_201_CREATED)
async def create_saved_link(
    request: SavedLinkCreate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new saved link in a group.
    """
    user_uuid = _to_uuid(user.user_id)
    group_uuid = _to_uuid(request.group_id)

    # Verify group exists and belongs to user
    result = await db.execute(
        select(LinkGroup).where(
            LinkGroup.id == group_uuid,
            LinkGroup.user_id == user_uuid,
        )
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link group not found: {request.group_id}",
        )

    # Get max sort order
    max_order = await db.execute(
        select(func.max(SavedLink.sort_order)).where(SavedLink.group_id == group_uuid)
    )
    next_order = (max_order.scalar() or 0) + 1

    link = SavedLink(
        url=request.url,
        title=request.title,
        description=request.description,
        group_id=group_uuid,
        user_id=user_uuid,
        tags=request.tags,
        auto_scrape=request.auto_scrape,
        scrape_frequency=request.scrape_frequency,
        sort_order=next_order,
    )

    db.add(link)
    await db.commit()
    await db.refresh(link)

    logger.info("Created saved link", link_id=str(link.id), url=request.url)

    return SavedLinkResponse(
        id=str(link.id),
        url=link.url,
        title=link.title,
        description=link.description,
        favicon_url=link.favicon_url,
        group_id=str(link.group_id),
        group_name=group.name,
        tags=link.tags,
        auto_scrape=link.auto_scrape,
        scrape_frequency=link.scrape_frequency,
        last_scraped_at=link.last_scraped_at,
        last_scrape_status=link.last_scrape_status,
        scrape_count=link.scrape_count,
        cached_word_count=link.cached_word_count,
        cached_content_preview=link.cached_content_preview,
        sort_order=link.sort_order,
        created_at=link.created_at,
        updated_at=link.updated_at,
    )


@router.post("/links/bulk", status_code=status.HTTP_201_CREATED)
async def create_bulk_links(
    request: BulkLinkCreate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create multiple saved links at once.
    """
    user_uuid = _to_uuid(user.user_id)
    group_uuid = _to_uuid(request.group_id)

    # Verify group exists
    result = await db.execute(
        select(LinkGroup).where(
            LinkGroup.id == group_uuid,
            LinkGroup.user_id == user_uuid,
        )
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link group not found: {request.group_id}",
        )

    # Get max sort order
    max_order = await db.execute(
        select(func.max(SavedLink.sort_order)).where(SavedLink.group_id == group_uuid)
    )
    next_order = (max_order.scalar() or 0) + 1

    created_links = []
    for i, url in enumerate(request.urls):
        link = SavedLink(
            url=url,
            group_id=group_uuid,
            user_id=user_uuid,
            auto_scrape=request.auto_scrape,
            sort_order=next_order + i,
        )
        db.add(link)
        created_links.append(link)

    await db.commit()

    logger.info(
        "Created bulk links",
        count=len(created_links),
        group_id=request.group_id,
    )

    return {
        "message": f"Created {len(created_links)} links",
        "count": len(created_links),
        "links": [str(l.id) for l in created_links],
    }


@router.get("/links", response_model=SavedLinkListResponse)
async def list_saved_links(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    group_id: Optional[str] = Query(None, description="Filter by group ID"),
    search: Optional[str] = Query(None, description="Search in URL, title, description"),
    auto_scrape_only: bool = Query(False, description="Only show auto-scrape links"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List saved links with optional filtering.
    """
    user_uuid = _to_uuid(user.user_id)

    query = (
        select(SavedLink)
        .options(selectinload(SavedLink.group))
        .where(SavedLink.user_id == user_uuid)
    )

    if group_id:
        query = query.where(SavedLink.group_id == _to_uuid(group_id))

    if search:
        search_pattern = f"%{search}%"
        query = query.where(
            or_(
                SavedLink.url.ilike(search_pattern),
                SavedLink.title.ilike(search_pattern),
                SavedLink.description.ilike(search_pattern),
            )
        )

    if auto_scrape_only:
        query = query.where(SavedLink.auto_scrape == True)

    # Get total count
    count_query = select(func.count(SavedLink.id)).where(SavedLink.user_id == user_uuid)
    if group_id:
        count_query = count_query.where(SavedLink.group_id == _to_uuid(group_id))
    total = (await db.execute(count_query)).scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(SavedLink.sort_order).offset(offset).limit(limit)

    result = await db.execute(query)
    links = result.scalars().all()

    return SavedLinkListResponse(
        links=[
            SavedLinkResponse(
                id=str(l.id),
                url=l.url,
                title=l.title,
                description=l.description,
                favicon_url=l.favicon_url,
                group_id=str(l.group_id),
                group_name=l.group.name if l.group else None,
                tags=l.tags,
                auto_scrape=l.auto_scrape,
                scrape_frequency=l.scrape_frequency,
                last_scraped_at=l.last_scraped_at,
                last_scrape_status=l.last_scrape_status,
                scrape_count=l.scrape_count,
                cached_word_count=l.cached_word_count,
                cached_content_preview=l.cached_content_preview,
                sort_order=l.sort_order,
                created_at=l.created_at,
                updated_at=l.updated_at,
            )
            for l in links
        ],
        total=total,
    )


@router.get("/links/{link_id}", response_model=SavedLinkResponse)
async def get_saved_link(
    link_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific saved link by ID.
    """
    user_uuid = _to_uuid(user.user_id)
    link_uuid = _to_uuid(link_id)

    result = await db.execute(
        select(SavedLink)
        .options(selectinload(SavedLink.group))
        .where(
            SavedLink.id == link_uuid,
            SavedLink.user_id == user_uuid,
        )
    )
    link = result.scalar_one_or_none()

    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved link not found: {link_id}",
        )

    return SavedLinkResponse(
        id=str(link.id),
        url=link.url,
        title=link.title,
        description=link.description,
        favicon_url=link.favicon_url,
        group_id=str(link.group_id),
        group_name=link.group.name if link.group else None,
        tags=link.tags,
        auto_scrape=link.auto_scrape,
        scrape_frequency=link.scrape_frequency,
        last_scraped_at=link.last_scraped_at,
        last_scrape_status=link.last_scrape_status,
        scrape_count=link.scrape_count,
        cached_word_count=link.cached_word_count,
        cached_content_preview=link.cached_content_preview,
        sort_order=link.sort_order,
        created_at=link.created_at,
        updated_at=link.updated_at,
    )


@router.put("/links/{link_id}", response_model=SavedLinkResponse)
async def update_saved_link(
    link_id: str,
    request: SavedLinkUpdate,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a saved link.
    """
    user_uuid = _to_uuid(user.user_id)
    link_uuid = _to_uuid(link_id)

    result = await db.execute(
        select(SavedLink)
        .options(selectinload(SavedLink.group))
        .where(
            SavedLink.id == link_uuid,
            SavedLink.user_id == user_uuid,
        )
    )
    link = result.scalar_one_or_none()

    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved link not found: {link_id}",
        )

    # Update fields
    if request.title is not None:
        link.title = request.title
    if request.description is not None:
        link.description = request.description
    if request.group_id is not None:
        # Verify new group exists
        new_group = await db.execute(
            select(LinkGroup).where(
                LinkGroup.id == _to_uuid(request.group_id),
                LinkGroup.user_id == user_uuid,
            )
        )
        if not new_group.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Link group not found: {request.group_id}",
            )
        link.group_id = _to_uuid(request.group_id)
    if request.tags is not None:
        link.tags = request.tags
    if request.auto_scrape is not None:
        link.auto_scrape = request.auto_scrape
    if request.scrape_frequency is not None:
        link.scrape_frequency = request.scrape_frequency
    if request.sort_order is not None:
        link.sort_order = request.sort_order

    link.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(link)

    # Reload group relationship
    await db.refresh(link, ["group"])

    return SavedLinkResponse(
        id=str(link.id),
        url=link.url,
        title=link.title,
        description=link.description,
        favicon_url=link.favicon_url,
        group_id=str(link.group_id),
        group_name=link.group.name if link.group else None,
        tags=link.tags,
        auto_scrape=link.auto_scrape,
        scrape_frequency=link.scrape_frequency,
        last_scraped_at=link.last_scraped_at,
        last_scrape_status=link.last_scrape_status,
        scrape_count=link.scrape_count,
        cached_word_count=link.cached_word_count,
        cached_content_preview=link.cached_content_preview,
        sort_order=link.sort_order,
        created_at=link.created_at,
        updated_at=link.updated_at,
    )


@router.delete("/links/{link_id}")
async def delete_saved_link(
    link_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a saved link.
    """
    user_uuid = _to_uuid(user.user_id)
    link_uuid = _to_uuid(link_id)

    result = await db.execute(
        select(SavedLink).where(
            SavedLink.id == link_uuid,
            SavedLink.user_id == user_uuid,
        )
    )
    link = result.scalar_one_or_none()

    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved link not found: {link_id}",
        )

    url = link.url
    await db.delete(link)
    await db.commit()

    return {"message": f"Link deleted: {url[:50]}..."}


# =============================================================================
# Scraping Integration Endpoints
# =============================================================================

@router.post("/groups/{group_id}/scrape")
async def scrape_group_links(
    group_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    storage_mode: str = Query("permanent", description="Storage mode: immediate or permanent"),
):
    """
    Scrape all links in a group.

    Initiates scraping for all saved links in the specified group.
    Returns a job ID for tracking progress.
    """
    from backend.services.scraper import get_scraper_service, StorageMode

    user_uuid = _to_uuid(user.user_id)
    group_uuid = _to_uuid(group_id)

    # Get group with links
    result = await db.execute(
        select(LinkGroup)
        .options(selectinload(LinkGroup.links))
        .where(
            LinkGroup.id == group_uuid,
            LinkGroup.user_id == user_uuid,
        )
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Link group not found: {group_id}",
        )

    if not group.links:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No links in this group to scrape",
        )

    # Parse storage mode
    try:
        mode = StorageMode(storage_mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid storage mode: {storage_mode}",
        )

    # Create scrape job with all URLs from the group
    urls = [link.url for link in group.links]

    service = get_scraper_service()
    job = await service.create_job(
        user_id=user.user_id,
        urls=urls,
        storage_mode=mode,
    )

    logger.info(
        "Started group scrape",
        group_id=group_id,
        group_name=group.name,
        urls_count=len(urls),
        job_id=job.id,
    )

    return {
        "job_id": job.id,
        "group_id": group_id,
        "group_name": group.name,
        "urls_count": len(urls),
        "storage_mode": storage_mode,
        "message": f"Started scraping {len(urls)} links from '{group.name}'",
    }


@router.get("/links/{link_id}/history", response_model=ScrapedContentListResponse)
async def get_link_scrape_history(
    link_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Get scrape history for a saved link.

    Returns all previous scrapes of this link with their content.
    """
    user_uuid = _to_uuid(user.user_id)
    link_uuid = _to_uuid(link_id)

    # Verify link ownership
    link_result = await db.execute(
        select(SavedLink).where(
            SavedLink.id == link_uuid,
            SavedLink.user_id == user_uuid,
        )
    )
    link = link_result.scalar_one_or_none()

    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved link not found: {link_id}",
        )

    # Get history
    result = await db.execute(
        select(ScrapedContentHistory)
        .where(ScrapedContentHistory.saved_link_id == link_uuid)
        .order_by(ScrapedContentHistory.scraped_at.desc())
        .limit(limit)
    )
    history = result.scalars().all()

    # Get total count
    total = (await db.execute(
        select(func.count(ScrapedContentHistory.id))
        .where(ScrapedContentHistory.saved_link_id == link_uuid)
    )).scalar() or 0

    return ScrapedContentListResponse(
        history=[
            ScrapedContentResponse(
                id=str(h.id),
                saved_link_id=str(h.saved_link_id),
                url=link.url,
                title=h.title,
                content=h.content,
                word_count=h.word_count,
                scraped_at=h.scraped_at,
                status=h.status,
                error_message=h.error_message,
                links_found=h.links_found,
                images_found=h.images_found,
                metadata=h.metadata,
                indexed_to_rag=h.indexed_to_rag,
            )
            for h in history
        ],
        total=total,
    )


@router.post("/links/{link_id}/scrape")
async def scrape_single_link(
    link_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    save_history: bool = Query(True, description="Save to scrape history"),
    index_to_rag: bool = Query(False, description="Index content to RAG pipeline"),
):
    """
    Scrape a single saved link and optionally save to history.

    Returns the scraped content immediately.
    """
    from backend.services.scraper import get_scraper_service, ScrapeConfig

    user_uuid = _to_uuid(user.user_id)
    link_uuid = _to_uuid(link_id)

    # Get link
    result = await db.execute(
        select(SavedLink).where(
            SavedLink.id == link_uuid,
            SavedLink.user_id == user_uuid,
        )
    )
    link = result.scalar_one_or_none()

    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved link not found: {link_id}",
        )

    # Scrape the URL
    service = get_scraper_service()

    try:
        page = await service.scrape_url_immediate(
            url=link.url,
            config=ScrapeConfig(),
        )

        # Update link metadata
        link.last_scraped_at = datetime.utcnow()
        link.last_scrape_status = "success"
        link.scrape_count = (link.scrape_count or 0) + 1
        link.cached_title = page.title
        link.cached_word_count = page.word_count
        link.cached_content_preview = page.content[:500] if page.content else None

        # Compute content hash
        content_hash = hashlib.sha256(page.content.encode()).hexdigest() if page.content else None
        link.last_content_hash = content_hash

        # Save to history if requested
        if save_history:
            history_entry = ScrapedContentHistory(
                saved_link_id=link_uuid,
                status="success",
                title=page.title,
                content=page.content,
                word_count=page.word_count,
                content_hash=content_hash,
                links_found=len(page.links) if page.links else 0,
                images_found=len(page.images) if page.images else 0,
                metadata=page.metadata,
                indexed_to_rag=index_to_rag,
            )
            db.add(history_entry)

        await db.commit()

        # Index to RAG if requested
        if index_to_rag:
            from backend.services.scraper import ScrapedPage
            await service.index_pages_content(
                pages=[page],
                source_id=f"saved_link_{link_id}",
            )

        return {
            "success": True,
            "url": link.url,
            "title": page.title,
            "content": page.content,
            "word_count": page.word_count,
            "links_found": len(page.links) if page.links else 0,
            "images_found": len(page.images) if page.images else 0,
            "metadata": page.metadata,
            "scraped_at": datetime.utcnow().isoformat(),
            "content_changed": link.last_content_hash != content_hash if link.last_content_hash else True,
            "indexed_to_rag": index_to_rag,
        }

    except Exception as e:
        # Update link with failure
        link.last_scraped_at = datetime.utcnow()
        link.last_scrape_status = "failed"

        if save_history:
            history_entry = ScrapedContentHistory(
                saved_link_id=link_uuid,
                status="failed",
                error_message=str(e),
            )
            db.add(history_entry)

        await db.commit()

        logger.error("Scrape failed", link_id=link_id, error=str(e))

        return {
            "success": False,
            "url": link.url,
            "error": str(e),
            "scraped_at": datetime.utcnow().isoformat(),
        }
