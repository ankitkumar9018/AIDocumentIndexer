"""
Generation Templates API Routes
================================

CRUD endpoints for managing document generation templates.
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.middleware.auth import get_current_user
from backend.db.database import get_async_session
from backend.db.models import TemplateCategory
from backend.services.generation_templates import GenerationTemplateService

router = APIRouter(prefix="/generation/templates", tags=["Generation Templates"])


# =============================================================================
# Request/Response Models
# =============================================================================

class TemplateSettings(BaseModel):
    """Generation settings stored in a template."""
    output_format: str = Field(default="pptx", description="Output format: pptx, docx, pdf, md, html, xlsx")
    theme: str = Field(default="business", description="Visual theme")
    font_family: Optional[str] = Field(default=None, description="Font family")
    layout_template: Optional[str] = Field(default=None, description="Layout template")
    include_toc: bool = Field(default=False, description="Include table of contents")
    include_sources: bool = Field(default=True, description="Include source references")
    use_existing_docs: bool = Field(default=False, description="Learn from existing documents")
    enable_animations: bool = Field(default=False, description="Enable PPTX animations")
    enable_images: bool = Field(default=False, description="Enable image generation")
    custom_colors: Optional[dict] = Field(default=None, description="Custom color overrides")

    class Config:
        extra = "allow"  # Allow additional settings


class CreateTemplateRequest(BaseModel):
    """Request to create a new template."""
    name: str = Field(..., min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(default=None, max_length=1000, description="Template description")
    category: str = Field(default="custom", description="Template category")
    settings: TemplateSettings = Field(..., description="Generation settings")
    default_collections: Optional[List[str]] = Field(default=None, description="Default collections for style learning")
    is_public: bool = Field(default=False, description="Whether template is publicly visible")
    tags: Optional[List[str]] = Field(default=None, description="Tags for filtering")
    thumbnail: Optional[str] = Field(default=None, description="Base64 encoded thumbnail")


class UpdateTemplateRequest(BaseModel):
    """Request to update a template."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    category: Optional[str] = Field(default=None)
    settings: Optional[TemplateSettings] = Field(default=None)
    default_collections: Optional[List[str]] = Field(default=None)
    is_public: Optional[bool] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    thumbnail: Optional[str] = Field(default=None)


class DuplicateTemplateRequest(BaseModel):
    """Request to duplicate a template."""
    new_name: Optional[str] = Field(default=None, max_length=255, description="Name for the duplicate")


class TemplateResponse(BaseModel):
    """Response containing a template."""
    id: str
    user_id: Optional[str]
    name: str
    description: Optional[str]
    category: str
    settings: dict
    default_collections: Optional[List[str]]
    is_public: bool
    is_system: bool
    use_count: int
    last_used_at: Optional[str]
    tags: Optional[List[str]]
    thumbnail: Optional[str]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class TemplateListResponse(BaseModel):
    """Response for template list."""
    templates: List[TemplateResponse]
    total: int
    limit: int
    offset: int


class CategoriesResponse(BaseModel):
    """Response for available categories."""
    categories: List[dict]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=TemplateListResponse)
async def list_templates(
    category: Optional[str] = Query(default=None, description="Filter by category"),
    search: Optional[str] = Query(default=None, description="Search in name/description"),
    include_system: bool = Query(default=True, description="Include system templates"),
    include_public: bool = Query(default=True, description="Include public templates"),
    limit: int = Query(default=50, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List available generation templates.

    Returns system templates, user's own templates, and optionally public templates.
    """
    service = GenerationTemplateService(db)

    user_id = uuid.UUID(current_user.get("sub")) if current_user.get("sub") else None

    templates, total = await service.list_templates(
        user_id=user_id,
        category=category,
        include_system=include_system,
        include_public=include_public,
        search=search,
        limit=limit,
        offset=offset,
    )

    return TemplateListResponse(
        templates=[
            TemplateResponse(
                id=str(t.id),
                user_id=str(t.user_id) if t.user_id else None,
                name=t.name,
                description=t.description,
                category=t.category,
                settings=t.settings or {},
                default_collections=t.default_collections,
                is_public=t.is_public,
                is_system=t.is_system,
                use_count=t.use_count,
                last_used_at=t.last_used_at.isoformat() if t.last_used_at else None,
                tags=t.tags,
                thumbnail=t.thumbnail,
                created_at=t.created_at.isoformat(),
                updated_at=t.updated_at.isoformat(),
            )
            for t in templates
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/categories", response_model=CategoriesResponse)
async def get_categories():
    """Get available template categories."""
    categories = [
        {"value": TemplateCategory.REPORT.value, "label": "Report", "description": "Business reports and analyses"},
        {"value": TemplateCategory.PROPOSAL.value, "label": "Proposal", "description": "Project and business proposals"},
        {"value": TemplateCategory.PRESENTATION.value, "label": "Presentation", "description": "Slides and pitch decks"},
        {"value": TemplateCategory.MEETING_NOTES.value, "label": "Meeting Notes", "description": "Meeting summaries and action items"},
        {"value": TemplateCategory.DOCUMENTATION.value, "label": "Documentation", "description": "Technical and user documentation"},
        {"value": TemplateCategory.CUSTOM.value, "label": "Custom", "description": "Custom templates"},
    ]
    return CategoriesResponse(categories=categories)


@router.get("/popular", response_model=List[TemplateResponse])
async def get_popular_templates(
    category: Optional[str] = Query(default=None, description="Filter by category"),
    limit: int = Query(default=10, ge=1, le=20, description="Max results"),
    db: AsyncSession = Depends(get_async_session),
):
    """Get most popular templates."""
    service = GenerationTemplateService(db)
    templates = await service.get_popular_templates(category=category, limit=limit)

    return [
        TemplateResponse(
            id=str(t.id),
            user_id=str(t.user_id) if t.user_id else None,
            name=t.name,
            description=t.description,
            category=t.category,
            settings=t.settings or {},
            default_collections=t.default_collections,
            is_public=t.is_public,
            is_system=t.is_system,
            use_count=t.use_count,
            last_used_at=t.last_used_at.isoformat() if t.last_used_at else None,
            tags=t.tags,
            thumbnail=t.thumbnail,
            created_at=t.created_at.isoformat(),
            updated_at=t.updated_at.isoformat(),
        )
        for t in templates
    ]


@router.get("/mine", response_model=List[TemplateResponse])
async def get_my_templates(
    limit: int = Query(default=50, ge=1, le=100, description="Max results"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Get current user's templates."""
    service = GenerationTemplateService(db)
    user_id = uuid.UUID(current_user.get("sub"))

    templates = await service.get_user_templates(user_id=user_id, limit=limit)

    return [
        TemplateResponse(
            id=str(t.id),
            user_id=str(t.user_id) if t.user_id else None,
            name=t.name,
            description=t.description,
            category=t.category,
            settings=t.settings or {},
            default_collections=t.default_collections,
            is_public=t.is_public,
            is_system=t.is_system,
            use_count=t.use_count,
            last_used_at=t.last_used_at.isoformat() if t.last_used_at else None,
            tags=t.tags,
            thumbnail=t.thumbnail,
            created_at=t.created_at.isoformat(),
            updated_at=t.updated_at.isoformat(),
        )
        for t in templates
    ]


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Get a template by ID."""
    service = GenerationTemplateService(db)
    user_id = uuid.UUID(current_user.get("sub"))

    template = await service.get_template(template_id, user_id)
    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")

    return TemplateResponse(
        id=str(template.id),
        user_id=str(template.user_id) if template.user_id else None,
        name=template.name,
        description=template.description,
        category=template.category,
        settings=template.settings or {},
        default_collections=template.default_collections,
        is_public=template.is_public,
        is_system=template.is_system,
        use_count=template.use_count,
        last_used_at=template.last_used_at.isoformat() if template.last_used_at else None,
        tags=template.tags,
        thumbnail=template.thumbnail,
        created_at=template.created_at.isoformat(),
        updated_at=template.updated_at.isoformat(),
    )


@router.post("", response_model=TemplateResponse, status_code=201)
async def create_template(
    request: CreateTemplateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Create a new template."""
    service = GenerationTemplateService(db)
    user_id = uuid.UUID(current_user.get("sub"))

    template = await service.create_template(
        user_id=user_id,
        name=request.name,
        description=request.description,
        category=request.category,
        settings=request.settings.model_dump(),
        default_collections=request.default_collections,
        is_public=request.is_public,
        tags=request.tags,
        thumbnail=request.thumbnail,
    )

    return TemplateResponse(
        id=str(template.id),
        user_id=str(template.user_id) if template.user_id else None,
        name=template.name,
        description=template.description,
        category=template.category,
        settings=template.settings or {},
        default_collections=template.default_collections,
        is_public=template.is_public,
        is_system=template.is_system,
        use_count=template.use_count,
        last_used_at=template.last_used_at.isoformat() if template.last_used_at else None,
        tags=template.tags,
        thumbnail=template.thumbnail,
        created_at=template.created_at.isoformat(),
        updated_at=template.updated_at.isoformat(),
    )


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: uuid.UUID,
    request: UpdateTemplateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Update a template."""
    service = GenerationTemplateService(db)
    user_id = uuid.UUID(current_user.get("sub"))

    updates = request.model_dump(exclude_unset=True)
    if "settings" in updates and updates["settings"]:
        updates["settings"] = updates["settings"].model_dump() if hasattr(updates["settings"], "model_dump") else updates["settings"]

    template = await service.update_template(template_id, user_id, **updates)
    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found or not owned by user")

    return TemplateResponse(
        id=str(template.id),
        user_id=str(template.user_id) if template.user_id else None,
        name=template.name,
        description=template.description,
        category=template.category,
        settings=template.settings or {},
        default_collections=template.default_collections,
        is_public=template.is_public,
        is_system=template.is_system,
        use_count=template.use_count,
        last_used_at=template.last_used_at.isoformat() if template.last_used_at else None,
        tags=template.tags,
        thumbnail=template.thumbnail,
        created_at=template.created_at.isoformat(),
        updated_at=template.updated_at.isoformat(),
    )


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Delete a template."""
    service = GenerationTemplateService(db)
    user_id = uuid.UUID(current_user.get("sub"))

    deleted = await service.delete_template(template_id, user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found or not owned by user")


@router.post("/{template_id}/duplicate", response_model=TemplateResponse, status_code=201)
async def duplicate_template(
    template_id: uuid.UUID,
    request: DuplicateTemplateRequest = DuplicateTemplateRequest(),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Duplicate a template."""
    service = GenerationTemplateService(db)
    user_id = uuid.UUID(current_user.get("sub"))

    template = await service.duplicate_template(
        template_id=template_id,
        user_id=user_id,
        new_name=request.new_name,
    )
    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")

    return TemplateResponse(
        id=str(template.id),
        user_id=str(template.user_id) if template.user_id else None,
        name=template.name,
        description=template.description,
        category=template.category,
        settings=template.settings or {},
        default_collections=template.default_collections,
        is_public=template.is_public,
        is_system=template.is_system,
        use_count=template.use_count,
        last_used_at=template.last_used_at.isoformat() if template.last_used_at else None,
        tags=template.tags,
        thumbnail=template.thumbnail,
        created_at=template.created_at.isoformat(),
        updated_at=template.updated_at.isoformat(),
    )


@router.post("/{template_id}/use", status_code=204)
async def record_template_use(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Record that a template was used (increments use count)."""
    service = GenerationTemplateService(db)
    await service.record_template_use(template_id)


@router.post("/seed", status_code=200)
async def seed_system_templates(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Seed system templates (admin only).

    This endpoint seeds the default system templates if they don't exist.
    """
    # Check if admin
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    service = GenerationTemplateService(db)
    count = await service.seed_system_templates()

    return {"message": f"Seeded {count} system templates", "count": count}
