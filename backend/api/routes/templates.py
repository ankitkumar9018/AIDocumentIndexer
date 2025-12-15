"""
AIDocumentIndexer - Prompt Templates API Routes
=================================================

API endpoints for managing prompt templates.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_async_session
from backend.api.middleware.auth import get_current_user
from backend.services.prompt_templates import get_prompt_template_service

router = APIRouter(prefix="/prompt-templates", tags=["Prompt Templates"])


# =============================================================================
# Request/Response Models
# =============================================================================

class TemplateVariable(BaseModel):
    """Template variable definition."""
    name: str
    description: str = ""
    default: str = ""
    required: bool = False


class CreateTemplateRequest(BaseModel):
    """Request to create a new template."""
    name: str = Field(..., min_length=1, max_length=255)
    prompt_text: str = Field(..., min_length=1)
    description: Optional[str] = None
    category: str = "general"
    system_prompt: Optional[str] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    is_public: bool = False
    tags: Optional[List[str]] = None
    variables: Optional[List[TemplateVariable]] = None


class UpdateTemplateRequest(BaseModel):
    """Request to update a template."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    prompt_text: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = None
    category: Optional[str] = None
    system_prompt: Optional[str] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    variables: Optional[List[TemplateVariable]] = None


class ApplyTemplateRequest(BaseModel):
    """Request to apply a template."""
    variables: Optional[dict] = None


class TemplateResponse(BaseModel):
    """Template response."""
    id: str
    user_id: Optional[str]
    name: str
    description: Optional[str]
    category: str
    tags: Optional[List[str]]
    prompt_text: str
    system_prompt: Optional[str]
    model_id: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    is_public: bool
    is_system: bool
    is_owner: bool
    use_count: int
    last_used_at: Optional[str]
    variables: List[dict]
    created_at: str
    updated_at: str


class TemplateListItem(BaseModel):
    """Template list item."""
    id: str
    name: str
    description: Optional[str]
    category: str
    is_public: bool
    is_system: bool
    is_owner: bool
    use_count: int
    created_at: str
    model_id: Optional[str]
    temperature: Optional[float]


class AppliedTemplateResponse(BaseModel):
    """Response from applying a template."""
    prompt_text: str
    system_prompt: Optional[str]
    model_id: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    template_id: str
    template_name: str


class CategoryResponse(BaseModel):
    """Category with count."""
    name: str
    count: int


# =============================================================================
# Routes
# =============================================================================

@router.get("", response_model=List[TemplateListItem])
async def list_templates(
    category: Optional[str] = None,
    search: Optional[str] = None,
    include_public: bool = True,
    include_system: bool = True,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """
    List prompt templates accessible to the current user.

    Returns the user's own templates plus public/system templates.
    """
    service = get_prompt_template_service()

    templates = await service.list_templates(
        db,
        user_id=str(current_user["sub"]),
        category=category,
        search=search,
        include_public=include_public,
        include_system=include_system,
        limit=limit,
        offset=offset,
    )

    return [
        TemplateListItem(
            id=t.id,
            name=t.name,
            description=t.description,
            category=t.category,
            is_public=t.is_public,
            is_system=t.is_system,
            is_owner=t.is_owner,
            use_count=t.use_count,
            created_at=t.created_at.isoformat(),
            model_id=t.model_id,
            temperature=t.temperature,
        )
        for t in templates
    ]


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_template(
    request: CreateTemplateRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """Create a new prompt template."""
    service = get_prompt_template_service()

    variables = None
    if request.variables:
        variables = [v.model_dump() for v in request.variables]

    template_id = await service.create_template(
        db,
        user_id=str(current_user["sub"]),
        name=request.name,
        prompt_text=request.prompt_text,
        description=request.description,
        category=request.category,
        system_prompt=request.system_prompt,
        model_id=request.model_id,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        is_public=request.is_public,
        tags=request.tags,
        variables=variables,
    )

    if not template_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create template",
        )

    return {"id": template_id, "message": "Template created successfully"}


@router.get("/categories", response_model=List[CategoryResponse])
async def get_categories(
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """Get all template categories with counts."""
    service = get_prompt_template_service()
    categories = await service.get_categories(db)

    return [CategoryResponse(**c) for c in categories]


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """Get a template by ID."""
    service = get_prompt_template_service()

    template = await service.get_template(
        db,
        template_id=template_id,
        user_id=str(current_user["sub"]),
    )

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found or access denied",
        )

    return TemplateResponse(**template)


@router.put("/{template_id}")
async def update_template(
    template_id: str,
    request: UpdateTemplateRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """Update a template."""
    service = get_prompt_template_service()

    variables = None
    if request.variables:
        variables = [v.model_dump() for v in request.variables]

    success = await service.update_template(
        db,
        template_id=template_id,
        user_id=str(current_user["sub"]),
        name=request.name,
        prompt_text=request.prompt_text,
        description=request.description,
        category=request.category,
        system_prompt=request.system_prompt,
        model_id=request.model_id,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        is_public=request.is_public,
        tags=request.tags,
        variables=variables,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found or you don't have permission to update it",
        )

    return {"message": "Template updated successfully"}


@router.delete("/{template_id}")
async def delete_template(
    template_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """Delete a template."""
    service = get_prompt_template_service()

    success = await service.delete_template(
        db,
        template_id=template_id,
        user_id=str(current_user["sub"]),
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found or you don't have permission to delete it",
        )

    return {"message": "Template deleted successfully"}


@router.post("/{template_id}/apply", response_model=AppliedTemplateResponse)
async def apply_template(
    template_id: str,
    request: ApplyTemplateRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """
    Apply a template and get the rendered prompt with settings.

    Variables in the template ({{variable_name}}) will be replaced
    with values from the request.
    """
    service = get_prompt_template_service()

    result = await service.apply_template(
        db,
        template_id=template_id,
        user_id=str(current_user["sub"]),
        variables=request.variables,
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found or access denied",
        )

    return AppliedTemplateResponse(**result)


@router.post("/{template_id}/duplicate", response_model=dict)
async def duplicate_template(
    template_id: str,
    new_name: Optional[str] = None,
    db: AsyncSession = Depends(get_async_session),
    current_user = Depends(get_current_user),
):
    """
    Create a copy of a template for the current user.

    The copy will be private by default.
    """
    service = get_prompt_template_service()

    new_template_id = await service.duplicate_template(
        db,
        template_id=template_id,
        user_id=str(current_user["sub"]),
        new_name=new_name,
    )

    if not new_template_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found or access denied",
        )

    return {"id": new_template_id, "message": "Template duplicated successfully"}
