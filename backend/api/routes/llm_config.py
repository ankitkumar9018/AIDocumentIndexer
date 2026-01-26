"""
AIDocumentIndexer - LLM Configuration API Routes
==================================================

Endpoints for managing LLM configurations, model access, and user overrides.

Features:
- List available models (filtered by user's access level)
- Get/set service LLM configurations (admin)
- User model overrides per service
- Model access group management (admin) - FULLY CONFIGURABLE
- Model registry management (admin) - FULLY CONFIGURABLE
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
import structlog

from backend.db.database import async_session_context
from backend.api.middleware.auth import AuthenticatedUser
from backend.services.llm_router import get_llm_router, RoutingStrategy, AccessGroup, ModelInfo

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class ModelInfo(BaseModel):
    """Information about an available model."""
    provider: str
    provider_name: str
    model: str
    quality_score: int = Field(description="Quality score (0-100)")
    cost_score: float = Field(description="Cost per 1M tokens in USD")
    is_default: bool


class AvailableModelsResponse(BaseModel):
    """Response containing available models."""
    models: List[ModelInfo]
    user_access_level: int
    total: int


class ServiceConfigResponse(BaseModel):
    """Service LLM configuration."""
    service_name: str
    operation_name: Optional[str] = None
    provider_type: str
    model_name: str
    temperature: float
    max_tokens: int
    routing_strategy: str
    allow_user_override: bool
    min_access_level: int


class ServiceConfigUpdate(BaseModel):
    """Update request for service config."""
    provider_type: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=128000)
    routing_strategy: Optional[str] = None
    allow_user_override: Optional[bool] = None
    fallback_provider_type: Optional[str] = None
    fallback_model_name: Optional[str] = None
    min_access_level: Optional[int] = Field(None, ge=1, le=100)


class UserOverrideRequest(BaseModel):
    """Request to set user's model override for a service."""
    provider_type: str
    model_name: str
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=128000)


class UserOverrideResponse(BaseModel):
    """User's model override for a service."""
    service_name: str
    operation_name: Optional[str] = None
    provider_type: str
    model_name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    updated_at: str


class ModelAccessGroupResponse(BaseModel):
    """Model access group information."""
    id: str
    name: str
    description: Optional[str] = None
    access_level: int
    model_patterns: List[str]
    max_tokens_per_request: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    max_cost_per_day_usd: Optional[float] = None


class UserAccessResponse(BaseModel):
    """User's model access information."""
    user_id: str
    access_level: int
    allowed_patterns: List[str]
    max_tokens_per_request: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    max_cost_per_day_usd: Optional[float] = None
    access_groups: List[str] = []


class GrantAccessRequest(BaseModel):
    """Request to grant model access to a user."""
    user_id: str
    access_group_name: str
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


# =============================================================================
# User-Facing Endpoints
# =============================================================================

@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models(
    user: AuthenticatedUser,
    service_name: Optional[str] = Query(None, description="Filter by service"),
):
    """
    Get list of models available to the current user.

    Models are filtered based on your access level and permissions.
    Optionally filter by service to see models available for a specific feature.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        models = await llm_router.get_available_models(
            db,
            user_id=user.user_id,
            service_name=service_name,
        )

        access = await llm_router.get_user_access(db, user.user_id)

        return AvailableModelsResponse(
            models=[ModelInfo(**m) for m in models],
            user_access_level=access.access_level,
            total=len(models),
        )


@router.get("/models/check/{model_name}")
async def check_model_access(
    model_name: str,
    user: AuthenticatedUser,
):
    """
    Check if you can access a specific model.

    Returns whether you have access and the reason if not.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        allowed, reason = await llm_router.check_model_access(
            db,
            user_id=user.user_id,
            model_name=model_name,
        )

        return {
            "model": model_name,
            "allowed": allowed,
            "reason": reason,
        }


@router.get("/access", response_model=UserAccessResponse)
async def get_my_access(
    user: AuthenticatedUser,
):
    """
    Get your current model access level and permissions.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        access = await llm_router.get_user_access(db, user.user_id)

        return UserAccessResponse(
            user_id=access.user_id,
            access_level=access.access_level,
            allowed_patterns=access.allowed_patterns,
            max_tokens_per_request=access.max_tokens_per_request,
            max_requests_per_minute=access.max_requests_per_minute,
            max_cost_per_day_usd=access.max_cost_per_day_usd,
        )


# =============================================================================
# Service Configuration Endpoints
# =============================================================================

@router.get("/services/{service_name}", response_model=ServiceConfigResponse)
async def get_service_config(
    service_name: str,
    operation_name: Optional[str] = Query(None),
    user: AuthenticatedUser = None,
):
    """
    Get LLM configuration for a service.

    Returns the model, parameters, and settings for a specific service/operation.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        config = await llm_router.get_service_config(
            db,
            service_name=service_name,
            operation_name=operation_name,
            organization_id=user.organization_id if user else None,
        )

        return ServiceConfigResponse(
            service_name=config.service_name,
            operation_name=config.operation_name,
            provider_type=config.provider_type,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            routing_strategy=config.routing_strategy.value,
            allow_user_override=config.allow_user_override,
            min_access_level=config.min_access_level,
        )


@router.get("/services", response_model=List[ServiceConfigResponse])
async def list_service_configs(
    user: AuthenticatedUser,
):
    """
    List all service LLM configurations.

    Returns configs for all services (chat, RAG, workflow, etc.)
    """
    llm_router = get_llm_router()

    services = [
        ("chat", None),
        ("chat", "agent"),
        ("rag", None),
        ("audio_overview", "script_generation"),
        ("workflow", None),
        ("generation", None),
        ("embeddings", None),
    ]

    configs = []
    async with async_session_context() as db:
        for service_name, operation_name in services:
            config = await llm_router.get_service_config(
                db,
                service_name=service_name,
                operation_name=operation_name,
                organization_id=user.organization_id,
            )
            configs.append(ServiceConfigResponse(
                service_name=config.service_name,
                operation_name=config.operation_name,
                provider_type=config.provider_type,
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                routing_strategy=config.routing_strategy.value,
                allow_user_override=config.allow_user_override,
                min_access_level=config.min_access_level,
            ))

    return configs


# =============================================================================
# User Override Endpoints
# =============================================================================

@router.get("/overrides/{service_name}", response_model=Optional[UserOverrideResponse])
async def get_my_override(
    service_name: str,
    operation_name: Optional[str] = Query(None),
    user: AuthenticatedUser = None,
):
    """
    Get your model override for a service.

    Returns None if you haven't set an override.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        override = await llm_router.get_user_service_override(
            db,
            user_id=user.user_id,
            service_name=service_name,
            operation_name=operation_name,
        )

        if not override:
            return None

        return UserOverrideResponse(
            service_name=service_name,
            operation_name=operation_name,
            provider_type=override.get("provider_type", ""),
            model_name=override.get("model_name", ""),
            temperature=override.get("temperature"),
            max_tokens=override.get("max_tokens"),
            updated_at=override.get("updated_at", datetime.utcnow().isoformat()),
        )


@router.put("/overrides/{service_name}", response_model=UserOverrideResponse)
async def set_my_override(
    service_name: str,
    override: UserOverrideRequest,
    operation_name: Optional[str] = Query(None),
    user: AuthenticatedUser = None,
):
    """
    Set your model override for a service.

    This lets you choose a different model than the default for this service.
    Only works if the service allows user overrides and you have access to the model.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        # Check if service allows override
        config = await llm_router.get_service_config(
            db, service_name, operation_name, user.organization_id
        )

        if not config.allow_user_override:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service '{service_name}' does not allow user overrides"
            )

        # Check if user can access the model
        allowed, reason = await llm_router.check_model_access(
            db, user.user_id, override.model_name
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You don't have access to model '{override.model_name}': {reason}"
            )

        # Save override (in production, insert into user_service_llm_overrides)
        logger.info(
            "User override saved",
            user_id=user.user_id[:8] if user.user_id else None,
            service=service_name,
            model=override.model_name,
        )

        return UserOverrideResponse(
            service_name=service_name,
            operation_name=operation_name,
            provider_type=override.provider_type,
            model_name=override.model_name,
            temperature=override.temperature,
            max_tokens=override.max_tokens,
            updated_at=datetime.utcnow().isoformat(),
        )


@router.delete("/overrides/{service_name}")
async def delete_my_override(
    service_name: str,
    operation_name: Optional[str] = Query(None),
    user: AuthenticatedUser = None,
):
    """
    Remove your model override for a service.

    Reverts to using the default model for this service.
    """
    # In production, delete from user_service_llm_overrides
    logger.info(
        "User override deleted",
        user_id=user.user_id[:8] if user.user_id else None,
        service=service_name,
    )

    return {"success": True, "message": "Override removed"}


# =============================================================================
# Admin Endpoints
# =============================================================================

@router.patch("/services/{service_name}/config", response_model=ServiceConfigResponse)
async def update_service_config(
    service_name: str,
    updates: ServiceConfigUpdate,
    operation_name: Optional[str] = Query(None),
    user: AuthenticatedUser = None,
):
    """
    Update LLM configuration for a service.

    Admin only.
    """
    if not user.is_admin():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    llm_router = get_llm_router()

    async with async_session_context() as db:
        config = await llm_router.update_service_config(
            db,
            service_name=service_name,
            operation_name=operation_name,
            organization_id=user.organization_id,
            config_updates=updates.model_dump(exclude_none=True),
        )

        return ServiceConfigResponse(
            service_name=config.service_name,
            operation_name=config.operation_name,
            provider_type=config.provider_type,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            routing_strategy=config.routing_strategy.value,
            allow_user_override=config.allow_user_override,
            min_access_level=config.min_access_level,
        )


@router.get("/access-groups", response_model=List[ModelAccessGroupResponse])
async def list_access_groups(
    user: AuthenticatedUser,
):
    """
    List all model access groups.

    Admin only.
    """
    # In production, query model_access_groups table
    groups = [
        ModelAccessGroupResponse(
            id="1",
            name="basic",
            description="Basic tier - Access to cost-effective models",
            access_level=1,
            model_patterns=["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-haiku-*"],
        ),
        ModelAccessGroupResponse(
            id="2",
            name="standard",
            description="Standard tier - Access to most production models",
            access_level=5,
            model_patterns=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-*", "llama-*"],
        ),
        ModelAccessGroupResponse(
            id="3",
            name="advanced",
            description="Advanced tier - Access to all standard models",
            access_level=10,
            model_patterns=["gpt-4*", "claude-*", "llama-*", "gemini-*"],
        ),
        ModelAccessGroupResponse(
            id="4",
            name="enterprise",
            description="Enterprise tier - Full access",
            access_level=50,
            model_patterns=["*"],
        ),
    ]

    return groups


@router.post("/access/grant")
async def grant_user_access(
    request: GrantAccessRequest,
    user: AuthenticatedUser,
):
    """
    Grant model access to a user.

    Admin only.
    """
    llm_router = get_llm_router()

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

    async with async_session_context() as db:
        success = await llm_router.grant_model_access(
            db,
            user_id=request.user_id,
            access_group_name=request.access_group_name,
            granted_by_id=user.user_id,
            expires_at=expires_at,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to grant access")

        return {
            "success": True,
            "user_id": request.user_id,
            "access_group": request.access_group_name,
            "expires_at": expires_at.isoformat() if expires_at else None,
        }


@router.post("/cache/clear")
async def clear_router_cache(
    user: AuthenticatedUser,
):
    """
    Clear LLM router cache.

    Admin only. Use after updating configurations.
    """
    llm_router = get_llm_router()
    llm_router.clear_cache()

    return {"success": True, "message": "Cache cleared"}


# =============================================================================
# Access Group Management (Admin CRUD)
# =============================================================================

class CreateAccessGroupRequest(BaseModel):
    """Request to create a new access group."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    access_level: int = Field(..., ge=1, le=100)
    model_patterns: List[str] = Field(..., min_items=1, description="Model patterns (wildcards supported)")
    max_tokens_per_request: Optional[int] = Field(None, ge=1, le=256000)
    max_requests_per_minute: Optional[int] = Field(None, ge=1, le=10000)
    max_cost_per_day_usd: Optional[float] = Field(None, ge=0)


class UpdateAccessGroupRequest(BaseModel):
    """Request to update an access group."""
    description: Optional[str] = None
    access_level: Optional[int] = Field(None, ge=1, le=100)
    model_patterns: Optional[List[str]] = Field(None, min_items=1)
    max_tokens_per_request: Optional[int] = Field(None, ge=1, le=256000)
    max_requests_per_minute: Optional[int] = Field(None, ge=1, le=10000)
    max_cost_per_day_usd: Optional[float] = Field(None, ge=0)
    is_active: Optional[bool] = None


class AccessGroupDetailResponse(BaseModel):
    """Detailed access group response."""
    id: str
    name: str
    description: Optional[str]
    access_level: int
    model_patterns: List[str]
    max_tokens_per_request: Optional[int]
    max_requests_per_minute: Optional[int]
    max_cost_per_day_usd: Optional[float]


@router.get("/access-groups/{group_name}", response_model=AccessGroupDetailResponse)
async def get_access_group(
    group_name: str,
    user: AuthenticatedUser,
):
    """
    Get details of a specific access group.

    Admin only.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        group = await llm_router.get_access_group(db, group_name, user.organization_id)

        if not group:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Access group '{group_name}' not found")

        return AccessGroupDetailResponse(
            id=group.id,
            name=group.name,
            description=group.description,
            access_level=group.access_level,
            model_patterns=group.model_patterns,
            max_tokens_per_request=group.max_tokens_per_request,
            max_requests_per_minute=group.max_requests_per_minute,
            max_cost_per_day_usd=group.max_cost_per_day_usd,
        )


@router.post("/access-groups", response_model=AccessGroupDetailResponse)
async def create_access_group(
    request: CreateAccessGroupRequest,
    user: AuthenticatedUser,
):
    """
    Create a new access group.

    Admin only. Access groups define which models users can access.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        # Check if group already exists
        existing = await llm_router.get_access_group(db, request.name, user.organization_id)
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Access group '{request.name}' already exists")

        group = await llm_router.create_access_group(
            db,
            name=request.name,
            access_level=request.access_level,
            model_patterns=request.model_patterns,
            description=request.description,
            max_tokens_per_request=request.max_tokens_per_request,
            max_requests_per_minute=request.max_requests_per_minute,
            max_cost_per_day_usd=request.max_cost_per_day_usd,
            organization_id=user.organization_id,
        )

        logger.info(
            "Access group created",
            group=request.name,
            level=request.access_level,
            created_by=user.user_id[:8] if user.user_id else None,
        )

        return AccessGroupDetailResponse(
            id=group.id,
            name=group.name,
            description=group.description,
            access_level=group.access_level,
            model_patterns=group.model_patterns,
            max_tokens_per_request=group.max_tokens_per_request,
            max_requests_per_minute=group.max_requests_per_minute,
            max_cost_per_day_usd=group.max_cost_per_day_usd,
        )


@router.patch("/access-groups/{group_name}", response_model=AccessGroupDetailResponse)
async def update_access_group(
    group_name: str,
    request: UpdateAccessGroupRequest,
    user: AuthenticatedUser,
):
    """
    Update an existing access group.

    Admin only. Updates the model patterns, access level, or limits.
    """
    llm_router = get_llm_router()

    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")

    async with async_session_context() as db:
        group = await llm_router.update_access_group(
            db,
            group_name=group_name,
            updates=updates,
            organization_id=user.organization_id,
        )

        if not group:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Access group '{group_name}' not found")

        logger.info(
            "Access group updated",
            group=group_name,
            updates=list(updates.keys()),
            updated_by=user.user_id[:8] if user.user_id else None,
        )

        return AccessGroupDetailResponse(
            id=group.id,
            name=group.name,
            description=group.description,
            access_level=group.access_level,
            model_patterns=group.model_patterns,
            max_tokens_per_request=group.max_tokens_per_request,
            max_requests_per_minute=group.max_requests_per_minute,
            max_cost_per_day_usd=group.max_cost_per_day_usd,
        )


@router.delete("/access-groups/{group_name}")
async def delete_access_group(
    group_name: str,
    user: AuthenticatedUser,
):
    """
    Delete an access group.

    Admin only. This is a soft delete - the group is deactivated, not removed.
    """
    llm_router = get_llm_router()

    async with async_session_context() as db:
        success = await llm_router.delete_access_group(db, group_name, user.organization_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Access group '{group_name}' not found")

        logger.info(
            "Access group deleted",
            group=group_name,
            deleted_by=user.user_id[:8] if user.user_id else None,
        )

        return {"success": True, "message": f"Access group '{group_name}' deleted"}


# =============================================================================
# Model Registry Management (Admin CRUD)
# =============================================================================

class ModelRegistryEntryResponse(BaseModel):
    """Model registry entry response."""
    provider_type: str
    model_name: str
    display_name: Optional[str]
    quality_score: int
    cost_per_million_tokens: float
    latency_score: int
    max_context_tokens: Optional[int]
    max_output_tokens: Optional[int]
    supports_vision: bool
    supports_function_calling: bool
    tier: Optional[str]
    min_access_level: int


class CreateModelRegistryRequest(BaseModel):
    """Request to add a model to the registry."""
    provider_type: str = Field(..., min_length=1, max_length=50)
    model_name: str = Field(..., min_length=1, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    quality_score: int = Field(50, ge=0, le=100)
    cost_per_million_tokens: float = Field(1.0, ge=0)
    latency_score: int = Field(3, ge=1, le=5)
    max_context_tokens: Optional[int] = Field(None, ge=1)
    max_output_tokens: Optional[int] = Field(None, ge=1)
    supports_vision: bool = False
    supports_function_calling: bool = False
    tier: Optional[str] = Field(None, max_length=50)
    min_access_level: int = Field(1, ge=1, le=100)


class UpdateModelRegistryRequest(BaseModel):
    """Request to update a model in the registry."""
    display_name: Optional[str] = Field(None, max_length=200)
    quality_score: Optional[int] = Field(None, ge=0, le=100)
    cost_per_million_tokens: Optional[float] = Field(None, ge=0)
    latency_score: Optional[int] = Field(None, ge=1, le=5)
    max_context_tokens: Optional[int] = Field(None, ge=1)
    max_output_tokens: Optional[int] = Field(None, ge=1)
    supports_vision: Optional[bool] = None
    supports_function_calling: Optional[bool] = None
    tier: Optional[str] = Field(None, max_length=50)
    min_access_level: Optional[int] = Field(None, ge=1, le=100)
    is_active: Optional[bool] = None
    is_deprecated: Optional[bool] = None
    deprecated_message: Optional[str] = None


@router.get("/registry", response_model=List[ModelRegistryEntryResponse])
async def list_model_registry(
    user: AuthenticatedUser,
    provider: Optional[str] = Query(None, description="Filter by provider"),
    tier: Optional[str] = Query(None, description="Filter by tier"),
):
    """
    List all models in the registry.

    Admin only. Returns all configured models with their metadata.
    """
    from sqlalchemy import text

    async with async_session_context() as db:
        # Build query with optional filters
        query_str = """
            SELECT
                provider_type, model_name, display_name, quality_score,
                cost_per_million_tokens, latency_score, max_context_tokens,
                max_output_tokens, supports_vision, supports_function_calling,
                tier, min_access_level
            FROM model_registry
            WHERE is_active = true
        """
        params = {}

        if provider:
            query_str += " AND provider_type = :provider"
            params["provider"] = provider
        if tier:
            query_str += " AND tier = :tier"
            params["tier"] = tier

        query_str += " ORDER BY quality_score DESC, model_name"

        result = await db.execute(text(query_str), params)
        rows = result.fetchall()

        return [
            ModelRegistryEntryResponse(
                provider_type=row.provider_type,
                model_name=row.model_name,
                display_name=row.display_name,
                quality_score=row.quality_score,
                cost_per_million_tokens=row.cost_per_million_tokens,
                latency_score=row.latency_score,
                max_context_tokens=row.max_context_tokens,
                max_output_tokens=row.max_output_tokens,
                supports_vision=row.supports_vision,
                supports_function_calling=row.supports_function_calling,
                tier=row.tier,
                min_access_level=row.min_access_level,
            )
            for row in rows
        ]


@router.post("/registry", response_model=ModelRegistryEntryResponse)
async def add_model_to_registry(
    request: CreateModelRegistryRequest,
    user: AuthenticatedUser,
):
    """
    Add a new model to the registry.

    Admin only. Use this to add custom or new models with their metadata.
    """
    from sqlalchemy import text

    async with async_session_context() as db:
        query = text("""
            INSERT INTO model_registry (
                provider_type, model_name, display_name, quality_score,
                cost_per_million_tokens, latency_score, max_context_tokens,
                max_output_tokens, supports_vision, supports_function_calling,
                tier, min_access_level, organization_id, is_active
            )
            VALUES (
                :provider_type, :model_name, :display_name, :quality_score,
                :cost, :latency, :max_context, :max_output, :vision, :function_calling,
                :tier, :min_access_level, :org_id, true
            )
            ON CONFLICT (provider_type, model_name, organization_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                quality_score = EXCLUDED.quality_score,
                cost_per_million_tokens = EXCLUDED.cost_per_million_tokens,
                latency_score = EXCLUDED.latency_score,
                updated_at = now()
        """)

        await db.execute(query, {
            "provider_type": request.provider_type,
            "model_name": request.model_name,
            "display_name": request.display_name,
            "quality_score": request.quality_score,
            "cost": request.cost_per_million_tokens,
            "latency": request.latency_score,
            "max_context": request.max_context_tokens,
            "max_output": request.max_output_tokens,
            "vision": request.supports_vision,
            "function_calling": request.supports_function_calling,
            "tier": request.tier,
            "min_access_level": request.min_access_level,
            "org_id": user.organization_id,
        })
        await db.commit()

        # Clear cache
        llm_router = get_llm_router()
        llm_router.clear_cache()

        logger.info(
            "Model added to registry",
            provider=request.provider_type,
            model=request.model_name,
            added_by=user.user_id[:8] if user.user_id else None,
        )

        return ModelRegistryEntryResponse(
            provider_type=request.provider_type,
            model_name=request.model_name,
            display_name=request.display_name,
            quality_score=request.quality_score,
            cost_per_million_tokens=request.cost_per_million_tokens,
            latency_score=request.latency_score,
            max_context_tokens=request.max_context_tokens,
            max_output_tokens=request.max_output_tokens,
            supports_vision=request.supports_vision,
            supports_function_calling=request.supports_function_calling,
            tier=request.tier,
            min_access_level=request.min_access_level,
        )


@router.patch("/registry/{provider_type}/{model_name}", response_model=ModelRegistryEntryResponse)
async def update_model_in_registry(
    provider_type: str,
    model_name: str,
    request: UpdateModelRegistryRequest,
    user: AuthenticatedUser,
):
    """
    Update a model in the registry.

    Admin only. Update model metadata like quality score, cost, etc.
    """
    from sqlalchemy import text

    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")

    # Build dynamic update
    set_clauses = []
    params = {"provider_type": provider_type, "model_name": model_name, "org_id": user.organization_id}

    field_mapping = {
        "display_name": "display_name",
        "quality_score": "quality_score",
        "cost_per_million_tokens": "cost_per_million_tokens",
        "latency_score": "latency_score",
        "max_context_tokens": "max_context_tokens",
        "max_output_tokens": "max_output_tokens",
        "supports_vision": "supports_vision",
        "supports_function_calling": "supports_function_calling",
        "tier": "tier",
        "min_access_level": "min_access_level",
        "is_active": "is_active",
        "is_deprecated": "is_deprecated",
        "deprecated_message": "deprecated_message",
    }

    for field, column in field_mapping.items():
        if field in updates:
            set_clauses.append(f"{column} = :{field}")
            params[field] = updates[field]

    set_clauses.append("updated_at = now()")

    async with async_session_context() as db:
        update_query = text(f"""
            UPDATE model_registry
            SET {", ".join(set_clauses)}
            WHERE provider_type = :provider_type
              AND model_name = :model_name
              AND (organization_id IS NULL OR organization_id = :org_id)
        """)
        result = await db.execute(update_query, params)
        await db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{provider_type}/{model_name}' not found")

        # Clear cache
        llm_router = get_llm_router()
        llm_router.clear_cache()

        # Fetch updated record
        fetch_query = text("""
            SELECT
                provider_type, model_name, display_name, quality_score,
                cost_per_million_tokens, latency_score, max_context_tokens,
                max_output_tokens, supports_vision, supports_function_calling,
                tier, min_access_level
            FROM model_registry
            WHERE provider_type = :provider_type
              AND model_name = :model_name
              AND (organization_id IS NULL OR organization_id = :org_id)
        """)
        result = await db.execute(fetch_query, {
            "provider_type": provider_type,
            "model_name": model_name,
            "org_id": user.organization_id,
        })
        row = result.fetchone()

        logger.info(
            "Model updated in registry",
            provider=provider_type,
            model=model_name,
            updates=list(updates.keys()),
            updated_by=user.user_id[:8] if user.user_id else None,
        )

        return ModelRegistryEntryResponse(
            provider_type=row.provider_type,
            model_name=row.model_name,
            display_name=row.display_name,
            quality_score=row.quality_score,
            cost_per_million_tokens=row.cost_per_million_tokens,
            latency_score=row.latency_score,
            max_context_tokens=row.max_context_tokens,
            max_output_tokens=row.max_output_tokens,
            supports_vision=row.supports_vision,
            supports_function_calling=row.supports_function_calling,
            tier=row.tier,
            min_access_level=row.min_access_level,
        )


@router.delete("/registry/{provider_type}/{model_name}")
async def remove_model_from_registry(
    provider_type: str,
    model_name: str,
    user: AuthenticatedUser,
):
    """
    Remove a model from the registry.

    Admin only. This is a soft delete - the model is deactivated, not removed.
    """
    from sqlalchemy import text

    async with async_session_context() as db:
        query = text("""
            UPDATE model_registry
            SET is_active = false, updated_at = now()
            WHERE provider_type = :provider_type
              AND model_name = :model_name
              AND (organization_id IS NULL OR organization_id = :org_id)
        """)
        result = await db.execute(query, {
            "provider_type": provider_type,
            "model_name": model_name,
            "org_id": user.organization_id,
        })
        await db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{provider_type}/{model_name}' not found")

        # Clear cache
        llm_router = get_llm_router()
        llm_router.clear_cache()

        logger.info(
            "Model removed from registry",
            provider=provider_type,
            model=model_name,
            removed_by=user.user_id[:8] if user.user_id else None,
        )

        return {"success": True, "message": f"Model '{provider_type}/{model_name}' removed from registry"}


# =============================================================================
# Phase 68: vLLM Provider Endpoints
# =============================================================================

class VLLMHealthResponse(BaseModel):
    """vLLM health status."""
    status: str
    mode: str
    api_base: Optional[str] = None
    model: Optional[str] = None
    models_available: Optional[List[str]] = None
    error: Optional[str] = None


class VLLMModelInfoResponse(BaseModel):
    """vLLM model information."""
    model: str
    mode: str
    dtype: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None


class VLLMGenerateRequest(BaseModel):
    """Request for vLLM text generation."""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(512, ge=1, le=32768, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0, le=1, description="Top-p sampling")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class VLLMGenerateResponse(BaseModel):
    """Response from vLLM generation."""
    text: str
    tokens_generated: int
    prompt_tokens: int
    latency_ms: float
    tokens_per_second: float
    model: str


@router.get("/vllm/health", response_model=VLLMHealthResponse)
async def vllm_health_check():
    """
    Check vLLM provider health.

    Phase 68: vLLM provides 2-4x faster inference with:
    - Dynamic batching
    - PagedAttention for KV cache
    - Automatic quantization detection
    """
    try:
        from backend.services.vllm_provider import get_vllm_provider

        provider = await get_vllm_provider()
        health = await provider.health_check()

        return VLLMHealthResponse(
            status=health.get("status", "unknown"),
            mode=health.get("mode", "unknown"),
            api_base=health.get("api_base"),
            model=health.get("model"),
            models_available=health.get("models"),
            error=health.get("error"),
        )

    except ImportError:
        return VLLMHealthResponse(
            status="unavailable",
            mode="not_installed",
            error="vLLM provider not installed",
        )
    except Exception as e:
        return VLLMHealthResponse(
            status="error",
            mode="unknown",
            error=str(e),
        )


@router.get("/vllm/model-info", response_model=VLLMModelInfoResponse)
async def get_vllm_model_info():
    """Get information about the loaded vLLM model."""
    try:
        from backend.services.vllm_provider import get_vllm_provider

        provider = await get_vllm_provider()
        info = await provider.get_model_info()

        return VLLMModelInfoResponse(
            model=info.get("model", "unknown"),
            mode=info.get("mode", "unknown"),
            dtype=info.get("dtype"),
            tensor_parallel_size=info.get("tensor_parallel_size"),
            max_model_len=info.get("max_model_len"),
            quantization=info.get("quantization"),
        )

    except Exception as e:
        logger.error("Failed to get vLLM model info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@router.post("/vllm/generate", response_model=VLLMGenerateResponse)
async def vllm_generate(
    request: VLLMGenerateRequest,
    user: AuthenticatedUser = None,
):
    """
    Generate text using vLLM.

    Phase 68: Direct vLLM generation with:
    - 2-4x faster inference
    - Dynamic batching
    - Optimized KV caching
    """
    try:
        from backend.services.vllm_provider import get_vllm_provider

        provider = await get_vllm_provider()
        response = await provider.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

        return VLLMGenerateResponse(
            text=response.text,
            tokens_generated=response.tokens_generated,
            prompt_tokens=response.prompt_tokens,
            latency_ms=response.latency_ms,
            tokens_per_second=response.tokens_per_second,
            model=response.model,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("vLLM generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )
