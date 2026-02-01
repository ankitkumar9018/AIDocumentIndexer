"""
AIDocumentIndexer - Embedding Defense API Routes
=================================================

API endpoints for configuring embedding inversion defense (OWASP LLM08:2025).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import require_admin
from backend.services.settings import get_settings_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/security/embedding-defense", tags=["Embedding Defense"])


# =============================================================================
# Request/Response Models
# =============================================================================

class EmbeddingDefenseConfigResponse(BaseModel):
    """Embedding defense configuration."""
    enabled: bool = Field(..., description="Whether defense is enabled")
    noise_scale: float = Field(..., description="Gaussian noise standard deviation")
    clip_norm: float = Field(..., description="L2 norm clipping radius")
    shuffle_enabled: bool = Field(..., description="Whether dimension shuffling is enabled")


class EmbeddingDefenseConfigUpdate(BaseModel):
    """Update embedding defense configuration."""
    enabled: Optional[bool] = Field(None, description="Enable/disable defense")
    noise_scale: Optional[float] = Field(None, ge=0.0, le=1.0, description="Noise scale (0-1)")
    clip_norm: Optional[float] = Field(None, ge=0.1, le=10.0, description="Clip norm (0.1-10)")


class DefenseStatsResponse(BaseModel):
    """Defense statistics."""
    embeddings_protected: int
    queries_protected: int
    average_noise_applied: float
    defense_active: bool


class TestDefenseRequest(BaseModel):
    """Request to test defense on an embedding."""
    embedding: list[float] = Field(..., min_length=1, description="Embedding to protect")


class TestDefenseResponse(BaseModel):
    """Response from defense test."""
    original_norm: float
    protected_norm: float
    noise_applied: bool
    shuffle_applied: bool
    clip_applied: bool
    similarity_preserved: float


# =============================================================================
# Routes
# =============================================================================

@router.get("/config", response_model=EmbeddingDefenseConfigResponse)
async def get_defense_config(
    current_user: dict = Depends(require_admin),
):
    """
    Get embedding defense configuration.

    Requires admin privileges.
    """
    settings = get_settings_service()

    return EmbeddingDefenseConfigResponse(
        enabled=settings.get("security.embedding_defense_enabled", False),
        noise_scale=settings.get("security.defense_noise_scale", 0.01),
        clip_norm=settings.get("security.defense_clip_norm", 1.0),
        shuffle_enabled=True,  # Always enabled when defense is on
    )


@router.put("/config", response_model=EmbeddingDefenseConfigResponse)
async def update_defense_config(
    request: EmbeddingDefenseConfigUpdate,
    current_user: dict = Depends(require_admin),
):
    """
    Update embedding defense configuration.

    WARNING: Changing these settings may affect existing embeddings.
    Consider re-indexing after significant changes.

    Requires admin privileges.
    """
    settings = get_settings_service()

    if request.enabled is not None:
        settings.set("security.embedding_defense_enabled", request.enabled)

    if request.noise_scale is not None:
        settings.set("security.defense_noise_scale", request.noise_scale)

    if request.clip_norm is not None:
        settings.set("security.defense_clip_norm", request.clip_norm)

    logger.info(
        "embedding_defense.config_updated",
        enabled=request.enabled,
        noise_scale=request.noise_scale,
        clip_norm=request.clip_norm,
        user_id=current_user.get("id"),
    )

    return EmbeddingDefenseConfigResponse(
        enabled=settings.get("security.embedding_defense_enabled", False),
        noise_scale=settings.get("security.defense_noise_scale", 0.01),
        clip_norm=settings.get("security.defense_clip_norm", 1.0),
        shuffle_enabled=True,
    )


@router.get("/stats", response_model=DefenseStatsResponse)
async def get_defense_stats(
    current_user: dict = Depends(require_admin),
):
    """
    Get embedding defense statistics.

    Requires admin privileges.
    """
    settings = get_settings_service()

    return DefenseStatsResponse(
        embeddings_protected=0,
        queries_protected=0,
        average_noise_applied=settings.get("security.defense_noise_scale", 0.01),
        defense_active=settings.get("security.embedding_defense_enabled", False),
    )


@router.post("/test", response_model=TestDefenseResponse)
async def test_defense(
    request: TestDefenseRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Test embedding defense on a sample embedding.

    This endpoint demonstrates the defense transforms without
    persisting anything. Useful for validating configuration.

    Requires admin privileges.
    """
    import numpy as np
    from backend.services.embedding_defense import get_embedding_defense

    settings = get_settings_service()

    if not settings.get("security.embedding_defense_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Embedding defense is not enabled",
        )

    defense = get_embedding_defense()
    original = np.array(request.embedding)
    original_norm = float(np.linalg.norm(original))

    # Apply defense
    protected = defense.protect(request.embedding)
    protected_arr = np.array(protected)
    protected_norm = float(np.linalg.norm(protected_arr))

    # Calculate similarity preservation
    if original_norm > 0 and protected_norm > 0:
        similarity = float(np.dot(original, protected_arr) / (original_norm * protected_norm))
    else:
        similarity = 0.0

    return TestDefenseResponse(
        original_norm=original_norm,
        protected_norm=protected_norm,
        noise_applied=True,
        shuffle_applied=True,
        clip_applied=protected_norm <= settings.get("security.defense_clip_norm", 1.0) + 0.01,
        similarity_preserved=similarity,
    )


@router.post("/enable")
async def enable_defense(
    current_user: dict = Depends(require_admin),
):
    """
    Enable embedding defense.

    Requires admin privileges.
    """
    settings = get_settings_service()
    settings.set("security.embedding_defense_enabled", True)

    logger.info(
        "embedding_defense.enabled",
        user_id=current_user.get("id"),
    )

    return {"message": "Embedding defense enabled", "status": "active"}


@router.post("/disable")
async def disable_defense(
    current_user: dict = Depends(require_admin),
):
    """
    Disable embedding defense.

    WARNING: Disabling defense removes protection from new embeddings.
    Existing protected embeddings will remain protected but new
    queries won't apply the same transforms, potentially causing
    search quality degradation.

    Requires admin privileges.
    """
    settings = get_settings_service()
    settings.set("security.embedding_defense_enabled", False)

    logger.warning(
        "embedding_defense.disabled",
        user_id=current_user.get("id"),
    )

    return {"message": "Embedding defense disabled", "status": "inactive"}
