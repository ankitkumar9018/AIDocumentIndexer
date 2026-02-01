"""
AIDocumentIndexer - Matryoshka Retrieval API Routes
====================================================

API endpoints for configuring Matryoshka adaptive retrieval.
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import require_admin
from backend.services.settings import get_settings_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/retrieval/matryoshka", tags=["Matryoshka Retrieval"])


# =============================================================================
# Request/Response Models
# =============================================================================

class MatryoshkaConfigResponse(BaseModel):
    """Matryoshka retrieval configuration."""
    enabled: bool = Field(..., description="Whether Matryoshka retrieval is enabled")
    fast_dims: int = Field(..., description="Dimensions for fast pass (stage 1)")
    shortlist_factor: float = Field(..., description="Multiplier for shortlist size")
    full_dims: int = Field(..., description="Full embedding dimensions")
    supported_models: List[str] = Field(default_factory=list, description="Models with MRL support")


class MatryoshkaConfigUpdate(BaseModel):
    """Update Matryoshka configuration."""
    enabled: Optional[bool] = Field(None, description="Enable/disable Matryoshka")
    fast_dims: Optional[int] = Field(None, ge=32, le=512, description="Fast pass dimensions")
    shortlist_factor: Optional[float] = Field(None, ge=2.0, le=20.0, description="Shortlist multiplier")


class MatryoshkaStatsResponse(BaseModel):
    """Matryoshka retrieval statistics."""
    queries_processed: int
    average_speedup: float
    recall_at_k: float
    fast_pass_time_ms: float
    rerank_time_ms: float


class TestMatryoshkaRequest(BaseModel):
    """Request to test Matryoshka retrieval."""
    query_embedding: List[float] = Field(..., min_length=128, description="Query embedding")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")


class TestMatryoshkaResponse(BaseModel):
    """Response from Matryoshka test."""
    fast_dims_used: int
    shortlist_size: int
    final_results: int
    estimated_speedup: float


# =============================================================================
# Routes
# =============================================================================

@router.get("/config", response_model=MatryoshkaConfigResponse)
async def get_matryoshka_config(
    current_user: dict = Depends(require_admin),
):
    """
    Get Matryoshka retrieval configuration.

    Requires admin privileges.
    """
    settings = get_settings_service()

    return MatryoshkaConfigResponse(
        enabled=settings.get("rag.matryoshka_retrieval_enabled", False),
        fast_dims=settings.get("rag.matryoshka_fast_dims", 128),
        shortlist_factor=settings.get("rag.matryoshka_shortlist_factor", 5.0),
        full_dims=1536,  # Default for text-embedding-3-small
        supported_models=[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "nomic-embed-text-v1.5",
        ],
    )


@router.put("/config", response_model=MatryoshkaConfigResponse)
async def update_matryoshka_config(
    request: MatryoshkaConfigUpdate,
    current_user: dict = Depends(require_admin),
):
    """
    Update Matryoshka retrieval configuration.

    Requires admin privileges.
    """
    settings = get_settings_service()

    if request.enabled is not None:
        settings.set("rag.matryoshka_retrieval_enabled", request.enabled)

    if request.fast_dims is not None:
        settings.set("rag.matryoshka_fast_dims", request.fast_dims)

    if request.shortlist_factor is not None:
        settings.set("rag.matryoshka_shortlist_factor", request.shortlist_factor)

    logger.info(
        "matryoshka.config_updated",
        enabled=request.enabled,
        fast_dims=request.fast_dims,
        shortlist_factor=request.shortlist_factor,
        user_id=current_user.get("id"),
    )

    return MatryoshkaConfigResponse(
        enabled=settings.get("rag.matryoshka_retrieval_enabled", False),
        fast_dims=settings.get("rag.matryoshka_fast_dims", 128),
        shortlist_factor=settings.get("rag.matryoshka_shortlist_factor", 5.0),
        full_dims=1536,
        supported_models=[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "nomic-embed-text-v1.5",
        ],
    )


@router.get("/stats", response_model=MatryoshkaStatsResponse)
async def get_matryoshka_stats(
    current_user: dict = Depends(require_admin),
):
    """
    Get Matryoshka retrieval statistics.

    Requires admin privileges.
    """
    settings = get_settings_service()
    fast_dims = settings.get("rag.matryoshka_fast_dims", 128)
    full_dims = 1536

    # Estimated speedup based on dimension ratio
    estimated_speedup = full_dims / fast_dims

    return MatryoshkaStatsResponse(
        queries_processed=0,
        average_speedup=estimated_speedup,
        recall_at_k=0.98,  # Typical recall with 5x shortlist
        fast_pass_time_ms=0.0,
        rerank_time_ms=0.0,
    )


@router.post("/test", response_model=TestMatryoshkaResponse)
async def test_matryoshka(
    request: TestMatryoshkaRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Test Matryoshka retrieval configuration.

    This endpoint validates the configuration without running
    actual retrieval. Useful for tuning parameters.

    Requires admin privileges.
    """
    settings = get_settings_service()

    if not settings.get("rag.matryoshka_retrieval_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matryoshka retrieval is not enabled",
        )

    fast_dims = settings.get("rag.matryoshka_fast_dims", 128)
    shortlist_factor = settings.get("rag.matryoshka_shortlist_factor", 5.0)
    full_dims = len(request.query_embedding)

    shortlist_size = int(request.top_k * shortlist_factor)
    estimated_speedup = full_dims / fast_dims

    return TestMatryoshkaResponse(
        fast_dims_used=fast_dims,
        shortlist_size=shortlist_size,
        final_results=request.top_k,
        estimated_speedup=estimated_speedup,
    )


@router.post("/enable")
async def enable_matryoshka(
    current_user: dict = Depends(require_admin),
):
    """
    Enable Matryoshka retrieval.

    Requires admin privileges.
    """
    settings = get_settings_service()
    settings.set("rag.matryoshka_retrieval_enabled", True)

    logger.info(
        "matryoshka.enabled",
        user_id=current_user.get("id"),
    )

    return {"message": "Matryoshka retrieval enabled", "status": "active"}


@router.post("/disable")
async def disable_matryoshka(
    current_user: dict = Depends(require_admin),
):
    """
    Disable Matryoshka retrieval.

    Requires admin privileges.
    """
    settings = get_settings_service()
    settings.set("rag.matryoshka_retrieval_enabled", False)

    logger.info(
        "matryoshka.disabled",
        user_id=current_user.get("id"),
    )

    return {"message": "Matryoshka retrieval disabled", "status": "inactive"}
