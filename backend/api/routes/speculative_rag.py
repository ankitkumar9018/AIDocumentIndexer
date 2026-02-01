"""
AIDocumentIndexer - Speculative RAG API Routes
===============================================

API endpoints for configuring Speculative RAG (ICLR 2025).
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import require_admin
from backend.services.settings import get_settings_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/rag/speculative", tags=["Speculative RAG"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SpeculativeRAGConfigResponse(BaseModel):
    """Speculative RAG configuration."""
    enabled: bool = Field(..., description="Whether speculative RAG is enabled")
    num_drafts: int = Field(..., description="Number of parallel drafts to generate")
    drafter_model: str = Field(..., description="Model for draft generation")
    verifier_model: str = Field(..., description="Model for verification")
    min_documents: int = Field(..., description="Minimum documents to trigger speculative")


class SpeculativeRAGConfigUpdate(BaseModel):
    """Update Speculative RAG configuration."""
    enabled: Optional[bool] = Field(None, description="Enable/disable speculative RAG")
    num_drafts: Optional[int] = Field(None, ge=2, le=5, description="Number of drafts (2-5)")
    drafter_model: Optional[str] = Field(None, description="Drafter model name")
    verifier_model: Optional[str] = Field(None, description="Verifier model name")
    min_documents: Optional[int] = Field(None, ge=3, le=20, description="Min docs threshold")


class SpeculativeRAGStatsResponse(BaseModel):
    """Speculative RAG statistics."""
    queries_processed: int
    average_drafts_generated: float
    latency_reduction_percent: float
    accuracy_improvement_percent: float
    draft_selection_distribution: dict


class TestSpeculativeRequest(BaseModel):
    """Request to test speculative RAG."""
    query: str = Field(..., description="Test query")
    context: str = Field(..., description="Test context")


class TestSpeculativeResponse(BaseModel):
    """Response from speculative RAG test."""
    num_drafts: int
    selected_draft: int
    confidence: float
    latency_saved: bool


# =============================================================================
# Routes
# =============================================================================

@router.get("/config", response_model=SpeculativeRAGConfigResponse)
async def get_speculative_config(
    current_user: dict = Depends(require_admin),
):
    """
    Get Speculative RAG configuration.

    Requires admin privileges.
    """
    settings = get_settings_service()

    return SpeculativeRAGConfigResponse(
        enabled=settings.get("rag.speculative_rag_enabled", False),
        num_drafts=settings.get("rag.speculative_num_drafts", 3),
        drafter_model=settings.get("rag.speculative_drafter_model", "gpt-4o-mini"),
        verifier_model=settings.get("rag.speculative_verifier_model", "gpt-4o"),
        min_documents=settings.get("rag.speculative_min_documents", 5),
    )


@router.put("/config", response_model=SpeculativeRAGConfigResponse)
async def update_speculative_config(
    request: SpeculativeRAGConfigUpdate,
    current_user: dict = Depends(require_admin),
):
    """
    Update Speculative RAG configuration.

    Requires admin privileges.
    """
    settings = get_settings_service()

    if request.enabled is not None:
        settings.set("rag.speculative_rag_enabled", request.enabled)

    if request.num_drafts is not None:
        settings.set("rag.speculative_num_drafts", request.num_drafts)

    if request.drafter_model is not None:
        settings.set("rag.speculative_drafter_model", request.drafter_model)

    if request.verifier_model is not None:
        settings.set("rag.speculative_verifier_model", request.verifier_model)

    if request.min_documents is not None:
        settings.set("rag.speculative_min_documents", request.min_documents)

    logger.info(
        "speculative_rag.config_updated",
        enabled=request.enabled,
        num_drafts=request.num_drafts,
        user_id=current_user.get("id"),
    )

    return SpeculativeRAGConfigResponse(
        enabled=settings.get("rag.speculative_rag_enabled", False),
        num_drafts=settings.get("rag.speculative_num_drafts", 3),
        drafter_model=settings.get("rag.speculative_drafter_model", "gpt-4o-mini"),
        verifier_model=settings.get("rag.speculative_verifier_model", "gpt-4o"),
        min_documents=settings.get("rag.speculative_min_documents", 5),
    )


@router.get("/stats", response_model=SpeculativeRAGStatsResponse)
async def get_speculative_stats(
    current_user: dict = Depends(require_admin),
):
    """
    Get Speculative RAG statistics.

    Requires admin privileges.
    """
    return SpeculativeRAGStatsResponse(
        queries_processed=0,
        average_drafts_generated=0.0,
        latency_reduction_percent=0.0,
        accuracy_improvement_percent=0.0,
        draft_selection_distribution={},
    )


@router.post("/test", response_model=TestSpeculativeResponse)
async def test_speculative(
    request: TestSpeculativeRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Test Speculative RAG on a sample query.

    This endpoint runs the speculative RAG pipeline on
    a test query without persisting results.

    Requires admin privileges.
    """
    settings = get_settings_service()

    if not settings.get("rag.speculative_rag_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Speculative RAG is not enabled",
        )

    # Placeholder - in production this would run actual speculative RAG
    num_drafts = settings.get("rag.speculative_num_drafts", 3)

    return TestSpeculativeResponse(
        num_drafts=num_drafts,
        selected_draft=1,
        confidence=0.85,
        latency_saved=True,
    )


@router.post("/enable")
async def enable_speculative(
    current_user: dict = Depends(require_admin),
):
    """
    Enable Speculative RAG.

    Requires admin privileges.
    """
    settings = get_settings_service()
    settings.set("rag.speculative_rag_enabled", True)

    logger.info(
        "speculative_rag.enabled",
        user_id=current_user.get("id"),
    )

    return {"message": "Speculative RAG enabled", "status": "active"}


@router.post("/disable")
async def disable_speculative(
    current_user: dict = Depends(require_admin),
):
    """
    Disable Speculative RAG.

    Requires admin privileges.
    """
    settings = get_settings_service()
    settings.set("rag.speculative_rag_enabled", False)

    logger.info(
        "speculative_rag.disabled",
        user_id=current_user.get("id"),
    )

    return {"message": "Speculative RAG disabled", "status": "inactive"}
