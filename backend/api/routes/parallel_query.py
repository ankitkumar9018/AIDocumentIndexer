"""
AIDocumentIndexer - Parallel Query API Routes
==============================================

Endpoints for parallel RAG + non-RAG knowledge enhancement.
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.parallel_knowledge import (
    ParallelKnowledgeService,
    OutputMode,
    MergeStrategy,
    get_parallel_knowledge_service,
)
from backend.services.rag import get_rag_service
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class ParallelQueryRequest(BaseModel):
    """Request for parallel knowledge query."""
    query: str = Field(..., description="The question to answer")
    output_mode: str = Field(
        default="separate",
        description="Output mode: separate, merged, rag_only, general, toggle"
    )
    merge_strategy: str = Field(
        default="synthesis",
        description="Merge strategy: synthesis, weighted, rag_primary, general_primary"
    )
    rag_provider_id: Optional[str] = Field(None, description="Provider ID for RAG query")
    rag_model: Optional[str] = Field(None, description="Model for RAG query")
    general_provider_id: Optional[str] = Field(None, description="Provider ID for general query")
    general_model: Optional[str] = Field(None, description="Model for general query")
    rag_top_k: int = Field(default=5, ge=1, le=20, description="Number of documents for RAG")


class KnowledgeSourceResponse(BaseModel):
    """A single knowledge source result."""
    source_type: str
    answer: str
    confidence: float
    model_used: str
    sources_count: int = 0
    latency_ms: int = 0


class ParallelQueryResponse(BaseModel):
    """Response from parallel knowledge query."""
    id: str
    query: str
    rag_result: Optional[KnowledgeSourceResponse] = None
    general_result: Optional[KnowledgeSourceResponse] = None
    merged_result: Optional[str] = None
    display_answer: str
    output_mode: str
    merge_strategy: Optional[str] = None
    total_latency_ms: int
    models_used: List[str]


class SwitchModeRequest(BaseModel):
    """Request to switch output mode for an existing query."""
    new_mode: str = Field(..., description="New output mode")
    merge_strategy: Optional[str] = Field(None, description="Merge strategy if switching to merged")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/query", response_model=ParallelQueryResponse)
async def parallel_query(
    request: ParallelQueryRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute a parallel RAG + non-RAG query.

    This endpoint:
    1. Sends the query to the RAG pipeline (document-grounded)
    2. Sends the query to a general LLM (without documents)
    3. Returns both results, optionally merged

    Output modes:
    - separate: Show both answers side by side
    - merged: Synthesize into one comprehensive answer
    - rag_only: Only use RAG result
    - general: Only use general knowledge result
    - toggle: UI can toggle between views
    """
    query_id = str(uuid4())

    logger.info(
        "Parallel query request",
        query_id=query_id,
        output_mode=request.output_mode,
        user_id=user.user_id,
    )

    # Parse output mode
    try:
        output_mode = OutputMode(request.output_mode)
    except ValueError:
        output_mode = OutputMode.SEPARATE

    # Parse merge strategy
    try:
        merge_strategy = MergeStrategy(request.merge_strategy)
    except ValueError:
        merge_strategy = MergeStrategy.SYNTHESIS

    # Get RAG service
    rag_service = get_rag_service()

    # Create parallel knowledge service
    parallel_service = get_parallel_knowledge_service(rag_service=rag_service)

    # Resolve provider IDs to provider types
    rag_provider = settings.DEFAULT_LLM_PROVIDER
    rag_model = request.rag_model or settings.DEFAULT_CHAT_MODEL
    general_provider = settings.DEFAULT_LLM_PROVIDER
    general_model = request.general_model or settings.DEFAULT_CHAT_MODEL

    # If provider IDs specified, look them up
    if request.rag_provider_id:
        from backend.services.llm_provider import LLMProviderService
        try:
            provider = await LLMProviderService.get_provider(db, request.rag_provider_id)
            if provider:
                rag_provider = provider.provider_type
                rag_model = request.rag_model or provider.default_chat_model or rag_model
        except Exception as e:
            logger.warning(f"Could not load RAG provider: {e}")

    if request.general_provider_id:
        from backend.services.llm_provider import LLMProviderService
        try:
            provider = await LLMProviderService.get_provider(db, request.general_provider_id)
            if provider:
                general_provider = provider.provider_type
                general_model = request.general_model or provider.default_chat_model or general_model
        except Exception as e:
            logger.warning(f"Could not load general provider: {e}")

    try:
        result = await parallel_service.query(
            query=request.query,
            rag_provider=rag_provider,
            rag_model=rag_model,
            general_provider=general_provider,
            general_model=general_model,
            output_mode=output_mode,
            merge_strategy=merge_strategy,
            rag_top_k=request.rag_top_k,
        )

        # Build response
        rag_response = None
        if result.rag_result:
            rag_response = KnowledgeSourceResponse(
                source_type=result.rag_result.source_type,
                answer=result.rag_result.answer,
                confidence=result.rag_result.confidence,
                model_used=result.rag_result.model_used,
                sources_count=len(result.rag_result.sources),
                latency_ms=result.rag_result.latency_ms,
            )

        general_response = None
        if result.general_result:
            general_response = KnowledgeSourceResponse(
                source_type=result.general_result.source_type,
                answer=result.general_result.answer,
                confidence=result.general_result.confidence,
                model_used=result.general_result.model_used,
                sources_count=0,
                latency_ms=result.general_result.latency_ms,
            )

        logger.info(
            "Parallel query completed",
            query_id=query_id,
            has_rag=result.rag_result is not None,
            has_general=result.general_result is not None,
            has_merged=result.merged_result is not None,
            latency_ms=result.total_latency_ms,
        )

        return ParallelQueryResponse(
            id=query_id,
            query=request.query,
            rag_result=rag_response,
            general_result=general_response,
            merged_result=result.merged_result,
            display_answer=result.get_display_answer(),
            output_mode=output_mode.value,
            merge_strategy=merge_strategy.value if result.merged_result else None,
            total_latency_ms=result.total_latency_ms,
            models_used=result.models_used,
        )

    except Exception as e:
        logger.error("Parallel query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parallel query failed: {str(e)}"
        )


@router.get("/modes")
async def get_output_modes(user: AuthenticatedUser):
    """
    Get available output modes and their descriptions.
    """
    return {
        "output_modes": [
            {
                "id": "separate",
                "name": "Side by Side",
                "description": "View RAG and general answers separately",
                "icon": "columns",
            },
            {
                "id": "merged",
                "name": "Merged",
                "description": "Synthesize both into one comprehensive answer",
                "icon": "merge",
            },
            {
                "id": "rag_only",
                "name": "Documents Only",
                "description": "Only use document-grounded answer",
                "icon": "file-text",
            },
            {
                "id": "general",
                "name": "General Knowledge",
                "description": "Only use general LLM knowledge",
                "icon": "brain",
            },
            {
                "id": "toggle",
                "name": "Toggle View",
                "description": "Switch between views interactively",
                "icon": "toggle-left",
            },
        ],
        "merge_strategies": [
            {
                "id": "synthesis",
                "name": "AI Synthesis",
                "description": "LLM creates a unified answer from both sources",
            },
            {
                "id": "weighted",
                "name": "Confidence Weighted",
                "description": "Use the higher confidence answer",
            },
            {
                "id": "rag_primary",
                "name": "Documents Primary",
                "description": "Documents as main, general supplements",
            },
            {
                "id": "general_primary",
                "name": "General Primary",
                "description": "General as main, documents validate",
            },
        ],
    }
