"""
AIDocumentIndexer - Experiments & Feedback API Routes
=====================================================

API endpoints for experiment tracking, feedback collection, A/B testing,
and feature flag management.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.deps import get_current_user
from backend.api.middleware.auth import require_admin

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])


# =============================================================================
# Feature Flag Management (synced with settings service)
# =============================================================================

# Maps experiment name → settings key
EXPERIMENT_SETTINGS_MAP: Dict[str, str] = {
    "attention_rag": "rag.attention_rag_enabled",
    "graph_o1": "rag.graph_o1_enabled",
    "tiered_reranking": "rerank.tiered_enabled",
    "adaptive_routing": "rag.adaptive_routing_enabled",
    "late_chunking": "vectorstore.late_chunking_enabled",
    "tree_of_thoughts": "rag.tree_of_thoughts_enabled",
    "generative_cache": "rag.semantic_cache_enabled",
    "llmlingua_compression": "rag.llmlingua_compression_enabled",
}

# Static metadata for each experiment (description, category, stability)
EXPERIMENT_METADATA: Dict[str, Dict[str, str]] = {
    "attention_rag": {
        "description": "6.3x context compression using attention scoring (AttentionRAG). Reduces context size while maintaining relevance.",
        "category": "Compression",
        "status": "beta",
    },
    "graph_o1": {
        "description": "Beam search reasoning over the knowledge graph (Graph-O1). 3-5x faster graph reasoning with 95%+ accuracy.",
        "category": "Retrieval",
        "status": "experimental",
    },
    "tiered_reranking": {
        "description": "4-stage reranking pipeline: BM25 \u2192 CrossEncoder \u2192 ColBERT \u2192 LLM. Doubles precision for complex queries.",
        "category": "Retrieval",
        "status": "stable",
    },
    "adaptive_routing": {
        "description": "Query-dependent strategy routing (DIRECT/HYBRID/TWO_STAGE/AGENTIC/GRAPH). Selects optimal retrieval based on query complexity.",
        "category": "Retrieval",
        "status": "stable",
    },
    "late_chunking": {
        "description": "Context-preserving chunking with +15-25% retrieval accuracy. Uses full document context for chunk embeddings.",
        "category": "Processing",
        "status": "beta",
    },
    "tree_of_thoughts": {
        "description": "Multi-path reasoning with beam search exploration. Generates and evaluates multiple reasoning chains.",
        "category": "Reasoning",
        "status": "experimental",
    },
    "generative_cache": {
        "description": "Semantic caching with 9x speedup over GPTCache. 68.8% cache hit rate for repeated queries.",
        "category": "Performance",
        "status": "stable",
    },
    "llmlingua_compression": {
        "description": "LLMLingua-2 context compression (3-6x). Reduces token usage while preserving answer quality.",
        "category": "Compression",
        "status": "beta",
    },
}


class ToggleExperimentRequest(BaseModel):
    """Request to toggle an experiment feature flag."""
    enabled: bool = Field(..., description="Whether to enable or disable the experiment")


@router.get("/")
async def get_feature_flags(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get all experiment feature flags with their current states.

    Reads enabled/disabled state from the settings service.
    Returns metadata (description, category, status) alongside each flag.
    """
    experiments = []

    try:
        from backend.services.settings import get_settings_service
        settings_svc = get_settings_service()

        # Batch-load all experiment settings
        settings_keys = list(EXPERIMENT_SETTINGS_MAP.values())
        settings_values = await settings_svc.get_settings_batch(settings_keys)

        for name, settings_key in EXPERIMENT_SETTINGS_MAP.items():
            meta = EXPERIMENT_METADATA.get(name, {})
            enabled = settings_values.get(settings_key)
            # Fall back to settings default if not in DB
            if enabled is None:
                enabled = settings_svc.get_default_value(settings_key)
            experiments.append({
                "name": name,
                "enabled": bool(enabled) if enabled is not None else False,
                "description": meta.get("description", ""),
                "category": meta.get("category", "General"),
                "status": meta.get("status", "experimental"),
            })

    except Exception as e:
        logger.warning("Failed to load experiment flags from settings, using defaults", error=str(e))
        for name, meta in EXPERIMENT_METADATA.items():
            experiments.append({
                "name": name,
                "enabled": False,
                "description": meta.get("description", ""),
                "category": meta.get("category", "General"),
                "status": meta.get("status", "experimental"),
            })

    return {"experiments": experiments}


@router.put("/{name}")
async def toggle_feature_flag(name: str, request: ToggleExperimentRequest, user: dict = Depends(require_admin)) -> Dict[str, Any]:
    """
    Toggle an experiment feature flag.

    Maps the experiment name to a settings key and persists via the settings service.
    """
    if name not in EXPERIMENT_SETTINGS_MAP:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown experiment",
        )

    settings_key = EXPERIMENT_SETTINGS_MAP[name]

    try:
        from backend.services.settings import get_settings_service
        settings_svc = get_settings_service()
        await settings_svc.set_setting(settings_key, request.enabled)

        logger.info("Toggled experiment", experiment=name, settings_key=settings_key, enabled=request.enabled)

        return {
            "name": name,
            "enabled": request.enabled,
            "settings_key": settings_key,
            "status": "updated",
        }

    except Exception as e:
        logger.error("Failed to toggle experiment", experiment=name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle experiment",
        )


# =============================================================================
# Request/Response Models
# =============================================================================

class FeedbackTypeEnum(str, Enum):
    """Types of user feedback."""
    CLICK = "click"
    DWELL = "dwell"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    RATING = "rating"
    SELECTED = "selected"


class RecordClickRequest(BaseModel):
    """Request to record a click."""
    query: str = Field(..., description="Search query")
    query_id: str = Field(..., description="Query identifier")
    document_id: str = Field(..., description="Document that was clicked")
    document_content: str = Field(..., description="Document content (for feedback)")
    rank_position: int = Field(..., ge=0, description="Position in search results")
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    reranker_model: str = Field("cohere-rerank-v4", description="Reranker used")


class RecordDwellRequest(BaseModel):
    """Request to record dwell time."""
    session_id: str = Field(..., description="Session identifier")
    document_id: str = Field(..., description="Document identifier")
    dwell_time_ms: int = Field(..., ge=0, description="Time spent viewing (ms)")


class RecordExplicitFeedbackRequest(BaseModel):
    """Request to record explicit feedback."""
    query_id: str = Field(..., description="Query identifier")
    document_id: str = Field(..., description="Document identifier")
    is_positive: bool = Field(..., description="Positive or negative feedback")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Optional 1-5 rating")
    comment: Optional[str] = Field(None, description="Optional user comment")


class FeedbackResponse(BaseModel):
    """Response from feedback recording."""
    feedback_id: str
    recorded: bool
    message: str


class FeedbackMetricsResponse(BaseModel):
    """Feedback metrics."""
    total_queries: int
    total_clicks: int
    total_positive: int
    total_negative: int
    avg_click_position: float
    mrr: float
    satisfaction_rate: float


class CreateExperimentRequest(BaseModel):
    """Request to create an experiment."""
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    retriever_config: Optional[Dict[str, Any]] = None
    reranker_config: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None


class LogEvaluationRequest(BaseModel):
    """Request to log an evaluation."""
    experiment_id: str = Field(..., description="Experiment to log to")
    query: str = Field(..., description="User query", max_length=10000)
    answer: str = Field(..., description="Generated answer", max_length=50000)
    contexts: List[str] = Field(
        ...,
        description="Retrieved contexts",
        min_length=1,
        max_length=50,  # Limit to prevent DoS (Phase 69)
    )
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    run_feedback: bool = Field(True, description="Run feedback evaluation")


class ExperimentSummaryResponse(BaseModel):
    """Experiment summary."""
    experiment_id: str
    name: str
    total_records: int
    avg_relevance: float
    avg_groundedness: float
    avg_latency_ms: float
    pass_rate: float
    status: str


# =============================================================================
# Feedback Endpoints
# =============================================================================

@router.post("/feedback/click", response_model=FeedbackResponse)
async def record_click(request: RecordClickRequest, user: dict = Depends(get_current_user)) -> FeedbackResponse:
    """
    Record a click on a search result.

    Used for implicit feedback to improve reranking.
    """
    logger.info(
        "Recording click",
        query_id=request.query_id,
        document_id=request.document_id,
        position=request.rank_position,
    )

    try:
        from backend.services.reranker_feedback import get_feedback_collector

        collector = await get_feedback_collector()
        feedback_id = await collector.record_click(
            query=request.query,
            query_id=request.query_id,
            document_id=request.document_id,
            document_content=request.document_content,
            rank_position=request.rank_position,
            session_id=request.session_id,
            user_id=request.user_id,
            reranker_model=request.reranker_model,
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            recorded=True,
            message="Click recorded successfully",
        )

    except Exception as e:
        logger.error("Failed to record click", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record click",
        )


@router.post("/feedback/dwell", response_model=FeedbackResponse)
async def record_dwell(request: RecordDwellRequest, user: dict = Depends(get_current_user)) -> FeedbackResponse:
    """
    Record dwell time on a search result.

    Call when user navigates away from result page.
    """
    try:
        from backend.services.reranker_feedback import get_feedback_collector

        collector = await get_feedback_collector()
        feedback_id = await collector.record_dwell_time(
            session_id=request.session_id,
            document_id=request.document_id,
            dwell_time_ms=request.dwell_time_ms,
        )

        if feedback_id:
            return FeedbackResponse(
                feedback_id=feedback_id,
                recorded=True,
                message="Dwell time recorded",
            )
        else:
            return FeedbackResponse(
                feedback_id="",
                recorded=False,
                message="No matching session found",
            )

    except Exception as e:
        logger.error("Failed to record dwell", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/feedback/explicit", response_model=FeedbackResponse)
async def record_explicit_feedback(request: RecordExplicitFeedbackRequest, user: dict = Depends(get_current_user)) -> FeedbackResponse:
    """
    Record explicit user feedback (thumbs up/down, rating).

    Strongest signal for reranking improvement.
    """
    logger.info(
        "Recording explicit feedback",
        query_id=request.query_id,
        is_positive=request.is_positive,
    )

    try:
        from backend.services.reranker_feedback import get_feedback_collector

        collector = await get_feedback_collector()
        feedback_id = await collector.record_explicit_feedback(
            query_id=request.query_id,
            document_id=request.document_id,
            is_positive=request.is_positive,
            rating=request.rating,
            comment=request.comment,
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            recorded=True,
            message="Feedback recorded successfully",
        )

    except Exception as e:
        logger.error("Failed to record feedback", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/feedback/metrics", response_model=FeedbackMetricsResponse)
async def get_feedback_metrics(user: dict = Depends(get_current_user)) -> FeedbackMetricsResponse:
    """Get aggregated feedback metrics."""
    try:
        from backend.services.reranker_feedback import get_feedback_collector

        collector = await get_feedback_collector()
        metrics = await collector.get_metrics()

        return FeedbackMetricsResponse(
            total_queries=metrics.total_queries,
            total_clicks=metrics.total_clicks,
            total_positive=metrics.total_explicit_positive,
            total_negative=metrics.total_explicit_negative,
            avg_click_position=metrics.avg_click_position,
            mrr=metrics.mrr,
            satisfaction_rate=metrics.satisfaction_rate,
        )

    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# =============================================================================
# TruLens Experiment Endpoints
# =============================================================================

@router.post("/create", response_model=Dict[str, str])
async def create_experiment(request: CreateExperimentRequest, user: dict = Depends(get_current_user)) -> Dict[str, str]:
    """
    Create a new experiment for tracking RAG quality.

    Experiments track metrics like relevance, groundedness, latency.
    """
    logger.info("Creating experiment", name=request.name)

    try:
        from backend.services.trulens_integration import get_trulens_manager

        manager = await get_trulens_manager()
        experiment_id = await manager.create_experiment(
            name=request.name,
            description=request.description,
            retriever_config=request.retriever_config,
            reranker_config=request.reranker_config,
            llm_config=request.llm_config,
        )

        return {
            "experiment_id": experiment_id,
            "name": request.name,
            "status": "created",
        }

    except Exception as e:
        logger.error("Failed to create experiment", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/log", response_model=Dict[str, str])
async def log_evaluation(request: LogEvaluationRequest, user: dict = Depends(get_current_user)) -> Dict[str, str]:
    """
    Log an evaluation for an experiment.

    Records query, answer, contexts, and runs quality evaluation.
    """
    try:
        from backend.services.trulens_integration import get_trulens_manager

        manager = await get_trulens_manager()
        record_id = await manager.log_evaluation(
            experiment_id=request.experiment_id,
            query=request.query,
            answer=request.answer,
            contexts=request.contexts,
            latency_ms=request.latency_ms,
            cost_usd=request.cost_usd,
            run_feedback=request.run_feedback,
        )

        return {
            "record_id": record_id,
            "experiment_id": request.experiment_id,
            "status": "logged",
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    except Exception as e:
        logger.error("Failed to log evaluation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/{experiment_id}/summary", response_model=ExperimentSummaryResponse)
async def get_experiment_summary(experiment_id: str, user: dict = Depends(get_current_user)) -> ExperimentSummaryResponse:
    """Get summary statistics for an experiment."""
    try:
        from backend.services.trulens_integration import get_trulens_manager

        manager = await get_trulens_manager()
        summary = await manager.get_experiment_summary(experiment_id)

        return ExperimentSummaryResponse(
            experiment_id=summary.experiment_id,
            name=summary.name,
            total_records=summary.total_records,
            avg_relevance=summary.avg_relevance,
            avg_groundedness=summary.avg_groundedness,
            avg_latency_ms=summary.avg_latency_ms,
            pass_rate=summary.pass_rate,
            status=summary.status.value,
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    except Exception as e:
        logger.error("Failed to get summary", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/compare", response_model=Dict[str, Any])
async def compare_experiments(experiment_ids: List[str], user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Compare multiple experiments.

    Returns metrics comparison and recommendations.
    """
    if len(experiment_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 experiments to compare",
        )

    try:
        from backend.services.trulens_integration import get_trulens_manager

        manager = await get_trulens_manager()
        comparison = await manager.compare_experiments(experiment_ids)

        return comparison

    except Exception as e:
        logger.error("Failed to compare experiments", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/list", response_model=List[ExperimentSummaryResponse])
async def list_experiments(user: dict = Depends(get_current_user)) -> List[ExperimentSummaryResponse]:
    """List all experiments with summaries."""
    try:
        from backend.services.trulens_integration import get_trulens_manager

        manager = await get_trulens_manager()
        summaries = await manager.get_all_experiments()

        return [
            ExperimentSummaryResponse(
                experiment_id=s.experiment_id,
                name=s.name,
                total_records=s.total_records,
                avg_relevance=s.avg_relevance,
                avg_groundedness=s.avg_groundedness,
                avg_latency_ms=s.avg_latency_ms,
                pass_rate=s.pass_rate,
                status=s.status.value,
            )
            for s in summaries
        ]

    except Exception as e:
        logger.error("Failed to list experiments", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/health")
async def experiments_health() -> Dict[str, Any]:
    """Check experiments service health."""
    health = {"feedback": "unknown", "trulens": "unknown"}

    try:
        from backend.services.reranker_feedback import get_feedback_collector
        collector = await get_feedback_collector()
        health["feedback"] = "healthy" if collector._initialized else "not_initialized"
    except Exception as e:
        health["feedback"] = f"error: {str(e)}"

    try:
        from backend.services.trulens_integration import get_trulens_manager
        manager = await get_trulens_manager()
        health["trulens"] = "healthy" if manager._initialized else "not_initialized"
    except Exception as e:
        health["trulens"] = f"error: {str(e)}"

    return {
        "status": "healthy" if all("healthy" in v for v in health.values()) else "degraded",
        "services": health,
    }
