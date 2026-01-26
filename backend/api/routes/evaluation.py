"""
AIDocumentIndexer - RAG Evaluation API Endpoints (Phase 66)
============================================================

Provides endpoints for evaluating RAG quality using RAGAS-inspired metrics:
- Single query evaluation
- Benchmark suite execution
- Quality metrics retrieval
- Evaluation history and trends
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
import structlog

from backend.db.database import AsyncSession, get_async_session
from backend.services.rag_evaluation import (
    RAGEvaluator,
    RAGBenchmark,
    EvaluationTracker,
    EvaluationResult,
    RAGASMetrics,
    BenchmarkResult,
    evaluate_rag_response,
    get_quality_level,
    format_metrics_report,
)
from backend.services.rag import get_rag_service
from backend.services.llm import LLMFactory
from backend.services.embeddings import get_embedding_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Global tracker for evaluation history
_evaluation_tracker = EvaluationTracker(max_history=1000)


# =============================================================================
# Request/Response Models
# =============================================================================

class EvaluateRequest(BaseModel):
    """Request for single evaluation."""
    query: str = Field(..., description="The user query", max_length=10000)
    answer: str = Field(..., description="The RAG-generated answer", max_length=50000)
    contexts: List[str] = Field(
        ...,
        description="Retrieved context chunks",
        min_length=1,
        max_length=50,  # Limit to prevent DoS (Phase 69)
    )
    ground_truth: Optional[str] = Field(None, description="Expected answer for recall calculation")


class EvaluateResponseRequest(BaseModel):
    """Request to evaluate a previous RAG response."""
    query: str = Field(..., description="The user query")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    ground_truth: Optional[str] = Field(None, description="Expected answer")


class BenchmarkRequest(BaseModel):
    """Request to run a benchmark."""
    test_cases: List[Dict[str, Any]] = Field(
        ...,
        description="List of test cases with query, optional ground_truth, and optional contexts",
        min_length=1,
        max_length=100,  # Limit to prevent DoS (Phase 69)
    )
    collection_filter: Optional[str] = Field(None, description="Filter to specific collection")
    use_rag_service: bool = Field(True, description="Generate answers using RAG service")


class MetricsResponse(BaseModel):
    """Response containing evaluation metrics."""
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    context_recall: float
    overall_score: float
    quality_level: str
    evaluation_time_ms: Optional[float] = None


class EvaluationResponse(BaseModel):
    """Full evaluation response."""
    query: str
    answer: str
    contexts: List[str]
    metrics: MetricsResponse
    ground_truth: Optional[str] = None
    issues: List[str] = []
    suggestions: List[str] = []
    timestamp: datetime


class BenchmarkResponse(BaseModel):
    """Benchmark result response."""
    test_count: int
    passing_rate: float
    duration_ms: float
    aggregate_metrics: Dict[str, float]
    quality_level: str
    timestamp: datetime


class TrendResponse(BaseModel):
    """Evaluation trend response."""
    metric: str
    values: List[Optional[float]]
    period_hours: int
    periods: int


# =============================================================================
# Helper Functions
# =============================================================================

def _metrics_to_response(metrics: RAGASMetrics) -> MetricsResponse:
    """Convert RAGASMetrics to response model."""
    return MetricsResponse(
        context_relevance=metrics.context_relevance,
        faithfulness=metrics.faithfulness,
        answer_relevance=metrics.answer_relevance,
        context_recall=metrics.context_recall,
        overall_score=metrics.overall_score,
        quality_level=get_quality_level(metrics.overall_score),
        evaluation_time_ms=metrics.evaluation_time_ms,
    )


def _result_to_response(result: EvaluationResult) -> EvaluationResponse:
    """Convert EvaluationResult to response model."""
    return EvaluationResponse(
        query=result.query,
        answer=result.answer,
        contexts=result.contexts,
        metrics=_metrics_to_response(result.metrics),
        ground_truth=result.ground_truth,
        issues=result.issues,
        suggestions=result.suggestions,
        timestamp=result.timestamp,
    )


async def _get_evaluator() -> RAGEvaluator:
    """Get configured evaluator instance."""
    try:
        # Get LLM for evaluation
        llm = LLMFactory.get_chat_model(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("Failed to initialize LLM for RAG evaluation", error=str(e))
        llm = None

    try:
        embedding_service = await get_embedding_service()
    except Exception as e:
        logger.warning("Failed to initialize embedding service for RAG evaluation", error=str(e))
        embedding_service = None

    return RAGEvaluator(llm=llm, embedding_service=embedding_service)


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    request: EvaluateRequest,
) -> EvaluationResponse:
    """
    Evaluate a RAG response using RAGAS-inspired metrics.

    Returns:
    - Context relevance: How relevant is the retrieved context?
    - Faithfulness: Is the answer grounded in the context?
    - Answer relevance: Does the answer address the question?
    - Context recall: Did we retrieve all needed information?
    - Overall score: Weighted average of all metrics
    """
    logger.info(
        "Evaluating RAG response",
        query_preview=request.query[:50],
        num_contexts=len(request.contexts),
    )

    try:
        evaluator = await _get_evaluator()
        result = await evaluator.evaluate(
            query=request.query,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth,
        )

        # Track evaluation
        _evaluation_tracker.record(result)

        response = _result_to_response(result)

        logger.info(
            "Evaluation complete",
            overall_score=result.metrics.overall_score,
            quality_level=get_quality_level(result.metrics.overall_score),
        )

        return response

    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Evaluation failed: {str(e)}")


@router.post("/evaluate-query", response_model=EvaluationResponse)
async def evaluate_query(
    request: EvaluateResponseRequest,
    session: AsyncSession = Depends(get_async_session),
) -> EvaluationResponse:
    """
    Evaluate a query by generating an answer and then evaluating it.

    This endpoint runs the full RAG pipeline and then evaluates the result.
    """
    logger.info(
        "Evaluating query with RAG pipeline",
        query_preview=request.query[:50],
    )

    try:
        # Get RAG service and generate response
        rag_service = get_rag_service()
        rag_response = await rag_service.query(
            question=request.query,
            session_id=request.session_id,
        )

        # Extract contexts from sources
        contexts = [
            s.full_content or s.snippet
            for s in rag_response.sources
        ]

        # Evaluate
        evaluator = await _get_evaluator()
        result = await evaluator.evaluate(
            query=request.query,
            answer=rag_response.content,
            contexts=contexts,
            ground_truth=request.ground_truth,
        )

        # Track evaluation
        _evaluation_tracker.record(result)

        return _result_to_response(result)

    except Exception as e:
        logger.error("Query evaluation failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query evaluation failed: {str(e)}")


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    request: BenchmarkRequest,
    session: AsyncSession = Depends(get_async_session),
) -> BenchmarkResponse:
    """
    Run a benchmark suite on test cases.

    Each test case should have:
    - query: The question to ask
    - ground_truth (optional): Expected answer
    - contexts (optional): Pre-defined contexts if not using RAG service
    """
    logger.info(
        "Running benchmark",
        test_count=len(request.test_cases),
        use_rag=request.use_rag_service,
    )

    try:
        evaluator = await _get_evaluator()

        rag_service = None
        if request.use_rag_service:
            rag_service = get_rag_service()

        benchmark = RAGBenchmark(
            evaluator=evaluator,
            rag_service=rag_service,
        )

        result = await benchmark.run_benchmark(
            test_cases=request.test_cases,
            session=session if request.use_rag_service else None,
        )

        quality_level = get_quality_level(
            result.aggregate_metrics.get("overall_score", 0)
        )

        logger.info(
            "Benchmark complete",
            test_count=result.test_count,
            passing_rate=result.passing_rate,
            overall_score=result.aggregate_metrics.get("overall_score", 0),
        )

        return BenchmarkResponse(
            test_count=result.test_count,
            passing_rate=result.passing_rate,
            duration_ms=result.duration_ms,
            aggregate_metrics=result.aggregate_metrics,
            quality_level=quality_level,
            timestamp=result.timestamp,
        )

    except Exception as e:
        logger.error("Benchmark failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Benchmark failed: {str(e)}")


@router.get("/metrics/recent")
async def get_recent_metrics(
    hours: int = 24,
) -> Dict[str, Any]:
    """
    Get aggregate metrics for recent evaluations.

    Returns average scores across all evaluations in the specified time window.
    """
    metrics = _evaluation_tracker.get_recent_metrics(hours=hours)

    if not metrics:
        return {
            "message": "No evaluations recorded in this time period",
            "hours": hours,
        }

    return {
        "hours": hours,
        "count": metrics.get("count", 0),
        "avg_overall": metrics.get("avg_overall", 0),
        "avg_faithfulness": metrics.get("avg_faithfulness", 0),
        "avg_relevance": metrics.get("avg_relevance", 0),
        "quality_level": get_quality_level(metrics.get("avg_overall", 0)),
    }


@router.get("/metrics/trend", response_model=TrendResponse)
async def get_metrics_trend(
    metric: str = "overall_score",
    periods: int = 7,
    period_hours: int = 24,
) -> TrendResponse:
    """
    Get trend of a metric over time.

    Returns metric values for each period, most recent last.
    """
    valid_metrics = [
        "overall_score",
        "context_relevance",
        "faithfulness",
        "answer_relevance",
        "context_recall",
    ]

    if metric not in valid_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metric. Choose from: {valid_metrics}",
        )

    trend = _evaluation_tracker.get_trend(
        metric=metric,
        periods=periods,
        period_hours=period_hours,
    )

    return TrendResponse(
        metric=metric,
        values=trend,
        period_hours=period_hours,
        periods=periods,
    )


@router.get("/health")
async def evaluation_health() -> Dict[str, Any]:
    """Check evaluation service health."""
    try:
        evaluator = await _get_evaluator()
        has_llm = evaluator.llm is not None
        has_embeddings = evaluator.embedding_service is not None

        recent = _evaluation_tracker.get_recent_metrics(hours=1)

        return {
            "status": "healthy",
            "llm_available": has_llm,
            "embeddings_available": has_embeddings,
            "recent_evaluations_1h": recent.get("count", 0),
            "features": {
                "context_relevance": True,
                "faithfulness": has_llm,
                "answer_relevance": has_llm,
                "context_recall": has_embeddings,
            },
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }
