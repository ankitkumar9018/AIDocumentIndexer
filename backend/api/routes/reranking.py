"""
AIDocumentIndexer - Reranking API Routes
==========================================

API endpoints for tiered reranking pipeline.
"""

from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.deps import get_current_user

from backend.services.tiered_reranking import (
    get_tiered_reranker,
    RerankCandidate,
    RerankerStage,
    QueryComplexity,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/reranking", tags=["reranking"])


# =============================================================================
# Request/Response Models
# =============================================================================

class RerankerStageEnum(str, Enum):
    """Reranking stages."""
    COLBERT = "colbert"
    CROSS_ENCODER = "cross"
    LLM = "llm"
    SEMANTIC = "semantic"


class CandidateInput(BaseModel):
    """Input candidate for reranking."""
    id: str = Field(..., description="Unique candidate ID")
    content: str = Field(..., description="Text content to rank")
    score: float = Field(0.0, description="Initial score from retrieval")
    document_id: Optional[str] = Field(None, description="Source document ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RerankRequest(BaseModel):
    """Request to rerank candidates."""
    query: str = Field(..., description="Search query")
    candidates: List[CandidateInput] = Field(..., description="Candidates to rerank")
    stages: Optional[List[RerankerStageEnum]] = Field(
        None, description="Stages to run (uses adaptive if not specified)"
    )
    final_top_k: Optional[int] = Field(None, ge=1, le=100, description="Final result count")


class RerankResultItem(BaseModel):
    """Single reranked result."""
    id: str
    content: str
    score: float
    stage: str
    stage_scores: Dict[str, float]
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RerankMetricsResponse(BaseModel):
    """Reranking metrics."""
    total_time_ms: float
    stage_times_ms: Dict[str, float]
    input_candidates: int
    output_candidates: int
    stages_executed: List[str]
    complexity: Optional[str] = None


class RerankResponse(BaseModel):
    """Response from reranking."""
    results: List[RerankResultItem]
    metrics: RerankMetricsResponse


class ConfigResponse(BaseModel):
    """Reranker configuration."""
    default_stages: List[str]
    colbert_top_k: int
    cross_encoder_top_k: int
    llm_top_k: int
    cross_encoder_model: str
    colbert_model: str
    llm_model: str
    enable_adaptive: bool
    use_llm_for_complex: bool
    min_score_threshold: float


class AnalyzeQueryResponse(BaseModel):
    """Query analysis result."""
    query: str
    complexity: str
    recommended_stages: List[str]
    explanation: str


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/rerank", response_model=RerankResponse)
async def rerank_candidates(request: RerankRequest, user: dict = Depends(get_current_user)) -> RerankResponse:
    """
    Rerank candidates using the tiered reranking pipeline.

    Stages:
    - colbert: Fast late interaction (filters to top 20)
    - cross: Cross-encoder (accurate, filters to top 10)
    - llm: LLM-based (highest quality, filters to top 5)

    If stages not specified, uses adaptive selection based on query complexity.
    """
    logger.info(
        "Reranking request",
        query_preview=request.query[:50],
        candidate_count=len(request.candidates),
        stages=request.stages,
    )

    if not request.candidates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No candidates provided")

    try:
        reranker = await get_tiered_reranker()

        # Convert input to RerankCandidate
        candidates = [
            RerankCandidate(
                id=c.id,
                content=c.content,
                score=c.score,
                metadata=c.metadata or {},
                document_id=c.document_id,
            )
            for c in request.candidates
        ]

        # Convert stages
        stages = None
        if request.stages:
            stages = [RerankerStage(s.value) for s in request.stages]

        # Run reranking
        results, metrics = await reranker.rerank(
            query=request.query,
            candidates=candidates,
            stages=stages,
            final_top_k=request.final_top_k,
        )

        # Convert results
        result_items = [
            RerankResultItem(
                id=r.candidate.id,
                content=r.candidate.content,
                score=r.score,
                stage=r.stage.value,
                stage_scores=r.stage_scores,
                document_id=r.candidate.document_id,
                metadata=r.candidate.metadata,
            )
            for r in results
        ]

        metrics_response = RerankMetricsResponse(
            total_time_ms=metrics.total_time_ms,
            stage_times_ms=metrics.stage_times_ms,
            input_candidates=metrics.input_candidates,
            output_candidates=metrics.output_candidates,
            stages_executed=metrics.stages_executed,
            complexity=metrics.complexity.value if metrics.complexity else None,
        )

        logger.info(
            "Reranking complete",
            output_count=len(result_items),
            total_time_ms=round(metrics.total_time_ms, 2),
        )

        return RerankResponse(results=result_items, metrics=metrics_response)

    except Exception as e:
        logger.error("Reranking failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Reranking failed")


@router.get("/config", response_model=ConfigResponse)
async def get_reranker_config(user: dict = Depends(get_current_user)) -> ConfigResponse:
    """Get current reranker configuration."""
    try:
        reranker = await get_tiered_reranker()
        config = reranker.config

        return ConfigResponse(
            default_stages=[s.value for s in config.default_stages],
            colbert_top_k=config.colbert_top_k,
            cross_encoder_top_k=config.cross_encoder_top_k,
            llm_top_k=config.llm_top_k,
            cross_encoder_model=config.cross_encoder_model,
            colbert_model=config.colbert_model,
            llm_model=config.llm_model,
            enable_adaptive=config.enable_adaptive,
            use_llm_for_complex=config.use_llm_for_complex,
            min_score_threshold=config.min_score_threshold,
        )
    except Exception as e:
        logger.error("Failed to get config", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get reranker config")


@router.post("/analyze-query", response_model=AnalyzeQueryResponse)
async def analyze_query(query: str, user: dict = Depends(get_current_user)) -> AnalyzeQueryResponse:
    """
    Analyze query complexity and get recommended stages.

    Useful for understanding how the adaptive pipeline will handle a query.
    """
    try:
        reranker = await get_tiered_reranker()
        complexity = reranker._complexity_analyzer.analyze(query)
        stages = reranker._get_adaptive_stages(query)

        explanations = {
            QueryComplexity.SIMPLE: "Simple factual query - ColBERT only for speed",
            QueryComplexity.MODERATE: "Moderate complexity - ColBERT + Cross-encoder for balance",
            QueryComplexity.COMPLEX: "Complex multi-hop query - Full pipeline with LLM verification",
            QueryComplexity.ANALYTICAL: "Analytical query - Full pipeline for deep understanding",
        }

        return AnalyzeQueryResponse(
            query=query,
            complexity=complexity.value,
            recommended_stages=[s.value for s in stages],
            explanation=explanations.get(complexity, "Unknown complexity"),
        )
    except Exception as e:
        logger.error("Query analysis failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Query analysis failed")


@router.get("/stages")
async def list_stages(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """List available reranking stages with descriptions."""
    return {
        "stages": [
            {
                "stage": "colbert",
                "name": "ColBERT Late Interaction",
                "description": "Fast reranking using token-level interactions. 180x fewer FLOPs than cross-encoder.",
                "typical_latency_ms": "10-50",
                "accuracy": "Good",
                "default_top_k": 20,
            },
            {
                "stage": "cross",
                "name": "Cross-Encoder",
                "description": "High-accuracy reranking using full attention between query and document.",
                "typical_latency_ms": "50-200",
                "accuracy": "Very Good",
                "default_top_k": 10,
            },
            {
                "stage": "llm",
                "name": "LLM Reranker",
                "description": "Highest quality reranking using LLM judgement. Best for complex queries.",
                "typical_latency_ms": "500-2000",
                "accuracy": "Excellent",
                "default_top_k": 5,
            },
        ],
        "complexity_levels": [
            {
                "level": "simple",
                "description": "Single fact queries (what is, who is)",
                "stages": ["colbert"],
            },
            {
                "level": "moderate",
                "description": "Multi-fact queries",
                "stages": ["colbert", "cross"],
            },
            {
                "level": "complex",
                "description": "Multi-hop reasoning queries",
                "stages": ["colbert", "cross", "llm"],
            },
            {
                "level": "analytical",
                "description": "Deep analysis queries",
                "stages": ["colbert", "cross", "llm"],
            },
        ],
    }


@router.get("/health")
async def reranker_health() -> Dict[str, Any]:
    """Check reranker service health."""
    try:
        reranker = await get_tiered_reranker()

        return {
            "status": "healthy",
            "initialized": reranker._initialized,
            "stages_available": list(reranker._stages.keys()),
            "adaptive_enabled": reranker.config.enable_adaptive,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
