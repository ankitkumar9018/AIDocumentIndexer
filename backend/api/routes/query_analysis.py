"""
AIDocumentIndexer - Query Analysis API Routes
==============================================

Endpoints for intelligent query analysis that auto-detects
what features and tools should be used.
"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.query_analyzer import (
    get_query_analyzer,
    QueryAnalyzerService,
    QueryComplexity,
)
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryAnalysisRequest(BaseModel):
    """Request for query analysis."""
    query: str = Field(..., description="Query to analyze")
    use_llm: bool = Field(True, description="Use LLM for deeper analysis")
    context: Optional[str] = Field(None, description="Optional conversation context")


class QueryAnalysisResponse(BaseModel):
    """Response from query analysis."""
    original_query: str
    complexity: str
    query_types: List[str]
    recommended_intelligence_level: str
    recommended_tools: List[str]
    enable_extended_thinking: bool
    enable_ensemble_voting: bool
    enable_parallel_knowledge: bool
    confidence: float
    reasoning: str


class QuickAnalysisRequest(BaseModel):
    """Request for quick (pattern-based) analysis."""
    query: str = Field(..., description="Query to analyze")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(
    request: QueryAnalysisRequest,
    user: AuthenticatedUser,
):
    """
    Analyze a query to determine optimal intelligence settings.

    This endpoint:
    1. Analyzes query complexity
    2. Detects query types (factual, analytical, mathematical, etc.)
    3. Recommends tools (calculator, fact-checker, etc.)
    4. Suggests intelligence level (basic/standard/enhanced/maximum)
    5. Recommends additional features (extended thinking, ensemble, etc.)

    Use this before sending a chat message to auto-configure settings.
    """
    service = get_query_analyzer()

    analysis = await service.analyze(
        query=request.query,
        use_llm=request.use_llm,
        context=request.context or "",
    )

    return QueryAnalysisResponse(
        original_query=analysis.original_query,
        complexity=analysis.complexity.value,
        query_types=[t.value for t in analysis.query_types],
        recommended_intelligence_level=analysis.recommended_intelligence_level,
        recommended_tools=analysis.recommended_tools,
        enable_extended_thinking=analysis.enable_extended_thinking,
        enable_ensemble_voting=analysis.enable_ensemble_voting,
        enable_parallel_knowledge=analysis.enable_parallel_knowledge,
        confidence=analysis.confidence,
        reasoning=analysis.reasoning,
    )


@router.post("/quick-analyze", response_model=QueryAnalysisResponse)
async def quick_analyze_query(
    request: QuickAnalysisRequest,
    user: AuthenticatedUser,
):
    """
    Quick pattern-based query analysis (no LLM call).

    Faster but less accurate than full analysis.
    Uses regex patterns to detect query characteristics.
    """
    service = get_query_analyzer()

    analysis = service.quick_analyze(request.query)

    return QueryAnalysisResponse(
        original_query=analysis.original_query,
        complexity=analysis.complexity.value,
        query_types=[t.value for t in analysis.query_types],
        recommended_intelligence_level=analysis.recommended_intelligence_level,
        recommended_tools=analysis.recommended_tools,
        enable_extended_thinking=analysis.enable_extended_thinking,
        enable_ensemble_voting=analysis.enable_ensemble_voting,
        enable_parallel_knowledge=analysis.enable_parallel_knowledge,
        confidence=analysis.confidence,
        reasoning=analysis.reasoning,
    )


@router.get("/complexity-levels")
async def get_complexity_levels(user: AuthenticatedUser):
    """
    Get available complexity levels and their descriptions.
    """
    return {
        "levels": [
            {
                "id": "simple",
                "name": "Simple",
                "description": "Direct lookup, single fact retrieval",
                "intelligence_level": "basic",
            },
            {
                "id": "moderate",
                "name": "Moderate",
                "description": "Some reasoning or multiple facts required",
                "intelligence_level": "standard",
            },
            {
                "id": "complex",
                "name": "Complex",
                "description": "Multi-step reasoning, analysis needed",
                "intelligence_level": "enhanced",
            },
            {
                "id": "highly_complex",
                "name": "Highly Complex",
                "description": "Deep analysis, multiple perspectives needed",
                "intelligence_level": "maximum",
            },
        ],
    }


@router.get("/query-types")
async def get_query_types(user: AuthenticatedUser):
    """
    Get available query types and their characteristics.
    """
    return {
        "types": [
            {
                "id": "factual",
                "name": "Factual",
                "description": "Looking up facts",
                "tools": [],
            },
            {
                "id": "analytical",
                "name": "Analytical",
                "description": "Requires analysis and reasoning",
                "tools": [],
            },
            {
                "id": "mathematical",
                "name": "Mathematical",
                "description": "Involves calculations",
                "tools": ["calculator"],
            },
            {
                "id": "comparative",
                "name": "Comparative",
                "description": "Comparing things",
                "tools": [],
            },
            {
                "id": "temporal",
                "name": "Temporal",
                "description": "Time/date related",
                "tools": ["date_calculator"],
            },
            {
                "id": "procedural",
                "name": "Procedural",
                "description": "How-to questions",
                "tools": ["code_executor"],
            },
            {
                "id": "verification",
                "name": "Verification",
                "description": "Fact-checking claims",
                "tools": ["fact_checker"],
            },
        ],
    }
