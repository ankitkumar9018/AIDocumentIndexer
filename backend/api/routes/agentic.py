"""
AIDocumentIndexer - Agentic RAG API Routes
==========================================

API endpoints for complex multi-step query processing using ReAct agents.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.middleware.auth import get_current_user
from backend.db.database import get_async_session
from backend.db.models import User

router = APIRouter(tags=["Agentic RAG"])


# =============================================================================
# Request/Response Models
# =============================================================================

class AgenticQueryRequest(BaseModel):
    """Request for agentic RAG query."""
    query: str = Field(..., min_length=1, max_length=5000, description="The complex question to process")
    collection_filter: Optional[str] = Field(None, description="Filter to specific collection")
    use_multi_agent: bool = Field(False, description="Use specialized multi-agent pipeline")
    stream: bool = Field(False, description="Stream progress updates")


class SubQueryResponse(BaseModel):
    """A decomposed sub-question."""
    query: str
    purpose: str
    completed: bool
    result: Optional[str] = None


class ReActStepResponse(BaseModel):
    """A single step in the ReAct loop."""
    step_number: int
    thought: str
    action: str
    action_input: str
    observation: Optional[str] = None


class AgenticQueryResponse(BaseModel):
    """Response from agentic RAG processing."""
    query: str
    final_answer: str
    sub_queries: List[SubQueryResponse]
    react_steps: List[ReActStepResponse]
    sources_used: List[Dict[str, Any]]
    confidence: float
    processing_time_ms: float
    iterations: int


class QueryComplexityResponse(BaseModel):
    """Response for query complexity analysis."""
    query: str
    is_complex: bool
    sub_queries: List[str]
    synthesis_approach: str
    recommendation: str


# =============================================================================
# Dependencies
# =============================================================================

async def get_agentic_service(db: AsyncSession = Depends(get_async_session)):
    """Get the agentic RAG service."""
    from backend.services.rag import RAGService
    from backend.services.agentic_rag import get_agentic_rag_service

    rag_service = RAGService(db)
    return get_agentic_rag_service(rag_service)


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/query",
    response_model=AgenticQueryResponse,
    summary="Process complex query",
    description="Process a complex query using agentic RAG with ReAct loop.",
)
async def process_agentic_query(
    request: AgenticQueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Process a complex query using agentic RAG.

    The agent will:
    1. Analyze query complexity
    2. Decompose into sub-questions if complex
    3. Use ReAct loop to iteratively find answers
    4. Synthesize final answer from gathered information
    """
    from backend.services.rag import RAGService
    from backend.services.agentic_rag import get_agentic_rag_service

    # Create services
    rag_service = RAGService(db)
    agentic_service = get_agentic_rag_service(rag_service)

    # Process query
    result = await agentic_service.process_query(
        query=request.query,
        collection_filter=request.collection_filter,
        access_tier=user.access_tier if hasattr(user, 'access_tier') else 100,
        user_id=str(user.id),
        use_multi_agent=request.use_multi_agent,
    )

    return AgenticQueryResponse(
        query=result.query,
        final_answer=result.final_answer,
        sub_queries=[
            SubQueryResponse(
                query=sq.query,
                purpose=sq.purpose,
                completed=sq.completed,
                result=sq.result,
            )
            for sq in result.sub_queries
        ],
        react_steps=[
            ReActStepResponse(
                step_number=step.step_number,
                thought=step.thought,
                action=step.action.value,
                action_input=step.action_input,
                observation=step.observation,
            )
            for step in result.react_steps
        ],
        sources_used=result.sources_used,
        confidence=result.confidence,
        processing_time_ms=result.processing_time_ms,
        iterations=result.iterations,
    )


@router.post(
    "/query/stream",
    summary="Stream complex query processing",
    description="Process a complex query with streaming progress updates.",
)
async def stream_agentic_query(
    request: AgenticQueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Process a query with streaming updates.

    Returns Server-Sent Events (SSE) with progress updates.
    """
    import json
    from backend.services.rag import RAGService
    from backend.services.agentic_rag import get_agentic_rag_service

    async def generate():
        rag_service = RAGService(db)
        agentic_service = get_agentic_rag_service(rag_service)

        async for update in agentic_service.process_query_stream(
            query=request.query,
            collection_filter=request.collection_filter,
            access_tier=user.access_tier if hasattr(user, 'access_tier') else 100,
            user_id=str(user.id),
        ):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.post(
    "/analyze",
    response_model=QueryComplexityResponse,
    summary="Analyze query complexity",
    description="Analyze whether a query requires agentic processing.",
)
async def analyze_query_complexity(
    request: AgenticQueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Analyze query complexity without fully processing.

    Returns information about:
    - Whether the query is complex
    - Suggested sub-queries
    - Recommended processing approach
    """
    from backend.services.rag import RAGService
    from backend.services.agentic_rag import get_agentic_rag_service

    rag_service = RAGService(db)
    agentic_service = get_agentic_rag_service(rag_service)

    is_complex, sub_queries, synthesis = await agentic_service.decompose_query(
        request.query
    )

    if is_complex:
        recommendation = "Use agentic RAG for best results - query requires multiple retrieval steps."
    else:
        recommendation = "Standard RAG should work well for this query."

    return QueryComplexityResponse(
        query=request.query,
        is_complex=is_complex,
        sub_queries=[sq.query for sq in sub_queries],
        synthesis_approach=synthesis,
        recommendation=recommendation,
    )
