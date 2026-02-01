"""
AIDocumentIndexer - Knowledge Analytics API Routes
===================================================

Lynkt-inspired endpoints for knowledge usage tracking and analysis.

Endpoints:
- GET /health - Knowledge base health metrics
- GET /usage - Usage report with patterns and signals
- GET /signals - Detected context signals
- GET /patterns - Detected usage patterns
- GET /documents/{id}/usage - Single document usage stats
- POST /track - Track a retrieval event
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.knowledge_analytics import (
    get_knowledge_analytics_service,
    KnowledgeAnalyticsService,
)
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class TrackRetrievalRequest(BaseModel):
    """Request to track a retrieval event."""
    query: str = Field(..., description="The query that triggered retrieval")
    documents: List[Dict[str, Any]] = Field(..., description="Retrieved documents")
    session_id: Optional[str] = Field(None, description="Session ID")


class ContextSignalResponse(BaseModel):
    """A detected context signal."""
    signal_type: str
    severity: str
    description: str
    affected_documents: List[str]
    suggested_action: str
    detected_at: str


class UsagePatternResponse(BaseModel):
    """A detected usage pattern."""
    pattern_type: str
    description: str
    frequency: int
    entities_involved: List[str]


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/health")
async def get_knowledge_health(
    user: AuthenticatedUser,
):
    """
    Get knowledge base health metrics.

    Returns overall health score and key metrics like:
    - Document activity rate
    - Average relevance scores
    - Query success rate
    - Signal and pattern counts
    """
    service = get_knowledge_analytics_service()
    health = await service.get_knowledge_health()
    return health


@router.get("/usage")
async def get_usage_report(
    user: AuthenticatedUser,
    period_days: int = 7,
):
    """
    Get a comprehensive usage report.

    Includes:
    - Usage summary (queries, retrievals, coverage)
    - Top documents
    - Detected signals
    - Detected patterns
    - Health metrics
    """
    service = get_knowledge_analytics_service()
    report = await service.get_usage_report()
    return report


@router.get("/reuse")
async def get_reuse_summary(
    user: AuthenticatedUser,
    period_days: int = 7,
    top_n: int = 10,
):
    """
    Get knowledge reuse summary.

    Shows how documents are being reused across queries:
    - Reuse rate
    - Top reused documents
    - Underutilized documents
    - Knowledge coverage
    """
    service = get_knowledge_analytics_service()
    summary = await service.get_reuse_summary(
        period_days=period_days,
        top_n=top_n,
    )

    return {
        "period": {
            "start": summary.period_start.isoformat(),
            "end": summary.period_end.isoformat(),
        },
        "total_queries": summary.total_queries,
        "unique_documents_used": summary.unique_documents_used,
        "total_retrievals": summary.total_retrievals,
        "avg_reuse_rate": summary.avg_reuse_rate,
        "knowledge_coverage": summary.knowledge_coverage,
        "top_documents": [
            {
                "id": d.document_id,
                "name": d.document_name,
                "retrievals": d.total_retrievals,
                "unique_queries": d.unique_queries,
                "avg_relevance": d.avg_relevance_score,
                "last_retrieved": d.last_retrieved.isoformat() if d.last_retrieved else None,
            }
            for d in summary.top_documents
        ],
        "underutilized_documents": summary.underutilized_documents,
    }


@router.get("/signals", response_model=List[ContextSignalResponse])
async def get_context_signals(
    user: AuthenticatedUser,
):
    """
    Get detected context signals.

    Signals include:
    - Gaps (missing knowledge areas)
    - Anomalies (low relevance patterns)
    - Stale documents (not used recently)
    """
    service = get_knowledge_analytics_service()
    signals = await service.detect_signals()

    return [
        ContextSignalResponse(
            signal_type=s.signal_type,
            severity=s.severity,
            description=s.description,
            affected_documents=s.affected_documents,
            suggested_action=s.suggested_action,
            detected_at=s.detected_at.isoformat(),
        )
        for s in signals
    ]


@router.get("/patterns", response_model=List[UsagePatternResponse])
async def get_usage_patterns(
    user: AuthenticatedUser,
):
    """
    Get detected usage patterns.

    Patterns include:
    - Temporal (peak usage times)
    - Topical (co-occurring documents)
    - User (high activity users)
    """
    service = get_knowledge_analytics_service()
    patterns = await service.detect_patterns()

    return [
        UsagePatternResponse(
            pattern_type=p.pattern_type,
            description=p.description,
            frequency=p.frequency,
            entities_involved=p.entities_involved,
        )
        for p in patterns
    ]


@router.get("/documents/{document_id}/usage")
async def get_document_usage(
    document_id: str,
    user: AuthenticatedUser,
):
    """
    Get usage statistics for a specific document.
    """
    service = get_knowledge_analytics_service()
    usage = service.get_document_usage(document_id)

    if not usage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No usage data for document: {document_id}",
        )

    return {
        "document_id": usage.document_id,
        "document_name": usage.document_name,
        "total_retrievals": usage.total_retrievals,
        "unique_queries": usage.unique_queries,
        "avg_relevance_score": usage.avg_relevance_score,
        "last_retrieved": usage.last_retrieved.isoformat() if usage.last_retrieved else None,
        "topics_matched": usage.topics_matched,
    }


@router.post("/track")
async def track_retrieval(
    request: TrackRetrievalRequest,
    user: AuthenticatedUser,
):
    """
    Track a document retrieval event.

    This should be called after each RAG query to track:
    - Which documents were retrieved
    - Relevance scores
    - User and session context
    """
    service = get_knowledge_analytics_service()

    await service.track_retrieval(
        query=request.query,
        retrieved_documents=request.documents,
        user_id=user.user_id,
        session_id=request.session_id,
    )

    return {"status": "tracked", "document_count": len(request.documents)}
