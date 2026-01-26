"""
AIDocumentIndexer - Real-Time Indexer API Routes
=================================================

API endpoints for document freshness tracking and incremental indexing.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.middleware.auth import get_current_user, require_admin
from backend.db.database import get_async_session
from backend.db.models import User
from backend.services.realtime_indexer import (
    get_realtime_indexer_service,
    RealTimeIndexerService,
    FreshnessLevel,
)

router = APIRouter(tags=["Indexer"])


# =============================================================================
# Request/Response Models
# =============================================================================

class FreshnessInfoResponse(BaseModel):
    """Response for document freshness info."""
    document_id: str
    filename: str
    last_modified: datetime
    freshness_level: str
    days_since_update: int
    recommendation: Optional[str] = None


class FreshnessSummaryResponse(BaseModel):
    """Response for freshness summary."""
    total_documents: int
    fresh: int
    current: int
    aging: int
    stale: int
    freshness_thresholds: dict


class StaleDocumentsResponse(BaseModel):
    """Response for stale documents list."""
    documents: List[FreshnessInfoResponse]
    total_count: int


class IndexingStatsResponse(BaseModel):
    """Response for indexing statistics."""
    documents_checked: int
    documents_updated: int
    chunks_added: int
    chunks_modified: int
    chunks_deleted: int
    processing_time_ms: float
    errors: List[str]


class QueueTaskRequest(BaseModel):
    """Request to queue a reindex task."""
    document_id: str
    priority: int = Field(default=0, ge=0, le=10, description="Priority (0-10, higher = more urgent)")


class TouchDocumentRequest(BaseModel):
    """Request to touch a document timestamp."""
    document_id: str


# =============================================================================
# Dependency
# =============================================================================

async def get_indexer_service(
    db: AsyncSession = Depends(get_async_session),
) -> RealTimeIndexerService:
    """Get the realtime indexer service."""
    return await get_realtime_indexer_service(db)


# =============================================================================
# Freshness Endpoints
# =============================================================================

@router.get(
    "/freshness/summary",
    response_model=FreshnessSummaryResponse,
    summary="Get freshness summary",
    description="Get overall freshness statistics for the document archive.",
)
async def get_freshness_summary(
    user: User = Depends(get_current_user),
    indexer: RealTimeIndexerService = Depends(get_indexer_service),
):
    """Get overall freshness statistics."""
    summary = await indexer.get_freshness_summary()
    return FreshnessSummaryResponse(**summary)


@router.get(
    "/freshness/{document_id}",
    response_model=FreshnessInfoResponse,
    summary="Get document freshness",
    description="Get freshness information for a specific document.",
)
async def get_document_freshness(
    document_id: str,
    user: User = Depends(get_current_user),
    indexer: RealTimeIndexerService = Depends(get_indexer_service),
):
    """Get freshness info for a document."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )

    freshness = await indexer.get_document_freshness(doc_uuid)

    if not freshness:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return FreshnessInfoResponse(
        document_id=str(freshness.document_id),
        filename=freshness.filename,
        last_modified=freshness.last_modified,
        freshness_level=freshness.freshness_level.value,
        days_since_update=freshness.days_since_update,
        recommendation=freshness.recommendation,
    )


@router.get(
    "/stale",
    response_model=StaleDocumentsResponse,
    summary="Get stale documents",
    description="Get list of documents that may contain outdated information.",
)
async def get_stale_documents(
    threshold_days: int = 90,
    limit: int = 100,
    user: User = Depends(get_current_user),
    indexer: RealTimeIndexerService = Depends(get_indexer_service),
):
    """Get list of stale documents."""
    stale_docs = await indexer.get_stale_documents(
        threshold_days=threshold_days,
        limit=limit,
    )

    return StaleDocumentsResponse(
        documents=[
            FreshnessInfoResponse(
                document_id=str(doc.document_id),
                filename=doc.filename,
                last_modified=doc.last_modified,
                freshness_level=doc.freshness_level.value,
                days_since_update=doc.days_since_update,
                recommendation=doc.recommendation,
            )
            for doc in stale_docs
        ],
        total_count=len(stale_docs),
    )


# =============================================================================
# Indexing Endpoints
# =============================================================================

@router.post(
    "/touch",
    summary="Touch document",
    description="Update document's last modified timestamp to mark it as current.",
)
async def touch_document(
    request: TouchDocumentRequest,
    user: User = Depends(get_current_user),
    indexer: RealTimeIndexerService = Depends(get_indexer_service),
):
    """Touch a document to update its timestamp."""
    try:
        doc_uuid = UUID(request.document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )

    success = await indexer.touch_document(doc_uuid)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return {"success": True, "message": "Document timestamp updated"}


@router.post(
    "/queue",
    summary="Queue reindex task",
    description="Queue a document for reindexing (admin only).",
)
async def queue_reindex_task(
    request: QueueTaskRequest,
    user: User = Depends(require_admin),
    indexer: RealTimeIndexerService = Depends(get_indexer_service),
):
    """Queue a document for reindexing."""
    try:
        doc_uuid = UUID(request.document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )

    await indexer.queue_reindex_task(doc_uuid, priority=request.priority)

    pending_count = await indexer.get_pending_task_count()

    return {
        "success": True,
        "message": "Document queued for reindexing",
        "pending_tasks": pending_count,
    }


@router.post(
    "/process",
    response_model=IndexingStatsResponse,
    summary="Process pending tasks",
    description="Process pending indexing tasks (admin only).",
)
async def process_pending_tasks(
    batch_size: int = 50,
    user: User = Depends(require_admin),
    indexer: RealTimeIndexerService = Depends(get_indexer_service),
):
    """Process pending indexing tasks."""
    stats = await indexer.process_pending_tasks(batch_size=batch_size)

    return IndexingStatsResponse(
        documents_checked=stats.documents_checked,
        documents_updated=stats.documents_updated,
        chunks_added=stats.chunks_added,
        chunks_modified=stats.chunks_modified,
        chunks_deleted=stats.chunks_deleted,
        processing_time_ms=stats.processing_time_ms,
        errors=stats.errors,
    )


@router.get(
    "/queue/count",
    summary="Get pending task count",
    description="Get the number of pending indexing tasks.",
)
async def get_pending_task_count(
    user: User = Depends(get_current_user),
    indexer: RealTimeIndexerService = Depends(get_indexer_service),
):
    """Get pending task count."""
    count = await indexer.get_pending_task_count()
    return {"pending_tasks": count}
