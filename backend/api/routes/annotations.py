"""
AIDocumentIndexer - Collaborative Annotations API Routes
========================================================

API endpoints for document annotations, highlights, and comments.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.middleware.auth import get_current_user
from backend.db.database import get_async_session
from backend.db.models import User

router = APIRouter(tags=["Annotations"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateAnnotationRequest(BaseModel):
    """Request to create an annotation."""
    document_id: str
    annotation_type: str = Field(..., description="highlight, comment, question, correction, or link")
    content: str = Field("", description="Annotation content/comment")
    selected_text: str = Field(..., description="Text that was selected")
    start_offset: int = Field(..., ge=0, description="Start position in document")
    end_offset: int = Field(..., ge=0, description="End position in document")
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    tags: Optional[List[str]] = None
    color: Optional[str] = Field(None, description="Highlight color (hex)")
    linked_annotation_id: Optional[str] = None
    linked_document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateAnnotationRequest(BaseModel):
    """Request to update an annotation."""
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddReplyRequest(BaseModel):
    """Request to add a reply."""
    content: str = Field(..., min_length=1, max_length=5000)


class ReplyResponse(BaseModel):
    """Response for annotation reply."""
    id: str
    annotation_id: str
    user_id: str
    user_name: str
    content: str
    created_at: str


class AnnotationResponse(BaseModel):
    """Response for annotation."""
    id: str
    document_id: str
    user_id: str
    user_name: str
    annotation_type: str
    content: str
    selected_text: str
    start_offset: int
    end_offset: int
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    status: str
    replies: List[ReplyResponse]
    tags: List[str]
    color: Optional[str] = None
    linked_annotation_id: Optional[str] = None
    linked_document_id: Optional[str] = None
    created_at: str
    updated_at: str
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None


class AnnotationSummaryResponse(BaseModel):
    """Response for annotation summary."""
    document_id: str
    total_count: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_user: Dict[str, int]
    recent_activity: List[Dict[str, Any]]


# =============================================================================
# Dependencies
# =============================================================================

def get_annotation_service():
    """Get the annotation service."""
    from backend.services.annotations import get_annotation_service
    return get_annotation_service()


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/",
    response_model=AnnotationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create annotation",
    description="Create a new annotation on a document.",
)
async def create_annotation(
    request: CreateAnnotationRequest,
    user: User = Depends(get_current_user),
):
    """Create a new annotation."""
    from backend.services.annotations import AnnotationType, get_annotation_service

    # Validate annotation type
    try:
        annotation_type = AnnotationType(request.annotation_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid annotation type. Must be one of: highlight, comment, question, correction, link",
        )

    service = get_annotation_service()

    annotation = await service.create_annotation(
        document_id=request.document_id,
        user_id=str(user.id),
        user_name=user.name if hasattr(user, 'name') else user.email,
        annotation_type=annotation_type,
        content=request.content,
        selected_text=request.selected_text,
        start_offset=request.start_offset,
        end_offset=request.end_offset,
        chunk_id=request.chunk_id,
        page_number=request.page_number,
        tags=request.tags,
        color=request.color,
        linked_annotation_id=request.linked_annotation_id,
        linked_document_id=request.linked_document_id,
        metadata=request.metadata,
    )

    return _annotation_to_response(annotation)


@router.get(
    "/{annotation_id}",
    response_model=AnnotationResponse,
    summary="Get annotation",
    description="Get an annotation by ID.",
)
async def get_annotation(
    annotation_id: str,
    user: User = Depends(get_current_user),
):
    """Get annotation by ID."""
    service = get_annotation_service()

    annotation = await service.get_annotation(annotation_id)
    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    return _annotation_to_response(annotation)


@router.patch(
    "/{annotation_id}",
    response_model=AnnotationResponse,
    summary="Update annotation",
    description="Update an annotation.",
)
async def update_annotation(
    annotation_id: str,
    request: UpdateAnnotationRequest,
    user: User = Depends(get_current_user),
):
    """Update an annotation."""
    service = get_annotation_service()

    annotation = await service.update_annotation(
        annotation_id=annotation_id,
        user_id=str(user.id),
        content=request.content,
        tags=request.tags,
        color=request.color,
        metadata=request.metadata,
    )

    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    return _annotation_to_response(annotation)


@router.delete(
    "/{annotation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete annotation",
    description="Delete an annotation.",
)
async def delete_annotation(
    annotation_id: str,
    user: User = Depends(get_current_user),
):
    """Delete an annotation."""
    service = get_annotation_service()

    success = await service.delete_annotation(
        annotation_id=annotation_id,
        user_id=str(user.id),
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )


@router.post(
    "/{annotation_id}/replies",
    response_model=ReplyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add reply",
    description="Add a reply to an annotation.",
)
async def add_reply(
    annotation_id: str,
    request: AddReplyRequest,
    user: User = Depends(get_current_user),
):
    """Add a reply to an annotation."""
    service = get_annotation_service()

    reply = await service.add_reply(
        annotation_id=annotation_id,
        user_id=str(user.id),
        user_name=user.name if hasattr(user, 'name') else user.email,
        content=request.content,
    )

    if not reply:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    return ReplyResponse(
        id=reply.id,
        annotation_id=reply.annotation_id,
        user_id=reply.user_id,
        user_name=reply.user_name,
        content=reply.content,
        created_at=reply.created_at.isoformat(),
    )


@router.post(
    "/{annotation_id}/resolve",
    response_model=AnnotationResponse,
    summary="Resolve annotation",
    description="Mark an annotation as resolved.",
)
async def resolve_annotation(
    annotation_id: str,
    user: User = Depends(get_current_user),
):
    """Resolve an annotation."""
    service = get_annotation_service()

    annotation = await service.resolve_annotation(
        annotation_id=annotation_id,
        user_id=str(user.id),
    )

    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    return _annotation_to_response(annotation)


@router.post(
    "/{annotation_id}/reopen",
    response_model=AnnotationResponse,
    summary="Reopen annotation",
    description="Reopen a resolved annotation.",
)
async def reopen_annotation(
    annotation_id: str,
    user: User = Depends(get_current_user),
):
    """Reopen a resolved annotation."""
    service = get_annotation_service()

    annotation = await service.reopen_annotation(
        annotation_id=annotation_id,
        user_id=str(user.id),
    )

    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found",
        )

    return _annotation_to_response(annotation)


@router.get(
    "/document/{document_id}",
    response_model=List[AnnotationResponse],
    summary="Get document annotations",
    description="Get all annotations for a document.",
)
async def get_document_annotations(
    document_id: str,
    include_resolved: bool = Query(False, description="Include resolved annotations"),
    annotation_type: Optional[str] = Query(None, description="Filter by type"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    user: User = Depends(get_current_user),
):
    """Get all annotations for a document."""
    from backend.services.annotations import AnnotationType

    service = get_annotation_service()

    # Parse annotation type filter
    type_filter = None
    if annotation_type:
        try:
            type_filter = AnnotationType(annotation_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid annotation type",
            )

    # Parse tags filter
    tags_filter = None
    if tags:
        tags_filter = [t.strip() for t in tags.split(",")]

    annotations = await service.get_document_annotations(
        document_id=document_id,
        include_resolved=include_resolved,
        annotation_type=type_filter,
        tags=tags_filter,
    )

    return [_annotation_to_response(a) for a in annotations]


@router.get(
    "/document/{document_id}/summary",
    response_model=AnnotationSummaryResponse,
    summary="Get annotation summary",
    description="Get summary statistics for document annotations.",
)
async def get_annotation_summary(
    document_id: str,
    user: User = Depends(get_current_user),
):
    """Get annotation summary for a document."""
    service = get_annotation_service()

    summary = await service.get_annotation_summary(document_id)

    return AnnotationSummaryResponse(
        document_id=summary.document_id,
        total_count=summary.total_count,
        by_type=summary.by_type,
        by_status=summary.by_status,
        by_user=summary.by_user,
        recent_activity=summary.recent_activity,
    )


@router.get(
    "/search",
    response_model=List[AnnotationResponse],
    summary="Search annotations",
    description="Search across annotations.",
)
async def search_annotations(
    query: str = Query(..., min_length=1, description="Search query"),
    document_ids: Optional[str] = Query(None, description="Comma-separated document IDs"),
    annotation_types: Optional[str] = Query(None, description="Comma-separated types"),
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
):
    """Search annotations by content."""
    from backend.services.annotations import AnnotationType

    service = get_annotation_service()

    # Parse document IDs
    doc_ids = None
    if document_ids:
        doc_ids = [d.strip() for d in document_ids.split(",")]

    # Parse annotation types
    types = None
    if annotation_types:
        try:
            types = [AnnotationType(t.strip()) for t in annotation_types.split(",")]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid annotation type",
            )

    annotations = await service.search_annotations(
        query=query,
        document_ids=doc_ids,
        user_id=None,  # Search all users' public annotations
        annotation_types=types,
        limit=limit,
    )

    return [_annotation_to_response(a) for a in annotations]


@router.get(
    "/user/me",
    response_model=List[AnnotationResponse],
    summary="Get my annotations",
    description="Get annotations created by the current user.",
)
async def get_my_annotations(
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
):
    """Get current user's annotations."""
    service = get_annotation_service()

    annotations = await service.get_user_annotations(
        user_id=str(user.id),
        limit=limit,
    )

    return [_annotation_to_response(a) for a in annotations]


# =============================================================================
# Helper Functions
# =============================================================================

def _annotation_to_response(annotation) -> AnnotationResponse:
    """Convert annotation to response model."""
    return AnnotationResponse(
        id=annotation.id,
        document_id=annotation.document_id,
        user_id=annotation.user_id,
        user_name=annotation.user_name,
        annotation_type=annotation.annotation_type.value,
        content=annotation.content,
        selected_text=annotation.selected_text,
        start_offset=annotation.start_offset,
        end_offset=annotation.end_offset,
        chunk_id=annotation.chunk_id,
        page_number=annotation.page_number,
        status=annotation.status.value,
        replies=[
            ReplyResponse(
                id=r.id,
                annotation_id=r.annotation_id,
                user_id=r.user_id,
                user_name=r.user_name,
                content=r.content,
                created_at=r.created_at.isoformat(),
            )
            for r in annotation.replies
        ],
        tags=annotation.tags,
        color=annotation.color,
        linked_annotation_id=annotation.linked_annotation_id,
        linked_document_id=annotation.linked_document_id,
        created_at=annotation.created_at.isoformat(),
        updated_at=annotation.updated_at.isoformat(),
        resolved_at=annotation.resolved_at.isoformat() if annotation.resolved_at else None,
        resolved_by=annotation.resolved_by,
    )
