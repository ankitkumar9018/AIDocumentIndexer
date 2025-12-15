"""
AIDocumentIndexer - Documents API Routes
========================================

Endpoints for document management, listing, and retrieval.
All endpoints enforce permission checking based on user access tier.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from backend.db.database import get_async_session
from backend.db.models import Document, Chunk, AccessTier, ProcessingStatus
from backend.api.middleware.auth import (
    get_user_context,
    require_admin,
    AuthenticatedUser,
    AdminUser,
)
from backend.services.permissions import (
    UserContext,
    Permission,
    get_permission_service,
)
from backend.services.vectorstore import get_vector_store, SearchType
from backend.services.audit import AuditAction, get_audit_service

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class DocumentBase(BaseModel):
    """Base document model."""
    name: str
    file_type: str
    collection: Optional[str] = None
    access_tier: int = Field(default=1, ge=1, le=100)


class DocumentCreate(DocumentBase):
    """Document creation request."""
    pass


class DocumentUpdate(BaseModel):
    """Document update request."""
    name: Optional[str] = None
    collection: Optional[str] = None
    access_tier: Optional[int] = Field(default=None, ge=1, le=100)
    tags: Optional[List[str]] = None


class DocumentResponse(BaseModel):
    """Document response model."""
    id: UUID
    name: str
    file_type: str
    file_size: int
    file_hash: str
    collection: Optional[str] = None
    access_tier: int
    access_tier_name: str
    status: str
    chunk_count: int
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    uploaded_by: Optional[UUID] = None
    tags: Optional[List[str]] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Paginated document list response."""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class DocumentSearchRequest(BaseModel):
    """Document search request."""
    query: str
    collection: Optional[str] = None
    file_types: Optional[List[str]] = None
    min_tier: Optional[int] = None
    max_tier: Optional[int] = None
    limit: int = Field(default=20, ge=1, le=100)


class SearchResultItem(BaseModel):
    """Individual search result."""
    document_id: str
    document_name: str
    chunk_id: str
    content_preview: str
    relevance_score: float
    page_number: Optional[int] = None


class SearchResultResponse(BaseModel):
    """Search results response."""
    results: List[SearchResultItem]
    total_results: int
    query: str


class ChunkResponse(BaseModel):
    """Document chunk response."""
    id: UUID
    content: str
    chunk_index: int
    page_number: Optional[int]
    metadata: dict

    class Config:
        from_attributes = True


class CollectionInfo(BaseModel):
    """Collection summary information."""
    name: str
    document_count: int


class CollectionsResponse(BaseModel):
    """Collections list response."""
    collections: List[CollectionInfo]


# =============================================================================
# Helper Functions
# =============================================================================

def get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def document_to_response(doc: Document, chunk_count: int = 0) -> DocumentResponse:
    """Convert Document model to response."""
    return DocumentResponse(
        id=doc.id,
        name=doc.title or doc.original_filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        file_hash=doc.file_hash,
        collection=doc.tags[0] if doc.tags else None,  # Use first tag as collection
        access_tier=doc.access_tier.level if doc.access_tier else 1,
        access_tier_name=doc.access_tier.name if doc.access_tier else "Unknown",
        status=doc.processing_status.value,
        chunk_count=chunk_count,
        word_count=doc.word_count,
        page_count=doc.page_count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        uploaded_by=doc.uploaded_by_id,
        tags=doc.tags,
    )


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    collection: Optional[str] = None,
    file_type: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = Query(default="created_at"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
):
    """
    List documents with pagination and filtering.

    Results are automatically filtered based on the user's access tier.
    Users can only see documents at or below their access tier level.
    """
    logger.info(
        "Listing documents",
        user_id=user.user_id,
        access_tier=user.access_tier_level,
        page=page,
        page_size=page_size,
        collection=collection,
    )

    # Build base query with access tier filtering
    base_query = (
        select(Document)
        .join(AccessTier, Document.access_tier_id == AccessTier.id)
        .where(AccessTier.level <= user.access_tier_level)
    )

    # Apply filters
    if collection:
        base_query = base_query.where(Document.tags.contains([collection]))

    if file_type:
        base_query = base_query.where(Document.file_type == file_type)

    if status:
        try:
            status_enum = ProcessingStatus(status)
            base_query = base_query.where(Document.processing_status == status_enum)
        except ValueError:
            pass  # Invalid status, ignore filter

    # Count total before pagination
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply sorting
    sort_column = getattr(Document, sort_by, Document.created_at)
    if sort_order == "desc":
        base_query = base_query.order_by(desc(sort_column))
    else:
        base_query = base_query.order_by(asc(sort_column))

    # Apply pagination
    offset = (page - 1) * page_size
    base_query = base_query.offset(offset).limit(page_size)

    # Include access tier relationship
    base_query = base_query.options(selectinload(Document.access_tier))

    # Execute query
    result = await db.execute(base_query)
    documents = result.scalars().all()

    # Get chunk counts for each document
    doc_ids = [doc.id for doc in documents]
    if doc_ids:
        chunk_counts_query = (
            select(Chunk.document_id, func.count(Chunk.id))
            .where(Chunk.document_id.in_(doc_ids))
            .group_by(Chunk.document_id)
        )
        chunk_result = await db.execute(chunk_counts_query)
        chunk_counts = {row[0]: row[1] for row in chunk_result.all()}
    else:
        chunk_counts = {}

    # Build response
    doc_responses = [
        document_to_response(doc, chunk_counts.get(doc.id, 0))
        for doc in documents
    ]

    return DocumentListResponse(
        documents=doc_responses,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(documents)) < total,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific document by ID.

    Returns 404 if document doesn't exist or user doesn't have access.
    """
    logger.info(
        "Getting document",
        document_id=str(document_id),
        user_id=user.user_id,
    )

    # Query with access tier check
    query = (
        select(Document)
        .join(AccessTier, Document.access_tier_id == AccessTier.id)
        .where(
            and_(
                Document.id == document_id,
                AccessTier.level <= user.access_tier_level,
            )
        )
        .options(selectinload(Document.access_tier))
    )

    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        # Check if document exists but user doesn't have access
        exists_query = select(Document.id).where(Document.id == document_id)
        exists_result = await db.execute(exists_query)
        if exists_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this document",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Get chunk count
    chunk_count_query = (
        select(func.count(Chunk.id))
        .where(Chunk.document_id == document_id)
    )
    chunk_result = await db.execute(chunk_count_query)
    chunk_count = chunk_result.scalar() or 0

    return document_to_response(document, chunk_count)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    update: DocumentUpdate,
    user: AuthenticatedUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update document metadata.

    Users can only update documents they have write access to:
    - Document owner can always update
    - Users with tier >= 90 can update any document at their tier level or below
    - Admins can update any document
    """
    logger.info(
        "Updating document",
        document_id=str(document_id),
        user_id=user.user_id,
        update=update.model_dump(exclude_none=True),
    )

    # Get document with access tier
    query = (
        select(Document)
        .join(AccessTier, Document.access_tier_id == AccessTier.id)
        .where(Document.id == document_id)
        .options(selectinload(Document.access_tier))
    )
    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Check write permission
    permission_service = get_permission_service()
    has_write = await permission_service.check_document_access(
        user_context=user,
        document_id=str(document_id),
        permission=Permission.WRITE,
        session=db,
    )

    if not has_write:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have write access to this document",
        )

    # If updating access tier, ensure user can assign that tier
    if update.access_tier is not None:
        can_assign = await permission_service.can_assign_tier(
            user_context=user,
            target_tier_level=update.access_tier,
        )
        if not can_assign:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cannot assign tier {update.access_tier}. Your tier: {user.access_tier_level}",
            )

        # Find the access tier by level
        tier_query = select(AccessTier).where(AccessTier.level == update.access_tier)
        tier_result = await db.execute(tier_query)
        new_tier = tier_result.scalar_one_or_none()

        if new_tier:
            document.access_tier_id = new_tier.id

    # Apply updates
    if update.name is not None:
        document.title = update.name

    if update.collection is not None:
        # Store collection as first tag
        current_tags = document.tags or []
        if current_tags:
            current_tags[0] = update.collection
        else:
            current_tags = [update.collection]
        document.tags = current_tags

    if update.tags is not None:
        document.tags = update.tags

    await db.commit()
    await db.refresh(document)

    # Log the update
    audit_service = get_audit_service()
    await audit_service.log_document_action(
        action=AuditAction.DOCUMENT_UPDATE,
        user_id=user.user_id,
        document_id=str(document_id),
        document_name=document.original_filename,
        changes=update.model_dump(exclude_none=True),
        ip_address=get_client_ip(request),
        session=db,
    )

    # Get chunk count
    chunk_count_query = (
        select(func.count(Chunk.id))
        .where(Chunk.document_id == document_id)
    )
    chunk_result = await db.execute(chunk_count_query)
    chunk_count = chunk_result.scalar() or 0

    return document_to_response(document, chunk_count)


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    user: AuthenticatedUser,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
    hard_delete: bool = Query(default=False, description="Permanently delete (admin only)"),
):
    """
    Delete a document and all its chunks.

    - Soft delete by default (marks as deleted)
    - Hard delete requires admin privileges and explicit flag
    """
    logger.info(
        "Deleting document",
        document_id=str(document_id),
        user_id=user.user_id,
        hard_delete=hard_delete,
    )

    # Get document
    query = select(Document).where(Document.id == document_id)
    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Check delete permission
    permission_service = get_permission_service()
    has_delete = await permission_service.check_document_access(
        user_context=user,
        document_id=str(document_id),
        permission=Permission.DELETE,
        session=db,
    )

    if not has_delete:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this document",
        )

    # Store document name for audit log
    document_name = document.original_filename

    # Hard delete requires admin
    if hard_delete:
        if not user.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Hard delete requires admin privileges",
            )
        # Permanently delete document and chunks (cascade)
        await db.delete(document)
        await db.commit()

        # Log the deletion
        audit_service = get_audit_service()
        await audit_service.log_document_action(
            action=AuditAction.DOCUMENT_DELETE,
            user_id=user.user_id,
            document_id=str(document_id),
            document_name=document_name,
            changes={"hard_delete": True},
            ip_address=get_client_ip(request),
        )

        return {
            "message": "Document permanently deleted",
            "document_id": str(document_id),
        }

    # Soft delete - mark status as failed with error message
    document.processing_status = ProcessingStatus.FAILED
    document.processing_error = "Deleted by user"
    await db.commit()

    # Log the deletion
    audit_service = get_audit_service()
    await audit_service.log_document_action(
        action=AuditAction.DOCUMENT_DELETE,
        user_id=user.user_id,
        document_id=str(document_id),
        document_name=document_name,
        changes={"hard_delete": False, "soft_delete": True},
        ip_address=get_client_ip(request),
        session=db,
    )

    return {
        "message": "Document deleted successfully",
        "document_id": str(document_id),
    }


@router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    document_id: UUID,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
):
    """
    Get all chunks for a document.

    Useful for debugging and viewing how documents were processed.
    """
    logger.info(
        "Getting document chunks",
        document_id=str(document_id),
        user_id=user.user_id,
        page=page,
        page_size=page_size,
    )

    # Check document access first
    doc_query = (
        select(Document.id)
        .join(AccessTier, Document.access_tier_id == AccessTier.id)
        .where(
            and_(
                Document.id == document_id,
                AccessTier.level <= user.access_tier_level,
            )
        )
    )
    doc_result = await db.execute(doc_query)
    if not doc_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this document",
        )

    # Get chunks
    offset = (page - 1) * page_size
    chunks_query = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(page_size)
    )

    result = await db.execute(chunks_query)
    chunks = result.scalars().all()

    return [
        ChunkResponse(
            id=chunk.id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            metadata={
                "section_title": chunk.section_title,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
            },
        )
        for chunk in chunks
    ]


@router.post("/search", response_model=SearchResultResponse)
async def search_documents(
    request: DocumentSearchRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Search documents using semantic and keyword search.

    Combines vector similarity with full-text search for best results.
    Results are filtered based on user's access tier.
    """
    logger.info(
        "Searching documents",
        query=request.query,
        user_id=user.user_id,
        access_tier=user.access_tier_level,
        collection=request.collection,
        limit=request.limit,
    )

    # Use custom vector store for search
    vector_store = get_vector_store()

    # Perform hybrid search with access tier filtering
    results = await vector_store.search(
        query=request.query,
        search_type=SearchType.HYBRID,
        top_k=request.limit,
        access_tier_level=user.access_tier_level,
        session=db,
    )

    # Build response
    search_results = []
    for result in results:
        # Get document info
        doc_query = (
            select(Document.original_filename)
            .where(Document.id == result.document_id)
        )
        doc_result = await db.execute(doc_query)
        doc_name = doc_result.scalar_one_or_none() or "Unknown"

        # Filter by collection if specified
        if request.collection:
            doc_tags_query = select(Document.tags).where(Document.id == result.document_id)
            tags_result = await db.execute(doc_tags_query)
            tags = tags_result.scalar_one_or_none() or []
            if request.collection not in tags:
                continue

        # Filter by file type if specified
        if request.file_types:
            doc_type_query = select(Document.file_type).where(Document.id == result.document_id)
            type_result = await db.execute(doc_type_query)
            file_type = type_result.scalar_one_or_none()
            if file_type not in request.file_types:
                continue

        search_results.append(
            SearchResultItem(
                document_id=str(result.document_id),
                document_name=doc_name,
                chunk_id=str(result.chunk_id),
                content_preview=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                relevance_score=result.score,
                page_number=result.page_number,
            )
        )

    return SearchResultResponse(
        results=search_results[:request.limit],
        total_results=len(search_results),
        query=request.query,
    )


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: UUID,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Reprocess a document (re-extract text, regenerate embeddings).

    Useful after OCR improvements or when processing settings change.
    Requires write permission on the document.
    """
    logger.info(
        "Reprocessing document",
        document_id=str(document_id),
        user_id=user.user_id,
    )

    # Check write permission
    permission_service = get_permission_service()
    has_write = await permission_service.check_document_access(
        user_context=user,
        document_id=str(document_id),
        permission=Permission.WRITE,
        session=db,
    )

    if not has_write:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have write access to this document",
        )

    # Get document
    query = select(Document).where(Document.id == document_id)
    result = await db.execute(query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Reset processing status
    document.processing_status = ProcessingStatus.PENDING
    document.processing_error = None
    await db.commit()

    # Document is marked as pending; Ray worker will pick it up automatically
    # when scanning for documents with PENDING status
    logger.info(
        "Document queued for reprocessing",
        document_id=str(document_id),
        status="pending",
    )

    return {
        "message": "Document queued for reprocessing",
        "document_id": str(document_id),
        "status": "pending",
    }


@router.get("/collections/list", response_model=CollectionsResponse)
async def list_collections(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all collections (tags) the user has access to.

    Collections are derived from document tags, showing only
    collections from documents the user can access.
    """
    # Get all tags from accessible documents
    query = (
        select(Document.tags)
        .join(AccessTier, Document.access_tier_id == AccessTier.id)
        .where(
            and_(
                AccessTier.level <= user.access_tier_level,
                Document.tags.isnot(None),
            )
        )
    )

    result = await db.execute(query)
    all_tags = result.scalars().all()

    # Count documents per tag
    tag_counts: dict[str, int] = {}
    for tags in all_tags:
        if tags:
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Build response
    collections = [
        CollectionInfo(name=tag, document_count=count)
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])
    ]

    return CollectionsResponse(collections=collections)
