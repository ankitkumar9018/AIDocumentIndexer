"""
AIDocumentIndexer - Temporary Upload API Routes
================================================

Endpoints for temporary document uploads that allow users to
chat with documents before deciding to save them permanently.
"""

import os
import tempfile
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import AuthenticatedUser
from backend.services.temp_documents import get_temp_document_service, TempDocumentService

logger = structlog.get_logger(__name__)

router = APIRouter()


# Request/Response Models
class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str
    expires_at: str
    message: str = "Temporary session created"


class TempDocumentInfo(BaseModel):
    """Information about a temporary document."""
    id: str
    filename: str
    token_count: int
    file_size: int
    file_type: str
    has_chunks: bool
    has_embeddings: bool


class SessionInfoResponse(BaseModel):
    """Response with session information."""
    id: str
    user_id: str
    document_count: int
    total_tokens: int
    documents: List[TempDocumentInfo]
    created_at: str
    expires_at: str


class UploadResponse(BaseModel):
    """Response for document upload."""
    doc_id: str
    filename: str
    token_count: int
    fits_in_context: bool
    message: str


class PromoteRequest(BaseModel):
    """Request to promote a document to permanent storage."""
    collection: Optional[str] = None


class PromoteResponse(BaseModel):
    """Response for document promotion."""
    permanent_doc_id: str
    message: str


# Routes
@router.post("/sessions", response_model=CreateSessionResponse)
async def create_temp_session(
    user: AuthenticatedUser,
    service: TempDocumentService = Depends(get_temp_document_service),
):
    """
    Create a new temporary document session.

    Sessions automatically expire after 24 hours.
    """
    session_id = await service.create_session(user.id)
    session = await service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create session")

    return CreateSessionResponse(
        session_id=session_id,
        expires_at=session.expires_at.isoformat(),
    )


@router.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(
    session_id: str,
    user: AuthenticatedUser,
    service: TempDocumentService = Depends(get_temp_document_service),
):
    """Get information about a temporary session."""
    info = await service.get_session_info(session_id)

    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")

    # Verify ownership
    if info["user_id"] != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

    return SessionInfoResponse(
        id=info["id"],
        user_id=info["user_id"],
        document_count=info["document_count"],
        total_tokens=info["total_tokens"],
        documents=[
            TempDocumentInfo(**doc)
            for doc in info["documents"]
        ],
        created_at=info["created_at"],
        expires_at=info["expires_at"],
    )


@router.post("/sessions/{session_id}/upload", response_model=UploadResponse)
async def upload_temp_document(
    session_id: str,
    user: AuthenticatedUser,
    file: UploadFile = File(...),
    create_embeddings: bool = Form(default=False),
    service: TempDocumentService = Depends(get_temp_document_service),
):
    """
    Upload a document to a temporary session.

    The document will be processed for text extraction but not saved to the database.
    Large documents will be automatically chunked.
    """
    # Verify session exists and belongs to user
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")

    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")

    # Save to temp file
    temp_dir = tempfile.mkdtemp(prefix="temp_upload_")
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Add to session
        doc = await service.add_document(
            session_id=session_id,
            file_path=temp_path,
            filename=file.filename,
            create_embeddings=create_embeddings,
        )

        if not doc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process document")

        # Check if document fits in context
        MAX_CONTEXT = 100000
        fits_in_context = doc.token_count <= MAX_CONTEXT

        return UploadResponse(
            doc_id=doc.id,
            filename=doc.filename,
            token_count=doc.token_count,
            fits_in_context=fits_in_context,
            message="Document uploaded successfully" + (
                "" if fits_in_context else ". Document will be chunked for queries."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {str(e)}")


@router.delete("/sessions/{session_id}/documents/{doc_id}")
async def remove_temp_document(
    session_id: str,
    doc_id: str,
    user: AuthenticatedUser,
    service: TempDocumentService = Depends(get_temp_document_service),
):
    """Remove a document from a temporary session."""
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")

    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

    success = await service.remove_document(session_id, doc_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return {"message": "Document removed"}


@router.post("/sessions/{session_id}/documents/{doc_id}/save", response_model=PromoteResponse)
async def promote_document(
    session_id: str,
    doc_id: str,
    request: PromoteRequest,
    user: AuthenticatedUser,
    service: TempDocumentService = Depends(get_temp_document_service),
):
    """
    Save a temporary document to permanent storage.

    This triggers the full processing pipeline and adds the document
    to the vector store.
    """
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")

    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

    permanent_id = await service.promote_to_permanent(
        session_id=session_id,
        doc_id=doc_id,
        user_id=user.id,
        access_tier_id=str(user.access_tier_id) if hasattr(user, 'access_tier_id') else None,
        collection=request.collection,
    )

    if not permanent_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save document")

    return PromoteResponse(
        permanent_doc_id=permanent_id,
        message="Document saved to library",
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user: AuthenticatedUser,
    service: TempDocumentService = Depends(get_temp_document_service),
):
    """Delete a temporary session and all its documents."""
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")

    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

    success = await service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete session")

    return {"message": "Session deleted"}


@router.get("/sessions/{session_id}/context")
async def get_session_context(
    session_id: str,
    user: AuthenticatedUser,
    query: Optional[str] = Query(None, description="Query for semantic search in large documents"),
    max_tokens: int = Query(50000, ge=1000, le=100000),
    service: TempDocumentService = Depends(get_temp_document_service),
):
    """
    Get the context from all documents in a session.

    For small documents, returns full content.
    For large documents, returns relevant chunks (optionally based on query).
    """
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")

    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

    context = await service.get_context(
        session_id=session_id,
        query=query,
        max_tokens=max_tokens,
    )

    if not context:
        return {"context": None, "message": "No documents in session"}

    return {
        "context": context,
        "token_estimate": len(context) // 4,  # Rough estimate
    }
