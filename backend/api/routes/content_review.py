"""
Content Review API Routes

Provides endpoints for viewing and editing generated content
BEFORE final document rendering.

This is the "man in the middle" step that gives users control
over content before it becomes a document.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, ValidationError
import structlog

from backend.db.database import get_async_session
from backend.api.middleware.auth import AuthenticatedUser
from backend.services.generator import (
    ContentStatus,
    EditAction,
    SlideContent,
    BulletPoint,
    PresentationContent,
    DocumentSection,
    DocumentContent,
    SheetContent,
    SpreadsheetContent,
    ContentEditRequest,
    ContentReviewService,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

# Global review service instance (would be injected in production)
_review_service: Optional[ContentReviewService] = None


async def get_llm_client():
    """Get the LLM client for content regeneration."""
    try:
        from backend.services.llm import EnhancedLLMFactory
        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="content_generation",
            prefer_fast=True,
        )
        return llm
    except Exception as e:
        logger.warning(f"Could not initialize LLM client: {e}")
        return None


def get_review_service() -> ContentReviewService:
    """Get or create the content review service."""
    global _review_service
    if _review_service is None:
        _review_service = ContentReviewService()
    return _review_service


async def get_review_service_with_llm() -> ContentReviewService:
    """Get or create the content review service with LLM client."""
    global _review_service
    if _review_service is None:
        llm_client = await get_llm_client()
        _review_service = ContentReviewService(llm_client=llm_client)
    elif _review_service.llm_client is None:
        # Lazily initialize LLM client if not set
        _review_service.llm_client = await get_llm_client()
    return _review_service


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateReviewSessionRequest(BaseModel):
    """Request to create a new review session."""
    content_type: str = Field(..., description="pptx, docx, or xlsx")
    job_id: Optional[str] = Field(default=None, description="Generation job ID")
    # Content can be provided directly or retrieved from job
    presentation: Optional[dict] = None
    document: Optional[dict] = None
    spreadsheet: Optional[dict] = None


class CreateReviewSessionResponse(BaseModel):
    """Response after creating a review session."""
    session_id: str
    content_type: str
    total_items: int
    message: str


class ContentPreview(BaseModel):
    """Preview of a content item for display."""
    item_id: str
    item_number: int
    title: str
    status: str
    preview_text: str
    char_counts: Optional[dict] = None


class ReviewStatusResponse(BaseModel):
    """Status of a review session."""
    session_id: str
    content_type: str
    total_items: int
    reviewed_items: int
    approved_items: int
    progress_percent: float
    can_render: bool
    status_breakdown: dict


class EditContentRequest(BaseModel):
    """Request to edit content."""
    item_id: str
    action: str = Field(..., description="approve, edit, delete, regenerate, enhance, shorten, expand, change_tone")
    field_name: Optional[str] = Field(default=None, description="Field to edit (for direct edits)")
    new_value: Optional[str] = Field(default=None, description="New value (for direct edits)")
    feedback: Optional[str] = Field(default=None, description="Feedback for LLM actions")


class EditContentResponse(BaseModel):
    """Response after editing content."""
    success: bool
    item_id: str
    message: str
    new_status: Optional[str] = None
    updated_content: Optional[dict] = None


class BatchApproveRequest(BaseModel):
    """Request to approve multiple items."""
    item_ids: List[str]


class SlideEditRequest(BaseModel):
    """Request to edit a slide directly."""
    title: Optional[str] = Field(default=None, max_length=80)
    subtitle: Optional[str] = Field(default=None, max_length=120)
    bullets: Optional[List[dict]] = Field(default=None, description="List of {text, sub_bullets}")
    body_text: Optional[str] = Field(default=None, max_length=600)
    speaker_notes: Optional[str] = Field(default=None, max_length=500)
    image_description: Optional[str] = Field(default=None, max_length=150)
    layout: Optional[str] = None


# =============================================================================
# Session Management
# =============================================================================

@router.post("/sessions", response_model=CreateReviewSessionResponse)
async def create_review_session(
    request: CreateReviewSessionRequest,
    # user: AuthenticatedUser = Depends(),  # Uncomment for auth
):
    """
    Create a new content review session.

    Call this after content is generated and before document rendering.
    """
    service = get_review_service()

    try:
        # Parse content based on type
        content = None
        if request.content_type == "pptx" and request.presentation:
            content = PresentationContent(**request.presentation)
        elif request.content_type == "docx" and request.document:
            content = DocumentContent(**request.document)
        elif request.content_type == "xlsx" and request.spreadsheet:
            content = SpreadsheetContent(**request.spreadsheet)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content required for type: {request.content_type}"
            )

        session = service.create_session(content, request.content_type)

        return CreateReviewSessionResponse(
            session_id=session.session_id,
            content_type=session.content_type,
            total_items=session.total_items,
            message=f"Review session created with {session.total_items} items to review"
        )

    except ValidationError as e:
        # Pydantic validation errors → 400 Bad Request
        logger.warning("Validation error creating review session", error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid content structure: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # True server errors → 500
        logger.error("Failed to create review session", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sessions/{session_id}/status", response_model=ReviewStatusResponse)
async def get_review_status(session_id: str):
    """Get the current status of a review session."""
    service = get_review_service()
    status = service.get_review_status(session_id)

    if "error" in status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=status["error"])

    return ReviewStatusResponse(
        session_id=session_id,
        content_type=status["content_type"],
        total_items=status["progress"]["total"],
        reviewed_items=status["progress"]["reviewed"],
        approved_items=status["progress"]["approved"],
        progress_percent=status["progress"]["progress_percent"],
        can_render=status["can_render"],
        status_breakdown=status["status_breakdown"]
    )


@router.delete("/sessions/{session_id}")
async def delete_review_session(session_id: str):
    """Delete a review session."""
    service = get_review_service()
    if service.delete_session(session_id):
        return {"success": True, "message": "Session deleted"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")


# =============================================================================
# Content Viewing
# =============================================================================

@router.get("/sessions/{session_id}/items")
async def list_all_items(session_id: str):
    """
    Get preview of all content items in the session.

    Use this to display the content list in the review UI.
    """
    service = get_review_service()
    items = service.get_all_items_preview(session_id)

    if not items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or empty")

    return {
        "session_id": session_id,
        "items": items,
        "total": len(items)
    }


@router.get("/sessions/{session_id}/items/{item_id}")
async def get_item_detail(session_id: str, item_id: str):
    """
    Get detailed content for a specific item.

    Use this when user clicks on an item to edit it.
    """
    service = get_review_service()
    detail = service.get_item_detail(session_id, item_id)

    if not detail:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

    return detail


# =============================================================================
# Content Editing
# =============================================================================

@router.post("/sessions/{session_id}/items/{item_id}/edit", response_model=EditContentResponse)
async def edit_item(
    session_id: str,
    item_id: str,
    request: EditContentRequest
):
    """
    Edit a content item.

    Actions:
    - approve: Accept the item as is
    - edit: Direct field edit (requires field_name and new_value)
    - delete: Remove the item
    - regenerate: Ask LLM to regenerate (requires feedback)
    - enhance: Ask LLM to improve
    - shorten: Ask LLM to make more concise
    - expand: Ask LLM to add detail
    - change_tone: Ask LLM to adjust tone (requires feedback)
    """
    # Use async service with LLM for regeneration/enhancement actions
    service = await get_review_service_with_llm()

    try:
        action = EditAction(request.action)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: {request.action}. Valid: {[a.value for a in EditAction]}"
        )

    edit_request = ContentEditRequest(
        content_type="",  # Will be determined from session
        item_id=item_id,
        action=action,
        field_name=request.field_name,
        new_value=request.new_value,
        feedback=request.feedback
    )

    result = service.edit_item(session_id, edit_request)

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error", "Edit failed"))

    return EditContentResponse(
        success=True,
        item_id=item_id,
        message=f"Item {request.action}d successfully",
        new_status=result.get("status"),
        updated_content=result.get("new_content")
    )


@router.put("/sessions/{session_id}/slides/{item_id}")
async def update_slide(
    session_id: str,
    item_id: str,
    request: SlideEditRequest
):
    """
    Update a slide with new content directly.

    This is a convenience endpoint for full slide updates.
    """
    service = get_review_service()

    # Build updates dict from request
    updates = {}
    if request.title is not None:
        updates["title"] = request.title
    if request.subtitle is not None:
        updates["subtitle"] = request.subtitle
    if request.bullets is not None:
        updates["bullets"] = [
            BulletPoint(text=b["text"], sub_bullets=b.get("sub_bullets", []))
            for b in request.bullets
        ]
    if request.body_text is not None:
        updates["body_text"] = request.body_text
    if request.speaker_notes is not None:
        updates["speaker_notes"] = request.speaker_notes
    if request.image_description is not None:
        updates["image_description"] = request.image_description
    if request.layout is not None:
        updates["layout"] = request.layout

    # Apply each update
    results = []
    for field_name, new_value in updates.items():
        edit_request = ContentEditRequest(
            content_type="pptx",
            item_id=item_id,
            action=EditAction.EDIT,
            field_name=field_name,
            new_value=new_value
        )
        result = service.edit_item(session_id, edit_request)
        results.append(result)

    # Get updated slide
    detail = service.get_item_detail(session_id, item_id)

    return {
        "success": all(r.get("success") for r in results),
        "item_id": item_id,
        "updated_slide": detail
    }


@router.post("/sessions/{session_id}/approve/{item_id}")
async def approve_item(session_id: str, item_id: str):
    """Quick approve a single item."""
    service = get_review_service()

    edit_request = ContentEditRequest(
        content_type="",
        item_id=item_id,
        action=EditAction.APPROVE
    )

    result = service.edit_item(session_id, edit_request)

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))

    return {
        "success": True,
        "item_id": item_id,
        "status": "approved",
        "progress": result.get("progress")
    }


@router.post("/sessions/{session_id}/batch-approve")
async def batch_approve(session_id: str, request: BatchApproveRequest):
    """Approve multiple items at once."""
    service = get_review_service()
    result = service.batch_approve(session_id, request.item_ids)

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))

    return result


@router.post("/sessions/{session_id}/approve-all")
async def approve_all_items(session_id: str):
    """Approve all remaining items and mark session as ready for rendering."""
    service = get_review_service()
    result = service.approve_all(session_id)

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))

    return result


# =============================================================================
# Content Retrieval for Rendering
# =============================================================================

@router.get("/sessions/{session_id}/approved-content")
async def get_approved_content(session_id: str):
    """
    Get the approved content ready for document rendering.

    This endpoint should be called ONLY after all items are approved.
    It returns the content in a format ready for the document generator.
    """
    service = get_review_service()

    # Check if all items are approved
    status = service.get_review_status(session_id)
    if "error" in status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=status["error"])

    if not status.get("can_render"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not all items approved. Progress: {status['progress']['approved']}/{status['progress']['total']}"
        )

    # Get the approved content
    content = service.get_approved_content(session_id)

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not retrieve approved content")

    return {
        "session_id": session_id,
        "content_type": status["content_type"],
        "ready_for_render": True,
        "content": content.model_dump()
    }


@router.get("/sessions/{session_id}/export")
async def export_content(session_id: str, force: bool = False):
    """
    Export content regardless of approval status.

    Use force=true to export even if not all items are approved.
    """
    service = get_review_service()

    if force:
        content = service.force_get_content(session_id)
    else:
        content = service.get_approved_content(session_id)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve content. Use force=true to export anyway."
        )

    return {
        "session_id": session_id,
        "content": content.model_dump(),
        "forced": force
    }
