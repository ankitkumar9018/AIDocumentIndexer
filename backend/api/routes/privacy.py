"""
AIDocumentIndexer - Privacy Control API Routes
================================================

Endpoints for managing user privacy settings, chat history,
and AI memory controls.

Based on ChatGPT's privacy controls:
- Data Controls (Settings → Data Controls)
- Memory management (Settings → Personalization → Memory)
- Temporary/Incognito chats
- Data export (GDPR/CCPA compliance)
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog
import io

from backend.db.database import async_session_context
from backend.api.middleware.auth import AuthenticatedUser
from backend.services.chat_privacy import (
    get_chat_privacy_service,
    PrivacyPreferences,
    ExportType,
    ExportFormat,
    ExportStatus,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class PrivacyPreferencesResponse(BaseModel):
    """User privacy preferences response."""
    user_id: str

    # Chat History Controls
    chat_history_enabled: bool = Field(
        description="Whether chat history is saved"
    )
    chat_history_visible_to_admins: bool = Field(
        description="Whether org admins can view your chat history (for support)"
    )

    # Memory Controls
    memory_enabled: bool = Field(
        description="Whether AI can learn and remember from your conversations"
    )
    memory_include_chat_history: bool = Field(
        description="Whether memory draws from past conversations"
    )
    memory_include_saved_facts: bool = Field(
        description="Whether to use explicitly saved memories"
    )

    # Data Training
    allow_training_data: bool = Field(
        description="Whether your conversations can be used to improve AI models"
    )

    # Retention
    auto_delete_history_days: Optional[int] = Field(
        None,
        description="Auto-delete chat history after this many days (null = never)"
    )
    auto_delete_memory_days: Optional[int] = Field(
        None,
        description="Auto-delete saved memories after this many days (null = never)"
    )


class PrivacyPreferencesUpdate(BaseModel):
    """Update request for privacy preferences."""
    chat_history_enabled: Optional[bool] = None
    chat_history_visible_to_admins: Optional[bool] = None
    memory_enabled: Optional[bool] = None
    memory_include_chat_history: Optional[bool] = None
    memory_include_saved_facts: Optional[bool] = None
    allow_training_data: Optional[bool] = None
    auto_delete_history_days: Optional[int] = Field(None, ge=1, le=365)
    auto_delete_memory_days: Optional[int] = Field(None, ge=1, le=365)


class ChatSessionSummary(BaseModel):
    """Summary of a chat session."""
    id: str
    title: Optional[str]
    is_active: bool
    created_at: Optional[str]
    updated_at: Optional[str]


class ChatHistoryResponse(BaseModel):
    """Response containing chat history."""
    sessions: List[ChatSessionSummary]
    total: int
    has_more: bool


class MemorySummary(BaseModel):
    """Summary of user's AI memories."""
    short_term_messages: int
    semantic_memory_count: int
    preference_count: int


class DataExportRequest(BaseModel):
    """Request to export user data."""
    export_type: str = Field(
        "all_data",
        description="Type of data to export: chat_history, memories, all_data"
    )
    format: str = Field(
        "json",
        description="Export format: json, csv, zip"
    )


class DataExportResponse(BaseModel):
    """Response for data export request."""
    request_id: str
    status: str
    export_type: str
    format: str
    requested_at: str
    download_url: Optional[str] = None


class DeleteHistoryResponse(BaseModel):
    """Response for delete history operation."""
    success: bool
    sessions_deleted: int
    message: str


# =============================================================================
# Privacy Preferences Endpoints
# =============================================================================

@router.get("/preferences", response_model=PrivacyPreferencesResponse)
async def get_privacy_preferences(
    user: AuthenticatedUser,
):
    """
    Get your current privacy preferences.

    Returns settings for:
    - Chat history saving
    - Admin visibility
    - AI memory/learning
    - Data training opt-out
    - Auto-deletion settings
    """
    service = get_chat_privacy_service()

    async with async_session_context() as db:
        prefs = await service.get_preferences(db, user.user_id)

        return PrivacyPreferencesResponse(
            user_id=prefs.user_id,
            chat_history_enabled=prefs.chat_history_enabled,
            chat_history_visible_to_admins=prefs.chat_history_visible_to_admins,
            memory_enabled=prefs.memory_enabled,
            memory_include_chat_history=prefs.memory_include_chat_history,
            memory_include_saved_facts=prefs.memory_include_saved_facts,
            allow_training_data=prefs.allow_training_data,
            auto_delete_history_days=prefs.auto_delete_history_days,
            auto_delete_memory_days=prefs.auto_delete_memory_days,
        )


@router.patch("/preferences", response_model=PrivacyPreferencesResponse)
async def update_privacy_preferences(
    updates: PrivacyPreferencesUpdate,
    user: AuthenticatedUser,
):
    """
    Update your privacy preferences.

    You can update any of the following:
    - chat_history_enabled: Toggle chat history saving
    - chat_history_visible_to_admins: Control admin access for support
    - memory_enabled: Toggle AI memory/learning
    - allow_training_data: Opt in/out of model training
    - auto_delete_history_days: Set auto-deletion period (1-365 days)
    """
    service = get_chat_privacy_service()

    # Only include non-None values
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}

    if not update_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No preferences to update")

    async with async_session_context() as db:
        prefs = await service.update_preferences(db, user.user_id, update_dict)

        return PrivacyPreferencesResponse(
            user_id=prefs.user_id,
            chat_history_enabled=prefs.chat_history_enabled,
            chat_history_visible_to_admins=prefs.chat_history_visible_to_admins,
            memory_enabled=prefs.memory_enabled,
            memory_include_chat_history=prefs.memory_include_chat_history,
            memory_include_saved_facts=prefs.memory_include_saved_facts,
            allow_training_data=prefs.allow_training_data,
            auto_delete_history_days=prefs.auto_delete_history_days,
            auto_delete_memory_days=prefs.auto_delete_memory_days,
        )


# =============================================================================
# Chat History Endpoints
# =============================================================================

@router.get("/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history(
    user: AuthenticatedUser,
    include_temporary: bool = Query(False, description="Include temporary/incognito chats"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Get your chat history.

    Only returns chats that belong to you.
    Temporary chats are excluded by default.
    """
    service = get_chat_privacy_service()

    async with async_session_context() as db:
        sessions = await service.get_chat_sessions(
            db,
            user.user_id,
            include_temporary=include_temporary,
            limit=limit + 1,  # Fetch one extra to check has_more
            offset=offset,
        )

        has_more = len(sessions) > limit
        if has_more:
            sessions = sessions[:limit]

        return ChatHistoryResponse(
            sessions=[ChatSessionSummary(**s) for s in sessions],
            total=len(sessions),
            has_more=has_more,
        )


@router.delete("/chat-history/{session_id}")
async def delete_chat_session(
    session_id: str,
    user: AuthenticatedUser,
):
    """
    Delete a specific chat session.

    Only works for sessions you own.
    This permanently deletes all messages in the session.
    """
    service = get_chat_privacy_service()

    async with async_session_context() as db:
        success = await service.delete_chat_session(db, user.user_id, session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or you don't have permission to delete it"
            )

        return {"success": True, "message": "Chat session deleted"}


@router.delete("/chat-history", response_model=DeleteHistoryResponse)
async def delete_all_chat_history(
    user: AuthenticatedUser,
    confirm: bool = Query(..., description="Must be true to confirm deletion"),
):
    """
    Delete ALL your chat history.

    This is a destructive operation that cannot be undone.
    You must pass confirm=true to proceed.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must pass confirm=true to delete all chat history"
        )

    service = get_chat_privacy_service()

    async with async_session_context() as db:
        count = await service.delete_all_chat_history(db, user.user_id)

        return DeleteHistoryResponse(
            success=True,
            sessions_deleted=count,
            message=f"Successfully deleted {count} chat session(s)"
        )


# =============================================================================
# Temporary Chat Endpoints
# =============================================================================

@router.post("/temporary-chat")
async def create_temporary_chat(
    user: AuthenticatedUser,
):
    """
    Create a temporary (incognito) chat session.

    Temporary chats:
    - Are automatically deleted after 30 days
    - Are NOT used for AI memory/learning
    - Are NOT used for model training
    - Don't appear in your regular chat history

    Use this for sensitive conversations you don't want saved.
    """
    service = get_chat_privacy_service()

    async with async_session_context() as db:
        session_id = await service.create_temporary_session(db, user.user_id)

        return {
            "session_id": session_id,
            "is_temporary": True,
            "message": "Temporary chat created. This session will be deleted after 30 days.",
        }


# =============================================================================
# Memory Management Endpoints
# =============================================================================

@router.get("/memories", response_model=MemorySummary)
async def get_memory_summary(
    user: AuthenticatedUser,
):
    """
    Get a summary of what the AI remembers about you.

    Returns counts of:
    - Short-term messages in current context
    - Semantic memories (facts, preferences)
    - Learned preferences
    """
    service = get_chat_privacy_service()

    async with async_session_context() as db:
        memories = await service.get_saved_memories(db, user.user_id)

        if memories:
            return MemorySummary(**memories[0])

        return MemorySummary(
            short_term_messages=0,
            semantic_memory_count=0,
            preference_count=0,
        )


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    user: AuthenticatedUser,
):
    """
    Delete a specific saved memory.

    Use this to make the AI forget a specific fact or preference.
    """
    service = get_chat_privacy_service()

    async with async_session_context() as db:
        success = await service.delete_memory(db, user.user_id, memory_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        return {"success": True, "message": "Memory deleted"}


@router.delete("/memories")
async def clear_all_memories(
    user: AuthenticatedUser,
    confirm: bool = Query(..., description="Must be true to confirm"),
):
    """
    Clear ALL your saved memories.

    This makes the AI forget everything it has learned about you.
    Cannot be undone.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must pass confirm=true to clear all memories"
        )

    service = get_chat_privacy_service()

    async with async_session_context() as db:
        count = await service.clear_all_memories(db, user.user_id)

        return {
            "success": True,
            "message": "All memories cleared. The AI will no longer remember your preferences."
        }


# =============================================================================
# Data Export Endpoints
# =============================================================================

@router.post("/export", response_model=DataExportResponse)
async def request_data_export(
    request: DataExportRequest,
    user: AuthenticatedUser,
):
    """
    Request an export of your data.

    Export types:
    - chat_history: All your chat sessions and messages
    - memories: All saved memories and preferences
    - all_data: Everything (chat history + memories + preferences)

    Formats:
    - json: Structured JSON format
    - csv: CSV format (chat sessions only)
    - zip: ZIP archive with all data files
    """
    try:
        export_type = ExportType(request.export_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid export_type. Must be one of: {[e.value for e in ExportType]}"
        )

    try:
        export_format = ExportFormat(request.format)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format. Must be one of: {[e.value for e in ExportFormat]}"
        )

    service = get_chat_privacy_service()

    async with async_session_context() as db:
        request_id = await service.request_data_export(
            db,
            user.user_id,
            export_type=export_type,
            export_format=export_format,
        )

        return DataExportResponse(
            request_id=request_id,
            status=ExportStatus.PENDING.value,
            export_type=export_type.value,
            format=export_format.value,
            requested_at=datetime.utcnow().isoformat(),
        )


@router.get("/export/{request_id}", response_model=DataExportResponse)
async def get_export_status(
    request_id: str,
    user: AuthenticatedUser,
):
    """
    Get the status of a data export request.
    """
    service = get_chat_privacy_service()

    async with async_session_context() as db:
        status = await service.get_export_status(db, user.user_id, request_id)

        return DataExportResponse(
            request_id=status["request_id"],
            status=status["status"],
            export_type=status["export_type"],
            format=status["format"],
            requested_at=status["requested_at"],
        )


@router.get("/export/download/{format}")
async def download_data_export(
    format: str,
    user: AuthenticatedUser,
):
    """
    Download your data directly.

    This generates and returns an immediate export of all your data.
    For large exports, use the async /export endpoint instead.
    """
    try:
        export_format = ExportFormat(format)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format. Must be one of: {[e.value for e in ExportFormat]}"
        )

    service = get_chat_privacy_service()

    async with async_session_context() as db:
        data = await service.generate_export(
            db,
            user.user_id,
            ExportType.ALL_DATA,
            export_format,
        )

        # Set appropriate content type and filename
        content_types = {
            ExportFormat.JSON: "application/json",
            ExportFormat.CSV: "text/csv",
            ExportFormat.ZIP: "application/zip",
        }

        extensions = {
            ExportFormat.JSON: "json",
            ExportFormat.CSV: "csv",
            ExportFormat.ZIP: "zip",
        }

        filename = f"my_data_export_{datetime.utcnow().strftime('%Y%m%d')}.{extensions[export_format]}"

        return StreamingResponse(
            io.BytesIO(data),
            media_type=content_types[export_format],
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
