"""
AIDocumentIndexer - Chat API Routes
===================================

Endpoints for AI-powered chat with RAG capabilities.
"""

from datetime import datetime
from typing import Optional, List, AsyncGenerator
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog
import json

from backend.services.rag import RAGService, RAGConfig, get_rag_service

logger = structlog.get_logger(__name__)

router = APIRouter()

# Initialize RAG service
_rag_service: Optional[RAGService] = None


def get_rag() -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = get_rag_service()
    return _rag_service


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatSource(BaseModel):
    """Source citation for a chat response."""
    document_id: UUID
    document_name: str
    chunk_id: UUID
    page_number: Optional[int] = None
    relevance_score: float
    snippet: str


class ChatRequest(BaseModel):
    """Chat completion request."""
    message: str
    session_id: Optional[UUID] = None
    include_sources: bool = True
    max_sources: int = Field(default=5, ge=1, le=20)
    collection_filter: Optional[str] = None
    query_only: bool = False  # If True, don't store in RAG


class ChatResponse(BaseModel):
    """Chat completion response."""
    session_id: UUID
    message_id: UUID
    content: str
    sources: List[ChatSource]
    created_at: datetime


class ChatStreamChunk(BaseModel):
    """Single chunk in a streaming response."""
    type: str  # "content", "source", "done", "error"
    data: str


class SessionResponse(BaseModel):
    """Chat session response."""
    id: UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    """List of chat sessions."""
    sessions: List[SessionResponse]
    total: int


class SessionMessagesResponse(BaseModel):
    """Messages in a session."""
    session_id: UUID
    messages: List[ChatMessage]
    sources: dict  # message_id -> sources


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    # user = Depends(get_current_user),
):
    """
    Create a chat completion with RAG.

    The AI will search the knowledge base for relevant context
    and generate a response with source citations.
    """
    logger.info(
        "Creating chat completion",
        message_length=len(request.message),
        session_id=str(request.session_id) if request.session_id else None,
        query_only=request.query_only,
    )

    # Get RAG service
    rag_service = get_rag()

    # Generate session ID if not provided
    session_id = request.session_id or uuid4()
    message_id = uuid4()

    try:
        # Query RAG service
        response = await rag_service.query(
            question=request.message,
            session_id=str(session_id) if not request.query_only else None,
            collection_filter=request.collection_filter,
            access_tier=100,  # Default tier; use user.access_tier when auth is enabled
        )

        # Convert sources to API format
        sources = []
        if request.include_sources:
            for source in response.sources[:request.max_sources]:
                # Parse UUIDs safely, falling back to generated ones
                try:
                    doc_uuid = UUID(source.document_id) if source.document_id else uuid4()
                except ValueError:
                    doc_uuid = uuid4()

                try:
                    chunk_uuid = UUID(source.chunk_id) if source.chunk_id else uuid4()
                except ValueError:
                    chunk_uuid = uuid4()

                sources.append(ChatSource(
                    document_id=doc_uuid,
                    document_name=source.document_name,
                    chunk_id=chunk_uuid,
                    page_number=source.page_number,
                    relevance_score=source.relevance_score,
                    snippet=source.snippet,
                ))

        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            content=response.content,
            sources=sources,
            created_at=datetime.now(),
        )

    except Exception as e:
        logger.error("Chat completion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@router.post("/completions/stream")
async def create_streaming_completion(
    request: ChatRequest,
    # user = Depends(get_current_user),
):
    """
    Create a streaming chat completion with RAG.

    Returns Server-Sent Events (SSE) for real-time streaming.
    """
    logger.info(
        "Creating streaming chat completion",
        message_length=len(request.message),
        session_id=str(request.session_id) if request.session_id else None,
    )

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response using RAG service."""
        rag_service = get_rag()
        session_id = request.session_id or uuid4()

        # Send session info first
        yield f"data: {json.dumps({'type': 'session', 'session_id': str(session_id)})}\n\n"

        try:
            # Stream from RAG service
            async for chunk in rag_service.query_stream(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
                collection_filter=request.collection_filter,
                access_tier=100,  # Default tier; use user.access_tier when auth is enabled
            ):
                if chunk.type == "content":
                    yield f"data: {json.dumps({'type': 'content', 'data': chunk.data})}\n\n"
                elif chunk.type == "sources" and request.include_sources:
                    # Format sources for API response
                    sources = chunk.data[:request.max_sources] if chunk.data else []
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                elif chunk.type == "done":
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                elif chunk.type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': str(chunk.data)})}\n\n"

        except Exception as e:
            logger.error("Streaming error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    # user = Depends(get_current_user),
):
    """
    List user's chat sessions.
    """
    logger.info("Listing chat sessions", page=page, page_size=page_size)

    # Return sample sessions for demo; connect to database when persistence is implemented
    mock_sessions = [
        SessionResponse(
            id=UUID("550e8400-e29b-41d4-a716-446655440100"),
            title="Q4 Strategy Questions",
            message_count=8,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        SessionResponse(
            id=UUID("550e8400-e29b-41d4-a716-446655440101"),
            title="Marketing Analysis",
            message_count=12,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]

    return SessionListResponse(
        sessions=mock_sessions,
        total=len(mock_sessions),
    )


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session(
    session_id: UUID,
    # user = Depends(get_current_user),
):
    """
    Get all messages in a chat session.
    """
    logger.info("Getting chat session", session_id=str(session_id))

    # Return sample messages for demo; connect to database when persistence is implemented
    return SessionMessagesResponse(
        session_id=session_id,
        messages=[
            ChatMessage(role="user", content="What were the key Q4 initiatives?"),
            ChatMessage(
                role="assistant",
                content="Based on your documents, the Q4 strategy focuses on digital transformation and customer experience improvements.",
            ),
            ChatMessage(role="user", content="Tell me more about the digital transformation plans."),
            ChatMessage(
                role="assistant",
                content="The digital transformation initiative includes cloud migration, AI integration, and process automation as outlined in the strategy presentation.",
            ),
        ],
        sources={
            "msg-1": [
                {
                    "document_name": "Q4 Strategy Presentation.pptx",
                    "page_number": 5,
                    "relevance_score": 0.92,
                }
            ]
        },
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    # user = Depends(get_current_user),
):
    """
    Delete a chat session and all its messages.
    """
    logger.info("Deleting chat session", session_id=str(session_id))

    # Session deletion is a no-op until database persistence is implemented
    logger.info("Session deletion requested", session_id=str(session_id))
    return {"message": "Session deleted successfully", "session_id": str(session_id)}


@router.patch("/sessions/{session_id}/title")
async def update_session_title(
    session_id: UUID,
    title: str = Query(..., min_length=1, max_length=200),
    # user = Depends(get_current_user),
):
    """
    Update the title of a chat session.
    """
    logger.info("Updating session title", session_id=str(session_id), title=title)

    # Title update is a no-op until database persistence is implemented
    logger.info("Session title update requested", session_id=str(session_id), new_title=title)
    return {"message": "Title updated", "session_id": str(session_id), "title": title}


@router.post("/sessions/new", response_model=SessionResponse)
async def create_session(
    title: Optional[str] = None,
    # user = Depends(get_current_user),
):
    """
    Create a new chat session.
    """
    logger.info("Creating new chat session", title=title)

    session_id = UUID("550e8400-e29b-41d4-a716-446655440150")

    return SessionResponse(
        id=session_id,
        title=title or "New Conversation",
        message_count=0,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@router.post("/feedback")
async def submit_feedback(
    message_id: UUID,
    rating: int = Query(..., ge=1, le=5),
    comment: Optional[str] = None,
    # user = Depends(get_current_user),
):
    """
    Submit feedback for a chat response.

    Used to improve RAG quality over time.
    """
    logger.info(
        "Submitting chat feedback",
        message_id=str(message_id),
        rating=rating,
        comment=comment,
    )

    # Log feedback for analysis; implement database storage for production use
    logger.info(
        "Chat feedback received",
        message_id=str(message_id),
        rating=rating,
        has_comment=comment is not None,
    )
    return {"message": "Feedback submitted", "message_id": str(message_id)}


# =============================================================================
# Session LLM Override Endpoints
# =============================================================================

class SessionLLMOverrideRequest(BaseModel):
    """Request to set session LLM override."""
    provider_id: str = Field(..., description="LLM provider ID to use for this session")
    model_override: Optional[str] = Field(None, description="Model to use (overrides provider default)")
    temperature_override: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature override")


class SessionLLMOverrideResponse(BaseModel):
    """Session LLM override response."""
    session_id: str
    provider_id: str
    provider_name: Optional[str]
    provider_type: str
    model: str
    temperature: Optional[float]


@router.get("/sessions/{session_id}/llm", response_model=SessionLLMOverrideResponse)
async def get_session_llm_override(
    session_id: UUID,
    # user = Depends(get_current_user),
):
    """
    Get the LLM configuration for a specific chat session.

    Returns the override if set, otherwise returns the default configuration.
    """
    from sqlalchemy import select
    from backend.db.database import async_session_context
    from backend.db.models import ChatSessionLLMOverride, LLMProvider
    from backend.services.llm import LLMConfigManager

    logger.info("Getting session LLM config", session_id=str(session_id))

    try:
        async with async_session_context() as db:
            # Check for existing override
            query = (
                select(ChatSessionLLMOverride)
                .where(ChatSessionLLMOverride.session_id == session_id)
            )
            result = await db.execute(query)
            override = result.scalar_one_or_none()

            if override:
                # Get provider details
                provider_query = select(LLMProvider).where(LLMProvider.id == override.provider_id)
                provider_result = await db.execute(provider_query)
                provider = provider_result.scalar_one_or_none()

                return SessionLLMOverrideResponse(
                    session_id=str(session_id),
                    provider_id=str(override.provider_id),
                    provider_name=provider.name if provider else None,
                    provider_type=provider.provider_type if provider else "unknown",
                    model=override.model_override or (provider.default_chat_model if provider else "default"),
                    temperature=override.temperature_override,
                )

        # No override - return default config
        config = await LLMConfigManager.get_config_for_operation("chat")
        return SessionLLMOverrideResponse(
            session_id=str(session_id),
            provider_id=config.provider_id or "env",
            provider_name=None if config.source == "env" else "Default Provider",
            provider_type=config.provider_type,
            model=config.model,
            temperature=config.temperature,
        )

    except Exception as e:
        logger.error("Failed to get session LLM config", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get session LLM config: {str(e)}")


@router.put("/sessions/{session_id}/llm", response_model=SessionLLMOverrideResponse)
async def set_session_llm_override(
    session_id: UUID,
    override_data: SessionLLMOverrideRequest,
    # user = Depends(get_current_user),
):
    """
    Set the LLM provider for a specific chat session.

    This allows users to use a different LLM for individual conversations.
    """
    from sqlalchemy import select
    from backend.db.database import async_session_context
    from backend.db.models import ChatSessionLLMOverride, LLMProvider, ChatSession
    from backend.services.llm import LLMConfigManager

    logger.info("Setting session LLM override", session_id=str(session_id), provider_id=override_data.provider_id)

    try:
        async with async_session_context() as db:
            # Verify provider exists and is active
            provider_query = select(LLMProvider).where(
                LLMProvider.id == override_data.provider_id,
                LLMProvider.is_active == True,
            )
            provider_result = await db.execute(provider_query)
            provider = provider_result.scalar_one_or_none()

            if not provider:
                raise HTTPException(
                    status_code=400,
                    detail="Provider not found or not active",
                )

            # Check if override already exists
            existing_query = select(ChatSessionLLMOverride).where(
                ChatSessionLLMOverride.session_id == session_id
            )
            existing_result = await db.execute(existing_query)
            existing = existing_result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.provider_id = override_data.provider_id
                existing.model_override = override_data.model_override
                existing.temperature_override = override_data.temperature_override
                await db.commit()
            else:
                # Create new
                new_override = ChatSessionLLMOverride(
                    session_id=session_id,
                    provider_id=override_data.provider_id,
                    model_override=override_data.model_override,
                    temperature_override=override_data.temperature_override,
                )
                db.add(new_override)
                await db.commit()

            # Invalidate cache for this session
            await LLMConfigManager.invalidate_cache(f"session:{session_id}")

            return SessionLLMOverrideResponse(
                session_id=str(session_id),
                provider_id=str(provider.id),
                provider_name=provider.name,
                provider_type=provider.provider_type,
                model=override_data.model_override or provider.default_chat_model,
                temperature=override_data.temperature_override,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to set session LLM override", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to set session LLM override: {str(e)}")


@router.delete("/sessions/{session_id}/llm")
async def delete_session_llm_override(
    session_id: UUID,
    # user = Depends(get_current_user),
):
    """
    Remove the LLM override for a chat session.

    The session will revert to using the default LLM configuration.
    """
    from sqlalchemy import select
    from backend.db.database import async_session_context
    from backend.db.models import ChatSessionLLMOverride
    from backend.services.llm import LLMConfigManager

    logger.info("Deleting session LLM override", session_id=str(session_id))

    try:
        async with async_session_context() as db:
            query = select(ChatSessionLLMOverride).where(
                ChatSessionLLMOverride.session_id == session_id
            )
            result = await db.execute(query)
            override = result.scalar_one_or_none()

            if not override:
                raise HTTPException(
                    status_code=404,
                    detail="No LLM override found for this session",
                )

            await db.delete(override)
            await db.commit()

            # Invalidate cache
            await LLMConfigManager.invalidate_cache(f"session:{session_id}")

            return {
                "message": "Session LLM override deleted. Will use default configuration.",
                "session_id": str(session_id),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session LLM override", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete session LLM override: {str(e)}")


# =============================================================================
# Chat Export Endpoints
# =============================================================================

class ExportFormat(str):
    """Export format options."""
    JSON = "json"
    MARKDOWN = "md"


@router.get("/sessions/{session_id}/export")
async def export_chat_session(
    session_id: UUID,
    format: str = Query("json", pattern="^(json|md)$"),
):
    """
    Export a chat session.

    Supported formats:
    - json: Full JSON export with metadata
    - md: Markdown formatted conversation
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from backend.db.database import async_session_context
    from backend.db.models import ChatSession, ChatMessage as ChatMessageModel
    from fastapi.responses import Response

    logger.info("Exporting chat session", session_id=str(session_id), format=format)

    try:
        async with async_session_context() as db:
            # Get session with messages
            query = (
                select(ChatSession)
                .where(ChatSession.id == session_id)
                .options(selectinload(ChatSession.messages))
            )
            result = await db.execute(query)
            session = result.scalar_one_or_none()

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            messages = sorted(session.messages, key=lambda m: m.created_at)

            if format == "json":
                # JSON export
                export_data = {
                    "session_id": str(session.id),
                    "title": session.title,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "message_count": len(messages),
                    "messages": [
                        {
                            "id": str(msg.id),
                            "role": msg.role.value,
                            "content": msg.content,
                            "model_used": msg.model_used,
                            "tokens_used": msg.tokens_used,
                            "created_at": msg.created_at.isoformat(),
                            "sources": msg.source_chunks,
                        }
                        for msg in messages
                    ],
                }

                filename = f"chat_{session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                return Response(
                    content=json.dumps(export_data, indent=2),
                    media_type="application/json",
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"'
                    },
                )

            elif format == "md":
                # Markdown export
                lines = [
                    f"# {session.title or 'Chat Session'}",
                    "",
                    f"**Session ID:** {session.id}",
                    f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"**Messages:** {len(messages)}",
                    "",
                    "---",
                    "",
                ]

                for msg in messages:
                    role_label = "**User:**" if msg.role.value == "user" else "**Assistant:**"
                    lines.append(role_label)
                    lines.append("")
                    lines.append(msg.content)
                    lines.append("")

                    if msg.model_used:
                        lines.append(f"*Model: {msg.model_used}*")

                    if msg.source_chunks:
                        lines.append("")
                        lines.append("**Sources:**")
                        for source in msg.source_chunks or []:
                            if isinstance(source, dict):
                                doc_name = source.get("document_name", "Unknown")
                                lines.append(f"- {doc_name}")
                        lines.append("")

                    lines.append("---")
                    lines.append("")

                content = "\n".join(lines)
                filename = f"chat_{session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"

                return Response(
                    content=content,
                    media_type="text/markdown",
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"'
                    },
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/sessions/{session_id}/token-count")
async def get_session_token_count(session_id: UUID):
    """
    Get the total token count for a chat session.

    Returns the sum of all tokens used in the session,
    useful for context window management.
    """
    from sqlalchemy import select, func
    from backend.db.database import async_session_context
    from backend.db.models import ChatSession, ChatMessage as ChatMessageModel

    try:
        async with async_session_context() as db:
            # Verify session exists
            session_query = select(ChatSession).where(ChatSession.id == session_id)
            result = await db.execute(session_query)
            session = result.scalar_one_or_none()

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Get token sum
            token_query = (
                select(func.sum(ChatMessageModel.tokens_used))
                .where(ChatMessageModel.session_id == session_id)
            )
            result = await db.execute(token_query)
            total_tokens = result.scalar() or 0

            # Get message count
            count_query = (
                select(func.count(ChatMessageModel.id))
                .where(ChatMessageModel.session_id == session_id)
            )
            result = await db.execute(count_query)
            message_count = result.scalar() or 0

            return {
                "session_id": str(session_id),
                "total_tokens": total_tokens,
                "message_count": message_count,
                "avg_tokens_per_message": total_tokens / message_count if message_count > 0 else 0,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get token count", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get token count: {str(e)}")
