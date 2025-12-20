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
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
import structlog
import json

from backend.services.rag import RAGService, RAGConfig, get_rag_service
from backend.services.response_cache import get_response_cache_service, ResponseCacheService
from backend.db.database import async_session_context, get_async_session
from backend.db.models import ChatSession as ChatSessionModel, ChatMessage as ChatMessageModel, MessageRole
from backend.services.agent_orchestrator import AgentOrchestrator, create_orchestrator
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


async def get_or_create_session(db, session_id: UUID, user_id: str) -> ChatSessionModel:
    """Get existing session or create a new one."""
    query = select(ChatSessionModel).where(ChatSessionModel.id == session_id)
    result = await db.execute(query)
    session = result.scalar_one_or_none()

    if not session:
        # Create new session
        session = ChatSessionModel(
            id=session_id,
            user_id=user_id,
            title="New Conversation",
            is_active=True,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

    return session


async def save_chat_messages(
    session_id: UUID,
    user_message: str,
    assistant_response: str,
    sources: Optional[List[dict]] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None,
):
    """Save user and assistant messages to the database."""
    try:
        async with async_session_context() as db:
            # Ensure session exists
            await get_or_create_session(db, session_id)

            # Save user message
            user_msg = ChatMessageModel(
                session_id=session_id,
                role=MessageRole.USER,
                content=user_message,
            )
            db.add(user_msg)

            # Save assistant message
            assistant_msg = ChatMessageModel(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=assistant_response,
                source_chunks=sources,
                model_used=model_used,
                tokens_used=tokens_used,
            )
            db.add(assistant_msg)

            # Update session title if it's the first message
            session_query = select(ChatSessionModel).where(ChatSessionModel.id == session_id)
            result = await db.execute(session_query)
            session = result.scalar_one()

            # Check message count - if this is the first exchange, generate title from user message
            msg_count_query = select(func.count(ChatMessageModel.id)).where(
                ChatMessageModel.session_id == session_id
            )
            count_result = await db.execute(msg_count_query)
            msg_count = count_result.scalar() or 0

            if msg_count == 0 and session.title == "New Conversation":
                # Generate title from first user message (truncate to 50 chars)
                new_title = user_message[:50] + "..." if len(user_message) > 50 else user_message
                session.title = new_title

            await db.commit()
            logger.info("Saved chat messages", session_id=str(session_id))

    except Exception as e:
        logger.error("Failed to save chat messages", error=str(e), session_id=str(session_id))

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


class AgentOptions(BaseModel):
    """Options for agent mode execution."""
    search_documents: bool = Field(default=True, description="Force search uploaded documents first")
    include_web_search: bool = Field(default=False, description="Include web search in research")
    require_approval: bool = Field(default=False, description="Show plan and require approval before execution")
    max_steps: int = Field(default=5, ge=1, le=10, description="Maximum number of steps in execution plan")
    collection: Optional[str] = Field(default=None, description="Target specific collection (None = all)")


class ChatRequest(BaseModel):
    """Chat completion request."""
    message: str
    session_id: Optional[UUID] = None
    include_sources: bool = True
    max_sources: int = Field(default=5, ge=1, le=20)
    collection_filter: Optional[str] = None
    query_only: bool = False  # If True, don't store in RAG
    mode: Optional[str] = Field(None, pattern="^(agent|chat|general)$")  # Execution mode: agent, chat (RAG), or general (LLM)
    agent_options: Optional[AgentOptions] = Field(default=None, description="Options for agent mode execution")
    include_collection_context: bool = Field(default=True, description="Include collection tags in LLM context")


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
    user: AuthenticatedUser,
):
    """
    Create a chat completion.

    Modes:
    - chat: RAG-powered response with document search and source citations
    - general: Direct LLM response without document search
    - agent: Multi-agent orchestration for complex tasks

    If no mode specified, defaults to 'chat' (RAG-powered).

    Response caching is enabled for low-temperature requests to reduce costs.
    """
    logger.info(
        "Creating chat completion",
        message_length=len(request.message),
        session_id=str(request.session_id) if request.session_id else None,
        query_only=request.query_only,
        mode=request.mode,
    )

    # Generate session ID if not provided
    session_id = request.session_id or uuid4()
    message_id = uuid4()

    # Get cache service
    cache_service = get_response_cache_service()

    try:
        # Route based on mode
        if request.mode == "agent":
            # Use agent orchestrator for complex multi-step tasks
            # Note: Non-streaming agent mode collects all output before returning
            async with async_session_context() as db:
                rag_service = get_rag()
                orchestrator = await create_orchestrator(db=db, rag_service=rag_service)

                # Build context with agent options
                agent_context = {"collection_filter": request.collection_filter}
                if request.agent_options:
                    agent_context["options"] = request.agent_options.model_dump()

                final_output = ""
                async for update in orchestrator.process_request(
                    request=request.message,
                    session_id=str(session_id),
                    user_id=user.user_id,
                    show_cost_estimation=True,
                    require_approval_above_usd=1.0,
                    context=agent_context,
                ):
                    if update.get("type") == "plan_completed":
                        final_output = update.get("output", "")
                    elif update.get("type") == "approval_required":
                        # For non-streaming, we can't wait for approval
                        return ChatResponse(
                            session_id=session_id,
                            message_id=message_id,
                            content="This request requires cost approval. Please use the streaming endpoint for agent mode.",
                            sources=[],
                            created_at=datetime.now(),
                        )
                    elif update.get("type") == "error":
                        raise HTTPException(status_code=500, detail=update.get("error", "Agent execution failed"))

                return ChatResponse(
                    session_id=session_id,
                    message_id=message_id,
                    content=final_output,
                    sources=[],  # Agents don't return traditional sources
                    created_at=datetime.now(),
                )

        elif request.mode == "general":
            # Use general chat service (no RAG)
            from backend.services.general_chat import get_general_chat_service
            general_service = get_general_chat_service()

            # Check cache for general chat (only for query_only mode to avoid caching conversational context)
            if request.query_only:
                async with async_session_context() as db:
                    # Check cache eligibility (low temperature)
                    temperature = 0.7  # Default temperature
                    if await cache_service.is_cache_eligible(db, temperature, "general", len(request.message)):
                        cached = await cache_service.get_cached_response(
                            db, request.message, "general-chat", temperature
                        )
                        if cached:
                            logger.info("Cache hit for general chat", prompt_length=len(request.message))
                            return ChatResponse(
                                session_id=session_id,
                                message_id=message_id,
                                content=cached.response_text,
                                sources=[],
                                created_at=datetime.now(),
                            )

            response = await general_service.query(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
            )

            # Cache the response for query_only requests
            if request.query_only:
                async with async_session_context() as db:
                    await cache_service.cache_response(
                        db=db,
                        prompt=request.message,
                        model_id="general-chat",
                        temperature=0.7,
                        response_text=response.content,
                        input_tokens=len(request.message) // 4,
                        output_tokens=len(response.content) // 4,
                    )

            # Save messages to database for session persistence
            if not request.query_only:
                await save_chat_messages(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response.content,
                    model_used="general-chat",
                )

            return ChatResponse(
                session_id=session_id,
                message_id=message_id,
                content=response.content,
                sources=[],  # No sources for general chat
                created_at=datetime.now(),
            )

        else:
            # Default: Use RAG service
            rag_service = get_rag()

            # Check cache for RAG queries (only for query_only mode)
            if request.query_only:
                async with async_session_context() as db:
                    temperature = rag_service.config.temperature
                    cache_key = f"{request.message}|{request.collection_filter or ''}"
                    if await cache_service.is_cache_eligible(db, temperature, "rag", len(cache_key)):
                        cached = await cache_service.get_cached_response(
                            db, cache_key, f"rag-{rag_service.config.chat_model or 'default'}", temperature
                        )
                        if cached:
                            logger.info("Cache hit for RAG query", prompt_length=len(request.message))
                            return ChatResponse(
                                session_id=session_id,
                                message_id=message_id,
                                content=cached.response_text,
                                sources=[],  # Cached responses don't include sources
                                created_at=datetime.now(),
                            )

            response = await rag_service.query(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
                collection_filter=request.collection_filter,
                access_tier=100,  # Default tier; use user.access_tier when auth is enabled
                include_collection_context=request.include_collection_context,
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

            # Cache the response for query_only requests
            if request.query_only:
                async with async_session_context() as db:
                    cache_key = f"{request.message}|{request.collection_filter or ''}"
                    await cache_service.cache_response(
                        db=db,
                        prompt=cache_key,
                        model_id=f"rag-{response.model}",
                        temperature=rag_service.config.temperature,
                        response_text=response.content,
                        input_tokens=response.tokens_used or len(request.message) // 4,
                        output_tokens=len(response.content) // 4,
                    )

            # Save messages to database for session persistence
            if not request.query_only:
                source_data = [
                    {
                        "document_id": str(s.document_id),
                        "document_name": s.document_name,
                        "chunk_id": str(s.chunk_id),
                        "page_number": s.page_number,
                        "relevance_score": s.relevance_score,
                        "snippet": s.snippet,
                    }
                    for s in sources
                ] if sources else None

                await save_chat_messages(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response.content,
                    sources=source_data,
                    model_used=response.model,
                    tokens_used=response.tokens_used,
                )

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
    user: AuthenticatedUser,
):
    """
    Create a streaming chat completion.

    Supports three modes:
    - chat: RAG-powered document search (default)
    - general: Direct LLM without document search
    - agent: Multi-agent orchestration for complex tasks

    Returns Server-Sent Events (SSE) for real-time streaming.
    """
    logger.info(
        "Creating streaming chat completion",
        message_length=len(request.message),
        session_id=str(request.session_id) if request.session_id else None,
        mode=request.mode,
    )

    async def generate_agent_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response using Agent orchestrator."""
        session_id = request.session_id or uuid4()
        user_id = user.user_id

        # Send session info first
        yield f"data: {json.dumps({'type': 'session', 'session_id': str(session_id)})}\n\n"

        try:
            async with async_session_context() as db:
                # Create orchestrator with RAG service for research
                rag_service = get_rag()
                orchestrator = await create_orchestrator(
                    db=db,
                    rag_service=rag_service,
                )

                # Build context with agent options
                agent_context = {"collection_filter": request.collection_filter}
                if request.agent_options:
                    agent_context["options"] = request.agent_options.model_dump()

                # Process through agent system
                async for update in orchestrator.process_request(
                    request=request.message,
                    session_id=str(session_id),
                    user_id=user_id,
                    show_cost_estimation=True,
                    require_approval_above_usd=1.0,
                    context=agent_context,
                ):
                    update_type = update.get("type", "unknown")

                    if update_type == "planning":
                        yield f"data: {json.dumps({'type': 'agent_status', 'status': 'planning', 'message': update.get('message', 'Planning...')})}\n\n"

                    elif update_type == "plan_created":
                        yield f"data: {json.dumps({'type': 'agent_plan', 'plan_id': update.get('plan_id'), 'summary': update.get('summary'), 'step_count': update.get('step_count')})}\n\n"

                    elif update_type == "estimating_cost":
                        yield f"data: {json.dumps({'type': 'agent_status', 'status': 'estimating', 'message': update.get('message', 'Estimating cost...')})}\n\n"

                    elif update_type == "cost_estimated":
                        yield f"data: {json.dumps({'type': 'agent_cost', 'estimated_cost_usd': update.get('estimated_cost_usd')})}\n\n"

                    elif update_type == "approval_required":
                        yield f"data: {json.dumps({'type': 'approval_required', 'plan_id': update.get('plan_id'), 'estimated_cost_usd': update.get('estimated_cost_usd'), 'threshold_usd': update.get('threshold_usd')})}\n\n"

                    elif update_type == "budget_exceeded":
                        yield f"data: {json.dumps({'type': 'error', 'data': update.get('message', 'Budget exceeded')})}\n\n"

                    elif update_type == "step_started":
                        yield f"data: {json.dumps({'type': 'agent_step', 'step': update.get('step_name'), 'agent': update.get('agent_type'), 'status': 'started'})}\n\n"

                    elif update_type == "step_completed":
                        yield f"data: {json.dumps({'type': 'agent_step', 'step': update.get('step_name'), 'agent': update.get('agent_type'), 'status': 'completed'})}\n\n"

                    elif update_type == "content":
                        yield f"data: {json.dumps({'type': 'content', 'data': update.get('data', '')})}\n\n"

                    elif update_type == "sources":
                        sources_data = update.get("data", [])
                        yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                    elif update_type == "plan_completed":
                        output = update.get("output", "")
                        yield f"data: {json.dumps({'type': 'content', 'data': output})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'total_cost_usd': update.get('total_cost_usd')})}\n\n"

                    elif update_type == "error":
                        yield f"data: {json.dumps({'type': 'error', 'data': update.get('error', 'Unknown error')})}\n\n"

        except Exception as e:
            logger.error("Agent streaming error", error=str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    async def generate_general_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response using general chat (no RAG)."""
        from backend.services.general_chat import get_general_chat_service

        session_id = request.session_id or uuid4()

        # Send session info first
        yield f"data: {json.dumps({'type': 'session', 'session_id': str(session_id)})}\n\n"

        try:
            general_service = get_general_chat_service()
            response = await general_service.query(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
            )
            # Send entire response as single content chunk
            yield f"data: {json.dumps({'type': 'content', 'data': response.content})}\n\n"

            # Save messages to database
            if not request.query_only:
                await save_chat_messages(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response.content,
                    model_used="general-chat",
                )

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error("General chat streaming error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    async def generate_rag_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response using RAG service."""
        rag_service = get_rag()
        session_id = request.session_id or uuid4()

        # Send session info first
        yield f"data: {json.dumps({'type': 'session', 'session_id': str(session_id)})}\n\n"

        # Accumulate content and sources for saving
        accumulated_content = []
        accumulated_sources = []

        try:
            # Stream from RAG service
            async for chunk in rag_service.query_stream(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
                collection_filter=request.collection_filter,
                access_tier=100,  # Default tier; use user.access_tier when auth is enabled
                include_collection_context=request.include_collection_context,
            ):
                if chunk.type == "content":
                    accumulated_content.append(chunk.data)
                    yield f"data: {json.dumps({'type': 'content', 'data': chunk.data})}\n\n"
                elif chunk.type == "sources" and request.include_sources:
                    # Format sources for API response
                    sources = chunk.data[:request.max_sources] if chunk.data else []
                    accumulated_sources = sources
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
                elif chunk.type == "done":
                    # Save messages to database before signaling done
                    if not request.query_only:
                        full_response = "".join(accumulated_content)
                        await save_chat_messages(
                            session_id=session_id,
                            user_message=request.message,
                            assistant_response=full_response,
                            sources=accumulated_sources if accumulated_sources else None,
                            model_used=rag_service.config.chat_model,
                        )
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                elif chunk.type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': str(chunk.data)})}\n\n"

        except Exception as e:
            logger.error("RAG streaming error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    # Route to appropriate stream generator based on mode
    if request.mode == "agent":
        stream_generator = generate_agent_stream()
    elif request.mode == "general":
        stream_generator = generate_general_stream()
    else:
        # Default to RAG (chat mode)
        stream_generator = generate_rag_stream()

    return StreamingResponse(
        stream_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user: AuthenticatedUser,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """
    List user's chat sessions.
    """
    logger.info("Listing chat sessions", page=page, page_size=page_size)

    try:
        async with async_session_context() as db:
            # Get total count
            count_query = select(func.count(ChatSessionModel.id)).where(
                ChatSessionModel.user_id == user.user_id,
                ChatSessionModel.is_active == True,
            )
            result = await db.execute(count_query)
            total = result.scalar() or 0

            # Get sessions with message count
            offset = (page - 1) * page_size
            sessions_query = (
                select(ChatSessionModel)
                .where(
                    ChatSessionModel.user_id == user.user_id,
                    ChatSessionModel.is_active == True,
                )
                .order_by(ChatSessionModel.updated_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            result = await db.execute(sessions_query)
            sessions = result.scalars().all()

            # Build response with message counts
            session_responses = []
            for session in sessions:
                # Get message count for this session
                msg_count_query = select(func.count(ChatMessageModel.id)).where(
                    ChatMessageModel.session_id == session.id
                )
                msg_result = await db.execute(msg_count_query)
                message_count = msg_result.scalar() or 0

                session_responses.append(SessionResponse(
                    id=session.id,
                    title=session.title or "New Conversation",
                    message_count=message_count,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                ))

            return SessionListResponse(
                sessions=session_responses,
                total=total,
            )

    except Exception as e:
        logger.error("Failed to list sessions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session(
    session_id: UUID,
    user: AuthenticatedUser,
):
    """
    Get all messages in a chat session.
    """
    logger.info("Getting chat session", session_id=str(session_id))

    try:
        async with async_session_context() as db:
            # Get session with messages
            query = (
                select(ChatSessionModel)
                .where(ChatSessionModel.id == session_id)
                .options(selectinload(ChatSessionModel.messages))
            )
            result = await db.execute(query)
            session = result.scalar_one_or_none()

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Convert messages to response format
            messages = []
            sources = {}
            for msg in sorted(session.messages, key=lambda m: m.created_at):
                messages.append(ChatMessage(
                    role=msg.role.value,
                    content=msg.content,
                ))
                # Include sources if available
                if msg.source_chunks:
                    sources[str(msg.id)] = msg.source_chunks

            return SessionMessagesResponse(
                session_id=session_id,
                messages=messages,
                sources=sources,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    user: AuthenticatedUser,
):
    """
    Delete a chat session and all its messages.
    """
    logger.info("Deleting chat session", session_id=str(session_id))

    try:
        async with async_session_context() as db:
            # Get the session
            query = select(ChatSessionModel).where(ChatSessionModel.id == session_id)
            result = await db.execute(query)
            session = result.scalar_one_or_none()

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Delete the session (cascade will delete messages)
            await db.delete(session)
            await db.commit()

            return {"message": "Session deleted successfully", "session_id": str(session_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.delete("/sessions")
async def delete_all_sessions(
    user: AuthenticatedUser,
    confirm: bool = Query(False, description="Must be true to confirm deletion"),
):
    """
    Delete all chat sessions for the current user.

    Requires confirm=true query parameter to prevent accidental deletion.
    """
    logger.info("Deleting all chat sessions for user", user_id=user.user_id, confirm=confirm)

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must confirm deletion with confirm=true query parameter"
        )

    try:
        async with async_session_context() as db:
            from sqlalchemy import delete

            # Delete all sessions for user (cascade will delete messages)
            result = await db.execute(
                delete(ChatSessionModel).where(ChatSessionModel.user_id == user.user_id)
            )
            await db.commit()

            deleted_count = result.rowcount
            logger.info("Deleted all sessions for user", user_id=user.user_id, deleted_count=deleted_count)

            return {"message": "All sessions deleted", "deleted_count": deleted_count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete all sessions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete all sessions: {str(e)}")


class BulkDeleteRequest(BaseModel):
    """Request to bulk delete sessions."""
    session_ids: List[UUID]


@router.post("/sessions/bulk-delete")
async def bulk_delete_sessions(
    request: BulkDeleteRequest,
    user: AuthenticatedUser,
):
    """
    Delete multiple chat sessions at once.

    Only deletes sessions owned by the current user.
    """
    logger.info("Bulk deleting chat sessions", user_id=user.user_id, count=len(request.session_ids))

    if not request.session_ids:
        return {"message": "No sessions to delete", "deleted_count": 0}

    try:
        async with async_session_context() as db:
            from sqlalchemy import delete

            # Delete specified sessions that belong to the user
            result = await db.execute(
                delete(ChatSessionModel).where(
                    ChatSessionModel.id.in_(request.session_ids),
                    ChatSessionModel.user_id == user.user_id,
                )
            )
            await db.commit()

            deleted_count = result.rowcount
            logger.info("Bulk deleted sessions", user_id=user.user_id, deleted_count=deleted_count)

            return {"message": "Sessions deleted", "deleted_count": deleted_count}

    except Exception as e:
        logger.error("Failed to bulk delete sessions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to bulk delete sessions: {str(e)}")


@router.patch("/sessions/{session_id}/title")
async def update_session_title(
    session_id: UUID,
    user: AuthenticatedUser,
    title: str = Query(..., min_length=1, max_length=200),
):
    """
    Update the title of a chat session.
    """
    logger.info("Updating session title", session_id=str(session_id), title=title)

    try:
        async with async_session_context() as db:
            # Get the session
            query = select(ChatSessionModel).where(ChatSessionModel.id == session_id)
            result = await db.execute(query)
            session = result.scalar_one_or_none()

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Update the title
            session.title = title
            await db.commit()

            return {"message": "Title updated", "session_id": str(session_id), "title": title}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update session title", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")


@router.post("/sessions/new", response_model=SessionResponse)
async def create_session(
    user: AuthenticatedUser,
    title: Optional[str] = None,
):
    """
    Create a new chat session.
    """
    logger.info("Creating new chat session", title=title)

    try:
        async with async_session_context() as db:
            # Create new session in database
            new_session = ChatSessionModel(
                user_id=user.user_id,
                title=title or "New Conversation",
                is_active=True,
            )
            db.add(new_session)
            await db.commit()
            await db.refresh(new_session)

            return SessionResponse(
                id=new_session.id,
                title=new_session.title,
                message_count=0,
                created_at=new_session.created_at,
                updated_at=new_session.updated_at,
            )

    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    user: AuthenticatedUser,
    message_id: UUID,
    rating: int = Query(..., ge=1, le=5),
    comment: Optional[str] = None,
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
    user: AuthenticatedUser,
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
    user: AuthenticatedUser,
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
    user: AuthenticatedUser,
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
