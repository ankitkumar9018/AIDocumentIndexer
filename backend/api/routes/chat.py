"""
AIDocumentIndexer - Chat API Routes
===================================

Endpoints for AI-powered chat with RAG capabilities.
"""

from datetime import datetime
from typing import Optional, List, AsyncGenerator
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
import structlog
import json

from backend.services.rag import RAGService, RAGConfig, get_rag_service
from backend.services.response_cache import get_response_cache_service, ResponseCacheService
from backend.services.knowledge_graph import get_kg_service
from backend.core.config import settings
from backend.db.database import async_session_context, get_async_session
from backend.db.models import (
    ChatSession as ChatSessionModel,
    ChatMessage as ChatMessageModel,
    ChatFeedback,
    MessageRole,
    User,
    AgentTrajectory,
)
from backend.services.agent_orchestrator import AgentOrchestrator, create_orchestrator
from backend.services.user_personalization import get_personalization_service, UserPersonalizationService
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


async def get_db_user_id(db, user_id: str, user_email: str = None) -> UUID:
    """
    Get a valid database user UUID from user_id string or email.
    Returns UUID of the user in the database.
    """
    db_user_id = None

    # First try: check if user_id is already a valid UUID
    try:
        db_user_id = UUID(user_id)
        # Verify it exists in users table
        user_query = select(User).where(User.id == db_user_id)
        user_result = await db.execute(user_query)
        if not user_result.scalar_one_or_none():
            db_user_id = None
    except (ValueError, TypeError):
        pass

    # Second try: look up user by email
    if not db_user_id and user_email:
        user_query = select(User).where(User.email == user_email)
        user_result = await db.execute(user_query)
        user = user_result.scalar_one_or_none()
        if user:
            db_user_id = user.id

    # SECURITY: Do NOT fall back to admin user - this would be privilege escalation
    # If no valid user found, return None and let the caller handle it appropriately
    if not db_user_id:
        logger.warning(
            "Could not find user in database",
            user_id=user_id,
            user_email=user_email,
        )

    return db_user_id


async def get_or_create_session(db, session_id: UUID, user_id: str, user_email: str = None) -> ChatSessionModel:
    """Get existing session or create a new one."""
    query = select(ChatSessionModel).where(ChatSessionModel.id == session_id)
    result = await db.execute(query)
    session = result.scalar_one_or_none()

    if not session:
        db_user_id = await get_db_user_id(db, user_id, user_email)

        if not db_user_id:
            raise ValueError(f"Could not find valid user for session creation")

        # Create new session
        session = ChatSessionModel(
            id=session_id,
            user_id=db_user_id,
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
    user_id: str,
    user_email: str = None,
    sources: Optional[List[dict]] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None,
    confidence_score: Optional[float] = None,
    confidence_level: Optional[str] = None,
) -> bool:
    """
    Save user and assistant messages to the database.

    Returns:
        True if messages were saved successfully, False otherwise.
    """
    try:
        async with async_session_context() as db:
            # Ensure session exists
            await get_or_create_session(db, session_id, user_id, user_email)

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
                confidence_score=confidence_score,
                confidence_level=confidence_level,
            )
            db.add(assistant_msg)

            # Update session title if it's still the default
            session_query = select(ChatSessionModel).where(ChatSessionModel.id == session_id)
            result = await db.execute(session_query)
            session = result.scalar_one_or_none()

            # Generate title from first user message if title is still default
            if session and session.title in ("New Conversation", "New Chat", None):
                # Generate title from first user message (truncate to 50 chars)
                new_title = user_message[:50] + "..." if len(user_message) > 50 else user_message
                session.title = new_title

            await db.commit()
            logger.info("Saved chat messages", session_id=str(session_id))
            return True

    except Exception as e:
        logger.error("Failed to save chat messages", error=str(e), session_id=str(session_id))
        return False

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
    content: str = Field(..., max_length=500000)
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None


class ChatSource(BaseModel):
    """Source citation for a chat response."""
    document_id: UUID
    document_name: str
    chunk_id: UUID
    page_number: Optional[int] = None
    relevance_score: float  # RRF score for ranking (may be tiny ~0.01-0.03)
    similarity_score: Optional[float] = None  # Original vector cosine similarity (0-1) for display
    snippet: str
    full_content: Optional[str] = None  # Full chunk content for source viewer modal
    collection: Optional[str] = None  # Collection/tag for grouping


class AgentOptions(BaseModel):
    """Options for agent mode execution."""
    search_documents: bool = Field(default=True, description="Force search uploaded documents first")
    include_web_search: bool = Field(default=False, description="Include web search in research")
    require_approval: bool = Field(default=False, description="Show plan and require approval before execution")
    max_steps: int = Field(default=5, ge=1, le=10, description="Maximum number of steps in execution plan")
    collection: Optional[str] = Field(default=None, description="Target specific collection (None = all)")
    language: Optional[str] = Field(default=None, description="Language for agent responses (en, de, es, fr, etc.). If None, uses request.language.")


class ImageAttachment(BaseModel):
    """Image attachment for vision-enabled chat."""
    data: Optional[str] = Field(None, description="Base64-encoded image data")
    url: Optional[str] = Field(None, description="URL to image (alternative to data)")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image")


class ChatRequest(BaseModel):
    """Chat completion request."""
    message: str = Field(..., min_length=1, max_length=100000, description="The user's message (max 100,000 characters)")
    session_id: Optional[UUID] = None
    include_sources: bool = True
    max_sources: int = Field(default=5, ge=1, le=20)
    collection_filter: Optional[str] = Field(None, max_length=100, description="Single collection filter (backward compatible)")
    collection_filters: Optional[List[str]] = Field(None, description="Multiple collection filters")
    query_only: bool = False  # If True, don't store in RAG
    mode: Optional[str] = Field(None, pattern="^(agent|chat|general|vision)$")  # Execution mode: agent, chat (RAG), general (LLM), or vision (multimodal)
    agent_options: Optional[AgentOptions] = Field(default=None, description="Options for agent mode execution")
    include_collection_context: bool = Field(default=True, description="Include collection tags in LLM context")
    temp_session_id: Optional[str] = Field(default=None, description="Temporary document session ID for quick chat")
    top_k: Optional[int] = Field(default=None, ge=3, le=25, description="Number of documents to search (3-25). Uses admin setting if not specified.")
    images: Optional[List[ImageAttachment]] = Field(default=None, description="Image attachments for vision mode")
    # Folder-scoped queries
    folder_id: Optional[str] = Field(default=None, description="Folder ID to scope query to. Only documents in this folder (and subfolders) will be searched.")
    include_subfolders: bool = Field(default=True, description="When folder_id is set, include documents in subfolders")
    # Language for responses
    language: Optional[str] = Field(default="auto", description="Language code for response. Use 'auto' to respond in the same language as the question, or specify: en, de, es, fr, it, pt, nl, pl, ru, zh, ja, ko, ar, hi")
    # Query enhancement toggle
    enhance_query: Optional[bool] = Field(default=None, description="Enable query enhancement (expansion + HyDE). None = use admin default setting.")
    # Restrict to documents only - when enabled in general mode, LLM won't use pre-trained knowledge
    restrict_to_documents: bool = Field(default=False, description="When enabled in general mode, the assistant will not use pre-trained knowledge and will suggest switching to document mode.")

    @property
    def effective_collection_filters(self) -> Optional[List[str]]:
        """Get effective collection filter(s) from either single or multi filter."""
        if self.collection_filters:
            return self.collection_filters
        elif self.collection_filter:
            return [self.collection_filter]
        return None

    @property
    def first_collection_filter(self) -> Optional[str]:
        """Get the first collection filter for backward-compatible single-filter APIs."""
        filters = self.effective_collection_filters
        return filters[0] if filters else None


class ContextSufficiencyInfo(BaseModel):
    """Context sufficiency check result for UI display."""
    is_sufficient: bool
    coverage_score: float  # 0-1
    has_conflicts: bool
    missing_aspects: List[str] = []
    confidence_level: str  # "high", "medium", "low"
    explanation: str = ""


class ChatResponse(BaseModel):
    """Chat completion response."""
    session_id: UUID
    message_id: UUID
    content: str
    sources: List[ChatSource]
    created_at: datetime
    # Confidence/verification fields
    confidence_score: Optional[float] = None  # 0-1 confidence in the answer
    confidence_level: Optional[str] = None  # "high", "medium", "low"
    # Confidence warning message for UI display
    confidence_warning: Optional[str] = None
    # Suggested follow-up questions
    suggested_questions: Optional[List[str]] = None
    # Context sufficiency check result (Phase 2 enhancement)
    context_sufficiency: Optional[ContextSufficiencyInfo] = None


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
                agent_context = {"collection_filter": request.first_collection_filter}
                # Pass language through agent context
                # Prioritize: explicit agent_options.language > request.language > "auto"
                agent_language = request.language or "auto"
                if request.agent_options and request.agent_options.language:
                    agent_language = request.agent_options.language
                agent_context["language"] = agent_language
                if request.agent_options:
                    agent_context["options"] = request.agent_options.model_dump()

                final_output = ""
                agent_sources = []  # Collect sources from agent execution
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
                    elif update.get("type") == "sources":
                        # Collect sources emitted by agents (research steps)
                        sources_data = update.get("data", [])
                        if sources_data:
                            agent_sources.extend(sources_data)
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
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=update.get("error", "Agent execution failed"))

                # Convert agent sources to ChatSource format if available
                formatted_sources = []
                if agent_sources and request.include_sources:
                    for src in agent_sources[:request.max_sources]:
                        if isinstance(src, dict):
                            formatted_sources.append(ChatSource(
                                document_id=src.get("document_id", ""),
                                document_name=src.get("document_name", src.get("title", "Unknown")),
                                content=src.get("content", ""),
                                score=src.get("score", 0.0),
                                chunk_index=src.get("chunk_index", 0),
                            ))

                return ChatResponse(
                    session_id=session_id,
                    message_id=message_id,
                    content=final_output,
                    sources=formatted_sources,
                    created_at=datetime.now(),
                )

        elif request.mode == "general":
            # Check if restrict_to_documents is enabled - return helpful message without using LLM
            if request.restrict_to_documents:
                restricted_response = (
                    "I'm currently restricted from using my pre-trained knowledge. "
                    "In this mode, I can only answer questions using information from your uploaded documents.\n\n"
                    "To get an answer, please either:\n"
                    "1. **Switch to Document Mode** - I'll search your documents for relevant information\n"
                    "2. **Disable 'No AI Knowledge'** toggle - Allow me to use my general knowledge\n\n"
                    "This restriction ensures all responses are strictly based on your document content."
                )

                # Save the interaction if not query_only
                if not request.query_only:
                    await save_chat_messages(
                        session_id=session_id,
                        user_message=request.message,
                        assistant_response=restricted_response,
                        user_id=user.user_id,
                        user_email=user.email,
                        model_used="restricted-mode",
                    )

                return ChatResponse(
                    session_id=session_id,
                    message_id=message_id,
                    content=restricted_response,
                    sources=[],
                    created_at=datetime.now(),
                )

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
                language=request.language or "en",
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
                    user_id=user.user_id,
                    user_email=user.email,
                    model_used="general-chat",
                )

            return ChatResponse(
                session_id=session_id,
                message_id=message_id,
                content=response.content,
                sources=[],  # No sources for general chat
                created_at=datetime.now(),
            )

        elif request.mode == "vision":
            # Vision mode: Use multimodal LLM for image analysis
            if not request.images:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vision mode requires at least one image attachment")

            from backend.services.llm import chat_with_vision
            import base64

            # Process images
            image_data_list = []
            image_urls = []

            for img in request.images:
                if img.data:
                    # Decode base64 image data
                    try:
                        decoded = base64.b64decode(img.data)
                        image_data_list.append((decoded, img.mime_type))
                    except Exception as e:
                        logger.warning("Failed to decode image data", error=str(e))
                elif img.url:
                    image_urls.append(img.url)

            # Use the first image (extend to multi-image later if needed)
            response_content = ""
            if image_data_list:
                image_bytes, mime_type = image_data_list[0]
                response_content = await chat_with_vision(
                    model=None,  # Use default vision model
                    text=request.message,
                    image_data=image_bytes,
                    image_type=mime_type,
                )
            elif image_urls:
                from backend.services.llm import create_vision_messages_from_urls, get_chat_llm
                llm = await get_chat_llm()
                # For URL-based images, use a different approach
                response_content = await chat_with_vision(
                    model=None,
                    text=request.message,
                    image_url=image_urls[0],
                )
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid image data or URL provided")

            # Save messages to database
            if not request.query_only:
                await save_chat_messages(
                    session_id=session_id,
                    user_message=f"[Image Analysis] {request.message}",
                    assistant_response=response_content,
                    user_id=user.user_id,
                    user_email=user.email,
                    model_used="vision",
                )

            return ChatResponse(
                session_id=session_id,
                message_id=message_id,
                content=response_content,
                sources=[],  # No sources for vision chat
                created_at=datetime.now(),
            )

        else:
            # Default: Use RAG service
            rag_service = get_rag()

            # Get temp document context if provided
            temp_context = None
            if request.temp_session_id:
                from backend.services.temp_documents import get_temp_document_service
                temp_service = get_temp_document_service()
                temp_session = await temp_service.get_session(request.temp_session_id)
                if temp_session and temp_session.user_id == user.user_id:
                    temp_context = await temp_service.get_context(
                        session_id=request.temp_session_id,
                        query=request.message,
                        max_tokens=50000,
                    )
                    logger.info(
                        "Using temp document context",
                        temp_session_id=request.temp_session_id,
                        context_length=len(temp_context) if temp_context else 0,
                    )

            # Check cache for RAG queries (only for query_only mode)
            if request.query_only:
                async with async_session_context() as db:
                    temperature = rag_service.config.temperature
                    cache_key = f"{request.message}|{request.first_collection_filter or ''}"
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

            # PHASE 40/48: Retrieve relevant memories to enhance context
            memory_context = None
            from backend.core.config import settings as app_settings
            if getattr(app_settings, "ENABLE_AGENT_MEMORY", False):
                try:
                    from backend.services.mem0_memory import get_memory_service

                    memory_service = await get_memory_service()
                    relevant_memories = await memory_service.get_relevant(
                        query=request.message,
                        user_id=user.user_id,
                        top_k=5,  # Get top 5 relevant memories
                    )

                    if relevant_memories:
                        # Format memories as context
                        memory_lines = []
                        for mem in relevant_memories:
                            if hasattr(mem, 'content'):
                                memory_lines.append(f"- {mem.content}")
                            elif isinstance(mem, dict):
                                memory_lines.append(f"- {mem.get('content', str(mem))}")

                        if memory_lines:
                            memory_context = "Relevant memories from previous conversations:\n" + "\n".join(memory_lines)
                            logger.info(
                                "Retrieved memories for context",
                                user_id=user.user_id,
                                memory_count=len(memory_lines),
                            )
                except Exception as e:
                    logger.warning("Failed to retrieve memories", error=str(e))

            # Combine temp_context with memory_context
            combined_context = temp_context
            if memory_context:
                if combined_context:
                    combined_context = f"{memory_context}\n\n{combined_context}"
                else:
                    combined_context = memory_context

            # Phase 60: Add KG entity context for enhanced query understanding
            kg_context = ""
            if getattr(settings, 'KG_ENABLED', True) and getattr(settings, 'KG_ENABLED_IN_CHAT', True):
                try:
                    kg_service = await get_kg_service()

                    # Extract entities from the user's query
                    entities, relations = await kg_service.extract_entities_from_text(
                        text=request.message,
                        document_language=request.language or "en",
                        use_fast_extraction=True,  # Quick extraction for queries
                    )

                    if entities:
                        # Get related context from KG
                        entity_names = [e.name for e in entities[:5]]  # Top 5 entities
                        kg_results = await kg_service.search_by_entities(
                            entity_names=entity_names,
                            collection=request.first_collection_filter,
                            max_hops=getattr(settings, 'KG_MAX_HOPS', 2),
                            limit=3,
                        )

                        if kg_results:
                            kg_context = "Knowledge Graph Context:\n"
                            for result in kg_results[:3]:
                                kg_context += f"- {result.entity.name}: {result.entity.description or 'Related entity'}\n"
                            kg_context += "\n"

                            logger.debug(
                                "KG context added to chat query",
                                entity_count=len(entities),
                                kg_results=len(kg_results),
                            )
                except Exception as e:
                    logger.debug("KG context enhancement skipped", error=str(e))

            # Combine all context: memory + temp docs + KG
            if kg_context:
                if combined_context:
                    combined_context = f"{kg_context}{combined_context}"
                else:
                    combined_context = kg_context

            # Phase 66: Get personalized prompt additions based on user preferences
            personalization_additions = ""
            if getattr(app_settings, "ENABLE_USER_PERSONALIZATION", True):
                try:
                    personalization_service = get_personalization_service()
                    personalization_additions = await personalization_service.get_personalized_prompt_additions(
                        user_id=user.user_id
                    )
                    if personalization_additions:
                        logger.debug(
                            "Applied user personalization",
                            user_id=user.user_id[:8] if user.user_id else "unknown",
                            has_additions=bool(personalization_additions),
                        )
                except Exception as e:
                    logger.debug("Personalization skipped", error=str(e))

            # Add personalization to combined context
            if personalization_additions:
                if combined_context:
                    combined_context = f"{combined_context}\n\nUser preferences:\n{personalization_additions}"
                else:
                    combined_context = f"User preferences:\n{personalization_additions}"

            response = await rag_service.query(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
                collection_filter=request.first_collection_filter,
                access_tier=user.access_tier_level,  # User's access tier for RLS
                user_id=user.user_id,  # User ID for private document access
                include_collection_context=request.include_collection_context,
                additional_context=combined_context,  # Include temp docs + memory context
                top_k=request.top_k,  # Per-query document count override
                folder_id=request.folder_id,  # Folder-scoped query
                include_subfolders=request.include_subfolders,
                language=request.language or "en",  # Language for response
                enhance_query=request.enhance_query,  # Per-query enhancement override
                organization_id=user.organization_id,  # Organization for multi-tenant isolation
                is_superadmin=user.is_superadmin,  # Superadmin can access all private docs
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
                        similarity_score=source.similarity_score if hasattr(source, 'similarity_score') else None,
                        snippet=source.snippet,
                        full_content=source.full_content if hasattr(source, 'full_content') else None,
                        collection=source.collection if hasattr(source, 'collection') else None,
                    ))

            # Cache the response for query_only requests
            if request.query_only:
                async with async_session_context() as db:
                    cache_key = f"{request.message}|{request.first_collection_filter or ''}"
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
                        "similarity_score": s.similarity_score,  # Original vector similarity (0-1)
                        "snippet": s.snippet,
                        "full_content": s.full_content,
                        "collection": s.collection,
                    }
                    for s in sources
                ] if sources else None

                await save_chat_messages(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response.content,
                    user_id=user.user_id,
                    user_email=user.email,
                    sources=source_data,
                    model_used=response.model,
                    tokens_used=response.tokens_used,
                    confidence_score=response.confidence_score,
                    confidence_level=response.confidence_level,
                )

                # PHASE 40/48: Extract and save important information to memory
                if getattr(app_settings, "ENABLE_AGENT_MEMORY", False):
                    try:
                        from backend.services.mem0_memory import get_memory_service, MemoryType

                        memory_service = await get_memory_service()

                        # Extract and save context from the conversation
                        # Only save if the response has high confidence and contains useful info
                        if response.confidence_score and response.confidence_score > 0.7:
                            # Save conversation context as memory
                            await memory_service.add(
                                user_id=user.user_id,
                                content=f"Q: {request.message[:200]}... A: {response.content[:500]}...",
                                memory_type=MemoryType.CONTEXT,
                                metadata={
                                    "session_id": str(session_id),
                                    "confidence": response.confidence_score,
                                    "collection": request.first_collection_filter,
                                },
                            )
                            logger.debug(
                                "Saved conversation to memory",
                                user_id=user.user_id,
                                session_id=str(session_id),
                            )
                    except Exception as e:
                        logger.warning("Failed to save memory", error=str(e))

                # Phase 66: Record query for personalization learning
                if getattr(app_settings, "ENABLE_USER_PERSONALIZATION", True):
                    try:
                        personalization_service = get_personalization_service()
                        # Extract topics from KG entities if available
                        detected_topics = None
                        if kg_context:
                            # Simple topic extraction from KG entities mentioned
                            detected_topics = [e.strip() for e in kg_context.split("- ")[1:] if ":" in e]
                            detected_topics = [t.split(":")[0] for t in detected_topics[:5]]

                        await personalization_service.record_query(
                            user_id=user.user_id,
                            query=request.message,
                            topics=detected_topics,
                        )
                    except Exception as e:
                        logger.debug("Failed to record query for personalization", error=str(e))

            # Convert context sufficiency result to API format
            context_sufficiency_info = None
            if response.context_sufficiency:
                context_sufficiency_info = ContextSufficiencyInfo(
                    is_sufficient=response.context_sufficiency.is_sufficient,
                    coverage_score=response.context_sufficiency.coverage_score,
                    has_conflicts=response.context_sufficiency.has_conflicts,
                    missing_aspects=response.context_sufficiency.missing_aspects,
                    confidence_level=response.context_sufficiency.confidence_level,
                    explanation=response.context_sufficiency.explanation,
                )

            return ChatResponse(
                session_id=session_id,
                message_id=message_id,
                content=response.content,
                sources=sources,
                created_at=datetime.now(),
                confidence_score=response.confidence_score,
                confidence_level=response.confidence_level,
                confidence_warning=response.confidence_warning if response.confidence_warning else None,
                suggested_questions=response.suggested_questions if response.suggested_questions else None,
                context_sufficiency=context_sufficiency_info,
            )

    except Exception as e:
        import traceback
        logger.error("Chat completion failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Chat completion failed: {str(e)}")


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
                agent_context = {"collection_filter": request.first_collection_filter}
                # Pass language through agent context
                # Prioritize: explicit agent_options.language > request.language > "auto"
                agent_language = request.language or "auto"
                if request.agent_options and request.agent_options.language:
                    agent_language = request.agent_options.language
                agent_context["language"] = agent_language
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
                        # Send planning event that frontend expects
                        yield f"data: {json.dumps({'type': 'planning', 'message': update.get('message', 'Planning...')})}\n\n"

                    elif update_type == "plan_created":
                        # Build steps array for frontend with expected format
                        step_count = update.get("step_count", 0)
                        summary = update.get("summary", "")
                        plan_id = update.get("plan_id", "")

                        # Create steps array in frontend-expected format
                        steps = []
                        for i in range(step_count):
                            steps.append({
                                "step_id": f"{plan_id}-step-{i}",
                                "step_number": i + 1,
                                "agent": "research" if i == 0 else "synthesis",
                                "task": f"Step {i + 1}",
                                "name": f"Step {i + 1}",
                                "status": "pending",
                                "estimated_cost_usd": 0.0,
                            })

                        yield f"data: {json.dumps({'type': 'plan_created', 'plan_id': plan_id, 'plan_summary': summary, 'steps': steps})}\n\n"

                    elif update_type == "estimating_cost":
                        yield f"data: {json.dumps({'type': 'planning', 'message': update.get('message', 'Estimating cost...')})}\n\n"

                    elif update_type == "cost_estimated":
                        yield f"data: {json.dumps({'type': 'cost_estimated', 'estimated_cost_usd': update.get('estimated_cost_usd')})}\n\n"

                    elif update_type == "approval_required":
                        yield f"data: {json.dumps({'type': 'approval_required', 'plan_id': update.get('plan_id'), 'estimated_cost': update.get('estimated_cost_usd'), 'threshold_usd': update.get('threshold_usd')})}\n\n"

                    elif update_type == "budget_exceeded":
                        yield f"data: {json.dumps({'type': 'error', 'error': update.get('message', 'Budget exceeded')})}\n\n"

                    elif update_type == "step_started":
                        # Send agent_step event with step details
                        step_index = update.get("step_index", 0)
                        yield f"data: {json.dumps({'type': 'agent_step', 'step_index': step_index, 'step': {'name': update.get('step_name', 'Processing'), 'agent': update.get('agent_type', 'research'), 'status': 'in_progress'}})}\n\n"

                    elif update_type == "step_completed":
                        # Send step_completed event with full output for detailed view
                        step_index = update.get("step_index", 0)
                        step_data = {
                            'type': 'step_completed',
                            'step_index': step_index,
                            'status': 'completed',
                            'step_name': update.get('step_name'),
                            'agent_type': update.get('agent_type'),
                            'output_preview': update.get('output_preview'),
                            'full_output': update.get('full_output'),  # Full output for expansion
                        }
                        yield f"data: {json.dumps(step_data)}\n\n"

                    elif update_type == "thinking":
                        # Forward thinking/reasoning content
                        yield f"data: {json.dumps({'type': 'thinking', 'content': update.get('content', '')})}\n\n"

                    elif update_type == "content":
                        yield f"data: {json.dumps({'type': 'content', 'data': update.get('data', '')})}\n\n"

                    elif update_type == "sources":
                        sources_data = update.get("data", [])
                        yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                    elif update_type == "plan_completed":
                        output = update.get("output", "")
                        # Send execution_complete first, then final content
                        yield f"data: {json.dumps({'type': 'execution_complete', 'result': output, 'total_cost_usd': update.get('total_cost_usd')})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'total_cost_usd': update.get('total_cost_usd')})}\n\n"

                    elif update_type == "error":
                        yield f"data: {json.dumps({'type': 'error', 'error': update.get('error', 'Unknown error')})}\n\n"

        except Exception as e:
            logger.error("Agent streaming error", error=str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    async def generate_general_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response using general chat (no RAG)."""
        from backend.services.general_chat import get_general_chat_service

        session_id = request.session_id or uuid4()

        # Send session info first
        yield f"data: {json.dumps({'type': 'session', 'session_id': str(session_id)})}\n\n"

        # Check if restrict_to_documents is enabled - return helpful message without using LLM
        if request.restrict_to_documents:
            restricted_response = (
                "I'm currently restricted from using my pre-trained knowledge. "
                "In this mode, I can only answer questions using information from your uploaded documents.\n\n"
                "To get an answer, please either:\n"
                "1. **Switch to Document Mode** - I'll search your documents for relevant information\n"
                "2. **Disable 'No AI Knowledge'** toggle - Allow me to use my general knowledge\n\n"
                "This restriction ensures all responses are strictly based on your document content."
            )

            yield f"data: {json.dumps({'type': 'content', 'data': restricted_response})}\n\n"

            # Save the interaction if not query_only
            if not request.query_only:
                await save_chat_messages(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=restricted_response,
                    user_id=user.user_id,
                    user_email=user.email,
                    model_used="restricted-mode",
                )

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        try:
            general_service = get_general_chat_service()
            response = await general_service.query(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
                language=request.language or "en",
            )
            # Send entire response as single content chunk
            yield f"data: {json.dumps({'type': 'content', 'data': response.content})}\n\n"

            # Save messages to database
            if not request.query_only:
                await save_chat_messages(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response.content,
                    user_id=user.user_id,
                    user_email=user.email,
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

        # Get temp document context if available
        temp_context = None
        if request.temp_session_id:
            from backend.services.temp_documents import get_temp_document_service
            temp_service = get_temp_document_service()
            temp_session = await temp_service.get_session(request.temp_session_id)
            if temp_session and temp_session.user_id == user.user_id:
                temp_context = await temp_service.get_context(
                    session_id=request.temp_session_id,
                    query=request.message,
                    max_tokens=50000,
                )

        # Accumulate content and sources for saving
        accumulated_content = []
        accumulated_sources = []

        try:
            # Stream from RAG service
            async for chunk in rag_service.query_stream(
                question=request.message,
                session_id=str(session_id) if not request.query_only else None,
                collection_filter=request.first_collection_filter,
                access_tier=user.access_tier_level,  # User's access tier for RLS
                user_id=user.user_id,  # User ID for private document access
                include_collection_context=request.include_collection_context,
                top_k=request.top_k,  # Per-query document count override
                folder_id=request.folder_id,  # Folder-scoped query
                include_subfolders=request.include_subfolders,
                language=request.language or "en",  # Language for response
                enhance_query=request.enhance_query,  # Per-query enhancement override
                organization_id=user.organization_id,  # Organization for multi-tenant isolation
                is_superadmin=user.is_superadmin,  # Superadmin can access all private docs
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
                            user_id=user.user_id,
                            user_email=user.email,
                            sources=accumulated_sources if accumulated_sources else None,
                            model_used=rag_service.config.chat_model,
                        )
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                elif chunk.type == "confidence":
                    # Send confidence information
                    yield f"data: {json.dumps({'type': 'confidence', 'score': chunk.data.get('score'), 'level': chunk.data.get('level')})}\n\n"
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
            # Get valid database user ID
            db_user_id = await get_db_user_id(db, user.user_id, user.email)
            if not db_user_id:
                # No valid user found, return empty list
                return SessionListResponse(sessions=[], total=0, page=page, page_size=page_size)

            # Get total count
            count_query = select(func.count(ChatSessionModel.id)).where(
                ChatSessionModel.user_id == db_user_id,
                ChatSessionModel.is_active == True,
            )
            result = await db.execute(count_query)
            total = result.scalar() or 0

            # Get sessions with message count
            offset = (page - 1) * page_size
            sessions_query = (
                select(ChatSessionModel)
                .where(
                    ChatSessionModel.user_id == db_user_id,
                    ChatSessionModel.is_active == True,
                )
                .order_by(ChatSessionModel.updated_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            result = await db.execute(sessions_query)
            sessions = result.scalars().all()

            # Get message counts for all sessions in a single query (avoid N+1)
            session_ids = [s.id for s in sessions]
            msg_counts_dict = {}
            if session_ids:
                msg_counts_query = (
                    select(
                        ChatMessageModel.session_id,
                        func.count(ChatMessageModel.id).label("count")
                    )
                    .where(ChatMessageModel.session_id.in_(session_ids))
                    .group_by(ChatMessageModel.session_id)
                )
                msg_counts_result = await db.execute(msg_counts_query)
                msg_counts_dict = {row.session_id: row.count for row in msg_counts_result}

            # Build response with pre-fetched message counts
            session_responses = []
            for session in sessions:
                session_responses.append(SessionResponse(
                    id=session.id,
                    title=session.title or "New Conversation",
                    message_count=msg_counts_dict.get(session.id, 0),
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                ))

            return SessionListResponse(
                sessions=session_responses,
                total=total,
            )

    except Exception as e:
        logger.error("Failed to list sessions", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list sessions: {str(e)}")


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
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

            # Convert messages to response format
            messages = []
            sources = {}
            for index, msg in enumerate(sorted(session.messages, key=lambda m: m.created_at)):
                messages.append(ChatMessage(
                    role=msg.role.value,
                    content=msg.content,
                    confidence_score=msg.confidence_score if hasattr(msg, 'confidence_score') else None,
                    confidence_level=msg.confidence_level if hasattr(msg, 'confidence_level') else None,
                ))
                # Include sources if available (keyed by message index for frontend)
                if msg.source_chunks:
                    # Handle both JSON string and parsed list
                    import json
                    chunks = msg.source_chunks
                    if isinstance(chunks, str):
                        try:
                            chunks = json.loads(chunks)
                        except json.JSONDecodeError:
                            chunks = []
                    sources[str(index)] = chunks

            return SessionMessagesResponse(
                session_id=session_id,
                messages=messages,
                sources=sources,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get session: {str(e)}")


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
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

            # Delete the session (cascade will delete messages)
            await db.delete(session)
            await db.commit()

            return {"message": "Session deleted successfully", "session_id": str(session_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete session: {str(e)}")


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
            status_code=status.HTTP_400_BAD_REQUEST,
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete all sessions: {str(e)}")


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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to bulk delete sessions: {str(e)}")


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
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

            # Update the title
            session.title = title
            await db.commit()

            return {"message": "Title updated", "session_id": str(session_id), "title": title}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update session title", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update title: {str(e)}")


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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create session: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    user: AuthenticatedUser,
    message_id: UUID,
    rating: int = Query(..., ge=1, le=5),
    comment: Optional[str] = None,
    session_id: Optional[UUID] = None,
    mode: Optional[str] = None,
):
    """
    Submit feedback for a chat response.

    Stores feedback in database and updates agent trajectories for agent mode responses.
    Used to improve RAG quality and agent prompts over time.
    """
    logger.info(
        "Submitting chat feedback",
        message_id=str(message_id),
        rating=rating,
        comment=comment,
        mode=mode,
    )

    try:
        async with async_session_context() as db:
            # Get valid user UUID from the database
            db_user_id = await get_db_user_id(db, user.user_id, user.email)
            if not db_user_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not find user in database")

            # Check if there's an agent trajectory for this message (for agent mode)
            trajectory_id = None
            if mode == "agent":
                # Look for trajectory with this message_id in metadata
                # Agent trajectories store message_id in their metadata
                trajectory_query = select(AgentTrajectory).where(
                    AgentTrajectory.trajectory_steps.contains({"message_id": str(message_id)})
                )
                result = await db.execute(trajectory_query)
                trajectory = result.scalar_one_or_none()

                if trajectory:
                    trajectory_id = trajectory.id
                    # Update the trajectory with user feedback
                    trajectory.user_rating = rating
                    trajectory.user_feedback = comment
                    logger.info(
                        "Updated agent trajectory with user feedback",
                        trajectory_id=str(trajectory_id),
                        rating=rating,
                    )

            # Create feedback record
            feedback = ChatFeedback(
                message_id=str(message_id),
                session_id=session_id,
                user_id=db_user_id,
                rating=rating,
                comment=comment,
                mode=mode,
                trajectory_id=trajectory_id,
            )
            db.add(feedback)
            await db.commit()

            logger.info(
                "Chat feedback stored",
                feedback_id=str(feedback.id),
                message_id=str(message_id),
                rating=rating,
                has_trajectory=trajectory_id is not None,
            )

            # Phase 66: Record feedback for personalization learning
            from backend.core.config import settings as app_settings
            if getattr(app_settings, "ENABLE_USER_PERSONALIZATION", True):
                try:
                    personalization_service = get_personalization_service()

                    # Get the original message to extract query and response details
                    message_query = select(ChatMessageModel).where(
                        ChatMessageModel.id == message_id
                    )
                    msg_result = await db.execute(message_query)
                    chat_message = msg_result.scalar_one_or_none()

                    if chat_message:
                        # Determine response format from content structure
                        response_format = "prose"
                        if chat_message.content.startswith(("- ", "* ", "1.")):
                            response_format = "bullets"
                        elif "\n## " in chat_message.content or "\n### " in chat_message.content:
                            response_format = "structured"

                        await personalization_service.record_feedback(
                            user_id=user.user_id,
                            query=chat_message.content[:500] if chat_message.role == MessageRole.USER else "",
                            response_id=str(message_id),
                            rating=rating,
                            feedback_text=comment,
                            response_length=len(chat_message.content),
                            response_format=response_format,
                        )
                        logger.debug(
                            "Recorded feedback for personalization",
                            user_id=user.user_id[:8] if user.user_id else "unknown",
                            rating=rating,
                        )
                except Exception as e:
                    logger.debug("Failed to record personalization feedback", error=str(e))

            return {
                "message": "Feedback submitted",
                "message_id": str(message_id),
                "feedback_id": str(feedback.id),
                "trajectory_updated": trajectory_id is not None,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to store feedback", error=str(e), message_id=str(message_id))
        # Still return success to not block the user experience
        return {"message": "Feedback submitted", "message_id": str(message_id)}


# =============================================================================
# Session LLM Override Endpoints
# =============================================================================

class SessionLLMOverrideRequest(BaseModel):
    """Request to set session LLM override."""
    provider_id: str = Field(..., description="LLM provider ID to use for this session")
    model_override: Optional[str] = Field(None, description="Model to use (overrides provider default)")
    temperature_override: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature override")
    temperature_manual_override: Optional[bool] = Field(None, description="Explicit flag: True if user manually set temperature")


class SessionLLMOverrideResponse(BaseModel):
    """Session LLM override response."""
    session_id: str
    provider_id: str
    provider_name: Optional[str]
    provider_type: str
    model: str
    temperature: Optional[float]
    temperature_manual_override: Optional[bool] = None  # Explicit flag for manual override


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
                    temperature_manual_override=override.temperature_manual_override if hasattr(override, 'temperature_manual_override') else None,
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
            temperature_manual_override=False,  # Default config is not a manual override
        )

    except Exception as e:
        logger.error("Failed to get session LLM config", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get session LLM config: {str(e)}")


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
                    status_code=status.HTTP_400_BAD_REQUEST,
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
                # Set manual override flag if temperature is being set
                if override_data.temperature_manual_override is not None:
                    existing.temperature_manual_override = override_data.temperature_manual_override
                elif override_data.temperature_override is not None:
                    # Auto-set flag if temperature is explicitly provided
                    existing.temperature_manual_override = True
                await db.commit()
            else:
                # Create new
                # Determine if temperature is manually overridden
                temp_manual = override_data.temperature_manual_override
                if temp_manual is None and override_data.temperature_override is not None:
                    temp_manual = True  # Auto-set flag if temperature is explicitly provided

                new_override = ChatSessionLLMOverride(
                    session_id=session_id,
                    provider_id=override_data.provider_id,
                    model_override=override_data.model_override,
                    temperature_override=override_data.temperature_override,
                    temperature_manual_override=temp_manual,
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
                temperature_manual_override=override_data.temperature_manual_override or (override_data.temperature_override is not None),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to set session LLM override", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to set session LLM override: {str(e)}")


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
                    status_code=status.HTTP_404_NOT_FOUND,
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete session LLM override: {str(e)}")


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
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

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
                                doc_id = source.get("document_id", "unknown")
                                doc_name = source.get("document_name") or f"Document {doc_id[:8]}"
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Export failed: {str(e)}")


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
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get token count: {str(e)}")


# =============================================================================
# @Mention Autocomplete Endpoints
# =============================================================================

class MentionSuggestion(BaseModel):
    """A single @mention autocomplete suggestion."""
    type: str  # folder, document, tag, type, recent
    value: str  # The full mention value to insert
    display: str  # Display text
    description: Optional[str] = None


class MentionAutocompleteResponse(BaseModel):
    """Response for mention autocomplete."""
    suggestions: List[MentionSuggestion]


class ParsedMentionInfo(BaseModel):
    """Information about parsed @mentions in a query."""
    clean_query: str
    has_filters: bool
    folder_count: int
    document_count: int
    tag_count: int
    date_filter: Optional[str] = None
    mentions: List[dict]


@router.get("/mentions/autocomplete", response_model=MentionAutocompleteResponse)
async def get_mention_autocomplete(
    partial: str = Query(..., min_length=1, description="Partial @mention text"),
    user: AuthenticatedUser = None,
):
    """
    Get autocomplete suggestions for @mentions in chat.

    Supports:
    - @folder:Name → Filter to folder
    - @document:file.pdf or @doc:file.pdf → Filter to document
    - @tag:collection → Filter by tag/collection
    - @recent:7d → Filter to recent documents (d=days, w=weeks, m=months)
    - @all → Search all documents

    Example: GET /mentions/autocomplete?partial=@folder:Mar
    Returns suggestions for folders starting with "Mar"
    """
    from backend.services.mention_parser import get_mention_autocomplete
    from backend.db.database import async_session_context

    autocomplete = get_mention_autocomplete()

    try:
        async with async_session_context() as db:
            suggestions = await autocomplete.get_suggestions(
                partial=partial,
                session=db,
                organization_id=user.organization_id if user else None,
                user_id=user.user_id if user else None,
                limit=10,
            )

            return MentionAutocompleteResponse(
                suggestions=[
                    MentionSuggestion(
                        type=s.type,
                        value=s.value,
                        display=s.display,
                        description=s.description,
                    )
                    for s in suggestions
                ]
            )

    except Exception as e:
        logger.error("Mention autocomplete failed", error=str(e))
        return MentionAutocompleteResponse(suggestions=[])


@router.post("/mentions/parse", response_model=ParsedMentionInfo)
async def parse_mentions(
    query: str = Query(..., min_length=1, description="Query with @mentions"),
    resolve: bool = Query(default=True, description="Resolve names to IDs"),
    user: AuthenticatedUser = None,
):
    """
    Parse @mentions from a query string.

    Returns the clean query (without @mentions) and extracted filter information.

    Example: POST /mentions/parse?query=What are sales? @folder:Marketing @recent:7d
    Returns:
    {
        "clean_query": "What are sales?",
        "has_filters": true,
        "folder_count": 1,
        "document_count": 0,
        "tag_count": 0,
        "date_filter": "2026-01-03T...",
        "mentions": [...]
    }
    """
    from backend.services.mention_parser import get_mention_parser
    from backend.db.database import async_session_context

    parser = get_mention_parser()
    parsed = parser.parse(query)

    # Resolve mentions if requested
    if resolve and (parsed.folder_names or parsed.document_names):
        try:
            async with async_session_context() as db:
                parsed = await parser.resolve_mentions(
                    parsed,
                    session=db,
                    organization_id=user.organization_id if user else None,
                    user_id=user.user_id if user else None,
                )
        except Exception as e:
            logger.warning("Failed to resolve mentions", error=str(e))

    return ParsedMentionInfo(
        clean_query=parsed.clean_query,
        has_filters=parsed.has_filters(),
        folder_count=len(parsed.folder_ids),
        document_count=len(parsed.document_ids),
        tag_count=len(parsed.collection_filters),
        date_filter=parsed.date_filter.isoformat() if parsed.date_filter else None,
        mentions=[
            {
                "type": m.mention_type,
                "value": m.value,
                "original": m.original,
                "resolved_id": m.resolved_id,
                "resolved_name": m.resolved_name,
            }
            for m in parsed.mentions
        ],
    )


# =============================================================================
# Phase 65: Advanced RAG Enhancement Endpoints
# =============================================================================


class LTRFeedbackRequest(BaseModel):
    """Request to record user click feedback for LTR training."""
    query: str = Field(..., description="The search query")
    doc_id: str = Field(..., description="ID of the clicked document/chunk")
    rank: int = Field(..., description="Original rank position (0-indexed)")
    clicked: bool = Field(True, description="Whether the result was clicked")
    dwell_time_seconds: float = Field(0.0, description="Time spent on the result")


class LTRFeedbackResponse(BaseModel):
    """Response from LTR feedback recording."""
    success: bool
    message: str
    total_feedback_samples: int


class LTRTrainRequest(BaseModel):
    """Request to trigger LTR model training."""
    force: bool = Field(False, description="Force training even if sample threshold not met")


class LTRTrainResponse(BaseModel):
    """Response from LTR training."""
    success: bool
    message: str
    metrics: Optional[dict] = None


class Phase65StatsResponse(BaseModel):
    """Statistics for all Phase 65 features."""
    initialized: bool
    config: dict
    semantic_cache: Optional[dict] = None
    ltr: Optional[dict] = None
    spell_correction: Optional[dict] = None
    gpu_search: Optional[dict] = None


class SpellCorrectionRequest(BaseModel):
    """Request to test spell correction."""
    query: str = Field(..., description="Query to spell-correct")


class SpellCorrectionResponse(BaseModel):
    """Response from spell correction."""
    original: str
    corrected: str
    is_corrected: bool
    corrections: list
    confidence: float


@router.post("/ltr/feedback", response_model=LTRFeedbackResponse)
async def record_ltr_feedback(
    request: LTRFeedbackRequest,
    user: AuthenticatedUser = None,
):
    """
    Record user click feedback for Learning-to-Rank model training.

    Phase 65: This endpoint collects implicit feedback (clicks, dwell time)
    to train an XGBoost-based ranker that learns to predict user preferences.

    Call this when a user clicks on a search result.
    """
    try:
        from backend.services.phase65_integration import get_phase65_pipeline

        pipeline = await get_phase65_pipeline()

        await pipeline.record_click_feedback(
            query=request.query,
            doc_id=request.doc_id,
            rank=request.rank,
            clicked=request.clicked,
            dwell_time_seconds=request.dwell_time_seconds,
        )

        stats = pipeline.get_stats()
        ltr_samples = stats.get("ltr", {}).get("feedback_samples", 0)

        return LTRFeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            total_feedback_samples=ltr_samples,
        )

    except Exception as e:
        logger.error("Failed to record LTR feedback", error=str(e))
        return LTRFeedbackResponse(
            success=False,
            message=f"Failed to record feedback: {str(e)}",
            total_feedback_samples=0,
        )


@router.post("/ltr/train", response_model=LTRTrainResponse)
async def train_ltr_model(
    request: LTRTrainRequest,
    user: AuthenticatedUser = None,
):
    """
    Trigger Learning-to-Rank model training.

    Phase 65: Trains an XGBoost ranker on collected click feedback.
    Requires minimum 100 feedback samples (or force=True to override).

    The trained model improves search result ordering based on user behavior.
    """
    try:
        from backend.services.phase65_integration import get_phase65_pipeline

        pipeline = await get_phase65_pipeline()

        result = await pipeline.train_ltr_model(force=request.force)

        if result.get("status") == "ltr_not_enabled":
            return LTRTrainResponse(
                success=False,
                message="LTR is not enabled in configuration",
                metrics=None,
            )

        return LTRTrainResponse(
            success=result.get("status") == "trained",
            message=result.get("message", "Training complete"),
            metrics=result.get("metrics"),
        )

    except Exception as e:
        logger.error("Failed to train LTR model", error=str(e))
        return LTRTrainResponse(
            success=False,
            message=f"Training failed: {str(e)}",
            metrics=None,
        )


@router.get("/phase65/stats", response_model=Phase65StatsResponse)
async def get_phase65_stats(
    user: AuthenticatedUser = None,
):
    """
    Get statistics for all Phase 65 features.

    Returns status and metrics for:
    - Semantic Cache (hit rate, entries)
    - Learning-to-Rank (trained status, feedback samples)
    - Spell Correction (vocabulary size)
    - GPU Search (backend, index size)
    """
    try:
        from backend.services.phase65_integration import get_phase65_pipeline

        pipeline = await get_phase65_pipeline()
        stats = pipeline.get_stats()

        return Phase65StatsResponse(
            initialized=stats.get("initialized", False),
            config=stats.get("config", {}),
            semantic_cache=stats.get("semantic_cache"),
            ltr=stats.get("ltr"),
            spell_correction=stats.get("spell_correction"),
            gpu_search=stats.get("gpu_search"),
        )

    except Exception as e:
        logger.error("Failed to get Phase 65 stats", error=str(e))
        return Phase65StatsResponse(
            initialized=False,
            config={"error": str(e)},
        )


@router.post("/spell-correct", response_model=SpellCorrectionResponse)
async def spell_correct_query(
    request: SpellCorrectionRequest,
    user: AuthenticatedUser = None,
):
    """
    Test spell correction on a query.

    Phase 65: Uses BK-tree based spell correction with O(log n) lookup.
    Supports domain vocabulary learned from the document corpus.
    """
    try:
        from backend.services.spell_correction import get_spell_corrector

        corrector = get_spell_corrector()
        result = await corrector.correct(request.query)

        return SpellCorrectionResponse(
            original=result.original,
            corrected=result.corrected,
            is_corrected=result.is_corrected,
            corrections=[(c[0], c[1], c[2]) for c in result.corrections],
            confidence=result.confidence,
        )

    except Exception as e:
        logger.error("Spell correction failed", error=str(e))
        return SpellCorrectionResponse(
            original=request.query,
            corrected=request.query,
            is_corrected=False,
            corrections=[],
            confidence=0.0,
        )
