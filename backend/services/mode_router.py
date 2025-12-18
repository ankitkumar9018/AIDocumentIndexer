"""
AIDocumentIndexer - Mode Router Service
========================================

Routes requests to either Agent mode (multi-agent orchestration),
Chat mode (RAG-powered), or General mode (direct LLM) based on:
1. Explicit user specification
2. User preferences
3. Auto-detection of request complexity

Features:
- Complexity detection using keywords, patterns, and heuristics
- User-configurable default mode
- Toggle to disable agent mode entirely
- General chat mode for non-document questions
- Smart fallback to general chat when no documents found
- Per-session mode override
"""

import re
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import ExecutionModePreference, ExecutionMode

logger = structlog.get_logger(__name__)


# =============================================================================
# Complexity Detection
# =============================================================================

class ComplexityLevel(str, Enum):
    """Request complexity levels."""
    SIMPLE = "simple"       # Direct answer, single step
    MODERATE = "moderate"   # Needs some research/context
    COMPLEX = "complex"     # Multi-step, needs orchestration


class ComplexityDetector:
    """
    Detects request complexity to determine execution mode.

    Uses heuristics including:
    - Keywords indicating complex tasks
    - Request length and structure
    - Multiple questions/parts
    - References to multiple documents
    """

    # Keywords that indicate complex tasks needing agent orchestration
    COMPLEX_KEYWORDS = [
        # Generation tasks
        "generate", "create", "write", "draft", "compose", "produce",
        "build", "make", "develop", "design",
        # Analysis tasks
        "analyze", "analyse", "evaluate", "assess", "review", "examine",
        "investigate", "study", "inspect",
        # Comparison tasks
        "compare", "contrast", "differentiate", "versus", "vs",
        # Aggregation tasks
        "summarize", "summarise", "combine", "aggregate", "consolidate",
        "compile", "synthesize", "synthesise",
        # Research tasks
        "research", "find all", "search for", "look up", "gather",
        "collect information", "investigate",
        # Document tasks
        "document", "report", "presentation", "powerpoint", "pptx",
        "word doc", "docx", "pdf", "spreadsheet", "excel",
        # Multi-step indicators
        "step by step", "step-by-step", "first then", "multiple",
        "several", "all of the", "each of the", "list all",
        # Planning tasks
        "plan", "outline", "roadmap", "strategy", "blueprint",
    ]

    # Patterns that indicate simple queries
    SIMPLE_PATTERNS = [
        r"^(what|who|when|where|why|how)\s+(is|are|was|were|do|does|did)\s+\w+\s*\??$",
        r"^(define|explain)\s+\w+\s*\??$",
        r"^(yes|no|true|false)\s*\??$",
        r"^hello|^hi|^hey|^greetings",
        r"^thanks|^thank you",
        r"^help$|^help\s*\?$",
    ]

    # Minimum length threshold for complex requests
    COMPLEX_LENGTH_THRESHOLD = 200

    # Multiple question indicators
    MULTI_QUESTION_PATTERNS = [
        r"\d+\.\s+",           # Numbered list: "1. first 2. second"
        r"(first|second|third|then|also|additionally)",
        r"\?\s+[A-Z]",         # Multiple sentences with questions
        r"and\s+(also|then|)",  # Chained requests
    ]

    def __init__(self):
        """Initialize detector with compiled patterns."""
        self._simple_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS
        ]
        self._multi_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.MULTI_QUESTION_PATTERNS
        ]

    def detect(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplexityLevel:
        """
        Detect complexity level of a request.

        Args:
            request: User request text
            context: Optional context (session history, etc.)

        Returns:
            ComplexityLevel enum value
        """
        request_lower = request.lower().strip()
        context = context or {}

        # Check for simple patterns first
        for pattern in self._simple_patterns:
            if pattern.match(request_lower):
                return ComplexityLevel.SIMPLE

        # Score-based detection
        complexity_score = 0.0

        # Check complex keywords
        keyword_matches = sum(
            1 for kw in self.COMPLEX_KEYWORDS if kw in request_lower
        )
        complexity_score += keyword_matches * 0.15

        # Check request length
        if len(request) > self.COMPLEX_LENGTH_THRESHOLD:
            complexity_score += 0.2
        elif len(request) > self.COMPLEX_LENGTH_THRESHOLD * 2:
            complexity_score += 0.4

        # Check for multiple questions/parts
        for pattern in self._multi_patterns:
            if pattern.search(request):
                complexity_score += 0.15

        # Count question marks (multiple questions)
        question_count = request.count("?")
        if question_count > 1:
            complexity_score += 0.1 * min(question_count - 1, 3)

        # Check context for multiple document references
        if context.get("document_count", 0) > 2:
            complexity_score += 0.2

        # Check if session has prior complex tasks
        if context.get("has_prior_complex_tasks", False):
            complexity_score += 0.1

        # Determine level based on score
        if complexity_score >= 0.6:
            return ComplexityLevel.COMPLEX
        elif complexity_score >= 0.3:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

    def should_use_agents(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if request should use agent orchestration.

        Args:
            request: User request text
            context: Optional context

        Returns:
            True if agents should be used
        """
        level = self.detect(request, context)
        # Use agents for complex and moderate requests
        return level in (ComplexityLevel.COMPLEX, ComplexityLevel.MODERATE)


# =============================================================================
# Mode Router
# =============================================================================

class ModeRouter:
    """
    Routes requests between Agent mode, Chat mode, and General mode.

    Responsibilities:
    - Get user preferences from database
    - Auto-detect complexity when enabled
    - Route to appropriate service
    - Handle mode toggles
    - Support general chat mode for non-document questions
    """

    def __init__(
        self,
        db: AsyncSession,
        rag_service: Any = None,  # RAGService instance
        orchestrator: Any = None,  # AgentOrchestrator instance
        general_chat_service: Any = None,  # GeneralChatService instance
    ):
        """
        Initialize router.

        Args:
            db: Database session
            rag_service: RAGService for chat mode (RAG-powered)
            orchestrator: AgentOrchestrator for agent mode
            general_chat_service: GeneralChatService for general LLM chat
        """
        self.db = db
        self.rag_service = rag_service
        self.orchestrator = orchestrator
        self.general_chat_service = general_chat_service
        self.detector = ComplexityDetector()

    def set_services(
        self,
        rag_service: Any = None,
        orchestrator: Any = None,
        general_chat_service: Any = None,
    ) -> None:
        """Set service instances (for dependency injection)."""
        if rag_service:
            self.rag_service = rag_service
        if orchestrator:
            self.orchestrator = orchestrator
        if general_chat_service:
            self.general_chat_service = general_chat_service

    async def get_user_preferences(
        self,
        user_id: str
    ) -> ExecutionModePreference:
        """
        Get user's execution mode preferences.

        Creates default preferences if none exist.

        Args:
            user_id: User UUID

        Returns:
            ExecutionModePreference record
        """
        user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

        result = await self.db.execute(
            select(ExecutionModePreference)
            .where(ExecutionModePreference.user_id == user_uuid)
        )
        prefs = result.scalar_one_or_none()

        if not prefs:
            # Create default preferences
            prefs = ExecutionModePreference(
                user_id=user_uuid,
                default_mode=ExecutionMode.AGENT.value,
                agent_mode_enabled=True,
                auto_detect_complexity=True,
                show_cost_estimation=True,
                require_approval_above_usd=1.0,
                general_chat_enabled=True,
                fallback_to_general=True,
            )
            self.db.add(prefs)
            await self.db.commit()
            await self.db.refresh(prefs)

            logger.info(
                "Created default mode preferences",
                user_id=str(user_id),
            )

        return prefs

    async def update_preferences(
        self,
        user_id: str,
        default_mode: Optional[str] = None,
        agent_mode_enabled: Optional[bool] = None,
        auto_detect_complexity: Optional[bool] = None,
        show_cost_estimation: Optional[bool] = None,
        require_approval_above_usd: Optional[float] = None,
        general_chat_enabled: Optional[bool] = None,
        fallback_to_general: Optional[bool] = None,
    ) -> ExecutionModePreference:
        """
        Update user's execution mode preferences.

        Args:
            user_id: User UUID
            default_mode: Default mode (agent/chat/general)
            agent_mode_enabled: Enable/disable agent mode
            auto_detect_complexity: Auto-detect complexity
            show_cost_estimation: Show cost before execution
            require_approval_above_usd: Cost threshold for approval
            general_chat_enabled: Enable/disable general chat mode
            fallback_to_general: Auto-fallback to general when no docs found

        Returns:
            Updated preferences
        """
        prefs = await self.get_user_preferences(user_id)

        if default_mode is not None:
            prefs.default_mode = default_mode
        if agent_mode_enabled is not None:
            prefs.agent_mode_enabled = agent_mode_enabled
        if auto_detect_complexity is not None:
            prefs.auto_detect_complexity = auto_detect_complexity
        if show_cost_estimation is not None:
            prefs.show_cost_estimation = show_cost_estimation
        if require_approval_above_usd is not None:
            prefs.require_approval_above_usd = require_approval_above_usd
        if general_chat_enabled is not None:
            prefs.general_chat_enabled = general_chat_enabled
        if fallback_to_general is not None:
            prefs.fallback_to_general = fallback_to_general

        await self.db.commit()
        await self.db.refresh(prefs)

        logger.info(
            "Updated mode preferences",
            user_id=str(user_id),
            default_mode=prefs.default_mode,
            agent_enabled=prefs.agent_mode_enabled,
            general_enabled=prefs.general_chat_enabled,
        )

        return prefs

    async def toggle_agent_mode(
        self,
        user_id: str,
        enabled: Optional[bool] = None
    ) -> bool:
        """
        Toggle agent mode on/off for user.

        Args:
            user_id: User UUID
            enabled: Explicit enable/disable, or None to toggle

        Returns:
            New agent_mode_enabled state
        """
        prefs = await self.get_user_preferences(user_id)

        if enabled is None:
            # Toggle current state
            prefs.agent_mode_enabled = not prefs.agent_mode_enabled
        else:
            prefs.agent_mode_enabled = enabled

        await self.db.commit()

        logger.info(
            "Toggled agent mode",
            user_id=str(user_id),
            enabled=prefs.agent_mode_enabled,
        )

        return prefs.agent_mode_enabled

    def determine_mode(
        self,
        request: str,
        prefs: ExecutionModePreference,
        explicit_mode: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionMode:
        """
        Determine which execution mode to use.

        Priority:
        1. Explicit mode parameter
        2. Agent mode disabled → use chat or general
        3. Auto-detect if enabled
        4. User's default mode

        Args:
            request: User request text
            prefs: User preferences
            explicit_mode: Explicitly specified mode
            context: Request context

        Returns:
            ExecutionMode to use
        """
        # 1. Explicit mode takes precedence
        if explicit_mode:
            mode = explicit_mode.lower()
            if mode in ("agent", "agents"):
                return ExecutionMode.AGENT
            elif mode in ("chat", "rag", "documents"):
                return ExecutionMode.CHAT
            elif mode in ("general", "llm", "direct"):
                return ExecutionMode.GENERAL

        # 2. If agent mode disabled, use chat or general based on default
        if not prefs.agent_mode_enabled:
            # If default is general, use general; otherwise use chat
            if prefs.default_mode == ExecutionMode.GENERAL.value:
                return ExecutionMode.GENERAL
            return ExecutionMode.CHAT

        # 3. Auto-detect complexity if enabled
        if prefs.auto_detect_complexity:
            if self.detector.should_use_agents(request, context):
                return ExecutionMode.AGENT
            else:
                return ExecutionMode.CHAT

        # 4. Fall back to user's default mode
        try:
            return ExecutionMode(prefs.default_mode)
        except ValueError:
            return ExecutionMode.CHAT

    async def route_request(
        self,
        request: str,
        session_id: str,
        user_id: str,
        explicit_mode: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Route request to appropriate service.

        Args:
            request: User request text
            session_id: Session UUID
            user_id: User UUID
            explicit_mode: Explicit mode override
            context: Additional context
            **kwargs: Additional parameters for services

        Yields:
            Response chunks from the appropriate service
        """
        # Get user preferences
        prefs = await self.get_user_preferences(user_id)

        # Determine execution mode
        mode = self.determine_mode(
            request=request,
            prefs=prefs,
            explicit_mode=explicit_mode,
            context=context,
        )

        logger.info(
            "Routing request",
            mode=mode.value,
            session_id=session_id,
            user_id=str(user_id),
            explicit_mode=explicit_mode,
            auto_detect=prefs.auto_detect_complexity,
            general_enabled=prefs.general_chat_enabled,
        )

        # Route to appropriate service
        if mode == ExecutionMode.GENERAL:
            async for chunk in self._route_to_general(
                request=request,
                session_id=session_id,
                user_id=user_id,
                **kwargs
            ):
                yield chunk
        elif mode == ExecutionMode.CHAT:
            async for chunk in self._route_to_chat(
                request=request,
                session_id=session_id,
                user_id=user_id,
                prefs=prefs,
                **kwargs
            ):
                yield chunk
        else:
            async for chunk in self._route_to_agents(
                request=request,
                session_id=session_id,
                user_id=user_id,
                prefs=prefs,
                **kwargs
            ):
                yield chunk

    async def _route_to_general(
        self,
        request: str,
        session_id: str,
        user_id: str,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Route to GeneralChatService for non-RAG questions."""
        if not self.general_chat_service:
            # Try to get the service dynamically
            from backend.services.general_chat import get_general_chat_service
            self.general_chat_service = get_general_chat_service()

        if not self.general_chat_service:
            yield {
                "type": "error",
                "error": "General chat service not available",
            }
            return

        # Yield mode indicator
        yield {
            "type": "mode_selected",
            "mode": "general",
            "message": "Using general knowledge (no document search)...",
        }

        # Delegate to GeneralChatService
        try:
            response = await self.general_chat_service.query(
                question=request,
                session_id=session_id,
                user_id=user_id,
            )
            yield {
                "type": "chat_response",
                "mode": "general",
                "data": {
                    "content": response.content,
                    "sources": [],
                    "is_general_response": True,
                    "model": response.model,
                    "processing_time_ms": response.processing_time_ms,
                },
                "complete": True,
            }
        except Exception as e:
            logger.error(
                "General chat failed",
                error=str(e),
                session_id=session_id,
            )
            yield {
                "type": "error",
                "mode": "general",
                "error": str(e),
            }

    async def _route_to_chat(
        self,
        request: str,
        session_id: str,
        user_id: str,
        prefs: Optional[ExecutionModePreference] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Route to RAGService for RAG-powered chat."""
        if not self.rag_service:
            yield {
                "type": "error",
                "error": "Chat service not available",
            }
            return

        # Delegate to RAGService
        try:
            # RAGService.query returns streaming response
            async for chunk in self.rag_service.query_stream(
                query=request,
                session_id=session_id,
                **kwargs
            ):
                yield {
                    "type": "chat_response",
                    "mode": "chat",
                    "data": chunk,
                }
        except AttributeError:
            # Non-streaming fallback
            response = await self.rag_service.query(
                question=request,
                session_id=session_id,
                **kwargs
            )

            # Check if we should fallback to general chat when no sources found
            if (prefs and prefs.fallback_to_general and prefs.general_chat_enabled
                and not response.sources):
                logger.info(
                    "No documents found, falling back to general chat",
                    session_id=session_id,
                )
                async for chunk in self._route_to_general(
                    request=request,
                    session_id=session_id,
                    user_id=user_id,
                    **kwargs
                ):
                    yield chunk
                return

            yield {
                "type": "chat_response",
                "mode": "chat",
                "data": {
                    "content": response.content,
                    "sources": [
                        {
                            "document_id": s.document_id,
                            "document_name": s.document_name,
                            "page_number": s.page_number,
                            "snippet": s.snippet,
                            "relevance_score": s.relevance_score,
                        }
                        for s in response.sources
                    ] if response.sources else [],
                    "is_general_response": False,
                    "model": response.model,
                    "processing_time_ms": response.processing_time_ms,
                },
                "complete": True,
            }

    async def _route_to_agents(
        self,
        request: str,
        session_id: str,
        user_id: str,
        prefs: ExecutionModePreference,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Route to AgentOrchestrator for complex tasks."""
        if not self.orchestrator:
            # Fall back to chat if orchestrator not available
            logger.warning("Orchestrator not available, falling back to chat")
            async for chunk in self._route_to_chat(
                request=request,
                session_id=session_id,
                user_id=user_id,
                **kwargs
            ):
                yield chunk
            return

        # Yield mode indicator
        yield {
            "type": "mode_selected",
            "mode": "agent",
            "message": "Processing with multi-agent system...",
        }

        # Delegate to orchestrator
        try:
            async for update in self.orchestrator.process_request(
                request=request,
                session_id=session_id,
                user_id=user_id,
                show_cost_estimation=prefs.show_cost_estimation,
                require_approval_above_usd=prefs.require_approval_above_usd,
                **kwargs
            ):
                yield {
                    "type": "agent_update",
                    "mode": "agent",
                    "data": update,
                }
        except Exception as e:
            logger.error(
                "Agent orchestration failed",
                error=str(e),
                session_id=session_id,
            )
            yield {
                "type": "error",
                "mode": "agent",
                "error": str(e),
                "fallback": "chat",
            }

            # Optionally fall back to chat on error
            if kwargs.get("fallback_to_chat", True):
                async for chunk in self._route_to_chat(
                    request=request,
                    session_id=session_id,
                    user_id=user_id,
                    **kwargs
                ):
                    yield chunk

    async def get_current_mode(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current execution mode status for user.

        Args:
            user_id: User UUID
            session_id: Optional session for session-specific mode

        Returns:
            Dict with mode information
        """
        prefs = await self.get_user_preferences(user_id)

        # Determine effective mode
        if not prefs.agent_mode_enabled:
            effective = prefs.default_mode if prefs.default_mode == ExecutionMode.GENERAL.value else ExecutionMode.CHAT.value
        else:
            effective = prefs.default_mode

        return {
            "default_mode": prefs.default_mode,
            "agent_mode_enabled": prefs.agent_mode_enabled,
            "auto_detect_complexity": prefs.auto_detect_complexity,
            "show_cost_estimation": prefs.show_cost_estimation,
            "require_approval_above_usd": prefs.require_approval_above_usd,
            "general_chat_enabled": prefs.general_chat_enabled,
            "fallback_to_general": prefs.fallback_to_general,
            "effective_mode": effective,
        }

    async def analyze_request_complexity(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze complexity of a request without routing.

        Useful for UI to show expected mode before submitting.

        Args:
            request: User request text
            context: Optional context

        Returns:
            Dict with complexity analysis
        """
        level = self.detector.detect(request, context)
        should_use_agents = self.detector.should_use_agents(request, context)

        return {
            "complexity_level": level.value,
            "recommended_mode": (
                ExecutionMode.AGENT.value if should_use_agents
                else ExecutionMode.CHAT.value
            ),
            "request_length": len(request),
            "estimated_steps": self._estimate_steps(level),
        }

    def _estimate_steps(self, level: ComplexityLevel) -> int:
        """Estimate number of agent steps based on complexity."""
        estimates = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 3,
            ComplexityLevel.COMPLEX: 5,
        }
        return estimates.get(level, 1)
