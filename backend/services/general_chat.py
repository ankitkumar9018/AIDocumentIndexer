"""
AIDocumentIndexer - General Chat Service
=========================================

LLM chat service without RAG - for general questions that don't require
document search. Uses the same LLM infrastructure as RAG but skips retrieval.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import time
import structlog

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.services.llm import (
    EnhancedLLMFactory,
    LLMConfigResult,
    LLMUsageTracker,
)
from backend.services.session_memory import get_session_memory_manager

logger = structlog.get_logger(__name__)


# System prompt for general chat mode
GENERAL_CHAT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question
to the best of your ability using your general knowledge.

Guidelines:
- Be concise and accurate
- If you're not sure about something, say so
- For complex topics, break down your explanation into clear steps
- Be helpful and friendly in your responses

Note: This system also has a document search capability. If the user's question
might be better answered by searching their documents, you can suggest they
switch to document search mode.
"""

# Language code to name mapping for multilingual support
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}


def _get_language_instruction(language: str, auto_detect: bool = False) -> str:
    """
    Get language instruction for the LLM prompt.

    Supports:
    - Queries in ANY language
    - Responses in the user's selected output language OR auto-detected from question

    Args:
        language: Language code for OUTPUT (en, de, es, etc.)
        auto_detect: If True and language is "en", respond in the same language as the question

    Returns:
        Language instruction string
    """
    if language == "en" and auto_detect:
        # Auto-detect mode: respond in the same language as the question
        # Make it VERY explicit for models that may default to document language
        return """
CRITICAL LANGUAGE AND SCRIPT REQUIREMENT (MUST FOLLOW):
1. FIRST, identify the language AND SCRIPT of the USER'S QUESTION (not any documents!)
2. THEN, respond ONLY in that SAME language AND SAME SCRIPT as the user's question
3. IMPORTANT about Indian languages:
   - Hinglish = Hindi written in LATIN/ROMAN script (like "kya hai", "accha hai") - respond in Latin script
   - If user writes in Devanagari (Hindi script like क्या), respond in Devanagari
   - NEVER respond in Gujarati, Bengali, Tamil, or other scripts unless user used them!
4. If the user asks in English, respond in English
5. If the user asks in German, respond in German
6. IGNORE the language of any source documents - they may be in any language
7. TRANSLATE all relevant information INTO the user's question language AND script

Example: "kya koi marketing campaign hai?" - Hinglish (Latin script), respond like: "Haan, yeh marketing campaigns hain..."
Example: "what marketing campaigns exist?" - English, respond in English.
"""

    if language == "en":
        return ""

    language_name = LANGUAGE_NAMES.get(language, "English")
    return f"""
LANGUAGE REQUIREMENT:
- Your response must be ENTIRELY in {language_name}
- The user's question may be in any language - understand it and respond in {language_name}
- Do NOT mix languages in your response - use only {language_name}
- Technical terms and proper nouns may remain in their original form if commonly used that way
"""


@dataclass
class GeneralChatResponse:
    """Response from general chat query."""
    content: str
    query: str
    model: str
    is_general_response: bool = True
    sources: List[Any] = field(default_factory=list)  # Always empty for general chat
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneralChatService:
    """
    LLM chat service without RAG.

    For answering general questions that don't require document search.
    Uses the same LLM configuration and memory management as RAG service.
    """

    def __init__(
        self,
        track_usage: bool = True,
        memory_window_k: int = 10,
    ):
        """
        Initialize General Chat Service.

        Args:
            track_usage: Whether to track LLM usage
            memory_window_k: Number of messages to keep in memory
        """
        self.track_usage = track_usage
        self.memory_window_k = memory_window_k

        # Use centralized session memory manager with LRU eviction (prevents memory leaks)
        self._session_memory = get_session_memory_manager(
            max_sessions=200,  # Reduced from 1000 to prevent memory bloat
            memory_window_k=memory_window_k,
            cleanup_stale_after_hours=4.0,  # Reduced from 24h to prevent stale sessions
        )

        logger.info(
            "GeneralChatService initialized",
            track_usage=track_usage,
            memory_window_k=memory_window_k,
        )

    async def get_llm_for_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> tuple:
        """Get LLM instance for a session using database-driven config."""
        # Use EnhancedLLMFactory which handles config resolution and model creation
        llm, config_result = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="chat",
            session_id=session_id,
            user_id=user_id,
            track_usage=self.track_usage,
        )

        return llm, config_result

    def _get_memory(self, session_id: str):
        """Get or create conversation memory for session using centralized manager."""
        return self._session_memory.get_memory(session_id)

    def clear_memory(self, session_id: str):
        """Clear conversation memory for session."""
        self._session_memory.clear_memory(session_id)

    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        language: str = "en",
    ) -> GeneralChatResponse:
        """
        Query the general chat service.

        Args:
            question: User's question
            session_id: Session ID for conversation memory
            user_id: User ID for usage tracking
            system_prompt: Optional custom system prompt
            language: Language code for response (en, de, es, fr, etc.)

        Returns:
            GeneralChatResponse with answer
        """
        start_time = time.time()

        logger.info(
            "Processing general chat query",
            question_length=len(question),
            session_id=session_id,
            language=language,
        )

        # Get LLM for this session
        llm, llm_config = await self.get_llm_for_session(
            session_id=session_id,
            user_id=user_id,
        )

        # Use custom system prompt or default, with language instruction
        # Support "auto" mode: respond in the same language as the question
        auto_detect = (language == "auto")
        effective_language = "en" if auto_detect else language
        base_prompt = system_prompt or GENERAL_CHAT_SYSTEM_PROMPT
        language_instruction = _get_language_instruction(effective_language, auto_detect=auto_detect)
        prompt = f"{base_prompt}\n{language_instruction}" if language_instruction else base_prompt

        # PHASE 15: Apply model-specific enhancements for small models
        # This wraps the system prompt with foundational behavioral hints
        model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        if model_name:
            from backend.services.rag_module.prompts import enhance_agent_system_prompt
            prompt = enhance_agent_system_prompt(prompt, model_name)

        # Build messages
        if session_id:
            # Use conversational prompt with history
            memory = self._get_memory(session_id)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

            messages = [
                SystemMessage(content=prompt),
                *chat_history,
                HumanMessage(content=question),
            ]
        else:
            # Single-turn query
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=question),
            ]

        # Generate response
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        processing_time_ms = (time.time() - start_time) * 1000

        # Track usage if enabled
        if self.track_usage and llm_config:
            # Estimate token counts
            input_text = prompt + question
            input_tokens = len(input_text) // 4  # ~4 chars per token
            output_tokens = len(content) // 4

            await LLMUsageTracker.log_usage(
                provider_type=llm_config.provider_type,
                model=llm_config.model,
                operation_type="general_chat",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_id=llm_config.provider_id,
                user_id=user_id,
                session_id=session_id,
                duration_ms=int(processing_time_ms),
                success=True,
            )

        # Update memory if using session
        if session_id:
            memory = self._get_memory(session_id)
            memory.save_context(
                {"input": question},
                {"output": content}
            )

        # Get model name from config
        model_name = llm_config.model if llm_config else "default"

        return GeneralChatResponse(
            content=content,
            query=question,
            model=model_name,
            is_general_response=True,
            sources=[],  # No sources for general chat
            processing_time_ms=processing_time_ms,
            metadata={
                "session_id": session_id,
                "provider": llm_config.provider_type if llm_config else "unknown",
                "mode": "general",
            },
        )


# Singleton instance with thread-safe initialization
import threading

_general_chat_service: Optional[GeneralChatService] = None
_general_chat_lock = threading.Lock()


def get_general_chat_service() -> GeneralChatService:
    """Get or create the general chat service singleton (thread-safe)."""
    global _general_chat_service

    # Fast path for existing service
    if _general_chat_service is not None:
        return _general_chat_service

    with _general_chat_lock:
        # Double-check after acquiring lock
        if _general_chat_service is None:
            _general_chat_service = GeneralChatService()
        return _general_chat_service
