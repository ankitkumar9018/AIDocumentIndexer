"""
AIDocumentIndexer - Session Memory Manager
==========================================

Thread-safe session memory management with LRU eviction to prevent memory leaks.
Provides in-memory conversation storage with:
- Dynamic memory window sizing based on model context window
- DB-based history rehydration on session load
- Token budget enforcement
- Configurable max sessions with LRU eviction
"""

import time
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import structlog

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage


logger = structlog.get_logger(__name__)


# =============================================================================
# Model-Tier Memory Configuration
# =============================================================================
# Research-backed memory window sizes by model parameter count.
# Key insight: small models degrade with too much history ("lost in the middle"),
# while large models benefit from extensive context.

MODEL_TIER_CONFIG = {
    # (min_params_b, max_params_b): {memory_k, history_summarize_after, context_budget_split}
    # context_budget_split: (system%, history%, chunks%, generation_buffer%)
    "tiny":   {"min_b": 0,   "max_b": 3,   "memory_k": 3,  "summarize_after": 2,  "budget": (10, 10, 60, 20)},
    "small":  {"min_b": 3,   "max_b": 9,   "memory_k": 6,  "summarize_after": 5,  "budget": (10, 15, 55, 20)},
    "medium": {"min_b": 9,   "max_b": 34,  "memory_k": 10, "summarize_after": 8,  "budget": (10, 15, 60, 15)},
    "large":  {"min_b": 34,  "max_b": 999, "memory_k": 15, "summarize_after": 12, "budget": (10, 15, 60, 15)},
}


def _estimate_model_params_b(model_name: str) -> float:
    """
    Estimate model parameter count in billions from model name.
    Uses common Ollama naming conventions (e.g., llama3.2:3b, deepseek-r1:14b).
    """
    if not model_name:
        return 3.0  # Default: assume small model

    model_lower = model_name.lower()

    # Direct size indicators in model name
    import re
    # Match patterns like :7b, :14b, :70b, :1.5b, :0.5b, -8b, etc.
    size_match = re.search(r'[:\-_](\d+\.?\d*)b', model_lower)
    if size_match:
        return float(size_match.group(1))

    # Match "8x7b" pattern (MoE models — use active params, not total)
    moe_match = re.search(r'(\d+)x(\d+\.?\d*)b', model_lower)
    if moe_match:
        # Active params ≈ single expert size for memory purposes
        return float(moe_match.group(2))

    # Known model name heuristics
    if "latest" in model_lower:
        # Common defaults
        if "llama3.2" in model_lower:
            return 3.0
        if "llama3.1" in model_lower or "llama3:" in model_lower:
            return 8.0
        if "mistral" in model_lower:
            return 7.0
        if "phi3:mini" in model_lower:
            return 3.8
    if "mini" in model_lower:
        return 3.8
    if "medium" in model_lower:
        return 14.0

    return 3.0  # Conservative default


def get_model_tier(model_name: str) -> Dict[str, Any]:
    """
    Get memory tier configuration for a given model.
    Returns the tier config dict with memory_k, summarize_after, budget.
    """
    params_b = _estimate_model_params_b(model_name)

    for tier_name, config in MODEL_TIER_CONFIG.items():
        if config["min_b"] <= params_b < config["max_b"]:
            return {**config, "tier": tier_name, "estimated_params_b": params_b}

    # Fallback to medium
    return {**MODEL_TIER_CONFIG["small"], "tier": "small", "estimated_params_b": params_b}


def get_dynamic_memory_k(model_name: str) -> int:
    """Get the recommended memory window (k) for a model based on its size tier."""
    tier = get_model_tier(model_name)
    return tier["memory_k"]


# =============================================================================
# Token Budget Calculator
# =============================================================================

def calculate_token_budget(
    context_window: int,
    model_name: str,
    system_prompt_tokens: int = 0,
    chunk_tokens: int = 0,
) -> Dict[str, int]:
    """
    Calculate token budget allocation for a given model and context window.

    Returns allocated budgets for each component:
    - system: system prompt + instructions
    - history: conversation history
    - chunks: retrieved document chunks
    - generation: reserved for model output

    The budget ensures total never exceeds context_window.
    """
    tier = get_model_tier(model_name)
    sys_pct, hist_pct, chunk_pct, gen_pct = tier["budget"]

    # Calculate base allocations
    system_budget = int(context_window * sys_pct / 100)
    history_budget = int(context_window * hist_pct / 100)
    chunks_budget = int(context_window * chunk_pct / 100)
    generation_budget = int(context_window * gen_pct / 100)

    # If system prompt is larger than allocated, steal from history first, then chunks
    if system_prompt_tokens > system_budget:
        overflow = system_prompt_tokens - system_budget
        system_budget = system_prompt_tokens
        # Steal from history first (less important than chunks)
        steal_from_history = min(overflow, history_budget // 2)
        history_budget -= steal_from_history
        overflow -= steal_from_history
        if overflow > 0:
            chunks_budget -= overflow

    # If chunks are provided and smaller than budget, give surplus to history
    if chunk_tokens > 0 and chunk_tokens < chunks_budget:
        surplus = chunks_budget - chunk_tokens
        history_budget += surplus // 2  # Give half the surplus to history
        chunks_budget = chunk_tokens + surplus // 2  # Keep some headroom

    return {
        "system": max(system_budget, 200),
        "history": max(history_budget, 100),
        "chunks": max(chunks_budget, 200),
        "generation": max(generation_budget, 256),
        "total": context_window,
        "tier": tier["tier"],
    }


def estimate_tokens(text: str) -> int:
    """Quick token estimation (~4 chars/token for English text)."""
    if not text:
        return 0
    return len(text) // 4


def trim_history_to_budget(
    messages: List,
    token_budget: int,
) -> List:
    """
    Trim conversation history to fit within token budget.
    Keeps most recent messages, drops oldest first.
    Always keeps at least the last user-assistant pair.
    """
    if not messages:
        return []

    # Estimate tokens per message
    msg_tokens = []
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        msg_tokens.append(estimate_tokens(content))

    total = sum(msg_tokens)
    if total <= token_budget:
        return list(messages)

    # Drop oldest messages until we fit
    trimmed = list(messages)
    trimmed_tokens = list(msg_tokens)

    while sum(trimmed_tokens) > token_budget and len(trimmed) > 2:
        trimmed.pop(0)
        trimmed_tokens.pop(0)

    return trimmed


# =============================================================================
# Session Info
# =============================================================================

@dataclass
class SessionInfo:
    """Metadata about a stored session."""
    session_id: str
    model_name: str = ""
    memory_k: int = 10
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    message_count: int = 0
    rehydrated: bool = False


# =============================================================================
# Session Memory Manager
# =============================================================================

class SessionMemoryManager:
    """
    Thread-safe session memory manager with LRU eviction and dynamic sizing.

    Features:
    - Dynamic memory_k based on model size tier
    - DB-based history rehydration on first access
    - Token budget enforcement
    - LRU eviction when max_sessions exceeded
    - Thread-safe with reentrant lock
    - Automatic cleanup of stale sessions
    """

    def __init__(
        self,
        max_sessions: int = 200,
        memory_window_k: int = 10,  # Default fallback, overridden per-model
        cleanup_stale_after_hours: Optional[float] = 4.0,
    ):
        self.max_sessions = max_sessions
        self.default_memory_window_k = memory_window_k
        self.cleanup_stale_after_hours = cleanup_stale_after_hours

        self._memory_store: OrderedDict[str, ConversationBufferWindowMemory] = OrderedDict()
        self._session_info: Dict[str, SessionInfo] = {}
        self._lock = threading.RLock()

        logger.info(
            "SessionMemoryManager initialized",
            max_sessions=max_sessions,
            default_memory_window_k=memory_window_k,
            cleanup_stale_hours=cleanup_stale_after_hours,
        )

    def get_memory(
        self,
        session_id: str,
        model_name: Optional[str] = None,
    ) -> ConversationBufferWindowMemory:
        """
        Get or create conversation memory for a session.
        Uses dynamic memory_k based on model size when model_name is provided.
        """
        with self._lock:
            if session_id in self._memory_store:
                self._memory_store.move_to_end(session_id)
                self._session_info[session_id].last_accessed_at = time.time()
                return self._memory_store[session_id]

            if self.cleanup_stale_after_hours:
                self._cleanup_stale_sessions()

            while len(self._memory_store) >= self.max_sessions:
                evicted_id, _ = self._memory_store.popitem(last=False)
                evicted_info = self._session_info.pop(evicted_id, None)
                logger.debug(
                    "Evicted LRU session",
                    session_id=evicted_id,
                    age_seconds=time.time() - evicted_info.created_at if evicted_info else 0,
                )

            # Dynamic memory_k based on model size
            if model_name:
                memory_k = get_dynamic_memory_k(model_name)
            else:
                memory_k = self.default_memory_window_k

            memory = ConversationBufferWindowMemory(
                k=memory_k,
                return_messages=True,
                memory_key="chat_history",
            )

            self._memory_store[session_id] = memory
            self._session_info[session_id] = SessionInfo(
                session_id=session_id,
                model_name=model_name or "",
                memory_k=memory_k,
            )

            logger.debug(
                "Created new session memory",
                session_id=session_id,
                model_name=model_name,
                memory_k=memory_k,
                total_sessions=len(self._memory_store),
            )

            return memory

    def get_memory_with_rehydration(
        self,
        session_id: str,
        model_name: Optional[str] = None,
        db_messages: Optional[List[Dict[str, str]]] = None,
    ) -> ConversationBufferWindowMemory:
        """
        Get memory with DB rehydration support.

        If the session doesn't exist in-memory and db_messages are provided,
        loads them into the new memory buffer. This restores conversation
        context after backend restarts.

        Args:
            session_id: Session identifier
            model_name: Model name for dynamic window sizing
            db_messages: List of {"role": "user"|"assistant", "content": "..."} from DB
        """
        with self._lock:
            if session_id in self._memory_store:
                self._memory_store.move_to_end(session_id)
                self._session_info[session_id].last_accessed_at = time.time()
                return self._memory_store[session_id]

        # Create new memory (outside lock for the get_memory call)
        memory = self.get_memory(session_id, model_name=model_name)

        # Rehydrate from DB messages if provided
        if db_messages:
            with self._lock:
                info = self._session_info.get(session_id)
                if info and not info.rehydrated:
                    memory_k = info.memory_k
                    # Take only the last memory_k pairs from DB
                    # Each pair = 1 user + 1 assistant message
                    pairs = []
                    i = 0
                    while i < len(db_messages) - 1:
                        if db_messages[i]["role"] == "user" and db_messages[i + 1]["role"] == "assistant":
                            pairs.append((db_messages[i]["content"], db_messages[i + 1]["content"]))
                            i += 2
                        else:
                            i += 1

                    # Load last memory_k pairs
                    for user_msg, ai_msg in pairs[-memory_k:]:
                        memory.save_context(
                            {"input": user_msg},
                            {"output": ai_msg},
                        )

                    info.rehydrated = True
                    info.message_count = len(pairs)

                    logger.info(
                        "Rehydrated session from DB",
                        session_id=session_id,
                        loaded_pairs=min(len(pairs), memory_k),
                        total_db_pairs=len(pairs),
                        memory_k=memory_k,
                    )

        return memory

    def clear_memory(self, session_id: str) -> bool:
        """Clear and remove memory for a specific session."""
        with self._lock:
            if session_id in self._memory_store:
                del self._memory_store[session_id]
                self._session_info.pop(session_id, None)
                logger.debug("Cleared session memory", session_id=session_id)
                return True
            return False

    def update_message_count(self, session_id: str, count: int = 1):
        """Update message count for a session."""
        with self._lock:
            if session_id in self._session_info:
                self._session_info[session_id].message_count += count

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get metadata about a session."""
        with self._lock:
            return self._session_info.get(session_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        with self._lock:
            oldest_session_age = 0
            if self._session_info:
                oldest = min(info.created_at for info in self._session_info.values())
                oldest_session_age = time.time() - oldest

            # Gather tier distribution
            tier_counts: Dict[str, int] = {}
            for info in self._session_info.values():
                if info.model_name:
                    tier = get_model_tier(info.model_name)["tier"]
                else:
                    tier = "unknown"
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            return {
                "active_sessions": len(self._memory_store),
                "max_sessions": self.max_sessions,
                "default_memory_window_k": self.default_memory_window_k,
                "oldest_session_age_seconds": oldest_session_age,
                "utilization_percent": (len(self._memory_store) / self.max_sessions) * 100,
                "tier_distribution": tier_counts,
            }

    def _cleanup_stale_sessions(self):
        """Remove sessions that haven't been accessed in cleanup_stale_after_hours."""
        if not self.cleanup_stale_after_hours:
            return

        cutoff_time = time.time() - (self.cleanup_stale_after_hours * 3600)
        stale_sessions = [
            session_id
            for session_id, info in self._session_info.items()
            if info.last_accessed_at < cutoff_time
        ]

        for session_id in stale_sessions:
            self._memory_store.pop(session_id, None)
            self._session_info.pop(session_id, None)

        if stale_sessions:
            logger.info(
                "Cleaned up stale sessions",
                count=len(stale_sessions),
                cutoff_hours=self.cleanup_stale_after_hours,
            )

    def clear_all(self):
        """Clear all session memories (use with caution)."""
        with self._lock:
            count = len(self._memory_store)
            self._memory_store.clear()
            self._session_info.clear()
            logger.info("Cleared all session memories", count=count)


# Singleton instance
_session_memory_manager: Optional[SessionMemoryManager] = None


def get_session_memory_manager(
    max_sessions: int = 1000,
    memory_window_k: int = 10,
    cleanup_stale_after_hours: Optional[float] = 24.0,
) -> SessionMemoryManager:
    """
    Get or create the session memory manager singleton.
    """
    global _session_memory_manager
    if _session_memory_manager is None:
        _session_memory_manager = SessionMemoryManager(
            max_sessions=max_sessions,
            memory_window_k=memory_window_k,
            cleanup_stale_after_hours=cleanup_stale_after_hours,
        )
    return _session_memory_manager
