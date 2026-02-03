"""
AIDocumentIndexer - Session Memory Manager
==========================================

Thread-safe session memory management with LRU eviction to prevent memory leaks.
Provides in-memory conversation storage with configurable max sessions.

For production deployments, consider Redis-backed storage using:
- RedisChatMessageHistory from langchain_community.chat_message_histories
- RunnableWithMessageHistory for automatic session management
"""

import time
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import structlog

# Phase 87 audit: still valid in langchain 0.3.x;
# future migration to langgraph state management recommended
from langchain.memory import ConversationBufferWindowMemory


logger = structlog.get_logger(__name__)


@dataclass
class SessionInfo:
    """Metadata about a stored session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    message_count: int = 0


class SessionMemoryManager:
    """
    Thread-safe session memory manager with LRU eviction.

    Prevents unbounded memory growth by limiting max sessions and
    evicting least-recently-used sessions when limit is reached.

    Features:
    - LRU eviction when max_sessions exceeded
    - Thread-safe with read-write lock
    - Session metadata tracking
    - Automatic cleanup of stale sessions (optional)

    For production with horizontal scaling, consider using Redis:
    ```python
    from langchain_community.chat_message_histories import RedisChatMessageHistory

    history = RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        key_prefix="chat_history:",
        ttl=3600  # 1 hour TTL
    )
    ```
    """

    def __init__(
        self,
        max_sessions: int = 200,  # Reduced from 1000 to prevent memory bloat
        memory_window_k: int = 10,
        cleanup_stale_after_hours: Optional[float] = 4.0,  # Auto-cleanup stale sessions
    ):
        """
        Initialize the session memory manager.

        Args:
            max_sessions: Maximum number of sessions to keep in memory (LRU eviction)
            memory_window_k: Number of messages to keep per session
            cleanup_stale_after_hours: Optional - cleanup sessions older than X hours
        """
        self.max_sessions = max_sessions
        self.memory_window_k = memory_window_k
        self.cleanup_stale_after_hours = cleanup_stale_after_hours

        # OrderedDict maintains insertion order for LRU tracking
        self._memory_store: OrderedDict[str, ConversationBufferWindowMemory] = OrderedDict()
        self._session_info: Dict[str, SessionInfo] = {}

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            "SessionMemoryManager initialized",
            max_sessions=max_sessions,
            memory_window_k=memory_window_k,
            cleanup_stale_hours=cleanup_stale_after_hours,
        )

    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Get or create conversation memory for a session.

        Uses LRU eviction if max_sessions exceeded.

        Args:
            session_id: Unique session identifier

        Returns:
            ConversationBufferWindowMemory for the session
        """
        with self._lock:
            if session_id in self._memory_store:
                # Move to end (most recently used)
                self._memory_store.move_to_end(session_id)
                self._session_info[session_id].last_accessed_at = time.time()
                return self._memory_store[session_id]

            # Optional: cleanup stale sessions before adding new one
            if self.cleanup_stale_after_hours:
                self._cleanup_stale_sessions()

            # Evict LRU session if at capacity
            while len(self._memory_store) >= self.max_sessions:
                evicted_id, evicted_memory = self._memory_store.popitem(last=False)
                evicted_info = self._session_info.pop(evicted_id, None)
                logger.debug(
                    "Evicted LRU session",
                    session_id=evicted_id,
                    age_seconds=time.time() - evicted_info.created_at if evicted_info else 0,
                )

            # Create new memory for session
            memory = ConversationBufferWindowMemory(
                k=self.memory_window_k,
                return_messages=True,
                memory_key="chat_history",
            )

            self._memory_store[session_id] = memory
            self._session_info[session_id] = SessionInfo(session_id=session_id)

            logger.debug(
                "Created new session memory",
                session_id=session_id,
                total_sessions=len(self._memory_store),
            )

            return memory

    def clear_memory(self, session_id: str) -> bool:
        """
        Clear and remove memory for a specific session.

        Args:
            session_id: Session ID to clear

        Returns:
            True if session was found and cleared, False otherwise
        """
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

            return {
                "active_sessions": len(self._memory_store),
                "max_sessions": self.max_sessions,
                "memory_window_k": self.memory_window_k,
                "oldest_session_age_seconds": oldest_session_age,
                "utilization_percent": (len(self._memory_store) / self.max_sessions) * 100,
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


# Singleton instance with sensible defaults
_session_memory_manager: Optional[SessionMemoryManager] = None


def get_session_memory_manager(
    max_sessions: int = 1000,
    memory_window_k: int = 10,
    cleanup_stale_after_hours: Optional[float] = 24.0,
) -> SessionMemoryManager:
    """
    Get or create the session memory manager singleton.

    Args:
        max_sessions: Maximum sessions to keep in memory (default: 1000)
        memory_window_k: Messages per session (default: 10)
        cleanup_stale_after_hours: Auto-cleanup sessions older than X hours (default: 24)

    Returns:
        SessionMemoryManager singleton instance
    """
    global _session_memory_manager
    if _session_memory_manager is None:
        _session_memory_manager = SessionMemoryManager(
            max_sessions=max_sessions,
            memory_window_k=memory_window_k,
            cleanup_stale_after_hours=cleanup_stale_after_hours,
        )
    return _session_memory_manager
