"""
AIDocumentIndexer - Agent Memory System
========================================

Phase 23B: Implements a 3-tier memory system for AI agents:

1. Message Buffer (Short-term)
   - Recent conversation context
   - In-memory, last N messages
   - Fast access, volatile

2. Core Memory (Working Memory)
   - User profile, current task, preferences
   - Redis-backed, pinned to context
   - Always included in prompts

3. Recall Memory (Long-term)
   - Full searchable history
   - Vector DB backed, retrieved on demand
   - Semantic search for relevant memories

Architecture based on:
- MemGPT/Letta architecture
- Personalization Agents (ICLR 2025 research)
- Multi-agent RAG systems

Usage:
    memory = AgentMemory(user_id="user_123", agent_id="agent_456")
    await memory.initialize()

    # Add interaction
    await memory.add_message("user", "What is the capital of France?")
    await memory.add_message("assistant", "The capital of France is Paris.")

    # Get context for prompt
    context = await memory.get_context_for_prompt(max_tokens=4000)

    # Search recall memory
    relevant = await memory.search_recall("previous discussion about France")
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class MemoryType(str, Enum):
    """Types of memory entries."""
    MESSAGE = "message"
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    ENTITY = "entity"
    SUMMARY = "summary"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def token_estimate(self) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(self.content) // 4 + 10  # +10 for role overhead


@dataclass
class CoreMemoryBlock:
    """A block of core memory (always in context)."""
    name: str
    content: str
    max_chars: int = 2000
    editable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "max_chars": self.max_chars,
            "editable": self.editable,
        }


@dataclass
class UserProfile:
    """User profile for personalization."""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    communication_style: Optional[str] = None
    expertise_level: str = "intermediate"
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "preferences": self.preferences,
            "facts": self.facts,
            "communication_style": self.communication_style,
            "expertise_level": self.expertise_level,
            "last_updated": self.last_updated.isoformat(),
        }

    def to_prompt_string(self) -> str:
        """Convert to string for inclusion in prompts."""
        parts = []
        if self.name:
            parts.append(f"User: {self.name}")
        if self.expertise_level:
            parts.append(f"Expertise: {self.expertise_level}")
        if self.communication_style:
            parts.append(f"Style: {self.communication_style}")
        if self.facts:
            parts.append(f"Known facts: {'; '.join(self.facts[:5])}")
        if self.preferences:
            prefs = [f"{k}: {v}" for k, v in list(self.preferences.items())[:3]]
            parts.append(f"Preferences: {', '.join(prefs)}")
        return "\n".join(parts) if parts else "No user profile available."


# =============================================================================
# Memory Tiers
# =============================================================================

class MessageBuffer:
    """
    Tier 1: Short-term message buffer.

    Stores recent conversation messages in memory.
    Fast access, limited size, volatile.
    """

    def __init__(self, max_messages: int = 20, max_tokens: int = 8000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: deque[Message] = deque(maxlen=max_messages)
        self._token_count = 0

    def add(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a message to the buffer."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        self._token_count += msg.token_estimate()

        # Trim if over token limit
        while self._token_count > self.max_tokens and len(self._messages) > 1:
            removed = self._messages.popleft()
            self._token_count -= removed.token_estimate()

        return msg

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get recent messages."""
        messages = list(self._messages)
        if limit:
            return messages[-limit:]
        return messages

    def get_for_prompt(self, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """Get messages formatted for LLM prompt."""
        result = []
        tokens = 0

        for msg in reversed(self._messages):
            msg_tokens = msg.token_estimate()
            if tokens + msg_tokens > max_tokens:
                break
            result.insert(0, {"role": msg.role, "content": msg.content})
            tokens += msg_tokens

        return result

    def clear(self):
        """Clear the buffer."""
        self._messages.clear()
        self._token_count = 0

    def __len__(self) -> int:
        return len(self._messages)


class CoreMemory:
    """
    Tier 2: Working memory (always in context).

    Stores user profile, current task, and key facts.
    Redis-backed for persistence, always included in prompts.
    """

    def __init__(self, user_id: str, agent_id: str):
        self.user_id = user_id
        self.agent_id = agent_id
        self._redis = None
        self._blocks: Dict[str, CoreMemoryBlock] = {}
        self._user_profile: Optional[UserProfile] = None
        self._current_task: Optional[str] = None
        self._initialized = False

    async def initialize(self):
        """Initialize core memory from Redis."""
        if self._initialized:
            return

        try:
            from backend.services.redis_client import get_redis_client
            self._redis = await get_redis_client()
        except Exception as e:
            logger.warning(f"Redis not available for core memory: {e}")

        # Load from Redis
        await self._load_from_storage()

        # Initialize default blocks if empty
        if not self._blocks:
            self._blocks = {
                "persona": CoreMemoryBlock(
                    name="persona",
                    content="I am a helpful AI assistant that answers questions based on the user's documents.",
                    max_chars=1000,
                    editable=True,
                ),
                "user": CoreMemoryBlock(
                    name="user",
                    content="",
                    max_chars=2000,
                    editable=True,
                ),
                "task": CoreMemoryBlock(
                    name="task",
                    content="",
                    max_chars=1000,
                    editable=True,
                ),
            }

        self._initialized = True

    async def _load_from_storage(self):
        """Load core memory from Redis."""
        if not self._redis:
            return

        try:
            key = f"agent:core_memory:{self.agent_id}:{self.user_id}"
            data = await self._redis.get(key)
            if data:
                parsed = json.loads(data)
                self._blocks = {
                    name: CoreMemoryBlock(**block)
                    for name, block in parsed.get("blocks", {}).items()
                }
                if parsed.get("user_profile"):
                    profile_data = parsed["user_profile"]
                    profile_data["last_updated"] = datetime.fromisoformat(profile_data["last_updated"])
                    self._user_profile = UserProfile(**profile_data)
                self._current_task = parsed.get("current_task")
        except Exception as e:
            logger.error(f"Failed to load core memory: {e}")

    async def _save_to_storage(self):
        """Save core memory to Redis."""
        if not self._redis:
            return

        try:
            key = f"agent:core_memory:{self.agent_id}:{self.user_id}"
            data = {
                "blocks": {name: block.to_dict() for name, block in self._blocks.items()},
                "user_profile": self._user_profile.to_dict() if self._user_profile else None,
                "current_task": self._current_task,
            }
            await self._redis.set(key, json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to save core memory: {e}")

    def get_block(self, name: str) -> Optional[CoreMemoryBlock]:
        """Get a memory block by name."""
        return self._blocks.get(name)

    async def update_block(self, name: str, content: str) -> bool:
        """Update a memory block."""
        if name not in self._blocks:
            return False

        block = self._blocks[name]
        if not block.editable:
            return False

        # Truncate if too long
        if len(content) > block.max_chars:
            content = content[:block.max_chars]

        block.content = content
        await self._save_to_storage()
        return True

    async def append_to_block(self, name: str, content: str, separator: str = "\n") -> bool:
        """Append content to a memory block."""
        if name not in self._blocks:
            return False

        block = self._blocks[name]
        if not block.editable:
            return False

        new_content = block.content + separator + content if block.content else content

        # Truncate from beginning if too long
        while len(new_content) > block.max_chars:
            # Remove first line
            if "\n" in new_content:
                new_content = new_content.split("\n", 1)[1]
            else:
                new_content = new_content[len(new_content) - block.max_chars:]

        block.content = new_content
        await self._save_to_storage()
        return True

    @property
    def user_profile(self) -> Optional[UserProfile]:
        return self._user_profile

    async def update_user_profile(self, **kwargs) -> UserProfile:
        """Update user profile."""
        if not self._user_profile:
            self._user_profile = UserProfile(user_id=self.user_id)

        for key, value in kwargs.items():
            if hasattr(self._user_profile, key):
                setattr(self._user_profile, key, value)

        self._user_profile.last_updated = datetime.utcnow()
        await self._save_to_storage()
        return self._user_profile

    async def add_user_fact(self, fact: str):
        """Add a fact about the user."""
        if not self._user_profile:
            self._user_profile = UserProfile(user_id=self.user_id)

        if fact not in self._user_profile.facts:
            self._user_profile.facts.append(fact)
            # Keep last 20 facts
            if len(self._user_profile.facts) > 20:
                self._user_profile.facts = self._user_profile.facts[-20:]
            await self._save_to_storage()

    async def set_current_task(self, task: str):
        """Set the current task."""
        self._current_task = task
        await self.update_block("task", task)

    def get_for_prompt(self) -> str:
        """Get core memory formatted for prompt."""
        parts = []

        # Add persona
        if self._blocks.get("persona") and self._blocks["persona"].content:
            parts.append(f"<persona>\n{self._blocks['persona'].content}\n</persona>")

        # Add user profile
        if self._user_profile:
            parts.append(f"<user_profile>\n{self._user_profile.to_prompt_string()}\n</user_profile>")
        elif self._blocks.get("user") and self._blocks["user"].content:
            parts.append(f"<user_info>\n{self._blocks['user'].content}\n</user_info>")

        # Add current task
        if self._current_task:
            parts.append(f"<current_task>\n{self._current_task}\n</current_task>")

        return "\n\n".join(parts)


class RecallMemory:
    """
    Tier 3: Long-term searchable memory.

    Stores full conversation history and facts.
    Vector DB backed, semantic search for retrieval.
    """

    def __init__(self, user_id: str, agent_id: str, embedding_service=None):
        self.user_id = user_id
        self.agent_id = agent_id
        self._embedding_service = embedding_service
        self._entries: List[MemoryEntry] = []
        self._db = None
        self._initialized = False

    async def initialize(self):
        """Initialize recall memory."""
        if self._initialized:
            return

        # Load from database
        await self._load_from_db()
        self._initialized = True

    async def _load_from_db(self):
        """Load memories from database."""
        try:
            from backend.db.database import async_session_context
            from sqlalchemy import text

            async with async_session_context() as session:
                result = await session.execute(
                    text("""
                        SELECT id, type, content, metadata, timestamp, importance, access_count
                        FROM agent_memories
                        WHERE user_id = :user_id AND agent_id = :agent_id
                        ORDER BY timestamp DESC
                        LIMIT 1000
                    """),
                    {"user_id": self.user_id, "agent_id": self.agent_id}
                )
                rows = result.fetchall()

                self._entries = [
                    MemoryEntry(
                        id=row[0],
                        type=MemoryType(row[1]),
                        content=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        timestamp=row[4],
                        importance=row[5] or 0.5,
                        access_count=row[6] or 0,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.debug(f"Could not load from DB (table may not exist): {e}")

    async def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.MESSAGE,
        metadata: Optional[Dict] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Add a memory entry."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
        )

        # Generate embedding if service available
        if self._embedding_service:
            try:
                embedding = self._embedding_service.embed_text(content)
                entry.embedding = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        self._entries.append(entry)

        # Persist to database
        await self._save_entry(entry)

        return entry

    async def _save_entry(self, entry: MemoryEntry):
        """Save a memory entry to database."""
        try:
            from backend.db.database import async_session_context
            from sqlalchemy import text

            async with async_session_context() as session:
                await session.execute(
                    text("""
                        INSERT INTO agent_memories
                        (id, user_id, agent_id, type, content, metadata, timestamp, importance, embedding)
                        VALUES (:id, :user_id, :agent_id, :type, :content, :metadata, :timestamp, :importance, :embedding)
                        ON CONFLICT (id) DO UPDATE SET
                            content = :content,
                            metadata = :metadata,
                            importance = :importance,
                            access_count = agent_memories.access_count + 1
                    """),
                    {
                        "id": entry.id,
                        "user_id": self.user_id,
                        "agent_id": self.agent_id,
                        "type": entry.type.value,
                        "content": entry.content,
                        "metadata": json.dumps(entry.metadata),
                        "timestamp": entry.timestamp,
                        "importance": entry.importance,
                        "embedding": json.dumps(entry.embedding) if entry.embedding else None,
                    }
                )
                await session.commit()
        except Exception as e:
            logger.debug(f"Could not save to DB (table may not exist): {e}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Search recall memory semantically.

        Args:
            query: Search query
            limit: Max results
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold

        Returns:
            Relevant memory entries
        """
        if not self._entries:
            return []

        # Filter by type and importance
        candidates = [
            e for e in self._entries
            if e.importance >= min_importance
            and (not memory_types or e.type in memory_types)
        ]

        if not candidates:
            return []

        # If we have embedding service, do semantic search
        if self._embedding_service:
            try:
                query_embedding = self._embedding_service.embed_text(query)

                # Score by cosine similarity
                scored = []
                for entry in candidates:
                    if entry.embedding:
                        score = self._cosine_similarity(query_embedding, entry.embedding)
                        scored.append((entry, score))

                if scored:
                    scored.sort(key=lambda x: x[1], reverse=True)
                    return [e for e, _ in scored[:limit]]
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Fallback to keyword matching
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for entry in candidates:
            content_lower = entry.content.lower()
            # Simple keyword overlap score
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0 or query_lower in content_lower:
                score = overlap + (1 if query_lower in content_lower else 0)
                scored.append((entry, score))

        scored.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        return [e for e, _ in scored[:limit]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def get_recent(
        self,
        limit: int = 20,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[MemoryEntry]:
        """Get recent memories."""
        candidates = self._entries
        if memory_types:
            candidates = [e for e in candidates if e.type in memory_types]

        # Sort by timestamp descending
        sorted_entries = sorted(candidates, key=lambda e: e.timestamp, reverse=True)
        return sorted_entries[:limit]

    async def summarize_history(self, llm_service=None) -> str:
        """Generate a summary of conversation history."""
        recent = await self.get_recent(50, [MemoryType.MESSAGE])

        if not recent:
            return "No conversation history."

        if llm_service:
            # Use LLM to summarize
            messages_text = "\n".join([
                f"{e.metadata.get('role', 'unknown')}: {e.content}"
                for e in reversed(recent)
            ])

            try:
                summary = await llm_service.generate(
                    f"Summarize this conversation in 2-3 sentences:\n\n{messages_text}"
                )
                return summary
            except Exception:
                pass

        # Simple summary
        return f"Conversation with {len(recent)} messages. Latest: {recent[0].content[:100]}..."


# =============================================================================
# Main Agent Memory Class
# =============================================================================

class AgentMemory:
    """
    Complete 3-tier agent memory system.

    Provides unified interface to:
    - Message Buffer (short-term)
    - Core Memory (working memory)
    - Recall Memory (long-term)
    """

    def __init__(
        self,
        user_id: str,
        agent_id: str,
        embedding_service=None,
        max_buffer_messages: int = 20,
        max_buffer_tokens: int = 8000,
    ):
        self.user_id = user_id
        self.agent_id = agent_id

        # Initialize tiers
        self.buffer = MessageBuffer(
            max_messages=max_buffer_messages,
            max_tokens=max_buffer_tokens,
        )
        self.core = CoreMemory(user_id=user_id, agent_id=agent_id)
        self.recall = RecallMemory(
            user_id=user_id,
            agent_id=agent_id,
            embedding_service=embedding_service,
        )

        self._initialized = False

    async def initialize(self):
        """Initialize all memory tiers."""
        if self._initialized:
            return

        await asyncio.gather(
            self.core.initialize(),
            self.recall.initialize(),
        )

        self._initialized = True
        logger.info(
            "Agent memory initialized",
            user_id=self.user_id,
            agent_id=self.agent_id,
        )

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
        persist: bool = True,
    ) -> Message:
        """
        Add a message to memory.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
            persist: Whether to save to recall memory

        Returns:
            The added message
        """
        # Add to buffer
        msg = self.buffer.add(role, content, metadata)

        # Persist to recall memory
        if persist:
            await self.recall.add(
                content=content,
                memory_type=MemoryType.MESSAGE,
                metadata={"role": role, **(metadata or {})},
                importance=0.5 if role == "user" else 0.3,
            )

        return msg

    async def add_fact(self, fact: str, importance: float = 0.7):
        """Add a learned fact."""
        await self.recall.add(
            content=fact,
            memory_type=MemoryType.FACT,
            importance=importance,
        )

    async def add_preference(self, preference: str, value: Any):
        """Add a user preference."""
        await self.core.update_user_profile(
            preferences={
                **(self.core.user_profile.preferences if self.core.user_profile else {}),
                preference: value,
            }
        )

        await self.recall.add(
            content=f"User preference: {preference} = {value}",
            memory_type=MemoryType.PREFERENCE,
            importance=0.8,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        include_buffer: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search across all memory tiers.

        Args:
            query: Search query
            limit: Max results
            include_buffer: Include buffer messages

        Returns:
            Relevant memories with source tier
        """
        results = []

        # Search recall memory
        recall_results = await self.recall.search(query, limit=limit)
        for entry in recall_results:
            results.append({
                "tier": "recall",
                "content": entry.content,
                "type": entry.type.value,
                "timestamp": entry.timestamp.isoformat(),
                "importance": entry.importance,
            })

        # Search buffer (simple keyword match)
        if include_buffer:
            query_lower = query.lower()
            for msg in self.buffer.get_messages():
                if query_lower in msg.content.lower():
                    results.append({
                        "tier": "buffer",
                        "content": msg.content,
                        "type": "message",
                        "role": msg.role,
                        "timestamp": msg.timestamp.isoformat(),
                    })

        return results[:limit]

    async def get_context_for_prompt(
        self,
        max_tokens: int = 4000,
        include_core: bool = True,
        include_recall: bool = False,
        recall_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get memory context for LLM prompt.

        Args:
            max_tokens: Token budget
            include_core: Include core memory
            include_recall: Include recall memory search
            recall_query: Query for recall search

        Returns:
            Context dict with messages and system info
        """
        context = {
            "messages": [],
            "system_prefix": "",
            "recalled_memories": [],
        }

        # Core memory goes in system prefix
        if include_core:
            context["system_prefix"] = self.core.get_for_prompt()

        # Reserve tokens for core memory
        core_tokens = len(context["system_prefix"]) // 4 if context["system_prefix"] else 0
        available_tokens = max_tokens - core_tokens

        # Get buffer messages
        context["messages"] = self.buffer.get_for_prompt(
            max_tokens=int(available_tokens * 0.8)
        )

        # Optionally add recalled memories
        if include_recall and recall_query:
            recalled = await self.recall.search(recall_query, limit=5)
            context["recalled_memories"] = [
                {
                    "content": e.content,
                    "type": e.type.value,
                    "relevance": e.importance,
                }
                for e in recalled
            ]

        return context

    @property
    def recent_messages(self) -> List[Dict[str, str]]:
        """Get recent messages for quick access."""
        return self.buffer.get_for_prompt()

    async def clear_buffer(self):
        """Clear the message buffer."""
        self.buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "buffer_size": len(self.buffer),
            "recall_size": len(self.recall._entries),
            "has_user_profile": self.core.user_profile is not None,
            "initialized": self._initialized,
        }


# =============================================================================
# Factory Function with LRU Eviction (Phase 70 Fix)
# =============================================================================

from collections import OrderedDict

# Phase 70: Memory instance cache with LRU eviction and TTL
# Prevents unbounded memory growth in long-running processes
_MEMORY_MAX_SIZE = int(os.getenv("AGENT_MEMORY_CACHE_MAX_SIZE", "1000"))
_MEMORY_TTL_SECONDS = int(os.getenv("AGENT_MEMORY_CACHE_TTL", "3600"))  # 1 hour default

# Stores (AgentMemory, last_access_time)
_memory_instances: OrderedDict[str, tuple] = OrderedDict()
_memory_lock = asyncio.Lock()


async def get_agent_memory(
    user_id: str,
    agent_id: str,
    embedding_service=None,
) -> AgentMemory:
    """
    Get or create agent memory instance.

    Phase 70: Now with LRU eviction and TTL to prevent memory leaks.

    Args:
        user_id: User ID
        agent_id: Agent ID
        embedding_service: Optional embedding service for semantic search

    Returns:
        AgentMemory instance
    """
    key = f"{agent_id}:{user_id}"
    current_time = time.time()

    async with _memory_lock:
        # Check if exists and not expired
        if key in _memory_instances:
            memory, last_access = _memory_instances[key]

            # Check TTL
            if current_time - last_access < _MEMORY_TTL_SECONDS:
                # Update access time and move to end (LRU)
                _memory_instances[key] = (memory, current_time)
                _memory_instances.move_to_end(key)
                return memory
            else:
                # Expired, remove it
                del _memory_instances[key]
                logger.debug("Evicted expired agent memory", key=key)

        # Create new instance
        memory = AgentMemory(
            user_id=user_id,
            agent_id=agent_id,
            embedding_service=embedding_service,
        )
        await memory.initialize()

        # Evict oldest if at capacity
        while len(_memory_instances) >= _MEMORY_MAX_SIZE:
            oldest_key, (oldest_memory, _) = _memory_instances.popitem(last=False)
            logger.debug("Evicted LRU agent memory", key=oldest_key)

        _memory_instances[key] = (memory, current_time)
        return memory


async def remove_agent_memory(user_id: str, agent_id: str) -> bool:
    """
    Remove a specific agent memory instance from cache.

    Phase 70: Added for explicit cleanup on user logout/session end.

    Args:
        user_id: User ID
        agent_id: Agent ID

    Returns:
        True if instance was found and removed
    """
    key = f"{agent_id}:{user_id}"
    async with _memory_lock:
        if key in _memory_instances:
            del _memory_instances[key]
            logger.debug("Removed agent memory", key=key)
            return True
        return False


def clear_memory_cache():
    """Clear all cached memory instances."""
    _memory_instances.clear()
    logger.info("Cleared all agent memory instances")


def get_memory_cache_stats() -> Dict[str, Any]:
    """
    Get memory cache statistics.

    Phase 70: Added for monitoring cache health.
    """
    current_time = time.time()
    expired_count = 0

    for key, (_, last_access) in _memory_instances.items():
        if current_time - last_access >= _MEMORY_TTL_SECONDS:
            expired_count += 1

    return {
        "total_instances": len(_memory_instances),
        "max_size": _MEMORY_MAX_SIZE,
        "ttl_seconds": _MEMORY_TTL_SECONDS,
        "expired_count": expired_count,
        "utilization": len(_memory_instances) / _MEMORY_MAX_SIZE if _MEMORY_MAX_SIZE > 0 else 0,
    }
