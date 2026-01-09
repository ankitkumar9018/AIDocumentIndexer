"""
AIDocumentIndexer - Enhanced Conversational Memory Service
============================================================

Implements multi-level memory architecture for conversation context:
1. Short-term: Recent messages in buffer (fast access)
2. Long-term: Summarized conversations (persistent)
3. Entity memory: Key facts about entities mentioned
4. Semantic memory: Embedding-based retrieval for relevant past context
5. User preferences: Learned preferences from interaction patterns
6. Cross-session memory: Persistent memory across sessions

Provides intelligent context management for multi-turn conversations
with semantic retrieval and preference learning capabilities.
"""

import json
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import structlog
import numpy as np

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memories stored."""
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    ENTITY = "entity"
    INSIGHT = "insight"


class MemoryImportance(str, Enum):
    """Importance levels for memories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationSummary:
    """Summary of a conversation segment."""
    summary: str
    key_topics: List[str]
    user_preferences: Dict[str, Any]
    entities_mentioned: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "key_topics": self.key_topics,
            "user_preferences": self.user_preferences,
            "entities_mentioned": self.entities_mentioned,
            "timestamp": self.timestamp.isoformat(),
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        return cls(
            summary=data.get("summary", ""),
            key_topics=data.get("key_topics", []),
            user_preferences=data.get("user_preferences", {}),
            entities_mentioned=data.get("entities_mentioned", []),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.utcnow(),
            message_count=data.get("message_count", 0),
        )


@dataclass
class SemanticMemory:
    """A memory with embedding for semantic retrieval."""
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data.get("memory_type", "fact")),
            importance=MemoryImportance(data.get("importance", "medium")),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if isinstance(data.get("last_accessed"), str) else datetime.utcnow(),
            access_count=data.get("access_count", 0),
        )


@dataclass
class UserPreference:
    """A learned user preference."""
    category: str  # e.g., "response_style", "topic_interest", "format_preference"
    key: str
    value: Any
    confidence: float = 0.5  # 0-1, how confident we are
    evidence_count: int = 1  # How many times observed
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        return cls(
            category=data["category"],
            key=data["key"],
            value=data["value"],
            confidence=data.get("confidence", 0.5),
            evidence_count=data.get("evidence_count", 1),
            last_updated=datetime.fromisoformat(data["last_updated"]) if isinstance(data.get("last_updated"), str) else datetime.utcnow(),
        )


@dataclass
class MemoryContext:
    """Complete memory context for a conversation turn."""
    conversation_id: str
    recent_messages: List[Message]
    summary: Optional[ConversationSummary]
    entity_facts: Dict[str, List[str]]  # entity -> facts
    semantic_memories: List[SemanticMemory] = field(default_factory=list)  # Relevant memories
    user_preferences: Dict[str, UserPreference] = field(default_factory=dict)  # Learned preferences
    formatted_context: str = ""  # Ready-to-use context string


class ConversationalMemory:
    """
    Enhanced multi-level memory architecture for conversations.

    Manages:
    - Short-term buffer of recent messages
    - Long-term summarized context
    - Entity facts extracted from conversations
    - Semantic memories with embedding-based retrieval
    - Learned user preferences
    - Cross-session persistent memory
    """

    def __init__(
        self,
        llm=None,
        cache=None,
        embeddings=None,
        max_short_term: int = 10,
        summary_ttl_days: int = 30,
        max_semantic_memories: int = 100,
        semantic_similarity_threshold: float = 0.7,
    ):
        """
        Initialize enhanced conversational memory.

        Args:
            llm: LangChain LLM for summarization
            cache: RedisCache or similar for persistence
            embeddings: Embedding model for semantic retrieval
            max_short_term: Max messages in short-term buffer
            summary_ttl_days: Days to keep summaries
            max_semantic_memories: Max semantic memories per user
            semantic_similarity_threshold: Min similarity for memory retrieval
        """
        self.llm = llm
        self.cache = cache
        self.embeddings = embeddings
        self.max_short_term = max_short_term
        self.summary_ttl = summary_ttl_days * 86400  # Convert to seconds
        self.max_semantic_memories = max_semantic_memories
        self.semantic_similarity_threshold = semantic_similarity_threshold

        # In-memory short-term buffers (per user)
        self._short_term: Dict[str, List[Message]] = {}
        # Entity memory (per user)
        self._entity_facts: Dict[str, Dict[str, List[str]]] = {}
        # Semantic memories (per user)
        self._semantic_memories: Dict[str, List[SemanticMemory]] = {}
        # User preferences (per user)
        self._user_preferences: Dict[str, Dict[str, UserPreference]] = {}

    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            user_id: User identifier
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        if user_id not in self._short_term:
            self._short_term[user_id] = []

        self._short_term[user_id].append(message)

        # Extract entities from user messages
        if role == "user":
            await self._extract_entities(user_id, content)
            # Learn preferences from user messages
            await self._learn_preferences(user_id, content)
            # Store important facts as semantic memories
            await self._extract_semantic_memories(user_id, content)

        # Check if we need to compact
        if len(self._short_term[user_id]) >= self.max_short_term:
            await self._compact_to_long_term(user_id)

        logger.debug(
            "Message added",
            user_id=user_id[:8],
            role=role,
            buffer_size=len(self._short_term[user_id]),
        )

    async def get_context(
        self,
        user_id: str,
        query: Optional[str] = None,
        include_summary: bool = True,
        include_entities: bool = True,
        include_semantic: bool = True,
        include_preferences: bool = True,
        max_recent: int = 5,
        max_semantic: int = 3,
    ) -> MemoryContext:
        """
        Get enhanced conversation context for a new turn.

        Args:
            user_id: User identifier
            query: Optional query for semantic memory retrieval
            include_summary: Include long-term summary
            include_entities: Include entity facts
            include_semantic: Include semantically relevant memories
            include_preferences: Include learned user preferences
            max_recent: Max recent messages to include
            max_semantic: Max semantic memories to include

        Returns:
            MemoryContext with formatted context
        """
        recent = self._short_term.get(user_id, [])[-max_recent:]

        summary = None
        if include_summary:
            summary = await self._get_summary(user_id)

        entity_facts = {}
        if include_entities:
            entity_facts = self._entity_facts.get(user_id, {})

        # Get semantically relevant memories
        semantic_memories = []
        if include_semantic and query:
            semantic_memories = await self._retrieve_semantic_memories(
                user_id, query, max_results=max_semantic
            )

        # Get user preferences
        user_preferences = {}
        if include_preferences:
            user_preferences = self._user_preferences.get(user_id, {})

        # Format context string
        formatted = self._format_context(
            recent, summary, entity_facts, semantic_memories, user_preferences
        )

        return MemoryContext(
            conversation_id=user_id,
            recent_messages=recent,
            summary=summary,
            entity_facts=entity_facts,
            semantic_memories=semantic_memories,
            user_preferences=user_preferences,
            formatted_context=formatted,
        )

    async def clear_history(self, user_id: str, clear_semantic: bool = False) -> None:
        """Clear conversation history for a user."""
        self._short_term.pop(user_id, None)
        self._entity_facts.pop(user_id, None)

        if clear_semantic:
            self._semantic_memories.pop(user_id, None)
            self._user_preferences.pop(user_id, None)
            if self.cache:
                await self.cache.delete(f"semantic_memories:{user_id}")
                await self.cache.delete(f"user_preferences:{user_id}")

        if self.cache:
            await self.cache.delete(f"conv_summary:{user_id}")

        logger.info("Conversation history cleared", user_id=user_id[:8])

    async def _compact_to_long_term(self, user_id: str) -> None:
        """Summarize and compact short-term to long-term memory."""
        messages = self._short_term[user_id]

        if not self.llm:
            # Without LLM, just trim the buffer
            self._short_term[user_id] = messages[-2:]
            return

        # Generate summary
        summary = await self._generate_summary(messages)

        # Extract topics and entities
        topics = await self._extract_topics(messages)
        entities = self._collect_entities(user_id)

        conv_summary = ConversationSummary(
            summary=summary,
            key_topics=topics,
            user_preferences={},
            entities_mentioned=entities,
            message_count=len(messages),
        )

        # Merge with existing summary
        existing = await self._get_summary(user_id)
        if existing:
            conv_summary = await self._merge_summaries(existing, conv_summary)

        # Store summary
        if self.cache:
            await self.cache.set(
                f"conv_summary:{user_id}",
                json.dumps(conv_summary.to_dict()),
                ttl=self.summary_ttl,
            )

        # Keep last 2 messages for continuity
        self._short_term[user_id] = messages[-2:]

        logger.info(
            "Conversation compacted",
            user_id=user_id[:8],
            messages_summarized=len(messages),
        )

    async def _generate_summary(self, messages: List[Message]) -> str:
        """Generate a summary of the conversation."""
        formatted = self._format_messages(messages)

        prompt = f"""Summarize this conversation concisely, capturing:
1. Main topics discussed
2. Key questions asked and answers given
3. Any decisions or conclusions reached

Conversation:
{formatted}

Provide a concise summary (2-4 sentences):"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
        except Exception as e:
            logger.warning("Summary generation failed", error=str(e))
            return "Previous conversation about document-related queries."

    async def _extract_topics(self, messages: List[Message]) -> List[str]:
        """Extract key topics from messages."""
        if not self.llm:
            return []

        text = " ".join(m.content for m in messages)

        prompt = f"""Extract 3-5 key topics from this conversation as a comma-separated list.
Be concise - use 1-3 words per topic.

Conversation text:
{text[:1000]}

Topics (comma-separated):"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            topics = [t.strip() for t in content.split(",") if t.strip()]
            return topics[:5]
        except Exception as e:
            logger.warning("Topic extraction failed", error=str(e))
            return []

    async def _extract_entities(self, user_id: str, text: str) -> None:
        """Extract and store entity mentions from text."""
        # Simple entity extraction (could be enhanced with NER)
        # Look for patterns like "the X document" or mentions of specific names

        if user_id not in self._entity_facts:
            self._entity_facts[user_id] = {}

        # Basic pattern matching for document references
        import re
        doc_pattern = r'(?:the|a|an)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:document|file|report|presentation))'
        matches = re.findall(doc_pattern, text)

        for match in matches:
            entity = match.strip()
            if entity and len(entity) > 2:
                if entity not in self._entity_facts[user_id]:
                    self._entity_facts[user_id][entity] = []
                # Store the context where it was mentioned
                self._entity_facts[user_id][entity].append(f"Mentioned: {text[:100]}")

    def _collect_entities(self, user_id: str) -> List[str]:
        """Collect all entities mentioned by user."""
        return list(self._entity_facts.get(user_id, {}).keys())

    async def _get_summary(self, user_id: str) -> Optional[ConversationSummary]:
        """Retrieve stored conversation summary."""
        if not self.cache:
            return None

        try:
            data = await self.cache.get(f"conv_summary:{user_id}")
            if data:
                parsed = json.loads(data) if isinstance(data, str) else data
                return ConversationSummary.from_dict(parsed)
        except Exception as e:
            logger.warning("Failed to get summary", error=str(e))

        return None

    async def _merge_summaries(
        self,
        old: ConversationSummary,
        new: ConversationSummary,
    ) -> ConversationSummary:
        """Merge old and new summaries."""
        if not self.llm:
            # Simple merge without LLM
            return ConversationSummary(
                summary=f"{old.summary} {new.summary}",
                key_topics=list(set(old.key_topics + new.key_topics))[:5],
                user_preferences={**old.user_preferences, **new.user_preferences},
                entities_mentioned=list(set(old.entities_mentioned + new.entities_mentioned)),
                message_count=old.message_count + new.message_count,
            )

        prompt = f"""Merge these two conversation summaries into one concise summary:

Previous summary:
{old.summary}

New summary:
{new.summary}

Merged summary (2-4 sentences, keep the most important information):"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            merged_summary = content.strip()
        except Exception as e:
            logger.warning("Summary merge failed", error=str(e))
            merged_summary = f"{old.summary} {new.summary}"

        return ConversationSummary(
            summary=merged_summary,
            key_topics=list(set(old.key_topics + new.key_topics))[:5],
            user_preferences={**old.user_preferences, **new.user_preferences},
            entities_mentioned=list(set(old.entities_mentioned + new.entities_mentioned)),
            message_count=old.message_count + new.message_count,
        )

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for LLM consumption."""
        lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _format_context(
        self,
        recent: List[Message],
        summary: Optional[ConversationSummary],
        entity_facts: Dict[str, List[str]],
        semantic_memories: Optional[List[SemanticMemory]] = None,
        user_preferences: Optional[Dict[str, UserPreference]] = None,
    ) -> str:
        """Format complete context for RAG prompt."""
        parts = []

        # Add user preferences
        if user_preferences:
            prefs_with_high_confidence = [
                p for p in user_preferences.values() if p.confidence >= 0.7
            ]
            if prefs_with_high_confidence:
                parts.append("User preferences:")
                for pref in prefs_with_high_confidence[:5]:
                    parts.append(f"- {pref.category}: {pref.key} = {pref.value}")
                parts.append("")

        # Add summary
        if summary and summary.summary:
            parts.append("Previous conversation summary:")
            parts.append(summary.summary)
            if summary.key_topics:
                parts.append(f"Topics discussed: {', '.join(summary.key_topics)}")
            parts.append("")

        # Add relevant semantic memories
        if semantic_memories:
            parts.append("Relevant memories from past conversations:")
            for mem in semantic_memories:
                importance_marker = "!" if mem.importance in [MemoryImportance.HIGH, MemoryImportance.CRITICAL] else "-"
                parts.append(f"{importance_marker} {mem.content}")
            parts.append("")

        # Add entity facts
        if entity_facts:
            parts.append("Known entities:")
            for entity, facts in list(entity_facts.items())[:5]:
                parts.append(f"- {entity}")
            parts.append("")

        # Add recent messages
        if recent:
            parts.append("Recent conversation:")
            for msg in recent:
                role = "User" if msg.role == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")

        return "\n".join(parts)

    # =========================================================================
    # Semantic Memory Methods
    # =========================================================================

    async def add_semantic_memory(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticMemory:
        """
        Add a new semantic memory for a user.

        Args:
            user_id: User identifier
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            metadata: Optional metadata

        Returns:
            The created SemanticMemory
        """
        # Generate embedding if available
        embedding = None
        if self.embeddings:
            try:
                embedding = await self._get_embedding(content)
            except Exception as e:
                logger.warning("Failed to generate embedding", error=str(e))

        memory = SemanticMemory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            metadata=metadata or {},
        )

        if user_id not in self._semantic_memories:
            self._semantic_memories[user_id] = []

        self._semantic_memories[user_id].append(memory)

        # Enforce max memories limit
        if len(self._semantic_memories[user_id]) > self.max_semantic_memories:
            # Remove oldest low-importance memories
            self._semantic_memories[user_id] = self._prune_memories(
                self._semantic_memories[user_id]
            )

        # Persist to cache
        await self._persist_semantic_memories(user_id)

        logger.debug(
            "Semantic memory added",
            user_id=user_id[:8],
            memory_type=memory_type.value,
            importance=importance.value,
        )

        return memory

    async def _retrieve_semantic_memories(
        self,
        user_id: str,
        query: str,
        max_results: int = 3,
    ) -> List[SemanticMemory]:
        """
        Retrieve semantically relevant memories for a query.

        Args:
            user_id: User identifier
            query: Query text
            max_results: Maximum memories to return

        Returns:
            List of relevant SemanticMemory objects
        """
        memories = self._semantic_memories.get(user_id, [])
        if not memories:
            # Try loading from cache
            await self._load_semantic_memories(user_id)
            memories = self._semantic_memories.get(user_id, [])

        if not memories:
            return []

        # If no embeddings available, return most recent important memories
        if not self.embeddings:
            important = [
                m for m in memories
                if m.importance in [MemoryImportance.HIGH, MemoryImportance.CRITICAL]
            ]
            return sorted(important, key=lambda x: x.created_at, reverse=True)[:max_results]

        try:
            query_embedding = await self._get_embedding(query)
        except Exception as e:
            logger.warning("Failed to get query embedding", error=str(e))
            return []

        # Calculate similarities
        scored_memories = []
        for mem in memories:
            if mem.embedding:
                similarity = self._cosine_similarity(query_embedding, mem.embedding)
                if similarity >= self.semantic_similarity_threshold:
                    scored_memories.append((similarity, mem))

        # Sort by similarity and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Update access info for retrieved memories
        results = []
        for _, mem in scored_memories[:max_results]:
            mem.last_accessed = datetime.utcnow()
            mem.access_count += 1
            results.append(mem)

        return results

    async def _extract_semantic_memories(self, user_id: str, text: str) -> None:
        """Extract and store important facts as semantic memories."""
        if not self.llm:
            return

        # Only process longer messages that might contain facts
        if len(text) < 50:
            return

        prompt = f"""Analyze this user message and extract any important facts that should be remembered for future conversations.
Focus on:
- User preferences
- Important decisions or requirements
- Specific constraints or needs
- Key entities or topics of interest

User message:
{text[:500]}

If there are important facts to remember, list them (one per line). If nothing important, respond with "NONE".

Important facts:"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            if "NONE" in content.upper():
                return

            facts = [f.strip() for f in content.strip().split("\n") if f.strip() and not f.startswith("-")]
            facts = [f.lstrip("- ").lstrip("• ") for f in facts]

            for fact in facts[:3]:  # Limit to 3 facts per message
                if len(fact) > 10:  # Skip very short facts
                    importance = self._assess_importance(fact)
                    await self.add_semantic_memory(
                        user_id,
                        fact,
                        memory_type=MemoryType.FACT,
                        importance=importance,
                        metadata={"source_text": text[:200]},
                    )

        except Exception as e:
            logger.warning("Failed to extract semantic memories", error=str(e))

    def _assess_importance(self, content: str) -> MemoryImportance:
        """Assess the importance of a memory based on content."""
        content_lower = content.lower()

        # Critical keywords
        critical_keywords = ["must", "never", "always", "critical", "important", "required"]
        if any(kw in content_lower for kw in critical_keywords):
            return MemoryImportance.HIGH

        # High importance indicators
        high_keywords = ["prefer", "want", "need", "should", "deadline", "budget"]
        if any(kw in content_lower for kw in high_keywords):
            return MemoryImportance.MEDIUM

        return MemoryImportance.LOW

    def _prune_memories(self, memories: List[SemanticMemory]) -> List[SemanticMemory]:
        """Prune memories to stay within limit, keeping important ones."""
        # Sort by importance (critical > high > medium > low), then by access count
        importance_order = {
            MemoryImportance.CRITICAL: 4,
            MemoryImportance.HIGH: 3,
            MemoryImportance.MEDIUM: 2,
            MemoryImportance.LOW: 1,
        }

        def memory_score(m: SemanticMemory) -> Tuple[int, int, float]:
            return (
                importance_order[m.importance],
                m.access_count,
                m.last_accessed.timestamp(),
            )

        sorted_memories = sorted(memories, key=memory_score, reverse=True)
        return sorted_memories[:self.max_semantic_memories]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if hasattr(self.embeddings, 'aembed_query'):
            return await self.embeddings.aembed_query(text)
        elif hasattr(self.embeddings, 'embed_query'):
            return self.embeddings.embed_query(text)
        raise ValueError("Embeddings model does not support embedding generation")

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))

    async def _persist_semantic_memories(self, user_id: str) -> None:
        """Persist semantic memories to cache."""
        if not self.cache:
            return

        memories = self._semantic_memories.get(user_id, [])
        data = [m.to_dict() for m in memories]

        try:
            await self.cache.set(
                f"semantic_memories:{user_id}",
                json.dumps(data),
                ttl=self.summary_ttl * 2,  # Keep memories longer
            )
        except Exception as e:
            logger.warning("Failed to persist semantic memories", error=str(e))

    async def _load_semantic_memories(self, user_id: str) -> None:
        """Load semantic memories from cache."""
        if not self.cache:
            return

        try:
            data = await self.cache.get(f"semantic_memories:{user_id}")
            if data:
                parsed = json.loads(data) if isinstance(data, str) else data
                self._semantic_memories[user_id] = [
                    SemanticMemory.from_dict(m) for m in parsed
                ]
        except Exception as e:
            logger.warning("Failed to load semantic memories", error=str(e))

    # =========================================================================
    # User Preference Learning Methods
    # =========================================================================

    async def _learn_preferences(self, user_id: str, text: str) -> None:
        """Learn user preferences from their messages."""
        # Pattern-based preference detection
        preferences = self._detect_preferences(text)

        for category, key, value in preferences:
            await self._update_preference(user_id, category, key, value)

    def _detect_preferences(self, text: str) -> List[Tuple[str, str, Any]]:
        """Detect preferences from text using patterns."""
        preferences = []
        text_lower = text.lower()

        # Response style preferences
        if any(word in text_lower for word in ["brief", "short", "concise", "summarize"]):
            preferences.append(("response_style", "length", "brief"))
        elif any(word in text_lower for word in ["detailed", "comprehensive", "thorough", "explain"]):
            preferences.append(("response_style", "length", "detailed"))

        # Format preferences
        if "bullet" in text_lower or "list" in text_lower:
            preferences.append(("format_preference", "structure", "bulleted"))
        elif "table" in text_lower:
            preferences.append(("format_preference", "structure", "tabular"))

        # Technical level
        if any(word in text_lower for word in ["technical", "expert", "advanced"]):
            preferences.append(("communication_style", "technical_level", "advanced"))
        elif any(word in text_lower for word in ["simple", "basic", "beginner", "explain like"]):
            preferences.append(("communication_style", "technical_level", "beginner"))

        # Topic interests (from explicit mentions)
        topic_patterns = [
            (r"interested in (.+?)(?:\.|,|$)", "topic_interest"),
            (r"focus on (.+?)(?:\.|,|$)", "topic_focus"),
            (r"working on (.+?)(?:\.|,|$)", "current_project"),
        ]

        for pattern, category in topic_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                topic = match.strip()[:50]  # Limit length
                if len(topic) > 3:
                    preferences.append((category, topic, True))

        return preferences

    async def _update_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        value: Any,
    ) -> None:
        """Update or create a user preference."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}

        pref_key = f"{category}:{key}"
        existing = self._user_preferences[user_id].get(pref_key)

        if existing:
            # Update existing preference
            if existing.value == value:
                # Same value observed again - increase confidence
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.evidence_count += 1
            else:
                # Different value - decrease confidence or switch
                if existing.confidence < 0.3:
                    # Switch to new value
                    existing.value = value
                    existing.confidence = 0.5
                    existing.evidence_count = 1
                else:
                    existing.confidence = max(0.1, existing.confidence - 0.2)
            existing.last_updated = datetime.utcnow()
        else:
            # Create new preference
            self._user_preferences[user_id][pref_key] = UserPreference(
                category=category,
                key=key,
                value=value,
            )

        # Persist preferences
        await self._persist_preferences(user_id)

    def get_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a specific user preference."""
        prefs = self._user_preferences.get(user_id, {})
        pref = prefs.get(f"{category}:{key}")
        if pref and pref.confidence >= 0.5:
            return pref.value
        return default

    def get_all_preferences(self, user_id: str, min_confidence: float = 0.5) -> Dict[str, UserPreference]:
        """Get all user preferences above confidence threshold."""
        prefs = self._user_preferences.get(user_id, {})
        return {k: v for k, v in prefs.items() if v.confidence >= min_confidence}

    async def _persist_preferences(self, user_id: str) -> None:
        """Persist user preferences to cache."""
        if not self.cache:
            return

        prefs = self._user_preferences.get(user_id, {})
        data = {k: v.to_dict() for k, v in prefs.items()}

        try:
            await self.cache.set(
                f"user_preferences:{user_id}",
                json.dumps(data),
                ttl=self.summary_ttl * 4,  # Keep preferences longer
            )
        except Exception as e:
            logger.warning("Failed to persist preferences", error=str(e))

    async def _load_preferences(self, user_id: str) -> None:
        """Load user preferences from cache."""
        if not self.cache:
            return

        try:
            data = await self.cache.get(f"user_preferences:{user_id}")
            if data:
                parsed = json.loads(data) if isinstance(data, str) else data
                self._user_preferences[user_id] = {
                    k: UserPreference.from_dict(v) for k, v in parsed.items()
                }
        except Exception as e:
            logger.warning("Failed to load preferences", error=str(e))

    # =========================================================================
    # Cross-Session Memory Methods
    # =========================================================================

    async def load_user_memory(self, user_id: str) -> None:
        """Load all persistent memory for a user (call at session start)."""
        await self._load_semantic_memories(user_id)
        await self._load_preferences(user_id)

        logger.debug(
            "User memory loaded",
            user_id=user_id[:8],
            semantic_count=len(self._semantic_memories.get(user_id, [])),
            preference_count=len(self._user_preferences.get(user_id, {})),
        )

    async def save_user_memory(self, user_id: str) -> None:
        """Persist all memory for a user (call at session end)."""
        await self._persist_semantic_memories(user_id)
        await self._persist_preferences(user_id)

        logger.debug("User memory saved", user_id=user_id[:8])

    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user."""
        return {
            "short_term_messages": len(self._short_term.get(user_id, [])),
            "entity_count": len(self._entity_facts.get(user_id, {})),
            "semantic_memory_count": len(self._semantic_memories.get(user_id, [])),
            "preference_count": len(self._user_preferences.get(user_id, {})),
            "high_confidence_preferences": len([
                p for p in self._user_preferences.get(user_id, {}).values()
                if p.confidence >= 0.7
            ]),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_memory_instances: Dict[str, ConversationalMemory] = {}


def get_conversation_memory(
    instance_id: str = "default",
    llm=None,
    cache=None,
    embeddings=None,
) -> ConversationalMemory:
    """
    Get or create a ConversationalMemory instance.

    Args:
        instance_id: Identifier for the memory instance
        llm: Optional LLM for summarization
        cache: Optional cache for persistence
        embeddings: Optional embedding model for semantic retrieval

    Returns:
        ConversationalMemory instance
    """
    if instance_id not in _memory_instances:
        _memory_instances[instance_id] = ConversationalMemory(
            llm=llm,
            cache=cache,
            embeddings=embeddings,
        )

    return _memory_instances[instance_id]


async def add_conversation_turn(
    user_id: str,
    user_message: str,
    assistant_response: str,
    memory: Optional[ConversationalMemory] = None,
) -> None:
    """
    Convenience function to add a complete conversation turn.

    Args:
        user_id: User identifier
        user_message: User's message
        assistant_response: Assistant's response
        memory: Optional ConversationalMemory (uses default if not provided)
    """
    mem = memory or get_conversation_memory()
    await mem.add_message(user_id, "user", user_message)
    await mem.add_message(user_id, "assistant", assistant_response)


async def get_conversation_context(
    user_id: str,
    query: Optional[str] = None,
    memory: Optional[ConversationalMemory] = None,
) -> str:
    """
    Get formatted conversation context for a user.

    Args:
        user_id: User identifier
        query: Optional query for semantic memory retrieval
        memory: Optional ConversationalMemory

    Returns:
        Formatted context string
    """
    mem = memory or get_conversation_memory()
    context = await mem.get_context(user_id, query=query)
    return context.formatted_context


async def get_enhanced_context(
    user_id: str,
    query: str,
    memory: Optional[ConversationalMemory] = None,
) -> MemoryContext:
    """
    Get full enhanced context including semantic memories and preferences.

    Args:
        user_id: User identifier
        query: Query for semantic memory retrieval
        memory: Optional ConversationalMemory

    Returns:
        MemoryContext with all context types
    """
    mem = memory or get_conversation_memory()
    return await mem.get_context(user_id, query=query)
