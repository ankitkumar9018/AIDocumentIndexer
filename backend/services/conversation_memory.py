"""
AIDocumentIndexer - Conversational Memory Service
==================================================

Implements multi-level memory architecture for conversation context:
1. Short-term: Recent messages in buffer (fast access)
2. Long-term: Summarized conversations (persistent)
3. Entity memory: Key facts about entities mentioned

Provides intelligent context management for multi-turn conversations.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

import structlog

logger = structlog.get_logger(__name__)


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
class MemoryContext:
    """Complete memory context for a conversation turn."""
    conversation_id: str
    recent_messages: List[Message]
    summary: Optional[ConversationSummary]
    entity_facts: Dict[str, List[str]]  # entity -> facts
    formatted_context: str  # Ready-to-use context string


class ConversationalMemory:
    """
    Multi-level memory architecture for conversations.

    Manages:
    - Short-term buffer of recent messages
    - Long-term summarized context
    - Entity facts extracted from conversations
    """

    def __init__(
        self,
        llm=None,
        cache=None,
        max_short_term: int = 10,
        summary_ttl_days: int = 30,
    ):
        """
        Initialize conversational memory.

        Args:
            llm: LangChain LLM for summarization
            cache: RedisCache or similar for persistence
            max_short_term: Max messages in short-term buffer
            summary_ttl_days: Days to keep summaries
        """
        self.llm = llm
        self.cache = cache
        self.max_short_term = max_short_term
        self.summary_ttl = summary_ttl_days * 86400  # Convert to seconds

        # In-memory short-term buffers (per user)
        self._short_term: Dict[str, List[Message]] = {}
        # Entity memory (per user)
        self._entity_facts: Dict[str, Dict[str, List[str]]] = {}

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
        include_summary: bool = True,
        include_entities: bool = True,
        max_recent: int = 5,
    ) -> MemoryContext:
        """
        Get conversation context for a new turn.

        Args:
            user_id: User identifier
            include_summary: Include long-term summary
            include_entities: Include entity facts
            max_recent: Max recent messages to include

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

        # Format context string
        formatted = self._format_context(recent, summary, entity_facts)

        return MemoryContext(
            conversation_id=user_id,
            recent_messages=recent,
            summary=summary,
            entity_facts=entity_facts,
            formatted_context=formatted,
        )

    async def clear_history(self, user_id: str) -> None:
        """Clear conversation history for a user."""
        self._short_term.pop(user_id, None)
        self._entity_facts.pop(user_id, None)

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
    ) -> str:
        """Format complete context for RAG prompt."""
        parts = []

        # Add summary
        if summary and summary.summary:
            parts.append("Previous conversation summary:")
            parts.append(summary.summary)
            if summary.key_topics:
                parts.append(f"Topics discussed: {', '.join(summary.key_topics)}")
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


# =============================================================================
# Convenience Functions
# =============================================================================

_memory_instances: Dict[str, ConversationalMemory] = {}


def get_conversation_memory(
    instance_id: str = "default",
    llm=None,
    cache=None,
) -> ConversationalMemory:
    """
    Get or create a ConversationalMemory instance.

    Args:
        instance_id: Identifier for the memory instance
        llm: Optional LLM for summarization
        cache: Optional cache for persistence

    Returns:
        ConversationalMemory instance
    """
    if instance_id not in _memory_instances:
        _memory_instances[instance_id] = ConversationalMemory(
            llm=llm,
            cache=cache,
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
    memory: Optional[ConversationalMemory] = None,
) -> str:
    """
    Get formatted conversation context for a user.

    Args:
        user_id: User identifier
        memory: Optional ConversationalMemory

    Returns:
        Formatted context string
    """
    mem = memory or get_conversation_memory()
    context = await mem.get_context(user_id)
    return context.formatted_context
