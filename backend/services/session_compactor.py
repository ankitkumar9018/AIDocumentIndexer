"""
AIDocumentIndexer - Session Compactor
=====================================

Compresses long conversations to fit within context windows.
Summarizes older messages while preserving key information.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

import structlog

from backend.services.llm import LLMFactory
from backend.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class CompactedSession:
    """Result of session compaction."""
    summary: str
    recent_messages: List[Message]
    key_facts: List[str]
    original_message_count: int
    compacted_message_count: int
    original_tokens: int
    compacted_tokens: int
    compression_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "recent_messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                }
                for m in self.recent_messages
            ],
            "key_facts": self.key_facts,
            "original_message_count": self.original_message_count,
            "compacted_message_count": self.compacted_message_count,
            "original_tokens": self.original_tokens,
            "compacted_tokens": self.compacted_tokens,
            "compression_ratio": self.compression_ratio,
        }

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM context."""
        messages = []

        # Add summary as system context
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"**Previous Conversation Summary:**\n{self.summary}\n\n**Key Facts:**\n" + "\n".join(f"- {f}" for f in self.key_facts),
            })

        # Add recent messages
        for msg in self.recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return messages


class SessionCompactor:
    """
    Compress long conversations to fit within context windows.

    This service:
    1. Identifies messages that can be summarized
    2. Extracts key facts that must be preserved
    3. Creates a concise summary of older messages
    4. Keeps recent messages intact
    5. Provides a compressed context for continued conversation
    """

    TOKENS_PER_CHAR = 0.25  # Rough estimate

    SUMMARY_PROMPT = """Summarize this conversation between a user and an AI assistant.

**Conversation to summarize:**
{conversation}

**Requirements:**
1. Capture the main topics discussed
2. Note any decisions made or conclusions reached
3. Record specific facts, numbers, or names mentioned
4. Keep user preferences or constraints stated
5. Be concise but complete

**Summary:**"""

    KEY_FACTS_PROMPT = """Extract the key facts from this conversation that MUST be remembered.

**Conversation:**
{conversation}

**Extract:**
1. Specific names, dates, numbers mentioned
2. User preferences or requirements stated
3. Decisions made or agreements reached
4. Technical details or specifications discussed
5. Action items or next steps identified

**Key Facts (one per line, max 10):**"""

    def __init__(
        self,
        provider: str = None,
        model: str = None,
    ):
        """Initialize the session compactor."""
        self.provider = provider or settings.DEFAULT_LLM_PROVIDER
        self.model = model or settings.DEFAULT_CHAT_MODEL

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.TOKENS_PER_CHAR)

    def estimate_message_tokens(self, message: Message) -> int:
        """Estimate tokens for a message."""
        # Role overhead + content
        overhead = 4  # Role tokens
        return overhead + self.estimate_tokens(message.content)

    async def compact(
        self,
        messages: List[Message],
        target_tokens: int = 4000,
        keep_recent: int = 5,
        min_messages_to_compact: int = 10,
    ) -> CompactedSession:
        """
        Compact a conversation to fit within token limit.

        Args:
            messages: Full list of conversation messages
            target_tokens: Target token count for compacted context
            keep_recent: Number of recent messages to keep unchanged
            min_messages_to_compact: Minimum messages before compacting

        Returns:
            CompactedSession with summary and recent messages
        """
        # Add token estimates to messages
        for msg in messages:
            msg.token_count = self.estimate_message_tokens(msg)

        original_tokens = sum(m.token_count for m in messages)

        logger.info(
            "Starting session compaction",
            message_count=len(messages),
            original_tokens=original_tokens,
            target_tokens=target_tokens,
        )

        # Check if compaction needed
        if len(messages) < min_messages_to_compact:
            return CompactedSession(
                summary="",
                recent_messages=messages,
                key_facts=[],
                original_message_count=len(messages),
                compacted_message_count=len(messages),
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                compression_ratio=1.0,
            )

        if original_tokens <= target_tokens:
            return CompactedSession(
                summary="",
                recent_messages=messages,
                key_facts=[],
                original_message_count=len(messages),
                compacted_message_count=len(messages),
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                compression_ratio=1.0,
            )

        # Split into messages to compact vs. recent
        messages_to_compact = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]

        # Check if we still need to compact
        recent_tokens = sum(m.token_count for m in recent_messages)
        if recent_tokens >= target_tokens * 0.8:
            # Even recent messages are too long, need aggressive compaction
            keep_recent = max(2, keep_recent // 2)
            messages_to_compact = messages[:-keep_recent]
            recent_messages = messages[-keep_recent:]

        # Generate summary
        summary = await self._generate_summary(messages_to_compact)

        # Extract key facts
        key_facts = await self._extract_key_facts(messages_to_compact)

        # Calculate new token count
        summary_tokens = self.estimate_tokens(summary)
        facts_tokens = self.estimate_tokens("\n".join(key_facts))
        recent_tokens = sum(m.token_count for m in recent_messages)
        compacted_tokens = summary_tokens + facts_tokens + recent_tokens

        # If still too long, truncate summary
        if compacted_tokens > target_tokens:
            max_summary_tokens = target_tokens - facts_tokens - recent_tokens - 100
            summary = self._truncate_to_tokens(summary, max_summary_tokens)
            summary_tokens = self.estimate_tokens(summary)
            compacted_tokens = summary_tokens + facts_tokens + recent_tokens

        compression_ratio = original_tokens / max(compacted_tokens, 1)

        result = CompactedSession(
            summary=summary,
            recent_messages=recent_messages,
            key_facts=key_facts,
            original_message_count=len(messages),
            compacted_message_count=keep_recent,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            compression_ratio=compression_ratio,
        )

        logger.info(
            "Session compaction complete",
            original_messages=len(messages),
            compacted_messages=keep_recent,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            compression_ratio=f"{compression_ratio:.2f}x",
        )

        return result

    async def _generate_summary(
        self,
        messages: List[Message],
    ) -> str:
        """Generate a summary of the messages."""
        # Format conversation
        conversation = self._format_conversation(messages)

        prompt = self.SUMMARY_PROMPT.format(conversation=conversation)

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.3,
                max_tokens=512,
            )

            response = await llm.ainvoke(prompt)
            return response.content.strip()

        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            # Fallback: simple concatenation
            return self._create_fallback_summary(messages)

    async def _extract_key_facts(
        self,
        messages: List[Message],
    ) -> List[str]:
        """Extract key facts from messages."""
        conversation = self._format_conversation(messages)

        prompt = self.KEY_FACTS_PROMPT.format(conversation=conversation)

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.2,
                max_tokens=256,
            )

            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            # Parse facts (one per line)
            facts = []
            for line in content.split('\n'):
                line = line.strip()
                # Remove numbering
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[-•]\s*', '', line)
                if line and len(line) > 10:
                    facts.append(line)

            return facts[:10]  # Max 10 facts

        except Exception as e:
            logger.error("Key facts extraction failed", error=str(e))
            return []

    def _format_conversation(
        self,
        messages: List[Message],
        max_chars: int = 8000,
    ) -> str:
        """Format messages as conversation text."""
        parts = []
        total_chars = 0

        for msg in messages:
            role = msg.role.upper()
            line = f"{role}: {msg.content}"

            if total_chars + len(line) > max_chars:
                # Truncate
                remaining = max_chars - total_chars - 50
                if remaining > 100:
                    parts.append(line[:remaining] + "...")
                parts.append("[Earlier messages truncated]")
                break

            parts.append(line)
            total_chars += len(line)

        return "\n\n".join(parts)

    def _create_fallback_summary(
        self,
        messages: List[Message],
    ) -> str:
        """Create a simple fallback summary."""
        # Get first and last user messages
        user_messages = [m for m in messages if m.role == "user"]

        if not user_messages:
            return "Previous conversation context."

        first = user_messages[0].content[:100]
        last = user_messages[-1].content[:100] if len(user_messages) > 1 else ""

        summary = f"Conversation started with: {first}"
        if last:
            summary += f"\n\nMost recent topic: {last}"

        return summary

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        chars_limit = int(max_tokens / self.TOKENS_PER_CHAR)

        if len(text) <= chars_limit:
            return text

        # Truncate at sentence boundary
        truncated = text[:chars_limit]
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?'),
        )

        if last_sentence > chars_limit * 0.6:
            return text[:last_sentence + 1] + "\n\n[Summary truncated]"
        else:
            return truncated + "..."

    def should_compact(
        self,
        messages: List[Message],
        context_limit: int = 8000,
    ) -> bool:
        """Check if compaction is recommended."""
        total_tokens = sum(self.estimate_message_tokens(m) for m in messages)
        return total_tokens > context_limit * 0.8


# Singleton instance
_session_compactor: Optional[SessionCompactor] = None


def get_session_compactor() -> SessionCompactor:
    """Get or create the session compactor singleton."""
    global _session_compactor
    if _session_compactor is None:
        _session_compactor = SessionCompactor()
    return _session_compactor
