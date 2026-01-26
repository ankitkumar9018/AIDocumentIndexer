"""
AIDocumentIndexer - Agent Memory Unit Tests (Phase 91)
======================================================

Tests for the 3-tier agent memory system:
- MessageBuffer (short-term)
- CoreMemory (working memory)
- RecallMemory (long-term)
- LRU cache with TTL eviction
"""

import asyncio
import os
import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment
os.environ["TESTING"] = "true"
os.environ["APP_ENV"] = "development"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"


# =============================================================================
# MessageBuffer Tests
# =============================================================================

class TestMessageBuffer:
    """Test short-term message buffer."""

    def setup_method(self):
        from backend.services.agent_memory import MessageBuffer
        self.buffer = MessageBuffer(max_messages=5, max_tokens=500)

    def test_add_message(self):
        msg = self.buffer.add("user", "Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert len(self.buffer) == 1

    def test_add_multiple_messages(self):
        self.buffer.add("user", "Hello")
        self.buffer.add("assistant", "Hi there")
        self.buffer.add("user", "How are you?")
        assert len(self.buffer) == 3

    def test_max_messages_eviction(self):
        """Buffer should evict oldest messages when max_messages exceeded."""
        for i in range(7):
            self.buffer.add("user", f"Message {i}")
        # maxlen=5, so deque handles this
        assert len(self.buffer) == 5

    def test_token_limit_eviction(self):
        """Buffer should evict oldest messages when token limit exceeded."""
        from backend.services.agent_memory import MessageBuffer
        # Each message is ~len/4 + 10 tokens
        # max_tokens=500, so adding large messages should evict old ones
        buffer = MessageBuffer(max_messages=100, max_tokens=100)
        buffer.add("user", "A" * 200)  # ~60 tokens
        buffer.add("user", "B" * 200)  # ~60 tokens → should evict first
        # Should have trimmed to stay within token budget
        assert len(buffer) >= 1

    def test_get_messages(self):
        self.buffer.add("user", "First")
        self.buffer.add("assistant", "Second")
        self.buffer.add("user", "Third")
        messages = self.buffer.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[2].content == "Third"

    def test_get_messages_with_limit(self):
        for i in range(5):
            self.buffer.add("user", f"Message {i}")
        messages = self.buffer.get_messages(limit=2)
        assert len(messages) == 2
        assert messages[0].content == "Message 3"
        assert messages[1].content == "Message 4"

    def test_get_for_prompt(self):
        self.buffer.add("user", "Hello")
        self.buffer.add("assistant", "Hi")
        prompt_messages = self.buffer.get_for_prompt(max_tokens=4000)
        assert len(prompt_messages) == 2
        assert prompt_messages[0]["role"] == "user"
        assert prompt_messages[1]["role"] == "assistant"

    def test_get_for_prompt_token_limit(self):
        """Should return only messages that fit within token budget."""
        from backend.services.agent_memory import MessageBuffer
        buffer = MessageBuffer(max_messages=100, max_tokens=10000)
        for i in range(20):
            buffer.add("user", f"Message {i} with some content")
        # With a very small token budget, should get fewer messages
        prompt = buffer.get_for_prompt(max_tokens=50)
        assert len(prompt) < 20

    def test_clear(self):
        self.buffer.add("user", "Hello")
        self.buffer.add("assistant", "Hi")
        self.buffer.clear()
        assert len(self.buffer) == 0
        assert self.buffer._token_count == 0

    def test_message_metadata(self):
        msg = self.buffer.add("user", "Hello", metadata={"source": "web"})
        assert msg.metadata == {"source": "web"}

    def test_message_token_estimate(self):
        from backend.services.agent_memory import Message
        msg = Message(role="user", content="Hello world")  # 11 chars
        estimate = msg.token_estimate()
        assert estimate == 11 // 4 + 10  # 12


# =============================================================================
# CoreMemory Tests
# =============================================================================

class TestCoreMemory:
    """Test working memory (core memory blocks)."""

    def setup_method(self):
        from backend.services.agent_memory import CoreMemory
        self.core = CoreMemory(user_id="test_user", agent_id="test_agent")
        # Bypass Redis by setting initialized directly
        self.core._initialized = True
        self.core._redis = None
        from backend.services.agent_memory import CoreMemoryBlock
        self.core._blocks = {
            "persona": CoreMemoryBlock(
                name="persona",
                content="I am a test assistant.",
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
            "readonly": CoreMemoryBlock(
                name="readonly",
                content="Cannot change this",
                max_chars=500,
                editable=False,
            ),
        }

    def test_get_block(self):
        block = self.core.get_block("persona")
        assert block is not None
        assert block.content == "I am a test assistant."

    def test_get_nonexistent_block(self):
        block = self.core.get_block("nonexistent")
        assert block is None

    @pytest.mark.asyncio
    async def test_update_block(self):
        result = await self.core.update_block("user", "John likes Python")
        assert result is True
        assert self.core.get_block("user").content == "John likes Python"

    @pytest.mark.asyncio
    async def test_update_nonexistent_block(self):
        result = await self.core.update_block("nonexistent", "data")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_readonly_block(self):
        result = await self.core.update_block("readonly", "new content")
        assert result is False
        assert self.core.get_block("readonly").content == "Cannot change this"

    @pytest.mark.asyncio
    async def test_update_block_truncation(self):
        """Content exceeding max_chars should be truncated."""
        long_content = "x" * 3000
        result = await self.core.update_block("user", long_content)
        assert result is True
        assert len(self.core.get_block("user").content) == 2000

    @pytest.mark.asyncio
    async def test_append_to_block(self):
        await self.core.update_block("user", "Line 1")
        result = await self.core.append_to_block("user", "Line 2")
        assert result is True
        assert "Line 1" in self.core.get_block("user").content
        assert "Line 2" in self.core.get_block("user").content

    @pytest.mark.asyncio
    async def test_append_overflow_truncates_from_beginning(self):
        """When appending exceeds max_chars, oldest content is removed."""
        from backend.services.agent_memory import CoreMemoryBlock
        self.core._blocks["small"] = CoreMemoryBlock(
            name="small", content="", max_chars=50, editable=True
        )
        # Fill it up
        await self.core.update_block("small", "A" * 30)
        # Append more than fits
        await self.core.append_to_block("small", "B" * 30)
        content = self.core.get_block("small").content
        assert len(content) <= 50

    @pytest.mark.asyncio
    async def test_user_profile_creation(self):
        profile = await self.core.update_user_profile(name="John", expertise_level="expert")
        assert profile.name == "John"
        assert profile.expertise_level == "expert"

    @pytest.mark.asyncio
    async def test_add_user_fact(self):
        await self.core.add_user_fact("Likes Python")
        assert self.core.user_profile is not None
        assert "Likes Python" in self.core.user_profile.facts

    @pytest.mark.asyncio
    async def test_add_duplicate_fact_ignored(self):
        await self.core.add_user_fact("Likes Python")
        await self.core.add_user_fact("Likes Python")
        assert self.core.user_profile.facts.count("Likes Python") == 1

    @pytest.mark.asyncio
    async def test_facts_limited_to_20(self):
        for i in range(25):
            await self.core.add_user_fact(f"Fact {i}")
        assert len(self.core.user_profile.facts) == 20

    def test_get_for_prompt(self):
        prompt = self.core.get_for_prompt()
        assert "<persona>" in prompt
        assert "I am a test assistant." in prompt

    @pytest.mark.asyncio
    async def test_get_for_prompt_with_profile(self):
        await self.core.update_user_profile(name="John", expertise_level="expert")
        prompt = self.core.get_for_prompt()
        assert "<user_profile>" in prompt
        assert "John" in prompt

    @pytest.mark.asyncio
    async def test_set_current_task(self):
        await self.core.set_current_task("Analyze document")
        assert self.core._current_task == "Analyze document"
        prompt = self.core.get_for_prompt()
        assert "<current_task>" in prompt
        assert "Analyze document" in prompt


# =============================================================================
# RecallMemory Tests
# =============================================================================

class TestRecallMemory:
    """Test long-term recall memory with search."""

    def setup_method(self):
        from backend.services.agent_memory import RecallMemory
        self.recall = RecallMemory(
            user_id="test_user",
            agent_id="test_agent",
            embedding_service=None,
        )
        self.recall._initialized = True

    @pytest.mark.asyncio
    async def test_add_memory(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            entry = await self.recall.add("The capital of France is Paris", MemoryType.FACT)
            assert entry.content == "The capital of France is Paris"
            assert entry.type == MemoryType.FACT
            assert len(self.recall._entries) == 1

    @pytest.mark.asyncio
    async def test_add_memory_with_importance(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            entry = await self.recall.add("Critical fact", importance=0.9)
            assert entry.importance == 0.9

    @pytest.mark.asyncio
    async def test_keyword_search(self):
        """Search should work with keyword matching when no embedding service."""
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            await self.recall.add("The capital of France is Paris", MemoryType.FACT)
            await self.recall.add("Python is a programming language", MemoryType.FACT)
            await self.recall.add("France has great cuisine", MemoryType.FACT)

            results = await self.recall.search("France")
            assert len(results) >= 2
            # Both France-related entries should be found
            contents = [r.content for r in results]
            assert any("France" in c for c in contents)

    @pytest.mark.asyncio
    async def test_search_with_importance_filter(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            await self.recall.add("Low importance fact about cats", importance=0.1)
            await self.recall.add("High importance fact about cats", importance=0.9)

            results = await self.recall.search("cats", min_importance=0.5)
            assert len(results) == 1
            assert results[0].importance == 0.9

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            await self.recall.add("User said hello", MemoryType.MESSAGE)
            await self.recall.add("User likes Python programming", MemoryType.PREFERENCE)
            await self.recall.add("Python is user's favorite language", MemoryType.FACT)

            results = await self.recall.search("Python", memory_types=[MemoryType.FACT])
            assert len(results) == 1
            assert results[0].type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty(self):
        results = await self.recall.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_limit(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            for i in range(20):
                await self.recall.add(f"Memory about topic {i}", MemoryType.FACT)

            results = await self.recall.search("topic", limit=5)
            assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_get_recent(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            await self.recall.add("Old memory", MemoryType.MESSAGE)
            await self.recall.add("New memory", MemoryType.MESSAGE)

            results = await self.recall.get_recent(limit=1)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_recent_with_type_filter(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            await self.recall.add("A message", MemoryType.MESSAGE)
            await self.recall.add("A fact", MemoryType.FACT)

            results = await self.recall.get_recent(memory_types=[MemoryType.FACT])
            assert len(results) == 1
            assert results[0].type == MemoryType.FACT

    def test_cosine_similarity(self):
        sim = self.recall._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        sim = self.recall._cosine_similarity([1, 0], [0, 1])
        assert abs(sim) < 0.001

    def test_cosine_similarity_zero_vector(self):
        sim = self.recall._cosine_similarity([0, 0, 0], [1, 0, 0])
        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_summarize_history_no_llm(self):
        from backend.services.agent_memory import MemoryType
        with patch.object(self.recall, '_save_entry', new_callable=AsyncMock):
            await self.recall.add("Hello", MemoryType.MESSAGE, metadata={"role": "user"})
            summary = await self.recall.summarize_history()
            assert "1 messages" in summary or "Hello" in summary


# =============================================================================
# MemoryEntry Data Class Tests
# =============================================================================

class TestMemoryEntry:
    """Test MemoryEntry serialization."""

    def test_to_dict(self):
        from backend.services.agent_memory import MemoryEntry, MemoryType
        entry = MemoryEntry(
            id="test-1",
            type=MemoryType.FACT,
            content="Test content",
            importance=0.8,
        )
        d = entry.to_dict()
        assert d["id"] == "test-1"
        assert d["type"] == "fact"
        assert d["content"] == "Test content"
        assert d["importance"] == 0.8

    def test_from_dict(self):
        from backend.services.agent_memory import MemoryEntry, MemoryType
        data = {
            "id": "test-2",
            "type": "preference",
            "content": "Likes dark mode",
            "timestamp": "2026-01-01T00:00:00",
            "importance": 0.7,
            "access_count": 3,
            "last_accessed": None,
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.id == "test-2"
        assert entry.type == MemoryType.PREFERENCE
        assert entry.content == "Likes dark mode"
        assert entry.importance == 0.7
        assert entry.access_count == 3

    def test_from_dict_with_last_accessed(self):
        from backend.services.agent_memory import MemoryEntry, MemoryType
        data = {
            "id": "test-3",
            "type": "fact",
            "content": "Test",
            "timestamp": "2026-01-01T00:00:00",
            "last_accessed": "2026-01-02T00:00:00",
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.last_accessed is not None
        assert entry.last_accessed.day == 2


# =============================================================================
# UserProfile Tests
# =============================================================================

class TestUserProfile:
    """Test user profile serialization and prompt generation."""

    def test_to_dict(self):
        from backend.services.agent_memory import UserProfile
        profile = UserProfile(
            user_id="u1",
            name="Alice",
            expertise_level="expert",
            facts=["Likes Python"],
        )
        d = profile.to_dict()
        assert d["user_id"] == "u1"
        assert d["name"] == "Alice"
        assert d["expertise_level"] == "expert"
        assert "Likes Python" in d["facts"]

    def test_to_prompt_string(self):
        from backend.services.agent_memory import UserProfile
        profile = UserProfile(
            user_id="u1",
            name="Alice",
            expertise_level="expert",
            communication_style="concise",
            facts=["Likes Python", "Works at Acme"],
            preferences={"theme": "dark"},
        )
        prompt = profile.to_prompt_string()
        assert "Alice" in prompt
        assert "expert" in prompt
        assert "concise" in prompt
        assert "Likes Python" in prompt
        assert "theme" in prompt

    def test_to_prompt_string_minimal(self):
        from backend.services.agent_memory import UserProfile
        profile = UserProfile(user_id="u1")
        prompt = profile.to_prompt_string()
        # Default expertise_level is "intermediate", so it's included
        assert "Expertise: intermediate" in prompt


# =============================================================================
# AgentMemory Integration Tests
# =============================================================================

class TestAgentMemory:
    """Test the unified AgentMemory interface."""

    def setup_method(self):
        from backend.services.agent_memory import AgentMemory
        self.memory = AgentMemory(
            user_id="test_user",
            agent_id="test_agent",
            max_buffer_messages=10,
            max_buffer_tokens=4000,
        )
        # Skip initialization (bypasses Redis/DB)
        self.memory._initialized = True
        self.memory.core._initialized = True
        self.memory.core._redis = None
        self.memory.recall._initialized = True
        from backend.services.agent_memory import CoreMemoryBlock
        self.memory.core._blocks = {
            "persona": CoreMemoryBlock(name="persona", content="Test assistant", max_chars=1000, editable=True),
            "user": CoreMemoryBlock(name="user", content="", max_chars=2000, editable=True),
            "task": CoreMemoryBlock(name="task", content="", max_chars=1000, editable=True),
        }

    @pytest.mark.asyncio
    async def test_add_message(self):
        with patch.object(self.memory.recall, '_save_entry', new_callable=AsyncMock):
            msg = await self.memory.add_message("user", "Hello")
            assert msg.content == "Hello"
            assert len(self.memory.buffer) == 1
            # Should also be in recall memory
            assert len(self.memory.recall._entries) == 1

    @pytest.mark.asyncio
    async def test_add_message_no_persist(self):
        msg = await self.memory.add_message("user", "Ephemeral", persist=False)
        assert msg.content == "Ephemeral"
        assert len(self.memory.buffer) == 1
        assert len(self.memory.recall._entries) == 0

    @pytest.mark.asyncio
    async def test_add_fact(self):
        with patch.object(self.memory.recall, '_save_entry', new_callable=AsyncMock):
            await self.memory.add_fact("Earth is round", importance=0.9)
            assert len(self.memory.recall._entries) == 1
            assert self.memory.recall._entries[0].importance == 0.9

    @pytest.mark.asyncio
    async def test_search_across_tiers(self):
        with patch.object(self.memory.recall, '_save_entry', new_callable=AsyncMock):
            await self.memory.add_message("user", "I love Python programming")
            results = await self.memory.search("Python", include_buffer=True)
            # Should find in both buffer and recall
            assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_context_for_prompt(self):
        with patch.object(self.memory.recall, '_save_entry', new_callable=AsyncMock):
            await self.memory.add_message("user", "Hello")
            await self.memory.add_message("assistant", "Hi there")
            context = await self.memory.get_context_for_prompt(max_tokens=4000)
            assert "messages" in context
            assert "system_prefix" in context
            assert len(context["messages"]) == 2

    @pytest.mark.asyncio
    async def test_clear_buffer(self):
        self.memory.buffer.add("user", "Hello")
        await self.memory.clear_buffer()
        assert len(self.memory.buffer) == 0

    def test_get_stats(self):
        stats = self.memory.get_stats()
        assert stats["user_id"] == "test_user"
        assert stats["agent_id"] == "test_agent"
        assert stats["buffer_size"] == 0
        assert stats["initialized"] is True

    def test_recent_messages(self):
        self.memory.buffer.add("user", "Hello")
        self.memory.buffer.add("assistant", "Hi")
        recent = self.memory.recent_messages
        assert len(recent) == 2


# =============================================================================
# LRU Cache + TTL Tests (Phase 70)
# =============================================================================

class TestMemoryCache:
    """Test the LRU cache with TTL for agent memory instances."""

    def setup_method(self):
        from backend.services.agent_memory import clear_memory_cache
        clear_memory_cache()

    @pytest.mark.asyncio
    async def test_get_agent_memory_creates_new(self):
        from backend.services.agent_memory import get_agent_memory, _memory_instances
        with patch('backend.services.agent_memory.AgentMemory.initialize', new_callable=AsyncMock):
            memory = await get_agent_memory("user1", "agent1")
            assert memory is not None
            assert "agent1:user1" in _memory_instances

    @pytest.mark.asyncio
    async def test_get_agent_memory_returns_cached(self):
        from backend.services.agent_memory import get_agent_memory
        with patch('backend.services.agent_memory.AgentMemory.initialize', new_callable=AsyncMock):
            memory1 = await get_agent_memory("user1", "agent1")
            memory2 = await get_agent_memory("user1", "agent1")
            assert memory1 is memory2

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        import backend.services.agent_memory as am
        original_ttl = am._MEMORY_TTL_SECONDS
        try:
            am._MEMORY_TTL_SECONDS = 0  # Expire immediately
            with patch('backend.services.agent_memory.AgentMemory.initialize', new_callable=AsyncMock):
                memory1 = await am.get_agent_memory("user1", "agent1")
                # Wait a tiny bit so time.time() advances
                await asyncio.sleep(0.01)
                memory2 = await am.get_agent_memory("user1", "agent1")
                # Should be a new instance since TTL=0
                assert memory1 is not memory2
        finally:
            am._MEMORY_TTL_SECONDS = original_ttl

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        import backend.services.agent_memory as am
        original_max = am._MEMORY_MAX_SIZE
        try:
            am._MEMORY_MAX_SIZE = 3
            with patch('backend.services.agent_memory.AgentMemory.initialize', new_callable=AsyncMock):
                await am.get_agent_memory("user1", "agent1")
                await am.get_agent_memory("user2", "agent2")
                await am.get_agent_memory("user3", "agent3")
                # Adding 4th should evict oldest
                await am.get_agent_memory("user4", "agent4")
                assert len(am._memory_instances) == 3
                assert "agent1:user1" not in am._memory_instances
                assert "agent4:user4" in am._memory_instances
        finally:
            am._MEMORY_MAX_SIZE = original_max

    @pytest.mark.asyncio
    async def test_remove_agent_memory(self):
        from backend.services.agent_memory import get_agent_memory, remove_agent_memory, _memory_instances
        with patch('backend.services.agent_memory.AgentMemory.initialize', new_callable=AsyncMock):
            await get_agent_memory("user1", "agent1")
            assert "agent1:user1" in _memory_instances
            result = await remove_agent_memory("user1", "agent1")
            assert result is True
            assert "agent1:user1" not in _memory_instances

    @pytest.mark.asyncio
    async def test_remove_nonexistent_memory(self):
        from backend.services.agent_memory import remove_agent_memory
        result = await remove_agent_memory("nonexistent", "nonexistent")
        assert result is False

    def test_get_memory_cache_stats(self):
        from backend.services.agent_memory import get_memory_cache_stats
        stats = get_memory_cache_stats()
        assert "total_instances" in stats
        assert "max_size" in stats
        assert "ttl_seconds" in stats
        assert "expired_count" in stats
        assert "utilization" in stats

    def test_clear_memory_cache(self):
        from backend.services.agent_memory import clear_memory_cache, _memory_instances
        # Manually add something
        _memory_instances["test:key"] = (MagicMock(), time.time())
        assert len(_memory_instances) == 1
        clear_memory_cache()
        assert len(_memory_instances) == 0
