"""
AIDocumentIndexer - A-Mem Agentic Memory System
================================================

Phase 48: Implements A-Mem for efficient agent memory management.

Key benefits (from research):
- 85-93% token reduction vs MemGPT baseline
- <$0.0003 per memory operation
- 1.1 seconds processing with Llama 3.2 1B locally
- Selective top-k retrieval mechanism

A-Mem uses an agentic approach where memory operations are performed
by a small, efficient LLM that decides what to remember, forget, and
retrieve based on relevance and recency.

Research sources:
- https://arxiv.org/pdf/2502.12110 (A-Mem Paper)
- https://github.com/Shichun-Liu/Agent-Memory-Paper-List
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


class MemoryOperation(str, Enum):
    """Memory operations the agent can perform."""
    STORE = "store"           # Store new memory
    RETRIEVE = "retrieve"     # Retrieve relevant memories
    UPDATE = "update"         # Update existing memory
    FORGET = "forget"         # Remove memory
    CONSOLIDATE = "consolidate"  # Merge related memories
    COMPRESS = "compress"     # Compress memory representation


class MemoryImportance(str, Enum):
    """Memory importance levels."""
    CRITICAL = "critical"     # Never forget (user preferences, key facts)
    HIGH = "high"             # Important, slow decay
    MEDIUM = "medium"         # Standard decay
    LOW = "low"               # Fast decay, can be forgotten
    EPHEMERAL = "ephemeral"   # Very short-lived, immediate tasks


@dataclass
class AMemConfig:
    """Configuration for A-Mem system."""
    # Memory capacity
    max_memories: int = 1000
    max_tokens_per_memory: int = 256

    # Retrieval settings
    top_k: int = 5
    relevance_threshold: float = 0.7

    # Decay settings
    decay_rate: float = 0.01         # Per-hour decay
    min_importance: float = 0.1      # Threshold for forgetting

    # Consolidation settings
    consolidation_threshold: float = 0.85  # Similarity for merging
    consolidation_interval: int = 100      # Operations between consolidations

    # Agent model settings
    agent_model: str = "llama-3.2-1b"  # Small model for memory ops
    use_local_model: bool = True

    # Token budget
    max_tokens_per_operation: int = 500
    max_context_tokens: int = 2048

    # Performance
    enable_caching: bool = True
    batch_operations: bool = True


@dataclass
class Memory:
    """A single memory unit in A-Mem."""
    id: str
    content: str
    importance: MemoryImportance
    score: float  # Current relevance score (0-1)
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    linked_memories: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'importance': self.importance.value,
            'score': self.score,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'metadata': self.metadata,
            'linked_memories': list(self.linked_memories),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            importance=MemoryImportance(data['importance']),
            score=data['score'],
            created_at=datetime.fromisoformat(data['created_at']),
            accessed_at=datetime.fromisoformat(data['accessed_at']),
            access_count=data.get('access_count', 0),
            metadata=data.get('metadata', {}),
            linked_memories=set(data.get('linked_memories', [])),
        )


@dataclass
class MemoryOperationResult:
    """Result of a memory operation."""
    operation: MemoryOperation
    success: bool
    memories_affected: List[str]
    tokens_used: int
    latency_ms: float
    cost_usd: float
    details: Dict[str, Any] = field(default_factory=dict)


class MemoryAgent:
    """
    The core memory agent that decides memory operations.

    Uses a small LLM to make intelligent decisions about:
    - What to remember
    - How to organize memories
    - When to forget
    - What to retrieve
    """

    DECISION_PROMPT = """You are a memory management agent. Given the context and query, decide the best memory operation.

Current memories (summarized):
{memory_summary}

Query/Input: {query}

Available operations:
- STORE: Save new information (if it's novel and important)
- RETRIEVE: Get relevant existing memories (if query asks for past info)
- UPDATE: Modify existing memory (if new info updates old)
- FORGET: Remove irrelevant memories (if cleanup needed)
- CONSOLIDATE: Merge similar memories (if duplicates exist)

Respond with JSON:
{{
    "operation": "STORE|RETRIEVE|UPDATE|FORGET|CONSOLIDATE",
    "reason": "brief explanation",
    "memory_ids": ["id1", "id2"],  // for RETRIEVE/UPDATE/FORGET/CONSOLIDATE
    "new_content": "content",  // for STORE/UPDATE
    "importance": "critical|high|medium|low|ephemeral"  // for STORE/UPDATE
}}"""

    def __init__(self, config: AMemConfig):
        self.config = config
        self._llm = None

    async def _get_llm(self):
        """Get or create LLM for memory operations."""
        if self._llm is not None:
            return self._llm

        try:
            if self.config.use_local_model:
                # Try to use local Ollama via factory (provider-agnostic)
                from backend.services.llm import LLMFactory
                self._llm = LLMFactory.get_chat_model(
                    provider="ollama",
                    model=self.config.agent_model,
                    temperature=0.1,
                )
            else:
                # Use API-based model via factory
                from backend.services.llm import LLMFactory
                self._llm = LLMFactory.get_chat_model(
                    model=self.config.agent_model,
                    temperature=0.1,
                )

            return self._llm

        except Exception as e:
            logger.warning(f"Failed to load memory agent LLM: {e}")
            return None

    async def decide_operation(
        self,
        query: str,
        memory_summary: str,
        available_memory_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Decide which memory operation to perform.

        Args:
            query: The input/query from the user
            memory_summary: Summary of current memories
            available_memory_ids: IDs of memories that can be affected

        Returns:
            Decision dict with operation, reason, and details
        """
        llm = await self._get_llm()

        if llm is None:
            # Fallback: simple heuristic-based decision
            return self._heuristic_decision(query, memory_summary)

        try:
            prompt = self.DECISION_PROMPT.format(
                memory_summary=memory_summary,
                query=query,
            )

            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                decision = json.loads(content[start:end])
            else:
                decision = self._heuristic_decision(query, memory_summary)

            return decision

        except Exception as e:
            logger.warning(f"Memory agent decision failed: {e}")
            return self._heuristic_decision(query, memory_summary)

    def _heuristic_decision(
        self,
        query: str,
        memory_summary: str,
    ) -> Dict[str, Any]:
        """Fallback heuristic-based decision."""
        query_lower = query.lower()

        # Simple keyword-based heuristics
        if any(kw in query_lower for kw in ['remember', 'save', 'store', 'note']):
            return {
                'operation': 'STORE',
                'reason': 'User explicitly wants to save information',
                'new_content': query,
                'importance': 'medium',
            }
        elif any(kw in query_lower for kw in ['recall', 'what did', 'remember when', 'previously']):
            return {
                'operation': 'RETRIEVE',
                'reason': 'User asking about past information',
                'memory_ids': [],
            }
        elif any(kw in query_lower for kw in ['forget', 'remove', 'delete']):
            return {
                'operation': 'FORGET',
                'reason': 'User wants to remove information',
                'memory_ids': [],
            }
        else:
            # Default: store if it seems informative
            if len(query.split()) > 5:
                return {
                    'operation': 'STORE',
                    'reason': 'New information to remember',
                    'new_content': query,
                    'importance': 'low',
                }
            else:
                return {
                    'operation': 'RETRIEVE',
                    'reason': 'Short query, likely asking for info',
                    'memory_ids': [],
                }


class AMemMemoryService:
    """
    A-Mem Agentic Memory Service.

    Provides intelligent memory management with:
    - Automatic importance scoring
    - Decay-based forgetting
    - Consolidation of similar memories
    - Efficient top-k retrieval

    Usage:
        service = AMemMemoryService()

        # Store memory
        await service.store("User prefers dark mode", importance="high")

        # Retrieve relevant memories
        memories = await service.retrieve("What are user preferences?")

        # Process with automatic operation selection
        result = await service.process("Remember that I like Python")
    """

    def __init__(self, config: Optional[AMemConfig] = None):
        self.config = config or AMemConfig()
        self._memories: Dict[str, Memory] = {}
        self._agent = MemoryAgent(self.config)
        self._operation_count = 0
        # LRU cache with max 5000 entries to prevent memory bloat
        self._embeddings_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._embeddings_cache_max_size = 5000
        self._cache_lock = asyncio.Lock()  # Thread-safe for async operations

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()[:12]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with thread-safe LRU caching."""
        # Check cache first (with lock)
        async with self._cache_lock:
            if text in self._embeddings_cache:
                # Move to end (most recently used)
                self._embeddings_cache.move_to_end(text)
                return self._embeddings_cache[text]

        # Generate embedding outside lock to avoid blocking
        try:
            from backend.services.embeddings import get_embedding_service
            service = get_embedding_service()
            embedding = service.embed_text(text)

            # Cache it with LRU eviction (with lock)
            if self.config.enable_caching:
                async with self._cache_lock:
                    # Evict oldest entries if at capacity
                    while len(self._embeddings_cache) >= self._embeddings_cache_max_size:
                        self._embeddings_cache.popitem(last=False)
                    self._embeddings_cache[text] = embedding

            return embedding

        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            # Return zero embedding
            return [0.0] * 768

    def _compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between embeddings."""
        try:
            import numpy as np

            a = np.array(embedding1)
            b = np.array(embedding2)

            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        except Exception as e:
            logger.debug("Cosine similarity calculation failed", error=str(e))
            return 0.0

    def _apply_decay(self, memory: Memory) -> float:
        """Apply time-based decay to memory score."""
        hours_since_access = (datetime.now() - memory.accessed_at).total_seconds() / 3600

        # Importance-based decay multiplier
        decay_multipliers = {
            MemoryImportance.CRITICAL: 0.0,    # No decay
            MemoryImportance.HIGH: 0.25,
            MemoryImportance.MEDIUM: 1.0,
            MemoryImportance.LOW: 2.0,
            MemoryImportance.EPHEMERAL: 4.0,
        }

        multiplier = decay_multipliers.get(memory.importance, 1.0)
        decay = self.config.decay_rate * multiplier * hours_since_access

        new_score = max(0.0, memory.score - decay)

        # Boost from access count
        access_boost = min(0.2, memory.access_count * 0.02)
        new_score = min(1.0, new_score + access_boost)

        return new_score

    def _get_memory_summary(self, limit: int = 10) -> str:
        """Get summary of current memories for agent."""
        # Sort by score and take top memories
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.score,
            reverse=True,
        )[:limit]

        if not sorted_memories:
            return "No memories stored yet."

        lines = []
        for mem in sorted_memories:
            preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
            lines.append(f"[{mem.id}] ({mem.importance.value}) {preview}")

        return "\n".join(lines)

    async def store(
        self,
        content: str,
        importance: Union[str, MemoryImportance] = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryOperationResult:
        """
        Store a new memory.

        Args:
            content: Content to remember
            importance: Importance level
            metadata: Optional metadata

        Returns:
            MemoryOperationResult
        """
        start_time = time.time()

        if isinstance(importance, str):
            importance = MemoryImportance(importance)

        # Generate ID and embedding
        memory_id = self._generate_id(content)
        embedding = await self._get_embedding(content)

        # Create memory
        memory = Memory(
            id=memory_id,
            content=content,
            importance=importance,
            score=1.0,  # Start with full score
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            metadata=metadata or {},
            embedding=embedding,
        )

        # Check capacity
        if len(self._memories) >= self.config.max_memories:
            await self._evict_lowest_score()

        self._memories[memory_id] = memory
        self._operation_count += 1

        # Maybe consolidate
        if self._operation_count % self.config.consolidation_interval == 0:
            asyncio.create_task(self._consolidate_memories())

        latency_ms = (time.time() - start_time) * 1000

        return MemoryOperationResult(
            operation=MemoryOperation.STORE,
            success=True,
            memories_affected=[memory_id],
            tokens_used=len(content.split()),
            latency_ms=latency_ms,
            cost_usd=0.0001,  # Estimated embedding cost
            details={'importance': importance.value},
        )

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        importance_filter: Optional[MemoryImportance] = None,
    ) -> List[Memory]:
        """
        Retrieve relevant memories.

        Args:
            query: Query to match against
            top_k: Number of memories to retrieve
            importance_filter: Only return memories of this importance or higher

        Returns:
            List of relevant memories
        """
        if not self._memories:
            return []

        top_k = top_k or self.config.top_k
        query_embedding = await self._get_embedding(query)

        # Score all memories
        scored = []
        for memory in self._memories.values():
            # Apply decay
            decayed_score = self._apply_decay(memory)

            # Compute relevance
            if memory.embedding:
                similarity = self._compute_similarity(query_embedding, memory.embedding)
            else:
                similarity = 0.0

            # Combined score
            combined = 0.6 * similarity + 0.4 * decayed_score

            # Filter by importance
            if importance_filter:
                importance_order = [
                    MemoryImportance.EPHEMERAL,
                    MemoryImportance.LOW,
                    MemoryImportance.MEDIUM,
                    MemoryImportance.HIGH,
                    MemoryImportance.CRITICAL,
                ]
                if importance_order.index(memory.importance) < importance_order.index(importance_filter):
                    continue

            if combined >= self.config.relevance_threshold:
                scored.append((memory, combined))

        # Use heapq for O(n) top-k selection instead of O(n log n) full sort
        import heapq
        if len(scored) <= top_k:
            # If we have fewer items than top_k, just sort them
            scored.sort(key=lambda x: x[1], reverse=True)
            top_items = scored
        else:
            # heapq.nlargest is O(n log k) which is faster than O(n log n) for small k
            top_items = heapq.nlargest(top_k, scored, key=lambda x: x[1])

        results = [mem for mem, _ in top_items]

        # Update access times and counts
        for mem in results:
            mem.accessed_at = datetime.now()
            mem.access_count += 1

        return results

    async def update(
        self,
        memory_id: str,
        new_content: Optional[str] = None,
        new_importance: Optional[MemoryImportance] = None,
    ) -> MemoryOperationResult:
        """
        Update an existing memory.

        Args:
            memory_id: ID of memory to update
            new_content: New content (optional)
            new_importance: New importance level (optional)

        Returns:
            MemoryOperationResult
        """
        start_time = time.time()

        if memory_id not in self._memories:
            return MemoryOperationResult(
                operation=MemoryOperation.UPDATE,
                success=False,
                memories_affected=[],
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0.0,
                details={'error': 'Memory not found'},
            )

        memory = self._memories[memory_id]

        if new_content:
            memory.content = new_content
            memory.embedding = await self._get_embedding(new_content)

        if new_importance:
            memory.importance = new_importance

        memory.accessed_at = datetime.now()
        memory.score = 1.0  # Refresh score on update

        latency_ms = (time.time() - start_time) * 1000

        return MemoryOperationResult(
            operation=MemoryOperation.UPDATE,
            success=True,
            memories_affected=[memory_id],
            tokens_used=len(new_content.split()) if new_content else 0,
            latency_ms=latency_ms,
            cost_usd=0.0001,
        )

    async def forget(
        self,
        memory_ids: Optional[List[str]] = None,
        below_score: Optional[float] = None,
    ) -> MemoryOperationResult:
        """
        Forget (remove) memories.

        Args:
            memory_ids: Specific IDs to forget
            below_score: Forget all memories below this score

        Returns:
            MemoryOperationResult
        """
        start_time = time.time()
        removed = []

        if memory_ids:
            for mid in memory_ids:
                if mid in self._memories:
                    del self._memories[mid]
                    removed.append(mid)

        if below_score is not None:
            to_remove = [
                mid for mid, mem in self._memories.items()
                if self._apply_decay(mem) < below_score
            ]
            for mid in to_remove:
                del self._memories[mid]
                removed.append(mid)

        latency_ms = (time.time() - start_time) * 1000

        return MemoryOperationResult(
            operation=MemoryOperation.FORGET,
            success=True,
            memories_affected=removed,
            tokens_used=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
        )

    async def _evict_lowest_score(self) -> bool:
        """Evict memory with lowest score. Returns True if eviction succeeded."""
        if not self._memories:
            return False

        # Find memory with lowest decayed score (excluding CRITICAL)
        lowest_id = None
        lowest_score = float('inf')

        for mid, mem in self._memories.items():
            if mem.importance == MemoryImportance.CRITICAL:
                continue

            score = self._apply_decay(mem)
            if score < lowest_score:
                lowest_score = score
                lowest_id = mid

        if lowest_id:
            del self._memories[lowest_id]
            logger.debug(f"Evicted memory {lowest_id} with score {lowest_score}")
            return True

        # All memories are CRITICAL — force-evict oldest by creation time
        if self._memories:
            oldest_id = min(self._memories, key=lambda mid: self._memories[mid].created_at)
            del self._memories[oldest_id]
            logger.warning(f"Force-evicted CRITICAL memory {oldest_id} (all memories are CRITICAL, store at capacity)")
            return True

        return False

    async def _consolidate_memories(self):
        """Consolidate similar memories."""
        start_time = time.time()
        consolidated = []

        memory_list = list(self._memories.values())

        for i, mem1 in enumerate(memory_list):
            if mem1.id in consolidated:
                continue

            for mem2 in memory_list[i + 1:]:
                if mem2.id in consolidated:
                    continue

                if mem1.embedding and mem2.embedding:
                    similarity = self._compute_similarity(mem1.embedding, mem2.embedding)

                    if similarity >= self.config.consolidation_threshold:
                        # Merge mem2 into mem1
                        mem1.content = f"{mem1.content}\n{mem2.content}"
                        mem1.embedding = await self._get_embedding(mem1.content)
                        mem1.score = max(mem1.score, mem2.score)
                        mem1.linked_memories.add(mem2.id)

                        # Remove mem2
                        if mem2.id in self._memories:
                            del self._memories[mem2.id]
                        consolidated.append(mem2.id)

        if consolidated:
            logger.info(
                f"Consolidated {len(consolidated)} memories in "
                f"{(time.time() - start_time) * 1000:.1f}ms"
            )

    async def process(
        self,
        input_text: str,
        context: Optional[str] = None,
    ) -> MemoryOperationResult:
        """
        Process input with automatic operation selection.

        The memory agent decides the best operation based on the input.

        Args:
            input_text: Input text to process
            context: Optional context for better decision making

        Returns:
            MemoryOperationResult
        """
        start_time = time.time()

        # Get memory summary for agent
        memory_summary = self._get_memory_summary()
        memory_ids = list(self._memories.keys())

        # Ask agent to decide
        decision = await self._agent.decide_operation(
            query=input_text,
            memory_summary=memory_summary,
            available_memory_ids=memory_ids,
        )

        operation = decision.get('operation', 'STORE').upper()

        # Execute the decided operation
        if operation == 'STORE':
            content = decision.get('new_content', input_text)
            importance = decision.get('importance', 'medium')
            result = await self.store(content, importance)

        elif operation == 'RETRIEVE':
            memories = await self.retrieve(input_text)
            result = MemoryOperationResult(
                operation=MemoryOperation.RETRIEVE,
                success=True,
                memories_affected=[m.id for m in memories],
                tokens_used=len(input_text.split()),
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0.0001,
                details={'memories': [m.to_dict() for m in memories]},
            )

        elif operation == 'UPDATE':
            memory_ids = decision.get('memory_ids', [])
            new_content = decision.get('new_content')
            if memory_ids:
                result = await self.update(memory_ids[0], new_content)
            else:
                result = await self.store(input_text)

        elif operation == 'FORGET':
            memory_ids = decision.get('memory_ids', [])
            result = await self.forget(memory_ids=memory_ids)

        elif operation == 'CONSOLIDATE':
            await self._consolidate_memories()
            result = MemoryOperationResult(
                operation=MemoryOperation.CONSOLIDATE,
                success=True,
                memories_affected=[],
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0.0,
            )

        else:
            # Default to store
            result = await self.store(input_text)

        result.details['agent_decision'] = decision
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        importance_counts = {}
        for mem in self._memories.values():
            imp = mem.importance.value
            importance_counts[imp] = importance_counts.get(imp, 0) + 1

        return {
            'total_memories': len(self._memories),
            'by_importance': importance_counts,
            'operation_count': self._operation_count,
            'cache_size': len(self._embeddings_cache),
            'config': {
                'max_memories': self.config.max_memories,
                'decay_rate': self.config.decay_rate,
                'top_k': self.config.top_k,
            },
        }

    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories as dictionaries."""
        return [mem.to_dict() for mem in self._memories.values()]

    def import_memories(self, memories: List[Dict[str, Any]]):
        """Import memories from dictionaries."""
        for mem_dict in memories:
            try:
                memory = Memory.from_dict(mem_dict)
                self._memories[memory.id] = memory
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")

    def clear(self):
        """Clear all memories."""
        self._memories.clear()
        self._embeddings_cache.clear()
        self._operation_count = 0
        logger.info("A-Mem memory cleared")


# Factory function
_service_instance: Optional[AMemMemoryService] = None


async def get_amem_service(
    config: Optional[AMemConfig] = None,
) -> AMemMemoryService:
    """Get or create A-Mem service instance."""
    global _service_instance

    if _service_instance is None or config is not None:
        _service_instance = AMemMemoryService(config)

    return _service_instance


__all__ = [
    "MemoryOperation",
    "MemoryImportance",
    "AMemConfig",
    "Memory",
    "MemoryOperationResult",
    "MemoryAgent",
    "AMemMemoryService",
    "get_amem_service",
]
