"""
Query Cache Service - LightRAG-inspired Q&A caching.

Caches query-answer pairs with similarity threshold matching.
If a new query is semantically similar (>= 0.95) to a cached query,
the cached answer is returned to save LLM calls.

Inspired by LightRAG's query caching implementation.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import structlog
import numpy as np

logger = structlog.get_logger(__name__)


@dataclass
class CachedQuery:
    """A cached query with its answer and embedding."""

    query: str
    answer: str
    embedding: List[float]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryCache:
    """
    Query-answer cache with semantic similarity matching.

    Features:
    - Semantic matching with configurable threshold (default 0.95)
    - LRU eviction when cache exceeds max size
    - TTL-based expiration
    - Thread-safe operations
    - Metrics tracking

    Example:
        >>> cache = QueryCache(similarity_threshold=0.95)
        >>> # Check for cached answer
        >>> cached = await cache.get_cached_answer(query, query_embedding)
        >>> if cached:
        ...     return cached.answer
        >>> # Generate and cache new answer
        >>> answer = await generate_answer(query)
        >>> await cache.cache_answer(query, answer, query_embedding)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 10000,
        ttl_hours: int = 24,
        enable_metrics: bool = True,
    ):
        """
        Initialize the query cache.

        Args:
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            max_entries: Maximum number of cached entries
            ttl_hours: Time-to-live for cache entries in hours
            enable_metrics: Whether to track cache metrics
        """
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = timedelta(hours=ttl_hours)
        self.enable_metrics = enable_metrics

        self._cache: Dict[str, CachedQuery] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get_cached_answer(
        self,
        query: str,
        query_embedding: List[float],
        collection_id: Optional[str] = None,
    ) -> Optional[CachedQuery]:
        """
        Check if a similar query exists in cache.

        Args:
            query: The query string
            query_embedding: The query's embedding vector
            collection_id: Optional collection scope for caching

        Returns:
            CachedQuery if found, None otherwise
        """
        async with self._lock:
            now = datetime.utcnow()

            # Search through cache for similar queries
            best_match: Optional[CachedQuery] = None
            best_similarity = 0.0

            expired_keys = []

            for key, cached in self._cache.items():
                # Check TTL
                if now - cached.created_at > self.ttl:
                    expired_keys.append(key)
                    continue

                # Check collection scope if specified
                if collection_id:
                    cached_collection = cached.metadata.get("collection_id")
                    if cached_collection and cached_collection != collection_id:
                        continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, cached.embedding)

                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_match = cached
                    best_similarity = similarity

            # Clean up expired entries
            for key in expired_keys:
                del self._cache[key]
                self._evictions += 1

            if best_match:
                # Update access stats
                best_match.last_accessed = now
                best_match.access_count += 1
                self._hits += 1

                logger.debug(
                    "Query cache hit",
                    similarity=round(best_similarity, 4),
                    access_count=best_match.access_count,
                )
                return best_match

            self._misses += 1
            return None

    async def cache_answer(
        self,
        query: str,
        answer: str,
        query_embedding: List[float],
        collection_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Cache a query-answer pair.

        Args:
            query: The query string
            answer: The generated answer
            query_embedding: The query's embedding vector
            collection_id: Optional collection scope
            metadata: Optional additional metadata

        Returns:
            The cache key
        """
        async with self._lock:
            # Generate cache key
            key = self._generate_key(query, collection_id)

            # Create cache entry
            entry_metadata = metadata or {}
            if collection_id:
                entry_metadata["collection_id"] = collection_id

            now = datetime.utcnow()
            cached = CachedQuery(
                query=query,
                answer=answer,
                embedding=query_embedding,
                created_at=now,
                last_accessed=now,
                metadata=entry_metadata,
            )

            # Evict if at capacity
            if len(self._cache) >= self.max_entries:
                await self._evict_lru()

            self._cache[key] = cached

            logger.debug("Query cached", key=key[:16], query_len=len(query))
            return key

    async def invalidate(
        self,
        query: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            query: Specific query to invalidate
            collection_id: Invalidate all entries for collection

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            if query:
                # Invalidate specific query
                key = self._generate_key(query, collection_id)
                if key in self._cache:
                    del self._cache[key]
                    return 1
                return 0

            if collection_id:
                # Invalidate all entries for collection
                keys_to_remove = [
                    k
                    for k, v in self._cache.items()
                    if v.metadata.get("collection_id") == collection_id
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                return len(keys_to_remove)

            # Clear all
            count = len(self._cache)
            self._cache.clear()
            return count

    async def _evict_lru(self) -> None:
        """Evict least recently used entries (20% of max)."""
        if not self._cache:
            return

        # Sort by last accessed time
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed,
        )

        # Remove oldest 20%
        evict_count = max(1, int(self.max_entries * 0.2))
        for key, _ in sorted_entries[:evict_count]:
            del self._cache[key]
            self._evictions += 1

        logger.debug("Evicted LRU entries", count=evict_count)

    def _generate_key(self, query: str, collection_id: Optional[str] = None) -> str:
        """Generate a cache key from query and collection."""
        content = f"{query}:{collection_id or 'default'}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))
        except Exception:
            return 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "evictions": self._evictions,
            "similarity_threshold": self.similarity_threshold,
        }

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0


# Global singleton
_query_cache: Optional[QueryCache] = None


def get_query_cache(
    similarity_threshold: float = 0.95,
    max_entries: int = 10000,
    ttl_hours: int = 24,
) -> QueryCache:
    """Get or create the global query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(
            similarity_threshold=similarity_threshold,
            max_entries=max_entries,
            ttl_hours=ttl_hours,
        )
    return _query_cache
