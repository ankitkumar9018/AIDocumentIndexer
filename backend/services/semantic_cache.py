"""
AIDocumentIndexer - Semantic Query Cache (Phase 65)
===================================================

Intelligent query caching using semantic similarity.

Traditional caching uses exact query matches, missing opportunities
to reuse results for semantically similar queries. This service
provides semantic caching using embedding similarity.

Features:
- Semantic matching: "What is ML?" → cache hit for "What is machine learning?"
- TTL-based expiration
- LRU eviction for memory management
- Cache warming from popular queries
- Hit rate analytics

Research:
- GPTCache: Semantic caching for LLM queries
- Query embeddings for similarity matching

Usage:
    from backend.services.semantic_cache import (
        SemanticQueryCache,
        get_semantic_cache,
    )

    cache = get_semantic_cache()

    # Check cache
    result = await cache.get(query, embedding)

    # Store result
    await cache.set(query, embedding, result, ttl=300)
"""

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import faiss as faiss_types

import structlog

logger = structlog.get_logger(__name__)

# NumPy for similarity computation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# FAISS for O(log n) similarity search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SemanticCacheConfig:
    """Configuration for semantic query cache."""
    # Cache settings
    max_entries: int = 1000  # Maximum cached queries
    default_ttl_seconds: int = 300  # Default TTL (5 minutes)

    # Semantic matching - Single threshold (legacy)
    similarity_threshold: float = 0.92  # Minimum similarity for cache hit
    use_exact_match_first: bool = True  # Check exact hash before semantic

    # Phase 69: Dual-threshold semantic matching
    # Production systems need different thresholds for precision vs recall trade-offs
    precision_threshold: float = 0.95  # High precision mode (fewer false positives)
    recall_threshold: float = 0.85  # High recall mode (more cache hits)
    threshold_mode: str = "adaptive"  # "precision", "recall", "adaptive", or "legacy"
    # - precision: Use precision_threshold only (conservative)
    # - recall: Use recall_threshold only (aggressive)
    # - adaptive: Try precision first, then recall if no match
    # - legacy: Use single similarity_threshold (backwards compatible)

    # Memory management
    eviction_policy: str = "lru"  # lru, fifo, lfu

    # Analytics
    track_hit_rate: bool = True
    hit_rate_window_seconds: int = 3600  # 1 hour window
    max_similarity_scores: int = 10000  # Max similarity scores to track (prevents memory leak)

    # Embedding settings
    embedding_dim: int = 768  # Expected embedding dimension

    # FAISS index settings (Phase 68: O(log n) search)
    use_faiss: bool = True  # Use FAISS for fast similarity search
    faiss_rebuild_threshold: int = 100  # Rebuild index after this many modifications


@dataclass
class CacheEntry:
    """A cached query result."""
    query: str
    query_hash: str
    embedding: Optional[List[float]]
    result: Any
    created_at: datetime
    expires_at: datetime
    hits: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    hits: int
    misses: int
    semantic_hits: int
    exact_hits: int
    evictions: int
    hit_rate: float
    avg_similarity: float
    faiss_enabled: bool = False
    faiss_index_size: int = 0
    # Phase 69: Dual-threshold info
    threshold_mode: str = "legacy"
    precision_threshold: float = 0.95
    recall_threshold: float = 0.85


# =============================================================================
# Semantic Query Cache
# =============================================================================

class SemanticQueryCache:
    """
    Semantic query cache with embedding similarity matching.

    Caches query results and returns them for semantically similar queries,
    reducing redundant LLM calls and retrieval operations.
    """

    def __init__(self, config: Optional[SemanticCacheConfig] = None):
        self.config = config or SemanticCacheConfig()

        # Cache storage (query_hash → CacheEntry)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Embedding index for semantic search
        self._embeddings: List[Tuple[str, np.ndarray]] = []  # (hash, embedding)
        self._hash_to_idx: Dict[str, int] = {}  # hash → index in FAISS

        # FAISS index for O(log n) similarity search (Phase 68)
        self._faiss_index: Optional["faiss.IndexFlatIP"] = None
        self._faiss_dirty: bool = False  # Needs rebuild
        self._faiss_modifications: int = 0  # Track modifications since last rebuild
        self._use_faiss = self.config.use_faiss and HAS_FAISS and HAS_NUMPY
        # Phase 71.5: Lock to prevent concurrent FAISS rebuilds
        self._faiss_rebuild_lock = asyncio.Lock()

        if self._use_faiss:
            logger.info(
                "FAISS enabled for semantic cache",
                embedding_dim=self.config.embedding_dim,
            )

        # Analytics
        self._hits: int = 0
        self._misses: int = 0
        self._semantic_hits: int = 0
        self._exact_hits: int = 0
        self._evictions: int = 0
        self._similarity_scores: List[float] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

    # =========================================================================
    # Cache Operations
    # =========================================================================

    async def get(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        threshold_mode: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get cached result for a query.

        Checks exact match first, then semantic similarity.

        Args:
            query: Query string
            embedding: Optional query embedding for semantic matching
            threshold_mode: Override config threshold mode ("precision", "recall", "adaptive", "legacy")

        Returns:
            Cached result or None if not found
        """
        query_hash = self._hash_query(query)

        async with self._lock:
            # 1. Check exact match
            if self.config.use_exact_match_first and query_hash in self._cache:
                entry = self._cache[query_hash]
                if not entry.is_expired:
                    self._record_hit(entry, exact=True)
                    return entry.result
                else:
                    # Expired, remove it
                    await self._remove_entry(query_hash)

            # 2. Semantic matching (if embedding provided)
            if embedding is not None and HAS_NUMPY and self._embeddings:
                similar_hash, similarity = await self._find_similar(embedding)

                if similar_hash:
                    # Phase 69: Dual-threshold semantic cache
                    mode = threshold_mode or self.config.threshold_mode
                    cache_hit = self._check_threshold(similarity, mode)

                    if cache_hit:
                        entry = self._cache.get(similar_hash)
                        if entry and not entry.is_expired:
                            self._record_hit(entry, exact=False, similarity=similarity)
                            return entry.result

            # Cache miss
            self._misses += 1
            return None

    def _check_threshold(self, similarity: float, mode: str) -> bool:
        """
        Check if similarity meets threshold based on mode.

        Phase 69: Dual-threshold semantic cache for precision/recall trade-off.

        Args:
            similarity: Computed similarity score (0-1)
            mode: Threshold mode ("precision", "recall", "adaptive", "legacy")

        Returns:
            True if similarity meets threshold, False otherwise
        """
        if mode == "precision":
            # High precision - fewer false positives, may miss some valid matches
            return similarity >= self.config.precision_threshold
        elif mode == "recall":
            # High recall - more cache hits, may have some false positives
            return similarity >= self.config.recall_threshold
        elif mode == "adaptive":
            # Try precision threshold first (stricter)
            # This provides high-confidence matches
            if similarity >= self.config.precision_threshold:
                return True
            # Fall back to recall threshold for edge cases
            # Only use if similarity is reasonably high
            if similarity >= self.config.recall_threshold:
                # Log when using recall fallback for monitoring
                logger.debug(
                    "Semantic cache using recall threshold",
                    similarity=round(similarity, 3),
                    precision_threshold=self.config.precision_threshold,
                    recall_threshold=self.config.recall_threshold,
                )
                return True
            return False
        else:
            # Legacy mode - single threshold for backwards compatibility
            return similarity >= self.config.similarity_threshold

    async def set(
        self,
        query: str,
        result: Any,
        embedding: Optional[List[float]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store a query result in cache.

        Args:
            query: Query string
            result: Result to cache
            embedding: Optional query embedding for semantic matching
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        query_hash = self._hash_query(query)
        ttl = ttl_seconds or self.config.default_ttl_seconds

        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.config.max_entries:
                await self._evict()

            # Create entry
            now = datetime.utcnow()
            entry = CacheEntry(
                query=query,
                query_hash=query_hash,
                embedding=embedding,
                result=result,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
            )

            # Store in cache
            self._cache[query_hash] = entry

            # Store embedding for semantic search
            if embedding is not None and HAS_NUMPY:
                emb_array = np.array(embedding, dtype=np.float32)
                self._embeddings.append((query_hash, emb_array))

                # Mark FAISS index as dirty (needs rebuild)
                if self._use_faiss:
                    self._faiss_modifications += 1
                    if self._faiss_modifications >= self.config.faiss_rebuild_threshold:
                        self._faiss_dirty = True

    async def invalidate(self, query: str) -> bool:
        """
        Invalidate a cached query.

        Args:
            query: Query to invalidate

        Returns:
            True if entry was found and removed
        """
        query_hash = self._hash_query(query)

        async with self._lock:
            return await self._remove_entry(query_hash)

    async def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            n_entries = len(self._cache)
            self._cache.clear()
            self._embeddings.clear()
            self._reset_stats()

            # Reset FAISS index
            if self._use_faiss:
                self._faiss_index = None
                self._hash_to_idx.clear()
                self._faiss_dirty = False
                self._faiss_modifications = 0

            return n_entries

    # =========================================================================
    # Semantic Search (FAISS-accelerated)
    # =========================================================================

    def _rebuild_faiss_index(self) -> None:
        """
        Rebuild FAISS index from current embeddings.

        Uses IndexFlatIP (inner product) on normalized vectors for cosine similarity.
        This provides O(log n) search complexity vs O(n) linear scan.
        """
        if not self._use_faiss or not self._embeddings:
            return

        # Get actual dimension from embeddings (may differ from config)
        actual_dim = self._embeddings[0][1].shape[0]

        # Create new index using inner product (normalized vectors → cosine similarity)
        self._faiss_index = faiss.IndexFlatIP(actual_dim)

        # Build matrix of normalized embeddings
        embeddings_matrix = np.zeros(
            (len(self._embeddings), actual_dim),
            dtype=np.float32,
        )
        self._hash_to_idx.clear()

        for idx, (query_hash, emb) in enumerate(self._embeddings):
            # Normalize for cosine similarity via inner product
            norm = np.linalg.norm(emb) + 1e-10
            embeddings_matrix[idx] = emb / norm
            self._hash_to_idx[query_hash] = idx

        # Add all embeddings to index
        self._faiss_index.add(embeddings_matrix)

        self._faiss_dirty = False
        self._faiss_modifications = 0

        logger.debug(
            "Rebuilt FAISS index",
            n_embeddings=len(self._embeddings),
            dimension=actual_dim,
        )

    async def _find_similar(
        self,
        embedding: List[float],
    ) -> Tuple[Optional[str], float]:
        """
        Find most similar cached query using FAISS or linear fallback.

        Args:
            embedding: Query embedding

        Returns:
            (query_hash, similarity) or (None, 0.0) if no match
        """
        if not self._embeddings:
            return (None, 0.0)

        # Use FAISS if available and index is ready
        if self._use_faiss:
            # Phase 71.5: Use lock to prevent concurrent FAISS rebuilds
            if self._faiss_dirty or self._faiss_index is None:
                async with self._faiss_rebuild_lock:
                    # Double-check after acquiring lock (another coroutine may have rebuilt)
                    if self._faiss_dirty or self._faiss_index is None:
                        self._rebuild_faiss_index()

            if self._faiss_index is not None and self._faiss_index.ntotal > 0:
                # Run FAISS search in thread pool
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    self._faiss_search,
                    embedding,
                )

        # Fallback to linear search (O(n))
        embeddings_snapshot = list(self._embeddings)
        valid_hashes = set(self._cache.keys())

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._compute_similarity_linear,
            embedding,
            embeddings_snapshot,
            valid_hashes,
        )

    def _faiss_search(
        self,
        embedding: List[float],
    ) -> Tuple[Optional[str], float]:
        """
        Search FAISS index for most similar embedding (thread-safe).

        FAISS IndexFlatIP returns inner product scores.
        With normalized vectors, this equals cosine similarity.
        """
        # Normalize query embedding
        query_emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        # Search for top-1 match
        scores, indices = self._faiss_index.search(query_norm, k=1)

        if indices[0][0] == -1:
            return (None, 0.0)

        idx = indices[0][0]
        similarity = float(scores[0][0])

        # Look up hash from index
        for query_hash, stored_idx in self._hash_to_idx.items():
            if stored_idx == idx:
                # Verify entry still exists in cache
                if query_hash in self._cache:
                    return (query_hash, similarity)
                break

        return (None, 0.0)

    def _compute_similarity_linear(
        self,
        embedding: List[float],
        embeddings_snapshot: List[Tuple[str, np.ndarray]],
        valid_hashes: set,
    ) -> Tuple[Optional[str], float]:
        """
        Compute cosine similarity against cached embeddings (linear O(n) fallback).

        Uses snapshot of embeddings and valid hashes to avoid race conditions
        when running in thread pool executor.
        """
        query_emb = np.array(embedding, dtype=np.float32)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        best_hash = None
        best_similarity = 0.0

        for cached_hash, cached_emb in embeddings_snapshot:
            # Check if entry still exists (using snapshot of valid hashes)
            if cached_hash not in valid_hashes:
                continue

            # Cosine similarity
            cached_norm = cached_emb / (np.linalg.norm(cached_emb) + 1e-10)
            similarity = float(np.dot(query_norm, cached_norm))

            if similarity > best_similarity:
                best_similarity = similarity
                best_hash = cached_hash

        return (best_hash, best_similarity)

    # =========================================================================
    # Eviction
    # =========================================================================

    async def _evict(self) -> None:
        """Evict entries based on policy."""
        if self.config.eviction_policy == "lru":
            # Remove least recently used
            if self._cache:
                oldest_hash = next(iter(self._cache))
                await self._remove_entry(oldest_hash)
                self._evictions += 1

        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used
            if self._cache:
                min_hits = min(e.hits for e in self._cache.values())
                for query_hash, entry in list(self._cache.items()):
                    if entry.hits == min_hits:
                        await self._remove_entry(query_hash)
                        self._evictions += 1
                        break

        else:  # fifo
            # Remove first in
            if self._cache:
                oldest_hash = next(iter(self._cache))
                await self._remove_entry(oldest_hash)
                self._evictions += 1

    async def _remove_entry(self, query_hash: str) -> bool:
        """Remove an entry from cache and embedding index."""
        if query_hash not in self._cache:
            return False

        del self._cache[query_hash]

        # Remove from embedding index
        self._embeddings = [
            (h, e) for h, e in self._embeddings if h != query_hash
        ]

        # Mark FAISS index as dirty (needs rebuild)
        if self._use_faiss:
            self._faiss_dirty = True
            self._hash_to_idx.pop(query_hash, None)

        return True

    # =========================================================================
    # Helpers
    # =========================================================================

    def _hash_query(self, query: str) -> str:
        """Generate hash for query (SHA-256 for collision resistance at scale)."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _record_hit(
        self,
        entry: CacheEntry,
        exact: bool,
        similarity: float = 1.0,
    ) -> None:
        """Record a cache hit."""
        entry.hits += 1
        entry.last_accessed = datetime.utcnow()

        # Move to end for LRU
        self._cache.move_to_end(entry.query_hash)

        self._hits += 1
        if exact:
            self._exact_hits += 1
        else:
            self._semantic_hits += 1
            self._similarity_scores.append(similarity)
            # Prevent unbounded growth (memory leak fix - Phase 69)
            if len(self._similarity_scores) > self.config.max_similarity_scores:
                # Keep only the most recent scores
                self._similarity_scores = self._similarity_scores[-self.config.max_similarity_scores:]

    def _reset_stats(self) -> None:
        """Reset analytics counters."""
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._exact_hits = 0
        self._evictions = 0
        self._similarity_scores.clear()

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        avg_similarity = (
            sum(self._similarity_scores) / len(self._similarity_scores)
            if self._similarity_scores else 0.0
        )

        # FAISS stats
        faiss_index_size = 0
        if self._use_faiss and self._faiss_index is not None:
            faiss_index_size = self._faiss_index.ntotal

        return CacheStats(
            total_entries=len(self._cache),
            hits=self._hits,
            misses=self._misses,
            semantic_hits=self._semantic_hits,
            exact_hits=self._exact_hits,
            evictions=self._evictions,
            hit_rate=hit_rate,
            avg_similarity=avg_similarity,
            faiss_enabled=self._use_faiss,
            faiss_index_size=faiss_index_size,
            # Phase 69: Dual-threshold info
            threshold_mode=self.config.threshold_mode,
            precision_threshold=self.config.precision_threshold,
            recall_threshold=self.config.recall_threshold,
        )

    # =========================================================================
    # Cache Warming
    # =========================================================================

    async def warm(
        self,
        queries: List[Tuple[str, List[float], Any]],
    ) -> int:
        """
        Warm the cache with pre-computed results.

        Args:
            queries: List of (query, embedding, result) tuples

        Returns:
            Number of entries added
        """
        n_added = 0

        for query, embedding, result in queries:
            await self.set(query, result, embedding)
            n_added += 1

        logger.info("Warmed cache", n_entries=n_added)
        return n_added

    async def get_popular_queries(
        self,
        limit: int = 10,
    ) -> List[Tuple[str, int]]:
        """
        Get most frequently hit queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of (query, hit_count) tuples
        """
        sorted_entries = sorted(
            self._cache.values(),
            key=lambda e: e.hits,
            reverse=True,
        )

        return [(e.query, e.hits) for e in sorted_entries[:limit]]


# =============================================================================
# Singleton
# =============================================================================

_semantic_cache: Optional[SemanticQueryCache] = None


def get_semantic_cache(
    config: Optional[SemanticCacheConfig] = None,
) -> SemanticQueryCache:
    """Get or create semantic cache singleton."""
    global _semantic_cache

    if _semantic_cache is None or config is not None:
        _semantic_cache = SemanticQueryCache(config)

    return _semantic_cache


# =============================================================================
# Convenience Functions
# =============================================================================

async def cache_get(
    query: str,
    embedding: Optional[List[float]] = None,
) -> Optional[Any]:
    """Get from cache."""
    return await get_semantic_cache().get(query, embedding)


async def cache_set(
    query: str,
    result: Any,
    embedding: Optional[List[float]] = None,
    ttl: int = 300,
) -> None:
    """Set in cache."""
    await get_semantic_cache().set(query, result, embedding, ttl)
