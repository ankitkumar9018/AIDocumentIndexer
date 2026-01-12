"""
AIDocumentIndexer - Embedding Deduplication Cache
==================================================

Provides caching for embeddings to avoid redundant API calls.
Uses content hashing to detect duplicate text and return cached
embeddings instead of re-computing.

Benefits:
- Reduces embedding API costs by 30-60%
- Speeds up document processing for similar content
- Enables efficient re-indexing of unchanged documents

Settings-aware: Respects cache.embedding_cache_enabled setting.

Note: This module is being migrated to the unified cache abstraction.
      See backend/services/cache/ for the new implementation.
"""

from typing import Any, Dict, List, Optional, Tuple

import structlog

from backend.services.cache import RedisBackedCache, CacheKeyGenerator

logger = structlog.get_logger(__name__)

# Cache key generator for embeddings
_embedding_keygen = CacheKeyGenerator(prefix="embed", normalize=True)


def invalidate_cache_settings():
    """Invalidate cached settings (call after settings change)."""
    # Delegate to the cache instance
    cache = get_embedding_cache()
    cache.invalidate_settings()


class EmbeddingCache(RedisBackedCache[List[float]]):
    """
    Cache for embedding vectors with content-based deduplication.

    Uses the unified cache abstraction with Redis backend and memory fallback.
    """

    def __init__(
        self,
        cache_ttl: int = 60 * 60 * 24 * 7,  # 7 days
        max_memory_items: int = 10000,
    ):
        """
        Initialize embedding cache.

        Args:
            cache_ttl: Time-to-live for cached embeddings (seconds)
            max_memory_items: Max items in memory fallback cache
        """
        super().__init__(
            prefix="embed",
            ttl_seconds=cache_ttl,
            max_items=max_memory_items,
            settings_key="cache.embedding_cache",
        )
        # Additional stats for backward compatibility
        self._cached_count = 0

    async def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None
        """
        cache_key = _embedding_keygen.content_key(text)
        result = await super().get(cache_key)

        if result is not None:
            logger.debug("Embedding cache hit", key=cache_key[:8])
        else:
            logger.debug("Embedding cache miss", key=cache_key[:8])

        return result

    async def set(self, text: str, embedding: List[float]) -> bool:
        """
        Cache an embedding.

        Args:
            text: Original text
            embedding: Embedding vector

        Returns:
            True if cached successfully
        """
        cache_key = _embedding_keygen.content_key(text)
        success = await super().set(cache_key, embedding)

        if success:
            self._cached_count += 1
            logger.debug("Embedding cached", key=cache_key[:8])

        return success

    async def get_many(
        self,
        texts: List[str],
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts to look up

        Returns:
            Tuple of (cached embeddings by index, indices that need computing)
        """
        cached = {}
        missing = []

        for idx, text in enumerate(texts):
            embedding = await self.get(text)
            if embedding is not None:
                cached[idx] = embedding
            else:
                missing.append(idx)

        return cached, missing

    async def set_many(
        self,
        texts: List[str],
        embeddings: List[List[float]],
    ) -> int:
        """
        Cache multiple embeddings.

        Args:
            texts: List of texts
            embeddings: Corresponding embeddings

        Returns:
            Number of items cached
        """
        count = 0
        for text, embedding in zip(texts, embeddings):
            if await self.set(text, embedding):
                count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = super().get_stats()
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "cached": self._cached_count,
            "hit_rate": stats.hit_rate,
            "memory_items": stats.size,
            "redis_hits": stats.redis_hits,
            "redis_misses": stats.redis_misses,
            "redis_errors": stats.redis_errors,
        }

    def clear_stats(self) -> None:
        """Reset statistics."""
        self.reset_stats()
        self._cached_count = 0

    async def clear(self) -> None:
        """Clear all cached embeddings."""
        await super().clear()
        self._cached_count = 0
        logger.info("Embedding cache cleared")


class CachedEmbeddingService:
    """
    Wrapper around embedding service with caching.

    Drop-in replacement that adds caching to any embedding service.
    """

    def __init__(
        self,
        embedding_service: Any,
        cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize cached embedding service.

        Args:
            embedding_service: Underlying embedding service
            cache: EmbeddingCache instance (creates default if None)
        """
        self.embedding_service = embedding_service
        self.cache = cache or EmbeddingCache()

    async def embed_texts(
        self,
        texts: List[str],
        **kwargs,
    ) -> List[List[float]]:
        """
        Embed texts with caching.

        Args:
            texts: Texts to embed
            **kwargs: Additional args for embedding service

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache
        cached, missing = await self.cache.get_many(texts)

        if not missing:
            # All cached
            return [cached[i] for i in range(len(texts))]

        # Compute missing embeddings
        missing_texts = [texts[i] for i in missing]
        new_embeddings = await self.embedding_service.embed_texts(missing_texts, **kwargs)

        # Cache new embeddings
        await self.cache.set_many(missing_texts, new_embeddings)

        # Merge results
        result = [None] * len(texts)
        for idx, embedding in cached.items():
            result[idx] = embedding
        for idx, embedding in zip(missing, new_embeddings):
            result[idx] = embedding

        logger.info(
            "Embeddings computed with caching",
            total=len(texts),
            cached=len(cached),
            computed=len(missing),
        )

        return result

    async def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embed a single query with caching."""
        embeddings = await self.embed_texts([query], **kwargs)
        return embeddings[0] if embeddings else []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# Singleton instances
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create embedding cache singleton."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache
