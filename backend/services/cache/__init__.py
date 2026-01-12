"""
AIDocumentIndexer - Unified Cache Abstraction
==============================================

Provides a unified caching framework that consolidates multiple cache implementations:

1. BaseCache - Abstract interface for all cache implementations
2. MemoryCache - In-memory LRU cache with TTL support
3. RedisCache - Redis-backed cache with memory fallback
4. CacheKeyGenerator - Standardized cache key generation

Benefits:
- Consistent API across all caches
- Shared LRU eviction logic
- Unified statistics tracking
- Settings-aware caching

Usage:
    from backend.services.cache import MemoryCache, RedisCache, CacheKeyGenerator

    # Simple memory cache
    cache = MemoryCache(prefix="embeddings", ttl_seconds=3600)
    await cache.set("key", value)
    value = await cache.get("key")

    # Redis cache with fallback
    redis_cache = RedisCache(prefix="search", ttl_seconds=300)
    await redis_cache.set("key", {"results": [...]})
"""

from backend.services.cache.base import (
    BaseCache,
    CacheEntry,
    CacheStats,
    CacheBackend,
)
from backend.services.cache.memory import MemoryCache, TTLDictCache
from backend.services.cache.redis import RedisBackedCache, HybridCache
from backend.services.cache.keys import CacheKeyGenerator, hash_content
from backend.services.cache.decorators import cached, async_cached

__all__ = [
    # Base classes
    "BaseCache",
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    # Implementations
    "MemoryCache",
    "TTLDictCache",
    "RedisBackedCache",
    "HybridCache",
    # Utilities
    "CacheKeyGenerator",
    "hash_content",
    # Decorators
    "cached",
    "async_cached",
]
