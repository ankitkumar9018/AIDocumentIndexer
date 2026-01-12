"""
AIDocumentIndexer - Redis Cache Implementation
===============================================

Redis-backed cache with automatic in-memory fallback.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, TypeVar

import structlog

from backend.services.cache.base import BaseCache, CacheEntry, CacheStats

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RedisBackedCache(BaseCache[T]):
    """
    Redis-backed cache with automatic in-memory fallback.

    Features:
    - Primary storage in Redis for distributed caching
    - Automatic fallback to in-memory when Redis unavailable
    - TTL support in both Redis and memory
    - LRU eviction for memory fallback
    - Statistics tracking for both backends

    Usage:
        cache = RedisBackedCache[Dict[str, Any]](
            prefix="search",
            ttl_seconds=300,
            max_items=1000,
        )

        await cache.set("key", {"results": [...]})
        value = await cache.get("key")
    """

    def __init__(
        self,
        prefix: str = "cache",
        ttl_seconds: int = 3600,
        max_items: int = 10000,
        settings_key: Optional[str] = None,
        redis_enabled_check: Optional[str] = None,
    ):
        """
        Initialize Redis-backed cache.

        Args:
            prefix: Key prefix for namespacing
            ttl_seconds: Default TTL in seconds
            max_items: Max items in memory fallback
            settings_key: Settings key for enable/TTL config
            redis_enabled_check: Settings key to check if Redis is enabled
        """
        super().__init__(
            prefix=prefix,
            ttl_seconds=ttl_seconds,
            max_items=max_items,
            settings_key=settings_key,
        )

        self.redis_enabled_check = redis_enabled_check

        # Memory fallback cache
        self._memory_cache: Dict[str, Any] = {}
        self._memory_timestamps: Dict[str, tuple[float, int]] = {}  # key -> (created_at, ttl)
        self._memory_order: List[str] = []

    async def _get_redis_client(self):
        """Get Redis client if available."""
        try:
            from backend.services.redis_client import get_redis_client
            return await get_redis_client()
        except Exception:
            return None

    async def _do_get(self, key: str) -> Optional[T]:
        """Get value from Redis or memory fallback."""
        # Try Redis first
        try:
            client = await self._get_redis_client()
            if client is not None:
                value = await client.get(key)
                if value:
                    self._stats.redis_hits += 1
                    return json.loads(value)
                self._stats.redis_misses += 1
                return None
        except Exception as e:
            self._stats.redis_errors += 1
            logger.debug("Redis get failed, using memory fallback", error=str(e))

        # Fall back to memory
        return self._memory_get(key)

    async def _do_set(self, key: str, value: T, ttl: int) -> bool:
        """Set value in Redis or memory fallback."""
        # Try Redis first
        try:
            client = await self._get_redis_client()
            if client is not None:
                await client.setex(key, ttl, json.dumps(value))
                return True
        except Exception as e:
            self._stats.redis_errors += 1
            logger.debug("Redis set failed, using memory fallback", error=str(e))

        # Fall back to memory
        self._memory_set(key, value, ttl)
        return True

    async def _do_delete(self, key: str) -> bool:
        """Delete value from Redis and memory."""
        deleted = False

        # Try Redis
        try:
            client = await self._get_redis_client()
            if client is not None:
                await client.delete(key)
                deleted = True
        except Exception as e:
            self._stats.redis_errors += 1
            logger.debug("Redis delete failed", error=str(e))

        # Always clean up memory too
        if key in self._memory_cache:
            self._memory_remove(key)
            deleted = True

        return deleted

    async def _do_clear(self) -> int:
        """Clear memory cache (Redis entries will expire via TTL)."""
        count = len(self._memory_cache)
        self._memory_cache.clear()
        self._memory_timestamps.clear()
        self._memory_order.clear()
        return count

    # =========================================================================
    # Memory fallback methods
    # =========================================================================

    def _memory_get(self, key: str) -> Optional[T]:
        """Get value from memory cache."""
        if key not in self._memory_cache:
            self._stats.memory_misses += 1
            return None

        # Check expiration
        if self._is_memory_expired(key):
            self._memory_remove(key)
            self._stats.memory_misses += 1
            return None

        # Update LRU order
        if key in self._memory_order:
            self._memory_order.remove(key)
        self._memory_order.append(key)

        self._stats.memory_hits += 1
        return self._memory_cache[key]

    def _memory_set(self, key: str, value: T, ttl: int) -> None:
        """Set value in memory cache."""
        # Evict expired entries periodically
        if len(self._memory_cache) % 100 == 0:
            self._memory_evict_expired()

        # Evict LRU if at capacity
        self._memory_evict_lru()

        # Store value with timestamp
        self._memory_cache[key] = value
        self._memory_timestamps[key] = (time.time(), ttl)

        # Update LRU order
        if key in self._memory_order:
            self._memory_order.remove(key)
        self._memory_order.append(key)

        self._stats.size = len(self._memory_cache)

    def _memory_remove(self, key: str) -> None:
        """Remove key from memory cache."""
        self._memory_cache.pop(key, None)
        self._memory_timestamps.pop(key, None)
        if key in self._memory_order:
            self._memory_order.remove(key)
        self._stats.size = len(self._memory_cache)

    def _is_memory_expired(self, key: str) -> bool:
        """Check if memory cache entry has expired."""
        if key not in self._memory_timestamps:
            return False
        created_at, ttl = self._memory_timestamps[key]
        return (time.time() - created_at) > ttl

    def _memory_evict_expired(self) -> None:
        """Remove expired entries from memory cache."""
        now = time.time()
        expired = []
        for key, (created_at, ttl) in list(self._memory_timestamps.items()):
            if (now - created_at) > ttl:
                expired.append(key)

        for key in expired:
            self._memory_remove(key)
            self._stats.evictions += 1

    def _memory_evict_lru(self) -> None:
        """Evict oldest entries if at capacity."""
        while len(self._memory_cache) >= self.max_items and self._memory_order:
            oldest = self._memory_order.pop(0)
            self._memory_cache.pop(oldest, None)
            self._memory_timestamps.pop(oldest, None)
            self._stats.evictions += 1

    def clear_memory_cache(self) -> None:
        """Clear only the in-memory fallback cache."""
        self._memory_cache.clear()
        self._memory_timestamps.clear()
        self._memory_order.clear()
        self._stats.size = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics with memory size."""
        self._stats.size = len(self._memory_cache)
        return self._stats


class HybridCache(RedisBackedCache[T]):
    """
    Two-level hybrid cache with L1 memory and L2 Redis.

    Unlike RedisBackedCache which uses memory as a fallback,
    HybridCache always checks memory first (L1) before Redis (L2),
    and populates L1 from L2 on cache hits.

    Benefits:
    - Sub-millisecond reads for frequently accessed data
    - Distributed consistency via Redis
    - Automatic L1 population from L2

    Usage:
        cache = HybridCache[Dict[str, Any]](
            prefix="embeddings",
            ttl_seconds=86400,
            l1_max_items=1000,  # Fast memory cache
            l2_max_items=100000,  # Large Redis cache
        )
    """

    def __init__(
        self,
        prefix: str = "hybrid",
        ttl_seconds: int = 3600,
        l1_max_items: int = 1000,
        l2_max_items: int = 100000,
        l1_ttl_fraction: float = 0.1,  # L1 TTL as fraction of L2
        settings_key: Optional[str] = None,
    ):
        super().__init__(
            prefix=prefix,
            ttl_seconds=ttl_seconds,
            max_items=l1_max_items,
            settings_key=settings_key,
        )

        self.l1_max_items = l1_max_items
        self.l2_max_items = l2_max_items
        self.l1_ttl_fraction = l1_ttl_fraction

    async def _do_get(self, key: str) -> Optional[T]:
        """Get value, checking L1 first then L2."""
        # Check L1 (memory) first
        value = self._memory_get(key)
        if value is not None:
            return value

        # Check L2 (Redis)
        try:
            client = await self._get_redis_client()
            if client is not None:
                redis_value = await client.get(key)
                if redis_value:
                    self._stats.redis_hits += 1
                    parsed = json.loads(redis_value)

                    # Populate L1 with shorter TTL
                    l1_ttl = int(self.default_ttl * self.l1_ttl_fraction)
                    self._memory_set(key, parsed, l1_ttl)

                    return parsed
                self._stats.redis_misses += 1
        except Exception as e:
            self._stats.redis_errors += 1
            logger.debug("Redis get failed in hybrid cache", error=str(e))

        return None

    async def _do_set(self, key: str, value: T, ttl: int) -> bool:
        """Set value in both L1 and L2."""
        success = True

        # Set in L2 (Redis) with full TTL
        try:
            client = await self._get_redis_client()
            if client is not None:
                await client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            self._stats.redis_errors += 1
            logger.debug("Redis set failed in hybrid cache", error=str(e))
            success = False

        # Set in L1 (memory) with shorter TTL
        l1_ttl = int(ttl * self.l1_ttl_fraction)
        self._memory_set(key, value, max(l1_ttl, 60))  # Minimum 60s in L1

        return success
