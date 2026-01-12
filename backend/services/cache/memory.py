"""
AIDocumentIndexer - Memory Cache Implementations
=================================================

In-memory cache implementations with LRU eviction and TTL support.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, TypeVar

import structlog

from backend.services.cache.base import BaseCache, CacheEntry, CacheStats

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class LRUEvictionMixin:
    """
    Mixin that provides LRU eviction logic for memory caches.

    Use this mixin to add consistent LRU eviction behavior to any cache class.
    """

    def __init__(self):
        self._access_order: List[str] = []

    def _update_access_order(self, key: str) -> None:
        """Update LRU order when key is accessed."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _remove_from_order(self, key: str) -> None:
        """Remove key from access order."""
        if key in self._access_order:
            self._access_order.remove(key)

    def _get_lru_key(self) -> Optional[str]:
        """Get the least recently used key."""
        return self._access_order[0] if self._access_order else None

    def _pop_lru_key(self) -> Optional[str]:
        """Pop and return the least recently used key."""
        return self._access_order.pop(0) if self._access_order else None


class TTLDictCache(Generic[T]):
    """
    Simple TTL-aware dictionary cache with LRU eviction.

    This is a lightweight cache for use cases where you don't need
    the full BaseCache interface (e.g., simple in-memory caching
    of computed values like settings or configurations).

    Usage:
        cache = TTLDictCache[str](ttl_seconds=300, max_items=100)
        cache.set("key", "value")
        value = cache.get("key")
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_items: int = 1000,
        name: str = "cache",
    ):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self.name = name

        self._cache: Dict[str, T] = {}
        self._timestamps: Dict[str, float] = {}  # key -> expire_time
        self._access_order: List[str] = []

        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: str) -> Optional[T]:
        """Get a value from cache."""
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        # Check expiration
        if time.time() > self._timestamps.get(key, 0):
            self._remove(key)
            self._stats["misses"] += 1
            return None

        # Update LRU order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self._stats["hits"] += 1
        return self._cache[key]

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        effective_ttl = ttl or self.ttl_seconds

        # Evict if at capacity
        self._evict_if_needed()

        self._cache[key] = value
        self._timestamps[key] = time.time() + effective_ttl

        # Update LRU order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if key in self._cache:
            self._remove(key)
            return True
        return False

    def clear(self) -> int:
        """Clear all entries."""
        count = len(self._cache)
        self._cache.clear()
        self._timestamps.clear()
        self._access_order.clear()
        return count

    def _remove(self, key: str) -> None:
        """Remove a key from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_if_needed(self) -> None:
        """Evict old entries if at capacity."""
        # First, evict expired entries
        now = time.time()
        expired = [k for k, exp in self._timestamps.items() if now > exp]
        for key in expired:
            self._remove(key)
            self._stats["evictions"] += 1

        # Then, evict LRU if still over capacity
        while len(self._cache) >= self.max_items and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
            self._timestamps.pop(oldest, None)
            self._stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "size": len(self._cache),
            "max_items": self.max_items,
            "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
        }

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)


class MemoryCache(BaseCache[T]):
    """
    Full-featured in-memory cache with LRU eviction and TTL support.

    Implements the BaseCache interface for consistent usage across
    the application.

    Features:
    - TTL-based expiration
    - LRU eviction when max capacity reached
    - Statistics tracking
    - Settings-aware configuration

    Usage:
        cache = MemoryCache[List[float]](
            prefix="embeddings",
            ttl_seconds=86400,
            max_items=10000,
        )

        # Simple get/set
        await cache.set("key", [0.1, 0.2, 0.3])
        value = await cache.get("key")

        # Get or compute
        value = await cache.get_or_set(
            "key",
            lambda: compute_expensive_value(),
        )
    """

    def __init__(
        self,
        prefix: str = "memory",
        ttl_seconds: int = 3600,
        max_items: int = 10000,
        settings_key: Optional[str] = None,
    ):
        super().__init__(
            prefix=prefix,
            ttl_seconds=ttl_seconds,
            max_items=max_items,
            settings_key=settings_key,
        )

        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order: List[str] = []

    async def _do_get(self, key: str) -> Optional[T]:
        """Get value from memory cache."""
        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired:
            await self._do_delete(key)
            return None

        # Update access tracking
        entry.hit_count += 1
        entry.last_accessed = datetime.utcnow()
        self._update_access_order(key)

        self._stats.memory_hits += 1
        return entry.value

    async def _do_set(self, key: str, value: T, ttl: int) -> bool:
        """Set value in memory cache."""
        # Evict if needed
        self._evict_expired()
        self._evict_lru()

        now = datetime.utcnow()
        self._cache[key] = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
        )

        self._update_access_order(key)
        self._stats.size = len(self._cache)

        return True

    async def _do_delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self._cache:
            del self._cache[key]
            self._remove_from_access_order(key)
            self._stats.size = len(self._cache)
            return True
        return False

    async def _do_clear(self) -> int:
        """Clear all entries."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        self._stats.size = 0
        return count

    def _update_access_order(self, key: str) -> None:
        """Update LRU order when key is accessed."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _remove_from_access_order(self, key: str) -> None:
        """Remove key from access order."""
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.utcnow()
        expired = [k for k, v in self._cache.items() if v.expires_at <= now]
        for key in expired:
            del self._cache[key]
            self._remove_from_access_order(key)
            self._stats.evictions += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entries if over capacity."""
        while len(self._cache) >= self.max_items and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
                self._stats.evictions += 1

    def get_sync(self, key: str) -> Optional[T]:
        """
        Synchronous get for use in sync contexts.

        Note: Does not check settings-based enable flag.
        """
        cache_key = self._make_key(key)
        if cache_key not in self._cache:
            self._stats.misses += 1
            return None

        entry = self._cache[cache_key]
        if entry.is_expired:
            del self._cache[cache_key]
            self._remove_from_access_order(cache_key)
            self._stats.misses += 1
            return None

        entry.hit_count += 1
        entry.last_accessed = datetime.utcnow()
        self._update_access_order(cache_key)
        self._stats.hits += 1
        self._stats.memory_hits += 1

        return entry.value

    def set_sync(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """
        Synchronous set for use in sync contexts.

        Note: Does not check settings-based enable flag.
        """
        cache_key = self._make_key(key)
        effective_ttl = ttl or self.default_ttl

        self._evict_expired()
        self._evict_lru()

        now = datetime.utcnow()
        self._cache[cache_key] = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + timedelta(seconds=effective_ttl),
        )

        self._update_access_order(cache_key)
        self._stats.size = len(self._cache)

        return True

    def items(self) -> List[tuple[str, CacheEntry[T]]]:
        """Get all non-expired cache entries."""
        now = datetime.utcnow()
        return [
            (k, v) for k, v in self._cache.items()
            if v.expires_at > now
        ]
