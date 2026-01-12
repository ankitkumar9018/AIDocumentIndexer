"""
AIDocumentIndexer - Base Cache Classes
=======================================

Abstract base classes and common types for the cache framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class CacheBackend(str, Enum):
    """Available cache backends."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"  # Memory L1 + Redis L2


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with metadata."""
    value: T
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return datetime.utcnow() >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age of this entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    total_bytes: int = 0

    # Backend-specific stats
    redis_hits: int = 0
    redis_misses: int = 0
    redis_errors: int = 0
    memory_hits: int = 0
    memory_misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def redis_hit_rate(self) -> float:
        """Calculate Redis hit rate."""
        total = self.redis_hits + self.redis_misses
        return self.redis_hits / total if total > 0 else 0.0

    @property
    def memory_hit_rate(self) -> float:
        """Calculate memory cache hit rate."""
        total = self.memory_hits + self.memory_misses
        return self.memory_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": round(self.hit_rate, 4),
            "total_bytes": self.total_bytes,
            "redis": {
                "hits": self.redis_hits,
                "misses": self.redis_misses,
                "errors": self.redis_errors,
                "hit_rate": round(self.redis_hit_rate, 4),
            },
            "memory": {
                "hits": self.memory_hits,
                "misses": self.memory_misses,
                "hit_rate": round(self.memory_hit_rate, 4),
            },
        }


class BaseCache(ABC, Generic[T]):
    """
    Abstract base class for all cache implementations.

    Provides a consistent interface for caching operations with:
    - TTL-based expiration
    - LRU eviction
    - Statistics tracking
    - Settings-aware behavior

    Subclasses must implement:
    - _do_get: Low-level get operation
    - _do_set: Low-level set operation
    - _do_delete: Low-level delete operation
    """

    def __init__(
        self,
        prefix: str = "cache",
        ttl_seconds: int = 3600,
        max_items: int = 10000,
        settings_key: Optional[str] = None,
    ):
        """
        Initialize the cache.

        Args:
            prefix: Key prefix for namespacing
            ttl_seconds: Default time-to-live in seconds
            max_items: Maximum number of items to store
            settings_key: Optional settings key for dynamic configuration
        """
        self.prefix = prefix
        self.default_ttl = ttl_seconds
        self.max_items = max_items
        self.settings_key = settings_key

        # Statistics
        self._stats = CacheStats(max_size=max_items)

        # Settings cache
        self._settings_enabled: Optional[bool] = None
        self._settings_ttl: Optional[int] = None

    # =========================================================================
    # Abstract methods to implement
    # =========================================================================

    @abstractmethod
    async def _do_get(self, key: str) -> Optional[T]:
        """Low-level get operation. Override in subclass."""
        pass

    @abstractmethod
    async def _do_set(self, key: str, value: T, ttl: int) -> bool:
        """Low-level set operation. Override in subclass."""
        pass

    @abstractmethod
    async def _do_delete(self, key: str) -> bool:
        """Low-level delete operation. Override in subclass."""
        pass

    @abstractmethod
    async def _do_clear(self) -> int:
        """Clear all entries. Returns number of entries cleared."""
        pass

    # =========================================================================
    # Public API
    # =========================================================================

    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.

        Args:
            key: Cache key (will be prefixed)

        Returns:
            Cached value or None if not found/expired
        """
        # Check if caching is enabled
        if not await self._is_enabled():
            return None

        cache_key = self._make_key(key)
        value = await self._do_get(cache_key)

        if value is not None:
            self._stats.hits += 1
        else:
            self._stats.misses += 1

        return value

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.

        Args:
            key: Cache key (will be prefixed)
            value: Value to cache
            ttl: Optional TTL override in seconds

        Returns:
            True if cached successfully
        """
        # Check if caching is enabled
        if not await self._is_enabled():
            return False

        cache_key = self._make_key(key)
        effective_ttl = ttl or await self._get_ttl()

        return await self._do_set(cache_key, value, effective_ttl)

    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key (will be prefixed)

        Returns:
            True if deleted successfully
        """
        cache_key = self._make_key(key)
        return await self._do_delete(cache_key)

    async def get_or_set(
        self,
        key: str,
        factory: Any,  # Callable[[], T] or Callable[[], Awaitable[T]]
        ttl: Optional[int] = None,
    ) -> T:
        """
        Get from cache or compute and store.

        Args:
            key: Cache key
            factory: Sync or async function to compute value if not cached
            ttl: Optional TTL override

        Returns:
            Cached or computed value
        """
        import asyncio

        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

    async def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        count = await self._do_clear()
        self._stats.evictions += count
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = CacheStats(max_size=self.max_items)

    def invalidate_settings(self) -> None:
        """Invalidate cached settings (call after settings change)."""
        self._settings_enabled = None
        self._settings_ttl = None

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.prefix}:{key}"

    async def _is_enabled(self) -> bool:
        """Check if caching is enabled in settings."""
        if self._settings_enabled is not None:
            return self._settings_enabled

        if not self.settings_key:
            return True

        try:
            from backend.services.settings import get_settings_service
            settings = get_settings_service()
            enabled = await settings.get_setting(f"{self.settings_key}_enabled")
            self._settings_enabled = enabled if enabled is not None else True
        except Exception:
            self._settings_enabled = True

        return self._settings_enabled

    async def _get_ttl(self) -> int:
        """Get TTL from settings or default."""
        if self._settings_ttl is not None:
            return self._settings_ttl

        if not self.settings_key:
            return self.default_ttl

        try:
            from backend.services.settings import get_settings_service
            settings = get_settings_service()
            ttl = await settings.get_setting(f"{self.settings_key}_ttl")
            self._settings_ttl = ttl if ttl is not None else self.default_ttl
        except Exception:
            self._settings_ttl = self.default_ttl

        return self._settings_ttl
