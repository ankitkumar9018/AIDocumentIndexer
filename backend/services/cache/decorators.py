"""
AIDocumentIndexer - Cache Decorators
======================================

Decorators for easy function-level caching.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from backend.services.cache.memory import MemoryCache

F = TypeVar("F", bound=Callable[..., Any])


def cached(
    ttl_seconds: int = 3600,
    max_items: int = 1000,
    prefix: str = "func",
    key_func: Optional[Callable[..., str]] = None,
    cache_instance: Optional[MemoryCache] = None,
):
    """
    Decorator to cache synchronous function results.

    Usage:
        @cached(ttl_seconds=3600)
        def expensive_computation(x: int, y: int) -> int:
            return x ** y

        @cached(key_func=lambda x, y: f"{x}:{y}")
        def custom_keyed_func(x: int, y: int) -> int:
            return x + y

    Args:
        ttl_seconds: Time-to-live for cached results
        max_items: Maximum items in cache
        prefix: Cache key prefix
        key_func: Optional function to generate cache key from args
        cache_instance: Optional existing cache to use
    """
    cache = cache_instance or MemoryCache(
        prefix=prefix,
        ttl_seconds=ttl_seconds,
        max_items=max_items,
    )

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Check cache (sync path)
            value = cache.get_sync(key)
            if value is not None:
                return value

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set_sync(key, result)
            return result

        # Attach cache for testing/stats
        wrapper._cache = cache  # type: ignore
        return wrapper  # type: ignore

    return decorator


def async_cached(
    ttl_seconds: int = 3600,
    max_items: int = 1000,
    prefix: str = "async_func",
    key_func: Optional[Callable[..., str]] = None,
    cache_instance: Optional[MemoryCache] = None,
):
    """
    Decorator to cache async function results.

    Usage:
        @async_cached(ttl_seconds=3600)
        async def fetch_data(url: str) -> dict:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        @async_cached(key_func=lambda text: hashlib.md5(text.encode()).hexdigest())
        async def get_embedding(text: str) -> List[float]:
            return await embedding_service.embed(text)

    Args:
        ttl_seconds: Time-to-live for cached results
        max_items: Maximum items in cache
        prefix: Cache key prefix
        key_func: Optional function to generate cache key from args
        cache_instance: Optional existing cache to use
    """
    cache = cache_instance or MemoryCache(
        prefix=prefix,
        ttl_seconds=ttl_seconds,
        max_items=max_items,
    )

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # Check cache
            value = await cache.get(key)
            if value is not None:
                return value

            # Compute and cache
            result = await func(*args, **kwargs)
            await cache.set(key, result)
            return result

        # Attach cache for testing/stats
        wrapper._cache = cache  # type: ignore
        return wrapper  # type: ignore

    return decorator


def memoize(
    ttl_seconds: int = 3600,
    max_items: int = 100,
):
    """
    Simple memoization decorator for pure functions.

    Automatically handles both sync and async functions.

    Usage:
        @memoize()
        def fibonacci(n: int) -> int:
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        @memoize(ttl_seconds=60)
        async def fetch_user(user_id: str) -> dict:
            return await db.get_user(user_id)
    """
    cache = MemoryCache(prefix="memo", ttl_seconds=ttl_seconds, max_items=max_items)

    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            value = cache.get_sync(key)
            if value is not None:
                return value
            result = func(*args, **kwargs)
            cache.set_sync(key, result)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            value = await cache.get(key)
            if value is not None:
                return value
            result = await func(*args, **kwargs)
            await cache.set(key, result)
            return result

        if asyncio.iscoroutinefunction(func):
            async_wrapper._cache = cache  # type: ignore
            return async_wrapper  # type: ignore
        else:
            sync_wrapper._cache = cache  # type: ignore
            return sync_wrapper  # type: ignore

    return decorator


class CacheAside:
    """
    Cache-aside pattern implementation for more complex caching scenarios.

    Usage:
        cache_aside = CacheAside(cache=my_cache)

        # With explicit load function
        async def load_user(user_id: str) -> User:
            return await db.get_user(user_id)

        user = await cache_aside.get_or_load(
            key=f"user:{user_id}",
            loader=lambda: load_user(user_id),
        )

        # Invalidate on update
        await cache_aside.invalidate(f"user:{user_id}")
    """

    def __init__(self, cache: MemoryCache):
        """
        Initialize cache-aside helper.

        Args:
            cache: Cache instance to use
        """
        self.cache = cache

    async def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or load from source.

        Args:
            key: Cache key
            loader: Async or sync function to load value if not cached
            ttl: Optional TTL override

        Returns:
            Cached or loaded value
        """
        # Try cache first
        value = await self.cache.get(key)
        if value is not None:
            return value

        # Load from source
        if asyncio.iscoroutinefunction(loader):
            value = await loader()
        else:
            value = loader()

        # Cache the result
        await self.cache.set(key, value, ttl)
        return value

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was deleted
        """
        return await self.cache.delete(key)

    async def refresh(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Force refresh a cache entry.

        Args:
            key: Cache key
            loader: Function to load fresh value
            ttl: Optional TTL override

        Returns:
            Fresh value
        """
        # Load fresh value
        if asyncio.iscoroutinefunction(loader):
            value = await loader()
        else:
            value = loader()

        # Update cache
        await self.cache.set(key, value, ttl)
        return value
