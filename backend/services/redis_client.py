"""
AIDocumentIndexer - Redis Client Service
=========================================

Provides Redis connection management for:
- Celery task queue broker
- Embedding deduplication cache
- Semantic query cache
- Session state caching

Settings-aware: Respects queue.celery_enabled and queue.redis_url settings.
"""

import asyncio
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as aioredis
import redis.exceptions as redis_exceptions
import structlog

logger = structlog.get_logger(__name__)

# Redis configuration from environment (fallback defaults)
DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Cache TTLs (in seconds)
EMBEDDING_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days
SEMANTIC_CACHE_TTL = 60 * 60  # 1 hour
SESSION_CACHE_TTL = 60 * 60 * 24  # 24 hours
SEARCH_CACHE_TTL = 60 * 5  # 5 minutes

# Global Redis client instance
_redis_client: Optional[aioredis.Redis] = None
_redis_enabled: Optional[bool] = None
_cached_redis_url: Optional[str] = None


async def _get_redis_settings() -> tuple[bool, str]:
    """Get Redis settings from database or defaults."""
    try:
        from backend.services.settings import get_settings_service

        settings = get_settings_service()
        enabled = await settings.get_setting("queue.celery_enabled")
        redis_url = await settings.get_setting("queue.redis_url")
        return enabled, redis_url or DEFAULT_REDIS_URL
    except Exception as e:
        logger.debug("Could not load Redis settings, using defaults", error=str(e))
        # Return defaults if settings not available (e.g., during startup)
        return False, DEFAULT_REDIS_URL


def is_redis_enabled_sync() -> bool:
    """
    Synchronous check if Redis is enabled.

    Uses cached value or falls back to environment/default.
    Useful for Celery initialization where async isn't available.
    """
    global _redis_enabled

    if _redis_enabled is not None:
        return _redis_enabled

    # Check environment override
    env_enabled = os.getenv("CELERY_ENABLED", "").lower()
    if env_enabled in ("true", "1", "yes"):
        return True
    if env_enabled in ("false", "0", "no"):
        return False

    # Default to disabled
    return False


async def is_redis_enabled() -> bool:
    """Check if Redis/Celery is enabled in settings."""
    global _redis_enabled

    # Check cached value first
    if _redis_enabled is not None:
        return _redis_enabled

    enabled, _ = await _get_redis_settings()
    _redis_enabled = enabled
    return enabled


async def get_redis_url() -> str:
    """Get Redis URL from settings."""
    global _cached_redis_url

    if _cached_redis_url is not None:
        return _cached_redis_url

    _, redis_url = await _get_redis_settings()
    _cached_redis_url = redis_url
    return redis_url


async def get_redis_client() -> Optional[aioredis.Redis]:
    """
    Get or create the Redis client singleton.

    Returns None if Redis is disabled in settings.
    """
    global _redis_client

    # Check if Redis is enabled
    if not await is_redis_enabled():
        logger.debug("Redis disabled in settings")
        return None

    if _redis_client is None:
        try:
            redis_url = await get_redis_url()
            _redis_client = aioredis.from_url(
                redis_url,
                password=REDIS_PASSWORD,
                db=REDIS_DB,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await _redis_client.ping()
            logger.info("Redis client connected", url=redis_url.split("@")[-1])
        except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, redis_exceptions.AuthenticationError, OSError) as e:
            logger.warning("Redis connection failed, using in-memory fallback", error=str(e), error_type=type(e).__name__)
            _redis_client = None
            raise

    return _redis_client


async def check_redis_connection() -> dict:
    """
    Check Redis connection status.

    Returns dict with connection status and details.
    """
    try:
        enabled = await is_redis_enabled()
        if not enabled:
            return {
                "connected": False,
                "enabled": False,
                "reason": "Redis disabled in settings"
            }

        redis_url = await get_redis_url()
        client = await get_redis_client()
        if client:
            await client.ping()
            # Mask password in URL
            masked_url = redis_url.split("@")[-1] if "@" in redis_url else redis_url
            return {
                "connected": True,
                "enabled": True,
                "url": f"redis://{masked_url}"
            }
        return {
            "connected": False,
            "enabled": True,
            "reason": "Client initialization failed"
        }
    except Exception as e:
        return {
            "connected": False,
            "enabled": True,
            "reason": str(e)
        }


def invalidate_redis_cache():
    """Invalidate cached Redis settings (call after settings change)."""
    global _redis_enabled, _cached_redis_url, _redis_client

    _redis_enabled = None
    _cached_redis_url = None
    # Note: We don't close the client here to avoid disrupting active connections
    # The client will be recreated with new settings on next request


async def close_redis_client():
    """Close the Redis client connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis client closed")


class RedisCache:
    """
    Generic Redis cache wrapper with TTL-aware in-memory fallback.

    Features:
    - Automatic fallback to in-memory when Redis unavailable
    - TTL support in both Redis and memory fallback
    - LRU eviction when memory cache exceeds max size
    - Statistics tracking for monitoring
    """

    def __init__(
        self,
        prefix: str = "cache",
        default_ttl: int = 3600,
        max_memory_items: int = 10000,
    ):
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items

        # In-memory fallback with TTL support
        self._fallback_cache: dict = {}  # key -> value
        self._fallback_timestamps: dict = {}  # key -> (created_at, ttl)
        self._fallback_order: list = []  # LRU tracking

        # Statistics
        self._stats = {
            "redis_hits": 0,
            "redis_misses": 0,
            "memory_hits": 0,
            "memory_misses": 0,
            "redis_errors": 0,
        }

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.prefix}:{key}"

    def _is_expired(self, cache_key: str) -> bool:
        """Check if an in-memory cache entry has expired."""
        if cache_key not in self._fallback_timestamps:
            return False

        import time
        created_at, ttl = self._fallback_timestamps[cache_key]
        return (time.time() - created_at) > ttl

    def _evict_expired(self):
        """Remove expired entries from memory cache."""
        import time
        current_time = time.time()
        expired_keys = []

        for key, (created_at, ttl) in list(self._fallback_timestamps.items()):
            if (current_time - created_at) > ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._fallback_cache.pop(key, None)
            self._fallback_timestamps.pop(key, None)
            if key in self._fallback_order:
                self._fallback_order.remove(key)

    def _evict_lru(self):
        """Evict oldest items if cache exceeds max size."""
        while len(self._fallback_cache) >= self.max_memory_items and self._fallback_order:
            oldest_key = self._fallback_order.pop(0)
            self._fallback_cache.pop(oldest_key, None)
            self._fallback_timestamps.pop(oldest_key, None)

    def _memory_set(self, cache_key: str, value: Any, ttl: int):
        """Set a value in memory cache with TTL."""
        import time

        # Evict expired entries periodically
        if len(self._fallback_cache) % 100 == 0:
            self._evict_expired()

        # Evict LRU if needed
        self._evict_lru()

        # Store value with timestamp
        self._fallback_cache[cache_key] = value
        self._fallback_timestamps[cache_key] = (time.time(), ttl)

        # Update LRU order
        if cache_key in self._fallback_order:
            self._fallback_order.remove(cache_key)
        self._fallback_order.append(cache_key)

    def _memory_get(self, cache_key: str) -> Optional[Any]:
        """Get a value from memory cache, respecting TTL."""
        if cache_key not in self._fallback_cache:
            return None

        if self._is_expired(cache_key):
            # Remove expired entry
            self._fallback_cache.pop(cache_key, None)
            self._fallback_timestamps.pop(cache_key, None)
            if cache_key in self._fallback_order:
                self._fallback_order.remove(cache_key)
            return None

        # Update LRU order
        if cache_key in self._fallback_order:
            self._fallback_order.remove(cache_key)
        self._fallback_order.append(cache_key)

        return self._fallback_cache[cache_key]

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        cache_key = self._make_key(key)

        try:
            client = await get_redis_client()
            if client is None:
                # Redis disabled, use in-memory
                value = self._memory_get(cache_key)
                if value is not None:
                    self._stats["memory_hits"] += 1
                else:
                    self._stats["memory_misses"] += 1
                return value

            value = await client.get(cache_key)
            if value:
                self._stats["redis_hits"] += 1
                return json.loads(value)
            self._stats["redis_misses"] += 1
        except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, OSError) as e:
            self._stats["redis_errors"] += 1
            logger.debug("Redis get connection failed, using memory fallback", error=str(e), error_type=type(e).__name__)
            value = self._memory_get(cache_key)
            if value is not None:
                self._stats["memory_hits"] += 1
            return value
        except (json.JSONDecodeError, ValueError) as e:
            self._stats["redis_errors"] += 1
            logger.warning("Redis get deserialization failed", error=str(e), key=cache_key[:50])
            return None

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        cache_key = self._make_key(key)
        ttl = ttl or self.default_ttl

        try:
            client = await get_redis_client()
            if client is None:
                # Redis disabled, use in-memory
                self._memory_set(cache_key, value, ttl)
                return True
            await client.setex(cache_key, ttl, json.dumps(value))
            return True
        except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, OSError) as e:
            self._stats["redis_errors"] += 1
            logger.debug("Redis set connection failed, using memory fallback", error=str(e), error_type=type(e).__name__)
            self._memory_set(cache_key, value, ttl)
            return True
        except (TypeError, ValueError) as e:
            self._stats["redis_errors"] += 1
            logger.warning("Redis set serialization failed", error=str(e), key=cache_key[:50])
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        cache_key = self._make_key(key)

        try:
            client = await get_redis_client()
            if client is None:
                self._fallback_cache.pop(cache_key, None)
                self._fallback_timestamps.pop(cache_key, None)
                if cache_key in self._fallback_order:
                    self._fallback_order.remove(cache_key)
                return True
            await client.delete(cache_key)
        except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, OSError) as e:
            self._stats["redis_errors"] += 1
            logger.debug("Redis delete failed", error=str(e), error_type=type(e).__name__)
            self._fallback_cache.pop(cache_key, None)
            self._fallback_timestamps.pop(cache_key, None)
            if cache_key in self._fallback_order:
                self._fallback_order.remove(cache_key)

        return True

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        cache_key = self._make_key(key)

        try:
            client = await get_redis_client()
            if client is None:
                return self._memory_get(cache_key) is not None
            return await client.exists(cache_key) > 0
        except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, OSError):
            return self._memory_get(cache_key) is not None

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "memory_size": len(self._fallback_cache),
            "max_memory_items": self.max_memory_items,
        }

    def clear_memory_cache(self):
        """Clear the in-memory fallback cache."""
        self._fallback_cache.clear()
        self._fallback_timestamps.clear()
        self._fallback_order.clear()


# Pre-configured cache instances
embedding_cache = RedisCache(prefix="embed", default_ttl=EMBEDDING_CACHE_TTL)
semantic_cache = RedisCache(prefix="semantic", default_ttl=SEMANTIC_CACHE_TTL)
session_cache = RedisCache(prefix="session", default_ttl=SESSION_CACHE_TTL)
search_cache = RedisCache(prefix="search", default_ttl=SEARCH_CACHE_TTL)


# =============================================================================
# Phase 75: Distributed Cache Invalidation via Redis Pub/Sub
# =============================================================================

# Channel names for cache invalidation
CACHE_INVALIDATION_CHANNEL = "cache:invalidation"
SETTINGS_CHANGE_CHANNEL = "settings:change"

# Global pub/sub listener
_pubsub_listener: Optional[asyncio.Task] = None
_invalidation_handlers: Dict[str, list] = {}


async def publish_cache_invalidation(
    cache_type: str,
    keys: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish a cache invalidation message to all instances.

    Args:
        cache_type: Type of cache (embedding, semantic, session, search, settings)
        keys: Specific keys to invalidate
        pattern: Pattern to match keys for invalidation
        metadata: Additional metadata

    Returns:
        True if published successfully
    """
    try:
        client = await get_redis_client()
        if client is None:
            logger.debug("Redis not available, skipping pub/sub")
            return False

        message = json.dumps({
            "type": cache_type,
            "keys": keys,
            "pattern": pattern,
            "metadata": metadata or {},
            "timestamp": time.time() if 'time' in dir() else 0,
        })

        await client.publish(CACHE_INVALIDATION_CHANNEL, message)
        logger.debug(
            "Published cache invalidation",
            cache_type=cache_type,
            keys_count=len(keys) if keys else 0,
            pattern=pattern,
        )
        return True

    except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, OSError) as e:
        logger.warning("Failed to publish cache invalidation", error=str(e), error_type=type(e).__name__)
        return False


async def publish_settings_change(
    setting_key: str,
    new_value: Any,
    old_value: Any = None,
) -> bool:
    """
    Publish a settings change notification to all instances.

    Args:
        setting_key: The setting key that changed
        new_value: New value
        old_value: Previous value (if known)

    Returns:
        True if published successfully
    """
    try:
        client = await get_redis_client()
        if client is None:
            return False

        import time as time_module
        message = json.dumps({
            "key": setting_key,
            "new_value": new_value,
            "old_value": old_value,
            "timestamp": time_module.time(),
        })

        await client.publish(SETTINGS_CHANGE_CHANNEL, message)
        logger.info(
            "Published settings change",
            key=setting_key,
        )
        return True

    except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, OSError) as e:
        logger.warning("Failed to publish settings change", error=str(e), error_type=type(e).__name__)
        return False


def register_invalidation_handler(
    cache_type: str,
    handler: Callable[[Dict[str, Any]], None],
):
    """
    Register a handler for cache invalidation messages.

    Args:
        cache_type: Type of cache to handle (embedding, semantic, etc.)
        handler: Async function to call when invalidation is received
    """
    if cache_type not in _invalidation_handlers:
        _invalidation_handlers[cache_type] = []
    _invalidation_handlers[cache_type].append(handler)
    logger.debug(f"Registered invalidation handler for {cache_type}")


async def _handle_invalidation_message(message: Dict[str, Any]):
    """Handle a received cache invalidation message."""
    try:
        cache_type = message.get("type", "")
        handlers = _invalidation_handlers.get(cache_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Invalidation handler failed: {e}")

        # Also handle built-in cache types
        if cache_type == "embedding":
            embedding_cache.clear_memory_cache()
        elif cache_type == "semantic":
            semantic_cache.clear_memory_cache()
        elif cache_type == "session":
            session_cache.clear_memory_cache()
        elif cache_type == "search":
            search_cache.clear_memory_cache()
        elif cache_type == "settings":
            invalidate_redis_cache()

    except Exception as e:
        logger.error(f"Failed to handle invalidation message: {e}")


async def _handle_settings_message(message: Dict[str, Any]):
    """Handle a received settings change message."""
    try:
        key = message.get("key", "")
        handlers = _invalidation_handlers.get("settings", [])

        # Notify registered handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Settings handler failed: {e}")

        # Invalidate related caches based on setting key
        if key.startswith("rag.") or key.startswith("embedding."):
            semantic_cache.clear_memory_cache()
        elif key.startswith("queue.") or key.startswith("cache."):
            invalidate_redis_cache()

        logger.debug(f"Processed settings change: {key}")

    except Exception as e:
        logger.error(f"Failed to handle settings message: {e}")


async def start_pubsub_listener():
    """
    Start the Redis pub/sub listener for distributed cache invalidation.

    Should be called during application startup.
    """
    global _pubsub_listener

    if _pubsub_listener is not None:
        logger.debug("Pub/sub listener already running")
        return

    try:
        client = await get_redis_client()
        if client is None:
            logger.info("Redis not available, pub/sub listener not started")
            return

        pubsub = client.pubsub()
        await pubsub.subscribe(CACHE_INVALIDATION_CHANNEL, SETTINGS_CHANGE_CHANNEL)

        async def listener():
            """Listen for pub/sub messages."""
            try:
                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue

                    try:
                        data = json.loads(message["data"])
                        channel = message["channel"]

                        if channel == CACHE_INVALIDATION_CHANNEL:
                            await _handle_invalidation_message(data)
                        elif channel == SETTINGS_CHANGE_CHANNEL:
                            await _handle_settings_message(data)

                    except json.JSONDecodeError:
                        logger.warning("Invalid pub/sub message format")
                    except Exception as e:
                        logger.error(f"Error processing pub/sub message: {e}")

            except asyncio.CancelledError:
                logger.info("Pub/sub listener cancelled")
            except Exception as e:
                logger.error(f"Pub/sub listener error: {e}")
            finally:
                await pubsub.unsubscribe()
                await pubsub.close()

        _pubsub_listener = asyncio.create_task(listener())
        logger.info("Started Redis pub/sub listener for distributed cache invalidation")

    except Exception as e:
        logger.warning(f"Failed to start pub/sub listener: {e}")


async def stop_pubsub_listener():
    """Stop the Redis pub/sub listener."""
    global _pubsub_listener

    if _pubsub_listener is not None:
        _pubsub_listener.cancel()
        try:
            await _pubsub_listener
        except asyncio.CancelledError:
            pass
        _pubsub_listener = None
        logger.info("Stopped Redis pub/sub listener")


class DistributedCache(RedisCache):
    """
    Extended Redis cache with distributed invalidation support.

    Automatically publishes invalidation messages when cache is modified.
    """

    def __init__(
        self,
        prefix: str = "cache",
        default_ttl: int = 3600,
        max_memory_items: int = 10000,
        cache_type: str = "general",
    ):
        super().__init__(prefix, default_ttl, max_memory_items)
        self.cache_type = cache_type

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value and optionally broadcast to other instances."""
        result = await super().set(key, value, ttl)
        return result

    async def delete(self, key: str, broadcast: bool = True) -> bool:
        """Delete value and optionally broadcast invalidation."""
        result = await super().delete(key)
        if broadcast:
            await publish_cache_invalidation(
                cache_type=self.cache_type,
                keys=[key],
            )
        return result

    async def invalidate_pattern(self, pattern: str, broadcast: bool = True) -> int:
        """
        Invalidate all keys matching a pattern.

        Args:
            pattern: Redis-style pattern (e.g., "user:*")
            broadcast: Whether to notify other instances

        Returns:
            Number of keys deleted
        """
        full_pattern = self._make_key(pattern)
        deleted = 0

        try:
            client = await get_redis_client()
            if client:
                # Use SCAN to find matching keys
                cursor = 0
                while True:
                    cursor, keys = await client.scan(cursor, match=full_pattern, count=100)
                    if keys:
                        await client.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
        except Exception as e:
            logger.warning(f"Pattern invalidation failed: {e}")

        # Also clear memory cache entries matching pattern
        import fnmatch
        to_delete = [
            k for k in self._fallback_cache.keys()
            if fnmatch.fnmatch(k, full_pattern)
        ]
        for k in to_delete:
            self._fallback_cache.pop(k, None)
            self._fallback_timestamps.pop(k, None)
            if k in self._fallback_order:
                self._fallback_order.remove(k)
        deleted += len(to_delete)

        if broadcast:
            await publish_cache_invalidation(
                cache_type=self.cache_type,
                pattern=pattern,
            )

        return deleted

    async def clear_all(self, broadcast: bool = True) -> bool:
        """Clear all cache entries for this prefix."""
        self.clear_memory_cache()

        try:
            client = await get_redis_client()
            if client:
                pattern = self._make_key("*")
                cursor = 0
                while True:
                    cursor, keys = await client.scan(cursor, match=pattern, count=100)
                    if keys:
                        await client.delete(*keys)
                    if cursor == 0:
                        break
        except Exception as e:
            logger.warning(f"Failed to clear Redis cache: {e}")

        if broadcast:
            await publish_cache_invalidation(
                cache_type=self.cache_type,
                pattern="*",
            )

        return True


# Pre-configured distributed cache instances
distributed_embedding_cache = DistributedCache(
    prefix="embed", default_ttl=EMBEDDING_CACHE_TTL, cache_type="embedding"
)
distributed_semantic_cache = DistributedCache(
    prefix="semantic", default_ttl=SEMANTIC_CACHE_TTL, cache_type="semantic"
)
distributed_search_cache = DistributedCache(
    prefix="search", default_ttl=SEARCH_CACHE_TTL, cache_type="search"
)
