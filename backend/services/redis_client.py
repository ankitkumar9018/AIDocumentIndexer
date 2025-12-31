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

import json
import os
from typing import Any, Optional

import redis.asyncio as aioredis
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
        except Exception as e:
            logger.warning("Redis connection failed, using in-memory fallback", error=str(e))
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
    """Generic Redis cache wrapper with fallback to in-memory."""

    def __init__(self, prefix: str = "cache", default_ttl: int = 3600):
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._fallback_cache: dict = {}  # In-memory fallback

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        cache_key = self._make_key(key)

        try:
            client = await get_redis_client()
            if client is None:
                # Redis disabled, use in-memory
                return self._fallback_cache.get(cache_key)
            value = await client.get(cache_key)
            if value:
                return json.loads(value)
        except Exception:
            # Fallback to in-memory
            return self._fallback_cache.get(cache_key)

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        cache_key = self._make_key(key)
        ttl = ttl or self.default_ttl

        try:
            client = await get_redis_client()
            if client is None:
                # Redis disabled, use in-memory
                self._fallback_cache[cache_key] = value
                return True
            await client.setex(cache_key, ttl, json.dumps(value))
            return True
        except Exception:
            # Fallback to in-memory
            self._fallback_cache[cache_key] = value
            return True

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        cache_key = self._make_key(key)

        try:
            client = await get_redis_client()
            if client is None:
                self._fallback_cache.pop(cache_key, None)
                return True
            await client.delete(cache_key)
        except Exception:
            self._fallback_cache.pop(cache_key, None)

        return True

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        cache_key = self._make_key(key)

        try:
            client = await get_redis_client()
            if client is None:
                return cache_key in self._fallback_cache
            return await client.exists(cache_key) > 0
        except Exception:
            return cache_key in self._fallback_cache


# Pre-configured cache instances
embedding_cache = RedisCache(prefix="embed", default_ttl=EMBEDDING_CACHE_TTL)
semantic_cache = RedisCache(prefix="semantic", default_ttl=SEMANTIC_CACHE_TTL)
session_cache = RedisCache(prefix="session", default_ttl=SESSION_CACHE_TTL)
search_cache = RedisCache(prefix="search", default_ttl=SEARCH_CACHE_TTL)
