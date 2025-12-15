"""
AIDocumentIndexer - Rate Limiting Middleware
=============================================

Token bucket rate limiting with:
- Per-user rate limits based on access tier
- Per-provider rate limits to respect API limits
- In-memory storage with optional Redis support
- Database-persisted configuration
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
import structlog

from backend.api.middleware.auth import get_user_context
from backend.db.database import get_async_session, async_session_context
from backend.db.models import RateLimitConfig, AccessTier
from backend.services.permissions import UserContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


# =============================================================================
# Rate Limit Storage
# =============================================================================

@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    tokens: float
    last_refill: float
    refill_rate: float  # tokens per second

    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to consume tokens from the bucket.

        Returns:
            Tuple of (success, retry_after_seconds)
        """
        now = time.time()

        # Refill tokens based on time elapsed
        time_passed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + time_passed * self.refill_rate
        )
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0
        else:
            # Calculate when enough tokens will be available
            tokens_needed = tokens - self.tokens
            retry_after = tokens_needed / self.refill_rate
            return False, retry_after


@dataclass
class UserRateLimitState:
    """Rate limit state for a user."""
    minute_bucket: TokenBucket
    hour_bucket: TokenBucket
    day_bucket: TokenBucket
    token_minute_bucket: TokenBucket
    token_day_bucket: TokenBucket


class InMemoryRateLimitStorage:
    """In-memory rate limit storage with automatic cleanup."""

    def __init__(self, cleanup_interval: int = 3600):
        self._user_states: Dict[str, UserRateLimitState] = {}
        self._provider_states: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    async def get_user_state(
        self,
        user_id: str,
        config: "RateLimitSettings"
    ) -> UserRateLimitState:
        """Get or create rate limit state for a user."""
        async with self._lock:
            # Periodic cleanup
            now = time.time()
            if now - self._last_cleanup > self._cleanup_interval:
                await self._cleanup_stale_entries()
                self._last_cleanup = now

            if user_id not in self._user_states:
                self._user_states[user_id] = self._create_user_state(config)

            return self._user_states[user_id]

    async def get_provider_state(
        self,
        provider_id: str,
        requests_per_minute: int = 100
    ) -> TokenBucket:
        """Get or create rate limit state for a provider."""
        async with self._lock:
            if provider_id not in self._provider_states:
                self._provider_states[provider_id] = TokenBucket(
                    capacity=requests_per_minute,
                    tokens=float(requests_per_minute),
                    last_refill=time.time(),
                    refill_rate=requests_per_minute / 60.0
                )
            return self._provider_states[provider_id]

    def _create_user_state(self, config: "RateLimitSettings") -> UserRateLimitState:
        """Create fresh rate limit state for a user."""
        now = time.time()
        return UserRateLimitState(
            minute_bucket=TokenBucket(
                capacity=config.requests_per_minute,
                tokens=float(config.requests_per_minute),
                last_refill=now,
                refill_rate=config.requests_per_minute / 60.0
            ),
            hour_bucket=TokenBucket(
                capacity=config.requests_per_hour,
                tokens=float(config.requests_per_hour),
                last_refill=now,
                refill_rate=config.requests_per_hour / 3600.0
            ),
            day_bucket=TokenBucket(
                capacity=config.requests_per_day,
                tokens=float(config.requests_per_day),
                last_refill=now,
                refill_rate=config.requests_per_day / 86400.0
            ),
            token_minute_bucket=TokenBucket(
                capacity=config.tokens_per_minute,
                tokens=float(config.tokens_per_minute),
                last_refill=now,
                refill_rate=config.tokens_per_minute / 60.0
            ),
            token_day_bucket=TokenBucket(
                capacity=config.tokens_per_day,
                tokens=float(config.tokens_per_day),
                last_refill=now,
                refill_rate=config.tokens_per_day / 86400.0
            ),
        )

    async def _cleanup_stale_entries(self) -> None:
        """Remove entries that haven't been used recently."""
        now = time.time()
        stale_threshold = 3600  # 1 hour

        # Find stale user entries
        stale_users = [
            user_id
            for user_id, state in self._user_states.items()
            if now - state.minute_bucket.last_refill > stale_threshold
        ]

        for user_id in stale_users:
            del self._user_states[user_id]

        if stale_users:
            logger.debug(
                "Cleaned up stale rate limit entries",
                count=len(stale_users)
            )

    async def reset_user(self, user_id: str) -> None:
        """Reset rate limits for a user."""
        async with self._lock:
            if user_id in self._user_states:
                del self._user_states[user_id]


# Global storage instance
_storage = InMemoryRateLimitStorage()


# =============================================================================
# Rate Limit Configuration
# =============================================================================

@dataclass
class RateLimitSettings:
    """Rate limit settings for a user/tier."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    tokens_per_minute: int = 100000
    tokens_per_day: int = 1000000
    operation_limits: Optional[Dict[str, int]] = None

    @classmethod
    def from_model(cls, model: RateLimitConfig) -> "RateLimitSettings":
        """Create from database model."""
        return cls(
            requests_per_minute=model.requests_per_minute,
            requests_per_hour=model.requests_per_hour,
            requests_per_day=model.requests_per_day,
            tokens_per_minute=model.tokens_per_minute,
            tokens_per_day=model.tokens_per_day,
            operation_limits=model.operation_limits,
        )


# Default settings by tier level
DEFAULT_TIER_LIMITS: Dict[int, RateLimitSettings] = {
    # Basic tier (level 0-25)
    0: RateLimitSettings(
        requests_per_minute=20,
        requests_per_hour=200,
        requests_per_day=1000,
        tokens_per_minute=20000,
        tokens_per_day=200000,
    ),
    # Standard tier (level 26-50)
    25: RateLimitSettings(
        requests_per_minute=60,
        requests_per_hour=600,
        requests_per_day=5000,
        tokens_per_minute=60000,
        tokens_per_day=600000,
    ),
    # Professional tier (level 51-75)
    50: RateLimitSettings(
        requests_per_minute=120,
        requests_per_hour=1200,
        requests_per_day=10000,
        tokens_per_minute=120000,
        tokens_per_day=1200000,
    ),
    # Enterprise tier (level 76-99)
    75: RateLimitSettings(
        requests_per_minute=300,
        requests_per_hour=3000,
        requests_per_day=30000,
        tokens_per_minute=300000,
        tokens_per_day=3000000,
    ),
    # Admin tier (level 100)
    100: RateLimitSettings(
        requests_per_minute=1000,
        requests_per_hour=10000,
        requests_per_day=100000,
        tokens_per_minute=1000000,
        tokens_per_day=10000000,
    ),
}


def get_default_settings_for_tier(tier_level: int) -> RateLimitSettings:
    """Get default rate limit settings based on tier level."""
    # Find the highest tier threshold that's <= user's tier
    applicable_tier = 0
    for tier_threshold in sorted(DEFAULT_TIER_LIMITS.keys()):
        if tier_level >= tier_threshold:
            applicable_tier = tier_threshold

    return DEFAULT_TIER_LIMITS.get(applicable_tier, DEFAULT_TIER_LIMITS[0])


# Cache for tier configurations
_tier_config_cache: Dict[str, Tuple[RateLimitSettings, float]] = {}
_cache_ttl = 300  # 5 minutes


async def get_tier_rate_limits(
    tier_id: str,
    tier_level: int,
    db: Optional[AsyncSession] = None
) -> RateLimitSettings:
    """
    Get rate limit settings for a tier.

    Checks database first, falls back to defaults.
    """
    cache_key = tier_id
    now = time.time()

    # Check cache
    if cache_key in _tier_config_cache:
        settings, cached_at = _tier_config_cache[cache_key]
        if now - cached_at < _cache_ttl:
            return settings

    # Try to load from database
    if db is not None:
        try:
            result = await db.execute(
                select(RateLimitConfig).where(
                    RateLimitConfig.tier_id == UUID(tier_id)
                )
            )
            config = result.scalar_one_or_none()

            if config:
                settings = RateLimitSettings.from_model(config)
                _tier_config_cache[cache_key] = (settings, now)
                return settings
        except Exception as e:
            logger.warning(
                "Failed to load rate limit config from database",
                tier_id=tier_id,
                error=str(e)
            )

    # Fall back to defaults
    settings = get_default_settings_for_tier(tier_level)
    _tier_config_cache[cache_key] = (settings, now)
    return settings


async def clear_tier_cache(tier_id: Optional[str] = None) -> None:
    """Clear rate limit configuration cache."""
    if tier_id:
        _tier_config_cache.pop(tier_id, None)
    else:
        _tier_config_cache.clear()


# =============================================================================
# Rate Limit Checker
# =============================================================================

@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    limit_type: Optional[str] = None  # 'minute', 'hour', 'day', 'tokens_minute', 'tokens_day'
    retry_after: float = 0
    remaining_minute: int = 0
    remaining_hour: int = 0
    remaining_day: int = 0
    limit_minute: int = 0
    limit_hour: int = 0
    limit_day: int = 0


class RateLimitChecker:
    """Check rate limits for requests."""

    def __init__(self, storage: InMemoryRateLimitStorage = _storage):
        self.storage = storage

    async def check_request_limit(
        self,
        user_id: str,
        tier_id: str,
        tier_level: int,
        operation: Optional[str] = None,
        db: Optional[AsyncSession] = None
    ) -> RateLimitResult:
        """
        Check if a request is allowed under rate limits.

        Args:
            user_id: User making the request
            tier_id: User's access tier ID
            tier_level: User's access tier level
            operation: Optional operation type for operation-specific limits
            db: Database session for loading configuration

        Returns:
            RateLimitResult indicating if request is allowed
        """
        settings = await get_tier_rate_limits(tier_id, tier_level, db)
        state = await self.storage.get_user_state(user_id, settings)

        # Check operation-specific limits first
        if operation and settings.operation_limits:
            op_limit = settings.operation_limits.get(operation)
            if op_limit is not None:
                # For simplicity, use minute bucket logic for operation limits
                pass  # TODO: Implement operation-specific buckets if needed

        # Check minute limit
        allowed, retry_after = state.minute_bucket.consume()
        if not allowed:
            return RateLimitResult(
                allowed=False,
                limit_type='minute',
                retry_after=retry_after,
                remaining_minute=int(state.minute_bucket.tokens),
                remaining_hour=int(state.hour_bucket.tokens),
                remaining_day=int(state.day_bucket.tokens),
                limit_minute=settings.requests_per_minute,
                limit_hour=settings.requests_per_hour,
                limit_day=settings.requests_per_day,
            )

        # Check hour limit
        allowed, retry_after = state.hour_bucket.consume()
        if not allowed:
            # Refund the minute token
            state.minute_bucket.tokens += 1
            return RateLimitResult(
                allowed=False,
                limit_type='hour',
                retry_after=retry_after,
                remaining_minute=int(state.minute_bucket.tokens),
                remaining_hour=int(state.hour_bucket.tokens),
                remaining_day=int(state.day_bucket.tokens),
                limit_minute=settings.requests_per_minute,
                limit_hour=settings.requests_per_hour,
                limit_day=settings.requests_per_day,
            )

        # Check day limit
        allowed, retry_after = state.day_bucket.consume()
        if not allowed:
            # Refund the minute and hour tokens
            state.minute_bucket.tokens += 1
            state.hour_bucket.tokens += 1
            return RateLimitResult(
                allowed=False,
                limit_type='day',
                retry_after=retry_after,
                remaining_minute=int(state.minute_bucket.tokens),
                remaining_hour=int(state.hour_bucket.tokens),
                remaining_day=int(state.day_bucket.tokens),
                limit_minute=settings.requests_per_minute,
                limit_hour=settings.requests_per_hour,
                limit_day=settings.requests_per_day,
            )

        return RateLimitResult(
            allowed=True,
            remaining_minute=int(state.minute_bucket.tokens),
            remaining_hour=int(state.hour_bucket.tokens),
            remaining_day=int(state.day_bucket.tokens),
            limit_minute=settings.requests_per_minute,
            limit_hour=settings.requests_per_hour,
            limit_day=settings.requests_per_day,
        )

    async def check_token_limit(
        self,
        user_id: str,
        tier_id: str,
        tier_level: int,
        tokens: int,
        db: Optional[AsyncSession] = None
    ) -> RateLimitResult:
        """
        Check if token usage is allowed under rate limits.

        Called before making LLM requests to check estimated token usage.
        """
        settings = await get_tier_rate_limits(tier_id, tier_level, db)
        state = await self.storage.get_user_state(user_id, settings)

        # Check minute token limit
        allowed, retry_after = state.token_minute_bucket.consume(tokens)
        if not allowed:
            return RateLimitResult(
                allowed=False,
                limit_type='tokens_minute',
                retry_after=retry_after,
            )

        # Check day token limit
        allowed, retry_after = state.token_day_bucket.consume(tokens)
        if not allowed:
            # Refund minute tokens
            state.token_minute_bucket.tokens += tokens
            return RateLimitResult(
                allowed=False,
                limit_type='tokens_day',
                retry_after=retry_after,
            )

        return RateLimitResult(allowed=True)

    async def check_provider_limit(
        self,
        provider_id: str,
        requests_per_minute: int = 100
    ) -> RateLimitResult:
        """
        Check provider-level rate limits.

        Used to respect API provider rate limits.
        """
        bucket = await self.storage.get_provider_state(
            provider_id,
            requests_per_minute
        )

        allowed, retry_after = bucket.consume()
        if not allowed:
            return RateLimitResult(
                allowed=False,
                limit_type='provider',
                retry_after=retry_after,
            )

        return RateLimitResult(allowed=True)

    async def get_current_usage(
        self,
        user_id: str,
        tier_id: str,
        tier_level: int,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, any]:
        """Get current rate limit usage for a user."""
        settings = await get_tier_rate_limits(tier_id, tier_level, db)
        state = await self.storage.get_user_state(user_id, settings)

        return {
            'requests': {
                'minute': {
                    'used': settings.requests_per_minute - int(state.minute_bucket.tokens),
                    'limit': settings.requests_per_minute,
                    'remaining': int(state.minute_bucket.tokens),
                },
                'hour': {
                    'used': settings.requests_per_hour - int(state.hour_bucket.tokens),
                    'limit': settings.requests_per_hour,
                    'remaining': int(state.hour_bucket.tokens),
                },
                'day': {
                    'used': settings.requests_per_day - int(state.day_bucket.tokens),
                    'limit': settings.requests_per_day,
                    'remaining': int(state.day_bucket.tokens),
                },
            },
            'tokens': {
                'minute': {
                    'used': settings.tokens_per_minute - int(state.token_minute_bucket.tokens),
                    'limit': settings.tokens_per_minute,
                    'remaining': int(state.token_minute_bucket.tokens),
                },
                'day': {
                    'used': settings.tokens_per_day - int(state.token_day_bucket.tokens),
                    'limit': settings.tokens_per_day,
                    'remaining': int(state.token_day_bucket.tokens),
                },
            },
        }


# Global checker instance
_checker = RateLimitChecker()


def get_rate_limit_checker() -> RateLimitChecker:
    """Get the global rate limit checker instance."""
    return _checker


# =============================================================================
# FastAPI Middleware / Dependencies
# =============================================================================

async def rate_limit_dependency(
    request: Request,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
) -> RateLimitResult:
    """
    FastAPI dependency for rate limiting.

    Usage:
        @router.post("/chat")
        async def chat(
            rate_limit: RateLimitResult = Depends(rate_limit_dependency),
            ...
        ):
            # Request is already allowed if we get here
            ...
    """
    checker = get_rate_limit_checker()

    # Determine operation from path
    operation = request.url.path.split('/')[-1] if request.url.path else None

    result = await checker.check_request_limit(
        user_id=user.user_id,
        tier_id=user.access_tier_id or "default",
        tier_level=user.access_tier_level,
        operation=operation,
        db=db
    )

    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "limit_type": result.limit_type,
                "retry_after": result.retry_after,
            },
            headers={
                "Retry-After": str(int(result.retry_after) + 1),
                "X-RateLimit-Limit": str(result.limit_minute),
                "X-RateLimit-Remaining": str(result.remaining_minute),
                "X-RateLimit-Reset": str(int(time.time() + 60)),
            }
        )

    return result


def add_rate_limit_headers(response: JSONResponse, result: RateLimitResult) -> None:
    """Add rate limit headers to a response."""
    response.headers["X-RateLimit-Limit-Minute"] = str(result.limit_minute)
    response.headers["X-RateLimit-Limit-Hour"] = str(result.limit_hour)
    response.headers["X-RateLimit-Limit-Day"] = str(result.limit_day)
    response.headers["X-RateLimit-Remaining-Minute"] = str(result.remaining_minute)
    response.headers["X-RateLimit-Remaining-Hour"] = str(result.remaining_hour)
    response.headers["X-RateLimit-Remaining-Day"] = str(result.remaining_day)
    response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))


# =============================================================================
# Utility Functions
# =============================================================================

async def reset_user_rate_limits(user_id: str) -> None:
    """Reset rate limits for a user (admin function)."""
    await _storage.reset_user(user_id)
    logger.info("Reset rate limits for user", user_id=user_id)


async def get_user_rate_limit_usage(
    user_id: str,
    tier_id: str,
    tier_level: int,
    db: Optional[AsyncSession] = None
) -> Dict:
    """Get current rate limit usage for a user."""
    checker = get_rate_limit_checker()
    return await checker.get_current_usage(user_id, tier_id, tier_level, db)
