"""
AIDocumentIndexer - Cost Limits Middleware
==========================================

Enforce per-user cost limits with:
- Daily and monthly spending caps
- Soft limits (warnings) vs hard limits (rejection)
- Alert generation at configurable thresholds
- Integration with LLM usage tracking
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, status
import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.middleware.auth import get_user_context
from backend.db.database import get_async_session, async_session_context
from backend.db.models import UserCostLimit, CostAlert, LLMUsageLog
from backend.services.permissions import UserContext

logger = structlog.get_logger(__name__)


# =============================================================================
# Cost Estimation
# =============================================================================

# Cost per 1K tokens for various models (as of late 2024)
MODEL_COSTS_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.001, "output": 0.005},
    # Default for unknown models
    "default": {"input": 0.01, "output": 0.03},
}

# Embedding model costs per 1K tokens
EMBEDDING_COSTS_PER_1K_TOKENS: Dict[str, float] = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.0001,
    "default": 0.0001,
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    is_embedding: bool = False
) -> float:
    """
    Estimate the cost of an LLM request.

    Args:
        model: Model name/ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens (0 for embeddings)
        is_embedding: Whether this is an embedding request

    Returns:
        Estimated cost in USD
    """
    if is_embedding:
        cost_per_1k = EMBEDDING_COSTS_PER_1K_TOKENS.get(
            model,
            EMBEDDING_COSTS_PER_1K_TOKENS["default"]
        )
        return (input_tokens / 1000) * cost_per_1k

    # Find matching model costs
    model_lower = model.lower()
    costs = MODEL_COSTS_PER_1K_TOKENS.get("default")

    for model_key, model_costs in MODEL_COSTS_PER_1K_TOKENS.items():
        if model_key in model_lower:
            costs = model_costs
            break

    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]

    return input_cost + output_cost


def estimate_tokens_from_text(text: str) -> int:
    """
    Rough estimate of token count from text.

    Uses the approximation of ~4 characters per token.
    """
    return len(text) // 4


# =============================================================================
# Cost Limit Configuration
# =============================================================================

@dataclass
class CostLimitSettings:
    """Cost limit settings for a user."""
    daily_limit_usd: float = 10.0
    monthly_limit_usd: float = 100.0
    enforce_hard_limit: bool = True
    alert_thresholds: List[int] = None  # Percentages: [50, 80, 100]

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = [50, 80, 100]


# Default settings by tier level
DEFAULT_TIER_COST_LIMITS: Dict[int, CostLimitSettings] = {
    0: CostLimitSettings(
        daily_limit_usd=1.0,
        monthly_limit_usd=10.0,
        enforce_hard_limit=True,
    ),
    25: CostLimitSettings(
        daily_limit_usd=5.0,
        monthly_limit_usd=50.0,
        enforce_hard_limit=True,
    ),
    50: CostLimitSettings(
        daily_limit_usd=20.0,
        monthly_limit_usd=200.0,
        enforce_hard_limit=True,
    ),
    75: CostLimitSettings(
        daily_limit_usd=100.0,
        monthly_limit_usd=1000.0,
        enforce_hard_limit=False,  # Soft limit for enterprise
    ),
    100: CostLimitSettings(
        daily_limit_usd=1000.0,
        monthly_limit_usd=10000.0,
        enforce_hard_limit=False,  # Admins get soft limits
    ),
}


def get_default_cost_limits_for_tier(tier_level: int) -> CostLimitSettings:
    """Get default cost limit settings based on tier level."""
    applicable_tier = 0
    for tier_threshold in sorted(DEFAULT_TIER_COST_LIMITS.keys()):
        if tier_level >= tier_threshold:
            applicable_tier = tier_threshold

    return DEFAULT_TIER_COST_LIMITS.get(applicable_tier, DEFAULT_TIER_COST_LIMITS[0])


# =============================================================================
# Cost Limit Result
# =============================================================================

@dataclass
class CostLimitResult:
    """Result of a cost limit check."""
    allowed: bool
    is_soft_limit: bool = False  # True if this is a warning, not a block
    limit_type: Optional[str] = None  # 'daily' or 'monthly'
    current_usage_usd: float = 0
    limit_usd: float = 0
    usage_percent: float = 0
    alerts_triggered: List[str] = None

    def __post_init__(self):
        if self.alerts_triggered is None:
            self.alerts_triggered = []


# =============================================================================
# Cost Limit Checker
# =============================================================================

class CostLimitChecker:
    """Check and enforce cost limits for users."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def get_or_create_user_cost_limit(
        self,
        db: AsyncSession,
        user_id: str,
        tier_level: int
    ) -> UserCostLimit:
        """Get or create cost limit record for a user."""
        result = await db.execute(
            select(UserCostLimit).where(
                UserCostLimit.user_id == UUID(user_id)
            )
        )
        cost_limit = result.scalar_one_or_none()

        if cost_limit is None:
            # Create new record with tier defaults
            defaults = get_default_cost_limits_for_tier(tier_level)
            cost_limit = UserCostLimit(
                id=uuid4(),
                user_id=UUID(user_id),
                daily_limit_usd=defaults.daily_limit_usd,
                monthly_limit_usd=defaults.monthly_limit_usd,
                enforce_hard_limit=defaults.enforce_hard_limit,
                alert_thresholds=defaults.alert_thresholds,
                current_daily_usage_usd=0.0,
                current_monthly_usage_usd=0.0,
                last_daily_reset=datetime.utcnow(),
                last_monthly_reset=datetime.utcnow(),
            )
            db.add(cost_limit)
            await db.flush()

        # Check for period resets
        await self._check_period_resets(db, cost_limit)

        return cost_limit

    async def _check_period_resets(
        self,
        db: AsyncSession,
        cost_limit: UserCostLimit
    ) -> None:
        """Reset usage counters if period has elapsed."""
        now = datetime.utcnow()
        needs_update = False

        # Check daily reset
        if cost_limit.last_daily_reset:
            daily_diff = now - cost_limit.last_daily_reset
            if daily_diff >= timedelta(days=1):
                cost_limit.current_daily_usage_usd = 0.0
                cost_limit.last_daily_reset = now
                needs_update = True

        # Check monthly reset
        if cost_limit.last_monthly_reset:
            monthly_diff = now - cost_limit.last_monthly_reset
            if monthly_diff >= timedelta(days=30):
                cost_limit.current_monthly_usage_usd = 0.0
                cost_limit.last_monthly_reset = now
                needs_update = True

        if needs_update:
            await db.flush()

    async def check_cost_limit(
        self,
        db: AsyncSession,
        user_id: str,
        tier_level: int,
        estimated_cost: float = 0
    ) -> CostLimitResult:
        """
        Check if a request is allowed under cost limits.

        Args:
            db: Database session
            user_id: User making the request
            tier_level: User's access tier level
            estimated_cost: Estimated cost of the request

        Returns:
            CostLimitResult indicating if request is allowed
        """
        cost_limit = await self.get_or_create_user_cost_limit(
            db, user_id, tier_level
        )

        alerts_triggered = []

        # Check daily limit
        projected_daily = cost_limit.current_daily_usage_usd + estimated_cost
        daily_percent = (projected_daily / cost_limit.daily_limit_usd) * 100

        # Check for alert thresholds
        if cost_limit.alert_thresholds:
            for threshold in cost_limit.alert_thresholds:
                current_percent = (cost_limit.current_daily_usage_usd / cost_limit.daily_limit_usd) * 100
                if current_percent < threshold <= daily_percent:
                    alerts_triggered.append(f"daily_{threshold}")

        if projected_daily > cost_limit.daily_limit_usd:
            return CostLimitResult(
                allowed=not cost_limit.enforce_hard_limit,
                is_soft_limit=not cost_limit.enforce_hard_limit,
                limit_type='daily',
                current_usage_usd=cost_limit.current_daily_usage_usd,
                limit_usd=cost_limit.daily_limit_usd,
                usage_percent=daily_percent,
                alerts_triggered=alerts_triggered,
            )

        # Check monthly limit
        projected_monthly = cost_limit.current_monthly_usage_usd + estimated_cost
        monthly_percent = (projected_monthly / cost_limit.monthly_limit_usd) * 100

        # Check monthly alert thresholds
        if cost_limit.alert_thresholds:
            for threshold in cost_limit.alert_thresholds:
                current_percent = (cost_limit.current_monthly_usage_usd / cost_limit.monthly_limit_usd) * 100
                if current_percent < threshold <= monthly_percent:
                    alerts_triggered.append(f"monthly_{threshold}")

        if projected_monthly > cost_limit.monthly_limit_usd:
            return CostLimitResult(
                allowed=not cost_limit.enforce_hard_limit,
                is_soft_limit=not cost_limit.enforce_hard_limit,
                limit_type='monthly',
                current_usage_usd=cost_limit.current_monthly_usage_usd,
                limit_usd=cost_limit.monthly_limit_usd,
                usage_percent=monthly_percent,
                alerts_triggered=alerts_triggered,
            )

        return CostLimitResult(
            allowed=True,
            current_usage_usd=cost_limit.current_daily_usage_usd,
            limit_usd=cost_limit.daily_limit_usd,
            usage_percent=daily_percent,
            alerts_triggered=alerts_triggered,
        )

    async def record_usage(
        self,
        db: AsyncSession,
        user_id: str,
        tier_level: int,
        cost_usd: float
    ) -> None:
        """
        Record usage cost for a user.

        Call this after a successful LLM request.
        """
        cost_limit = await self.get_or_create_user_cost_limit(
            db, user_id, tier_level
        )

        cost_limit.current_daily_usage_usd += cost_usd
        cost_limit.current_monthly_usage_usd += cost_usd

        await db.flush()

        # Check and create alerts
        await self._check_and_create_alerts(db, cost_limit)

    async def _check_and_create_alerts(
        self,
        db: AsyncSession,
        cost_limit: UserCostLimit
    ) -> None:
        """Create alerts if thresholds are crossed."""
        if not cost_limit.alert_thresholds:
            return

        for threshold in cost_limit.alert_thresholds:
            # Check daily threshold
            daily_percent = (cost_limit.current_daily_usage_usd / cost_limit.daily_limit_usd) * 100
            if daily_percent >= threshold:
                await self._create_alert_if_not_exists(
                    db, cost_limit.id, "daily", threshold, cost_limit.current_daily_usage_usd
                )

            # Check monthly threshold
            monthly_percent = (cost_limit.current_monthly_usage_usd / cost_limit.monthly_limit_usd) * 100
            if monthly_percent >= threshold:
                await self._create_alert_if_not_exists(
                    db, cost_limit.id, "monthly", threshold, cost_limit.current_monthly_usage_usd
                )

    async def _create_alert_if_not_exists(
        self,
        db: AsyncSession,
        cost_limit_id: UUID,
        alert_type: str,
        threshold: int,
        usage: float
    ) -> None:
        """Create an alert if one doesn't already exist for this threshold/period."""
        # Check if alert already exists for today (daily) or this month (monthly)
        now = datetime.utcnow()

        if alert_type == "daily":
            start_of_period = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_of_period = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        result = await db.execute(
            select(CostAlert).where(
                CostAlert.cost_limit_id == cost_limit_id,
                CostAlert.alert_type == alert_type,
                CostAlert.threshold_percent == threshold,
                CostAlert.created_at >= start_of_period,
            )
        )
        existing = result.scalar_one_or_none()

        if existing is None:
            alert = CostAlert(
                id=uuid4(),
                cost_limit_id=cost_limit_id,
                alert_type=alert_type,
                threshold_percent=threshold,
                usage_at_alert_usd=usage,
                notified=False,
                acknowledged=False,
            )
            db.add(alert)
            await db.flush()

            logger.info(
                "Cost alert created",
                cost_limit_id=str(cost_limit_id),
                alert_type=alert_type,
                threshold=threshold,
                usage=usage
            )

    async def get_user_cost_status(
        self,
        db: AsyncSession,
        user_id: str,
        tier_level: int
    ) -> Dict:
        """Get current cost limit status for a user."""
        cost_limit = await self.get_or_create_user_cost_limit(
            db, user_id, tier_level
        )

        return {
            'daily': {
                'used': cost_limit.current_daily_usage_usd,
                'limit': cost_limit.daily_limit_usd,
                'remaining': max(0, cost_limit.daily_limit_usd - cost_limit.current_daily_usage_usd),
                'percent': (cost_limit.current_daily_usage_usd / cost_limit.daily_limit_usd) * 100,
                'reset_at': cost_limit.last_daily_reset + timedelta(days=1) if cost_limit.last_daily_reset else None,
            },
            'monthly': {
                'used': cost_limit.current_monthly_usage_usd,
                'limit': cost_limit.monthly_limit_usd,
                'remaining': max(0, cost_limit.monthly_limit_usd - cost_limit.current_monthly_usage_usd),
                'percent': (cost_limit.current_monthly_usage_usd / cost_limit.monthly_limit_usd) * 100,
                'reset_at': cost_limit.last_monthly_reset + timedelta(days=30) if cost_limit.last_monthly_reset else None,
            },
            'enforce_hard_limit': cost_limit.enforce_hard_limit,
            'alert_thresholds': cost_limit.alert_thresholds,
        }

    async def update_user_cost_limits(
        self,
        db: AsyncSession,
        user_id: str,
        tier_level: int,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        enforce_hard_limit: Optional[bool] = None,
        alert_thresholds: Optional[List[int]] = None
    ) -> UserCostLimit:
        """Update cost limits for a user."""
        cost_limit = await self.get_or_create_user_cost_limit(
            db, user_id, tier_level
        )

        if daily_limit is not None:
            cost_limit.daily_limit_usd = daily_limit
        if monthly_limit is not None:
            cost_limit.monthly_limit_usd = monthly_limit
        if enforce_hard_limit is not None:
            cost_limit.enforce_hard_limit = enforce_hard_limit
        if alert_thresholds is not None:
            cost_limit.alert_thresholds = alert_thresholds

        await db.flush()
        return cost_limit

    async def get_pending_alerts(
        self,
        db: AsyncSession,
        user_id: Optional[str] = None,
        unacknowledged_only: bool = True
    ) -> List[CostAlert]:
        """Get pending cost alerts."""
        query = select(CostAlert)

        if user_id:
            query = query.join(UserCostLimit).where(
                UserCostLimit.user_id == UUID(user_id)
            )

        if unacknowledged_only:
            query = query.where(CostAlert.acknowledged == False)

        query = query.order_by(CostAlert.created_at.desc())

        result = await db.execute(query)
        return list(result.scalars().all())

    async def acknowledge_alert(
        self,
        db: AsyncSession,
        alert_id: str
    ) -> Optional[CostAlert]:
        """Acknowledge a cost alert."""
        result = await db.execute(
            select(CostAlert).where(CostAlert.id == UUID(alert_id))
        )
        alert = result.scalar_one_or_none()

        if alert:
            alert.acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            await db.flush()

        return alert


# Global checker instance
_checker = CostLimitChecker()


def get_cost_limit_checker() -> CostLimitChecker:
    """Get the global cost limit checker instance."""
    return _checker


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def cost_limit_dependency(
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
) -> CostLimitResult:
    """
    FastAPI dependency for cost limit checking.

    This is a pre-check that verifies the user hasn't exceeded their limits.
    Actual cost recording should be done after the LLM request completes.

    Usage:
        @router.post("/chat")
        async def chat(
            cost_check: CostLimitResult = Depends(cost_limit_dependency),
            ...
        ):
            # Request is allowed if we get here (or it's a soft limit)
            ...
    """
    checker = get_cost_limit_checker()

    result = await checker.check_cost_limit(
        db=db,
        user_id=user.user_id,
        tier_level=user.access_tier_level,
        estimated_cost=0  # Pre-check without specific estimate
    )

    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "Cost limit exceeded",
                "limit_type": result.limit_type,
                "current_usage_usd": result.current_usage_usd,
                "limit_usd": result.limit_usd,
                "usage_percent": result.usage_percent,
            }
        )

    return result


async def check_estimated_cost(
    db: AsyncSession,
    user_id: str,
    tier_level: int,
    model: str,
    estimated_input_tokens: int,
    estimated_output_tokens: int = 0,
    is_embedding: bool = False
) -> CostLimitResult:
    """
    Check if an estimated cost is within limits.

    Call this before making an LLM request.
    """
    checker = get_cost_limit_checker()
    estimated_cost = estimate_cost(
        model=model,
        input_tokens=estimated_input_tokens,
        output_tokens=estimated_output_tokens,
        is_embedding=is_embedding
    )

    return await checker.check_cost_limit(
        db=db,
        user_id=user_id,
        tier_level=tier_level,
        estimated_cost=estimated_cost
    )


async def record_actual_cost(
    db: AsyncSession,
    user_id: str,
    tier_level: int,
    model: str,
    input_tokens: int,
    output_tokens: int,
    is_embedding: bool = False
) -> float:
    """
    Record actual cost after an LLM request.

    Call this after a successful LLM request.

    Returns:
        The actual cost in USD
    """
    checker = get_cost_limit_checker()
    actual_cost = estimate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        is_embedding=is_embedding
    )

    await checker.record_usage(
        db=db,
        user_id=user_id,
        tier_level=tier_level,
        cost_usd=actual_cost
    )

    return actual_cost


# =============================================================================
# Utility Functions
# =============================================================================

async def get_user_cost_status(
    db: AsyncSession,
    user_id: str,
    tier_level: int
) -> Dict:
    """Get current cost limit status for a user."""
    checker = get_cost_limit_checker()
    return await checker.get_user_cost_status(db, user_id, tier_level)


async def reset_user_cost_tracking(
    db: AsyncSession,
    user_id: str,
    tier_level: int,
    reset_daily: bool = True,
    reset_monthly: bool = False
) -> None:
    """Reset cost tracking for a user (admin function)."""
    cost_limit = await get_cost_limit_checker().get_or_create_user_cost_limit(
        db, user_id, tier_level
    )

    if reset_daily:
        cost_limit.current_daily_usage_usd = 0.0
        cost_limit.last_daily_reset = datetime.utcnow()

    if reset_monthly:
        cost_limit.current_monthly_usage_usd = 0.0
        cost_limit.last_monthly_reset = datetime.utcnow()

    await db.flush()
    logger.info(
        "Reset cost tracking for user",
        user_id=user_id,
        reset_daily=reset_daily,
        reset_monthly=reset_monthly
    )
