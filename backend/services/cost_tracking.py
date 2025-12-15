"""
AIDocumentIndexer - LiteLLM Cost Tracking Service
==================================================

Tracks LLM usage costs across all providers using LiteLLM's
cost tracking capabilities.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

# Try to import litellm for cost tracking
try:
    import litellm
    from litellm import completion_cost
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("litellm not installed. Cost tracking will use estimates.")


# =============================================================================
# Cost Data (fallback when litellm not available)
# =============================================================================

# Approximate costs per 1K tokens (USD)
MODEL_COSTS = {
    # OpenAI
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},

    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-2": {"input": 0.008, "output": 0.024},
    "claude-instant": {"input": 0.0008, "output": 0.0024},

    # Embeddings
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},

    # Ollama (local - no cost)
    "llama2": {"input": 0.0, "output": 0.0},
    "llama3": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "mixtral": {"input": 0.0, "output": 0.0},
}


# =============================================================================
# Enums and Types
# =============================================================================

class UsageType(str, Enum):
    """Type of LLM usage."""
    CHAT = "chat"
    EMBEDDING = "embedding"
    GENERATION = "generation"
    COLLABORATION = "collaboration"
    RAG = "rag"


class CostPeriod(str, Enum):
    """Time period for cost aggregation."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    ALL_TIME = "all_time"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UsageRecord:
    """A single usage record."""
    id: str
    user_id: str
    model: str
    provider: str
    usage_type: UsageType
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostSummary:
    """Cost summary for a time period."""
    period: CostPeriod
    start_date: datetime
    end_date: datetime
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    by_model: Dict[str, float]
    by_provider: Dict[str, float]
    by_usage_type: Dict[str, float]
    by_user: Dict[str, float]


@dataclass
class UserCostSummary:
    """Cost summary for a specific user."""
    user_id: str
    period: CostPeriod
    total_cost: float
    total_requests: int
    total_tokens: int
    by_model: Dict[str, float]
    by_usage_type: Dict[str, float]
    daily_costs: List[Dict[str, Any]]


@dataclass
class CostAlert:
    """Cost alert configuration."""
    id: str
    user_id: Optional[str]  # None for system-wide
    threshold: float  # USD
    period: CostPeriod
    enabled: bool = True
    notification_sent: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Cost Tracking Service
# =============================================================================

class CostTrackingService:
    """
    Service for tracking LLM usage costs.

    Uses LiteLLM's cost tracking when available, falls back to
    estimates otherwise.
    """

    def __init__(self):
        self._records: List[UsageRecord] = []
        self._alerts: Dict[str, CostAlert] = {}
        self._alert_callbacks: List[callable] = []

    def _estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a request."""
        # Try litellm first
        if LITELLM_AVAILABLE:
            try:
                cost = completion_cost(
                    model=model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )
                return cost
            except Exception:
                pass

        # Fallback to our estimates
        model_lower = model.lower()

        # Find matching model
        for model_key, costs in MODEL_COSTS.items():
            if model_key in model_lower:
                input_cost = (input_tokens / 1000) * costs["input"]
                output_cost = (output_tokens / 1000) * costs["output"]
                return input_cost + output_cost

        # Default to GPT-4 pricing if unknown
        default_costs = MODEL_COSTS["gpt-4"]
        return (input_tokens / 1000) * default_costs["input"] + \
               (output_tokens / 1000) * default_costs["output"]

    def _get_provider(self, model: str) -> str:
        """Determine provider from model name."""
        model_lower = model.lower()

        if "gpt" in model_lower or "text-embedding" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "ollama"
        elif "gemini" in model_lower:
            return "google"
        else:
            return "unknown"

    async def _check_alerts(self, user_id: str):
        """Check and trigger alerts if thresholds exceeded."""
        for alert in self._alerts.values():
            if alert.user_id and alert.user_id != user_id:
                continue

            if not alert.enabled or alert.notification_sent:
                continue

            # Calculate cost for period
            cost = await self.get_user_cost(
                user_id=alert.user_id or user_id,
                period=alert.period,
            )

            if cost >= alert.threshold:
                alert.notification_sent = True
                for callback in self._alert_callbacks:
                    try:
                        await callback(alert, cost)
                    except Exception as e:
                        logger.error("Alert callback failed", error=str(e))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def record_usage(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        usage_type: UsageType = UsageType.CHAT,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a usage event."""
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        provider = self._get_provider(model)

        record = UsageRecord(
            id=str(uuid4()),
            user_id=user_id,
            model=model,
            provider=provider,
            usage_type=usage_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            request_id=request_id,
            metadata=metadata or {},
        )

        self._records.append(record)

        logger.debug(
            "Recorded usage",
            user_id=user_id,
            model=model,
            tokens=record.total_tokens,
            cost=f"${cost:.6f}",
        )

        # Check alerts
        await self._check_alerts(user_id)

        return record

    async def get_user_cost(
        self,
        user_id: str,
        period: CostPeriod = CostPeriod.MONTH,
    ) -> float:
        """Get total cost for a user in a period."""
        now = datetime.utcnow()

        if period == CostPeriod.HOUR:
            start = now - timedelta(hours=1)
        elif period == CostPeriod.DAY:
            start = now - timedelta(days=1)
        elif period == CostPeriod.WEEK:
            start = now - timedelta(weeks=1)
        elif period == CostPeriod.MONTH:
            start = now - timedelta(days=30)
        else:
            start = datetime.min

        total = sum(
            r.cost for r in self._records
            if r.user_id == user_id and r.timestamp >= start
        )

        return total

    async def get_user_summary(
        self,
        user_id: str,
        period: CostPeriod = CostPeriod.MONTH,
    ) -> UserCostSummary:
        """Get detailed cost summary for a user."""
        now = datetime.utcnow()

        if period == CostPeriod.HOUR:
            start = now - timedelta(hours=1)
        elif period == CostPeriod.DAY:
            start = now - timedelta(days=1)
        elif period == CostPeriod.WEEK:
            start = now - timedelta(weeks=1)
        elif period == CostPeriod.MONTH:
            start = now - timedelta(days=30)
        else:
            start = datetime.min

        # Filter records
        records = [
            r for r in self._records
            if r.user_id == user_id and r.timestamp >= start
        ]

        # Aggregate by model
        by_model: Dict[str, float] = {}
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + r.cost

        # Aggregate by usage type
        by_usage: Dict[str, float] = {}
        for r in records:
            key = r.usage_type.value
            by_usage[key] = by_usage.get(key, 0) + r.cost

        # Daily costs
        daily: Dict[str, float] = {}
        for r in records:
            day = r.timestamp.strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + r.cost

        daily_costs = [
            {"date": d, "cost": c}
            for d, c in sorted(daily.items())
        ]

        return UserCostSummary(
            user_id=user_id,
            period=period,
            total_cost=sum(r.cost for r in records),
            total_requests=len(records),
            total_tokens=sum(r.total_tokens for r in records),
            by_model=by_model,
            by_usage_type=by_usage,
            daily_costs=daily_costs,
        )

    async def get_system_summary(
        self,
        period: CostPeriod = CostPeriod.MONTH,
    ) -> CostSummary:
        """Get system-wide cost summary."""
        now = datetime.utcnow()

        if period == CostPeriod.HOUR:
            start = now - timedelta(hours=1)
        elif period == CostPeriod.DAY:
            start = now - timedelta(days=1)
        elif period == CostPeriod.WEEK:
            start = now - timedelta(weeks=1)
        elif period == CostPeriod.MONTH:
            start = now - timedelta(days=30)
        else:
            start = datetime.min

        # Filter records
        records = [r for r in self._records if r.timestamp >= start]

        # Aggregate by model
        by_model: Dict[str, float] = {}
        for r in records:
            by_model[r.model] = by_model.get(r.model, 0) + r.cost

        # Aggregate by provider
        by_provider: Dict[str, float] = {}
        for r in records:
            by_provider[r.provider] = by_provider.get(r.provider, 0) + r.cost

        # Aggregate by usage type
        by_usage: Dict[str, float] = {}
        for r in records:
            key = r.usage_type.value
            by_usage[key] = by_usage.get(key, 0) + r.cost

        # Aggregate by user
        by_user: Dict[str, float] = {}
        for r in records:
            by_user[r.user_id] = by_user.get(r.user_id, 0) + r.cost

        return CostSummary(
            period=period,
            start_date=start,
            end_date=now,
            total_cost=sum(r.cost for r in records),
            total_requests=len(records),
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            by_model=by_model,
            by_provider=by_provider,
            by_usage_type=by_usage,
            by_user=by_user,
        )

    async def get_recent_usage(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[UsageRecord]:
        """Get recent usage records."""
        records = self._records

        if user_id:
            records = [r for r in records if r.user_id == user_id]

        return sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]

    def create_alert(
        self,
        threshold: float,
        period: CostPeriod = CostPeriod.MONTH,
        user_id: Optional[str] = None,
    ) -> CostAlert:
        """Create a cost alert."""
        alert = CostAlert(
            id=str(uuid4()),
            user_id=user_id,
            threshold=threshold,
            period=period,
        )

        self._alerts[alert.id] = alert

        logger.info(
            "Created cost alert",
            alert_id=alert.id,
            threshold=threshold,
            period=period.value,
            user_id=user_id,
        )

        return alert

    def delete_alert(self, alert_id: str) -> bool:
        """Delete a cost alert."""
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            return True
        return False

    def list_alerts(
        self,
        user_id: Optional[str] = None,
    ) -> List[CostAlert]:
        """List cost alerts."""
        alerts = list(self._alerts.values())

        if user_id:
            alerts = [a for a in alerts if a.user_id is None or a.user_id == user_id]

        return alerts

    def register_alert_callback(self, callback: callable):
        """Register a callback for when alerts are triggered."""
        self._alert_callbacks.append(callback)

    def reset_alert(self, alert_id: str):
        """Reset an alert's notification status."""
        if alert_id in self._alerts:
            self._alerts[alert_id].notification_sent = False

    async def estimate_request_cost(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Estimate cost for a potential request."""
        # Rough token estimation (1 token ≈ 4 chars)
        input_tokens = len(prompt) // 4

        cost = self._estimate_cost(model, input_tokens, max_output_tokens)

        return {
            "model": model,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": max_output_tokens,
            "estimated_cost": round(cost, 6),
            "currency": "USD",
        }

    def get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get current model pricing."""
        return MODEL_COSTS.copy()


# =============================================================================
# Module-level singleton and helpers
# =============================================================================

_cost_service: Optional[CostTrackingService] = None


def get_cost_service() -> CostTrackingService:
    """Get the cost tracking service singleton."""
    global _cost_service
    if _cost_service is None:
        _cost_service = CostTrackingService()
    return _cost_service


# Convenience function for recording usage
async def track_usage(
    user_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    usage_type: UsageType = UsageType.CHAT,
    **kwargs,
) -> UsageRecord:
    """Track LLM usage."""
    service = get_cost_service()
    return await service.record_usage(
        user_id=user_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_type=usage_type,
        **kwargs,
    )
