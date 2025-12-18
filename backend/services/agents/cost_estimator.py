"""
AIDocumentIndexer - Agent Cost Estimator
=========================================

Estimates execution costs before running agent plans.
Enforces budget limits and tracks actual spending.

Features:
- Pre-execution cost estimation per step
- Budget checking against user limits
- Model pricing integration
- Historical cost analysis
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import (
    AgentDefinition,
    AgentTrajectory,
    LLMProvider,
    UserCostLimit,
    CostAlert,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StepCostEstimate:
    """Cost estimate for a single plan step."""
    step_id: str
    step_name: str
    agent_type: str
    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "agent_type": self.agent_type,
            "model": self.model,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass
class CostEstimate:
    """Total cost estimate for an execution plan."""
    plan_id: str
    total_cost_usd: float
    steps: List[StepCostEstimate] = field(default_factory=list)
    currency: str = "USD"
    confidence: float = 0.8  # Estimation confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "total_cost_usd": self.total_cost_usd,
            "steps": [s.to_dict() for s in self.steps],
            "currency": self.currency,
            "confidence": self.confidence,
        }


@dataclass
class BudgetCheckResult:
    """Result of budget check."""
    allowed: bool
    remaining_budget: float
    estimated_cost: float = 0.0
    reason: Optional[str] = None
    budget_type: str = "daily"  # daily, monthly, per_request


# =============================================================================
# Model Pricing
# =============================================================================

# Default pricing per 1M tokens (in USD)
# These are approximations and should be updated with actual pricing
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

    # Ollama (local - free)
    "llama3.2": {"input": 0.0, "output": 0.0},
    "llama3.1": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "mixtral": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "nomic-embed-text": {"input": 0.0, "output": 0.0},

    # Default fallback
    "default": {"input": 1.00, "output": 3.00},
}

# Average tokens per task type (estimates)
TASK_TOKEN_ESTIMATES = {
    "generation": {"input": 500, "output": 1000},
    "evaluation": {"input": 800, "output": 500},
    "research": {"input": 300, "output": 800},
    "tool_execution": {"input": 400, "output": 300},
    "decomposition": {"input": 400, "output": 600},
    "synthesis": {"input": 1000, "output": 800},
    "default": {"input": 500, "output": 500},
}


# =============================================================================
# Cost Estimator
# =============================================================================

class AgentCostEstimator:
    """
    Estimates and tracks costs for agent execution.

    Integrates with:
    - Model pricing tables
    - User budget limits
    - Historical usage data
    """

    def __init__(self, db: Optional[AsyncSession] = None):
        """
        Initialize cost estimator.

        Args:
            db: Database session for budget lookups
        """
        self.db = db
        self._price_cache: Dict[str, Dict[str, float]] = {}

    def set_db(self, db: AsyncSession) -> None:
        """Set database session."""
        self.db = db

    def get_model_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name

        Returns:
            Dict with input/output prices per 1M tokens
        """
        # Check cache first
        if model in self._price_cache:
            return self._price_cache[model]

        # Look up in pricing table
        model_lower = model.lower()

        # Try exact match
        if model_lower in MODEL_PRICING:
            pricing = MODEL_PRICING[model_lower]
        else:
            # Try partial match
            for key, pricing in MODEL_PRICING.items():
                if key in model_lower or model_lower in key:
                    break
            else:
                # Default pricing
                pricing = MODEL_PRICING["default"]

        self._price_cache[model] = pricing
        return pricing

    def calculate_token_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for specific token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = self.get_model_pricing(model)

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def estimate_step_tokens(
        self,
        task_type: str,
        context_length: int = 0,
    ) -> tuple[int, int]:
        """
        Estimate tokens for a task type.

        Args:
            task_type: Type of task
            context_length: Additional context in characters

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        base = TASK_TOKEN_ESTIMATES.get(task_type, TASK_TOKEN_ESTIMATES["default"])

        # Add context overhead (roughly 4 chars per token)
        context_tokens = context_length // 4

        return (
            base["input"] + context_tokens,
            base["output"],
        )

    async def estimate_plan_cost(
        self,
        plan: Any,  # ExecutionPlan
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> CostEstimate:
        """
        Estimate total cost for an execution plan.

        Args:
            plan: ExecutionPlan to estimate
            agent_configs: Optional agent configurations with models

        Returns:
            CostEstimate with per-step breakdown
        """
        agent_configs = agent_configs or {}
        step_costs = []
        total_cost = 0.0

        for step in plan.steps:
            # Get model for this agent type
            agent_config = agent_configs.get(step.agent_type, {})
            model = agent_config.get("model", "gpt-4o")  # Default to gpt-4o

            # Estimate tokens
            task_type = step.task.type.value if hasattr(step.task.type, 'value') else str(step.task.type)
            context_len = len(step.task.description or "")
            input_tokens, output_tokens = self.estimate_step_tokens(task_type, context_len)

            # Calculate cost
            step_cost = self.calculate_token_cost(model, input_tokens, output_tokens)

            step_estimate = StepCostEstimate(
                step_id=step.id,
                step_name=step.task.name,
                agent_type=step.agent_type,
                model=model,
                estimated_input_tokens=input_tokens,
                estimated_output_tokens=output_tokens,
                estimated_cost_usd=step_cost,
            )
            step_costs.append(step_estimate)
            total_cost += step_cost

        return CostEstimate(
            plan_id=plan.id,
            total_cost_usd=total_cost,
            steps=step_costs,
        )

    async def check_budget(
        self,
        user_id: str,
        estimated_cost: float,
    ) -> BudgetCheckResult:
        """
        Check if user has budget for estimated cost.

        Args:
            user_id: User UUID
            estimated_cost: Estimated cost in USD

        Returns:
            BudgetCheckResult
        """
        if not self.db:
            # No DB - allow by default
            return BudgetCheckResult(
                allowed=True,
                remaining_budget=float('inf'),
                estimated_cost=estimated_cost,
            )

        try:
            # Get user's cost summary
            user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

            result = await self.db.execute(
                select(UserCostLimit)
                .where(UserCostLimit.user_id == user_uuid)
            )
            cost_limit = result.scalar_one_or_none()

            if not cost_limit:
                # No limits set - allow
                return BudgetCheckResult(
                    allowed=True,
                    remaining_budget=float('inf'),
                    estimated_cost=estimated_cost,
                )

            # Calculate remaining budget
            daily_limit = cost_limit.daily_limit_usd or float('inf')
            daily_usage = await self._get_daily_usage(user_uuid)
            remaining = daily_limit - daily_usage

            if estimated_cost > remaining:
                return BudgetCheckResult(
                    allowed=False,
                    remaining_budget=remaining,
                    estimated_cost=estimated_cost,
                    reason=f"Would exceed daily budget. Remaining: ${remaining:.4f}, Estimated: ${estimated_cost:.4f}",
                    budget_type="daily",
                )

            return BudgetCheckResult(
                allowed=True,
                remaining_budget=remaining,
                estimated_cost=estimated_cost,
            )

        except Exception as e:
            logger.error(f"Budget check error: {e}")
            # Allow on error to avoid blocking
            return BudgetCheckResult(
                allowed=True,
                remaining_budget=float('inf'),
                estimated_cost=estimated_cost,
                reason=f"Budget check error: {e}",
            )

    async def _get_daily_usage(self, user_id: uuid.UUID) -> float:
        """Get user's cost usage for today."""
        if not self.db:
            return 0.0

        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Sum costs from trajectories today
        result = await self.db.execute(
            select(func.sum(AgentTrajectory.total_cost_usd))
            .where(and_(
                AgentTrajectory.created_at >= today_start,
                # Need to join with execution plan to get user_id
                # For now, sum all costs (simplified)
            ))
        )

        total = result.scalar()
        return float(total) if total else 0.0

    async def get_user_cost_limit(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Get user's cost limits.

        Args:
            user_id: User UUID

        Returns:
            Dict with limit info
        """
        if not self.db:
            return {"daily_limit_usd": float('inf'), "monthly_limit_usd": float('inf')}

        try:
            user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

            result = await self.db.execute(
                select(UserCostLimit)
                .where(UserCostLimit.user_id == user_uuid)
            )
            cost_limit = result.scalar_one_or_none()

            if cost_limit:
                return {
                    "daily_limit_usd": cost_limit.daily_limit_usd or float('inf'),
                    "monthly_limit_usd": cost_limit.monthly_limit_usd or float('inf'),
                    "current_daily_usd": cost_limit.current_daily_usd or 0.0,
                    "current_monthly_usd": cost_limit.current_monthly_usd or 0.0,
                }

            return {"daily_limit_usd": float('inf'), "monthly_limit_usd": float('inf')}

        except Exception as e:
            logger.error(f"Error getting cost limit: {e}")
            return {"daily_limit_usd": float('inf'), "monthly_limit_usd": float('inf')}

    async def record_actual_cost(
        self,
        user_id: str,
        cost_usd: float,
        operation: str = "agent_execution",
    ) -> None:
        """
        Record actual cost after execution.

        Args:
            user_id: User UUID
            cost_usd: Actual cost in USD
            operation: Type of operation
        """
        if not self.db or cost_usd <= 0:
            return

        try:
            user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

            # Update user cost limit record
            result = await self.db.execute(
                select(UserCostLimit)
                .where(UserCostLimit.user_id == user_uuid)
            )
            cost_limit = result.scalar_one_or_none()

            if cost_limit:
                cost_limit.current_daily_usd = (cost_limit.current_daily_usd or 0) + cost_usd
                cost_limit.current_monthly_usd = (cost_limit.current_monthly_usd or 0) + cost_usd
                cost_limit.total_cost_usd = (cost_limit.total_cost_usd or 0) + cost_usd
            else:
                # Create new cost limit record
                cost_limit = UserCostLimit(
                    user_id=user_uuid,
                    current_daily_usd=cost_usd,
                    current_monthly_usd=cost_usd,
                    total_cost_usd=cost_usd,
                )
                self.db.add(cost_limit)

            await self.db.commit()

            logger.info(
                "Recorded cost",
                user_id=str(user_id),
                cost_usd=cost_usd,
                operation=operation,
            )

            # Check for cost alerts
            await self._check_cost_alerts(user_uuid, cost_limit)

        except Exception as e:
            logger.error(f"Error recording cost: {e}")

    async def _check_cost_alerts(
        self,
        user_id: uuid.UUID,
        cost_limit: UserCostLimit,
    ) -> None:
        """Check and trigger cost alerts if thresholds exceeded."""
        if not self.db:
            return

        try:
            # Get user's alerts
            result = await self.db.execute(
                select(CostAlert)
                .where(and_(
                    CostAlert.user_id == user_id,
                    CostAlert.is_active == True,
                ))
            )
            alerts = result.scalars().all()

            for alert in alerts:
                triggered = False

                if alert.alert_type == "daily_threshold":
                    if (cost_limit.current_daily_usd or 0) >= alert.threshold_usd:
                        triggered = True
                elif alert.alert_type == "monthly_threshold":
                    if (cost_limit.current_monthly_usd or 0) >= alert.threshold_usd:
                        triggered = True

                if triggered:
                    alert.last_triggered_at = datetime.utcnow()
                    alert.trigger_count = (alert.trigger_count or 0) + 1

                    logger.warning(
                        "Cost alert triggered",
                        user_id=str(user_id),
                        alert_type=alert.alert_type,
                        threshold=alert.threshold_usd,
                    )

            await self.db.commit()

        except Exception as e:
            logger.error(f"Error checking cost alerts: {e}")

    async def get_cost_breakdown(
        self,
        user_id: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get cost breakdown for user.

        Args:
            user_id: User UUID
            days: Number of days to analyze

        Returns:
            Cost breakdown by day and agent
        """
        if not self.db:
            return {"daily_costs": [], "agent_costs": {}}

        try:
            user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Get trajectories for cost analysis
            result = await self.db.execute(
                select(AgentTrajectory)
                .where(AgentTrajectory.created_at >= cutoff)
                .order_by(AgentTrajectory.created_at)
            )
            trajectories = result.scalars().all()

            # Aggregate by day
            daily_costs = {}
            agent_costs = {}

            for t in trajectories:
                day = t.created_at.date().isoformat()
                cost = t.total_cost_usd or 0

                daily_costs[day] = daily_costs.get(day, 0) + cost

                agent_id = str(t.agent_id) if t.agent_id else "unknown"
                agent_costs[agent_id] = agent_costs.get(agent_id, 0) + cost

            return {
                "daily_costs": [
                    {"date": d, "cost_usd": c}
                    for d, c in sorted(daily_costs.items())
                ],
                "agent_costs": agent_costs,
                "total_cost_usd": sum(daily_costs.values()),
            }

        except Exception as e:
            logger.error(f"Error getting cost breakdown: {e}")
            return {"daily_costs": [], "agent_costs": {}}
