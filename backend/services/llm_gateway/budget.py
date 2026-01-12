"""
AIDocumentIndexer - Budget Management
======================================

Manage spending budgets for LLM usage with soft and hard limits.

Features:
- Organization and user-level budgets
- Daily, weekly, monthly periods
- Soft limits (warnings) and hard limits (blocking)
- Automatic reset on period boundaries
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.base import BaseService, CRUDService, ServiceException, NotFoundException

logger = structlog.get_logger(__name__)


class BudgetPeriod(str, Enum):
    """Budget period types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class BudgetStatus(str, Enum):
    """Budget status."""
    OK = "ok"
    WARNING = "warning"  # Soft limit reached
    EXCEEDED = "exceeded"  # Hard limit reached
    DISABLED = "disabled"


@dataclass
class Budget:
    """Budget definition."""
    id: str
    organization_id: str
    user_id: Optional[str]  # None = organization-wide
    name: str
    period: BudgetPeriod
    limit_amount: float  # In USD
    soft_limit_percent: float = 80.0  # Warn at 80% by default
    spent_amount: float = 0.0
    reset_at: Optional[datetime] = None
    is_hard_limit: bool = True  # Block when exceeded
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        """Get remaining budget."""
        return max(0, self.limit_amount - self.spent_amount)

    @property
    def percent_used(self) -> float:
        """Get percentage of budget used."""
        if self.limit_amount <= 0:
            return 0
        return (self.spent_amount / self.limit_amount) * 100

    @property
    def status(self) -> BudgetStatus:
        """Get current budget status."""
        if not self.is_active:
            return BudgetStatus.DISABLED
        if self.spent_amount >= self.limit_amount:
            return BudgetStatus.EXCEEDED
        if self.percent_used >= self.soft_limit_percent:
            return BudgetStatus.WARNING
        return BudgetStatus.OK


class BudgetManager(BaseService):
    """
    Manages spending budgets for LLM usage.

    Provides:
    - Budget creation and management
    - Spend tracking and enforcement
    - Automatic period resets
    - Alerts and notifications
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__(session, organization_id, user_id)
        # In-memory budget cache for fast lookups
        self._budget_cache: Dict[str, Budget] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._cache_loaded_at: Optional[datetime] = None

    async def create_budget(
        self,
        name: str,
        period: BudgetPeriod,
        limit_amount: float,
        user_id: Optional[str] = None,
        soft_limit_percent: float = 80.0,
        is_hard_limit: bool = True,
    ) -> Budget:
        """
        Create a new budget.

        Args:
            name: Budget name (e.g., "Monthly API Budget")
            period: Budget period (daily, weekly, monthly)
            limit_amount: Spending limit in USD
            user_id: User ID for user-specific budget (None for org-wide)
            soft_limit_percent: Percentage at which to warn
            is_hard_limit: Whether to block requests when exceeded

        Returns:
            Created Budget
        """
        session = await self.get_session()

        budget_id = str(uuid.uuid4())
        reset_at = self._calculate_next_reset(period)

        budget = Budget(
            id=budget_id,
            organization_id=str(self._organization_id),
            user_id=user_id,
            name=name,
            period=period,
            limit_amount=limit_amount,
            soft_limit_percent=soft_limit_percent,
            spent_amount=0.0,
            reset_at=reset_at,
            is_hard_limit=is_hard_limit,
            is_active=True,
        )

        # Store in database
        from backend.db.models import Budget

        db_budget = Budget(
            id=uuid.UUID(budget_id),
            organization_id=self._organization_id,
            user_id=uuid.UUID(user_id) if user_id else None,
            name=name,
            period=period.value,
            limit_amount=limit_amount,
            soft_limit_percent=soft_limit_percent,
            spent_amount=0.0,
            reset_at=reset_at,
            is_hard_limit=is_hard_limit,
            is_active=True,
        )

        session.add(db_budget)
        await session.commit()

        # Update cache
        self._budget_cache[budget_id] = budget

        self.log_info(
            "Budget created",
            budget_id=budget_id,
            name=name,
            period=period.value,
            limit=limit_amount,
        )

        return budget

    async def get_budget(self, budget_id: str) -> Optional[Budget]:
        """Get a budget by ID."""
        # Check cache first
        if budget_id in self._budget_cache:
            return self._budget_cache[budget_id]

        session = await self.get_session()
        from backend.db.models import Budget

        result = await session.execute(
            select(Budget).where(
                Budget.id == uuid.UUID(budget_id),
                Budget.organization_id == self._organization_id,
            )
        )
        db_budget = result.scalar_one_or_none()

        if not db_budget:
            return None

        budget = self._db_to_budget(db_budget)
        self._budget_cache[budget_id] = budget
        return budget

    async def get_active_budgets(
        self,
        user_id: Optional[str] = None,
    ) -> List[Budget]:
        """
        Get all active budgets for the organization.

        Args:
            user_id: Filter to specific user (also includes org-wide budgets)

        Returns:
            List of active budgets
        """
        session = await self.get_session()
        from backend.db.models import Budget

        query = select(Budget).where(
            Budget.organization_id == self._organization_id,
            Budget.is_active == True,
        )

        if user_id:
            # Get both user-specific and org-wide budgets
            query = query.where(
                (Budget.user_id == uuid.UUID(user_id)) |
                (Budget.user_id == None)
            )

        result = await session.execute(query)
        db_budgets = result.scalars().all()

        budgets = [self._db_to_budget(b) for b in db_budgets]

        # Update cache
        for budget in budgets:
            self._budget_cache[budget.id] = budget

        return budgets

    async def check_budget(
        self,
        estimated_cost: float,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check if a request can proceed within budget constraints.

        Args:
            estimated_cost: Estimated cost of the request in USD
            user_id: User making the request

        Returns:
            Dict with 'allowed', 'warnings', 'blocked_by' keys
        """
        budgets = await self.get_active_budgets(user_id)

        result = {
            "allowed": True,
            "warnings": [],
            "blocked_by": None,
            "budgets": [],
        }

        for budget in budgets:
            # Check if reset is needed
            if budget.reset_at and budget.reset_at <= datetime.utcnow():
                await self._reset_budget(budget)

            budget_info = {
                "id": budget.id,
                "name": budget.name,
                "status": budget.status.value,
                "percent_used": round(budget.percent_used, 1),
                "remaining": round(budget.remaining, 4),
            }
            result["budgets"].append(budget_info)

            # Check if would exceed
            if budget.spent_amount + estimated_cost > budget.limit_amount:
                if budget.is_hard_limit:
                    result["allowed"] = False
                    result["blocked_by"] = budget.name
                else:
                    result["warnings"].append(
                        f"Budget '{budget.name}' would be exceeded"
                    )
            elif budget.status == BudgetStatus.WARNING:
                result["warnings"].append(
                    f"Budget '{budget.name}' is at {budget.percent_used:.1f}%"
                )

        return result

    async def record_spend(
        self,
        amount: float,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Budget]:
        """
        Record spending against applicable budgets.

        Args:
            amount: Amount spent in USD
            user_id: User who made the request
            model: LLM model used
            metadata: Additional metadata

        Returns:
            List of updated budgets
        """
        budgets = await self.get_active_budgets(user_id)
        session = await self.get_session()
        from backend.db.models import Budget

        updated_budgets = []

        for budget in budgets:
            # Check if reset is needed
            if budget.reset_at and budget.reset_at <= datetime.utcnow():
                await self._reset_budget(budget)

            # Update spend
            budget.spent_amount += amount

            # Update in database
            result = await session.execute(
                select(Budget).where(Budget.id == uuid.UUID(budget.id))
            )
            db_budget = result.scalar_one_or_none()
            if db_budget:
                db_budget.spent_amount = budget.spent_amount
                updated_budgets.append(budget)

            # Update cache
            self._budget_cache[budget.id] = budget

            # Check for alerts
            if budget.status == BudgetStatus.WARNING:
                self.log_warning(
                    "Budget warning threshold reached",
                    budget_id=budget.id,
                    budget_name=budget.name,
                    percent_used=budget.percent_used,
                )
            elif budget.status == BudgetStatus.EXCEEDED:
                self.log_warning(
                    "Budget exceeded",
                    budget_id=budget.id,
                    budget_name=budget.name,
                    spent=budget.spent_amount,
                    limit=budget.limit_amount,
                )

        await session.commit()
        return updated_budgets

    async def update_budget(
        self,
        budget_id: str,
        limit_amount: Optional[float] = None,
        soft_limit_percent: Optional[float] = None,
        is_hard_limit: Optional[bool] = None,
        is_active: Optional[bool] = None,
    ) -> Budget:
        """Update budget settings."""
        session = await self.get_session()
        from backend.db.models import Budget

        result = await session.execute(
            select(Budget).where(
                Budget.id == uuid.UUID(budget_id),
                Budget.organization_id == self._organization_id,
            )
        )
        db_budget = result.scalar_one_or_none()

        if not db_budget:
            raise NotFoundException("Budget", budget_id)

        if limit_amount is not None:
            db_budget.limit_amount = limit_amount
        if soft_limit_percent is not None:
            db_budget.soft_limit_percent = soft_limit_percent
        if is_hard_limit is not None:
            db_budget.is_hard_limit = is_hard_limit
        if is_active is not None:
            db_budget.is_active = is_active

        db_budget.updated_at = datetime.utcnow()
        await session.commit()

        budget = self._db_to_budget(db_budget)
        self._budget_cache[budget_id] = budget
        return budget

    async def delete_budget(self, budget_id: str):
        """Delete a budget."""
        session = await self.get_session()
        from backend.db.models import Budget

        result = await session.execute(
            select(Budget).where(
                Budget.id == uuid.UUID(budget_id),
                Budget.organization_id == self._organization_id,
            )
        )
        db_budget = result.scalar_one_or_none()

        if not db_budget:
            raise NotFoundException("Budget", budget_id)

        await session.delete(db_budget)
        await session.commit()

        # Remove from cache
        self._budget_cache.pop(budget_id, None)

        self.log_info("Budget deleted", budget_id=budget_id)

    async def _reset_budget(self, budget: Budget):
        """Reset a budget for a new period."""
        session = await self.get_session()
        from backend.db.models import Budget

        result = await session.execute(
            select(Budget).where(Budget.id == uuid.UUID(budget.id))
        )
        db_budget = result.scalar_one_or_none()

        if db_budget:
            # Archive old spend (could store history here)
            old_spent = db_budget.spent_amount

            # Reset
            db_budget.spent_amount = 0.0
            db_budget.reset_at = self._calculate_next_reset(BudgetPeriod(db_budget.period))
            await session.commit()

            # Update in-memory
            budget.spent_amount = 0.0
            budget.reset_at = db_budget.reset_at

            self.log_info(
                "Budget reset",
                budget_id=budget.id,
                budget_name=budget.name,
                previous_spent=old_spent,
                new_reset_at=budget.reset_at.isoformat(),
            )

    def _calculate_next_reset(self, period: BudgetPeriod) -> datetime:
        """Calculate the next reset time for a budget period."""
        now = datetime.utcnow()

        if period == BudgetPeriod.DAILY:
            # Reset at midnight UTC
            return datetime(now.year, now.month, now.day) + timedelta(days=1)

        elif period == BudgetPeriod.WEEKLY:
            # Reset on Monday at midnight
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            return datetime(now.year, now.month, now.day) + timedelta(days=days_until_monday)

        elif period == BudgetPeriod.MONTHLY:
            # Reset on 1st of next month
            if now.month == 12:
                return datetime(now.year + 1, 1, 1)
            return datetime(now.year, now.month + 1, 1)

        elif period == BudgetPeriod.QUARTERLY:
            # Reset on 1st of next quarter
            quarter_month = ((now.month - 1) // 3 + 1) * 3 + 1
            if quarter_month > 12:
                return datetime(now.year + 1, quarter_month - 12, 1)
            return datetime(now.year, quarter_month, 1)

        elif period == BudgetPeriod.YEARLY:
            # Reset on Jan 1
            return datetime(now.year + 1, 1, 1)

        return now + timedelta(days=30)  # Default fallback

    def _db_to_budget(self, db_budget) -> Budget:
        """Convert database model to Budget dataclass."""
        return Budget(
            id=str(db_budget.id),
            organization_id=str(db_budget.organization_id),
            user_id=str(db_budget.user_id) if db_budget.user_id else None,
            name=db_budget.name,
            period=BudgetPeriod(db_budget.period),
            limit_amount=float(db_budget.limit_amount),
            soft_limit_percent=float(db_budget.soft_limit_percent),
            spent_amount=float(db_budget.spent_amount),
            reset_at=db_budget.reset_at,
            is_hard_limit=db_budget.is_hard_limit,
            is_active=db_budget.is_active,
        )

    async def get_budget_summary(
        self,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a summary of all budgets."""
        budgets = await self.get_active_budgets(user_id)

        total_limit = sum(b.limit_amount for b in budgets)
        total_spent = sum(b.spent_amount for b in budgets)
        total_remaining = sum(b.remaining for b in budgets)

        # Helper to safely get enum value (handles both enum and string)
        def safe_enum_value(val, default=""):
            if val is None:
                return default
            return val.value if hasattr(val, 'value') else val

        return {
            "total_budgets": len(budgets),
            "total_limit_usd": round(total_limit, 2),
            "total_spent_usd": round(total_spent, 4),
            "total_remaining_usd": round(total_remaining, 4),
            "overall_percent_used": round((total_spent / total_limit * 100) if total_limit > 0 else 0, 1),
            "budgets": [
                {
                    "id": b.id,
                    "name": b.name,
                    "period": safe_enum_value(b.period),
                    "limit": b.limit_amount,
                    "spent": round(b.spent_amount, 4),
                    "remaining": round(b.remaining, 4),
                    "percent_used": round(b.percent_used, 1),
                    "status": safe_enum_value(b.status),
                    "reset_at": b.reset_at.isoformat() if b.reset_at else None,
                    "is_hard_limit": b.is_hard_limit,
                }
                for b in budgets
            ],
            "warnings": [
                b.name for b in budgets if safe_enum_value(b.status) == "warning"
            ],
            "exceeded": [
                b.name for b in budgets if safe_enum_value(b.status) == "exceeded"
            ],
        }
