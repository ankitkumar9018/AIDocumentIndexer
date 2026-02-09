"""
AIDocumentIndexer - Usage Tracking
===================================

Track LLM usage for analytics and billing.

Features:
- Per-request logging
- Aggregated statistics
- Cost calculation
- Export capabilities
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import structlog
from sqlalchemy import select, func, and_, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.base import BaseService

logger = structlog.get_logger(__name__)


# Token pricing (per 1M tokens, as of 2024)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},

    # Anthropic
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},

    # Default for unknown models
    "default": {"input": 1.00, "output": 3.00},
}


@dataclass
class UsageRecord:
    """A single usage record."""
    id: str
    organization_id: str
    user_id: Optional[str]
    virtual_key_id: Optional[str]
    model: str
    provider: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime


class UsageTracker(BaseService):
    """
    Tracks and analyzes LLM usage.

    Provides:
    - Request-level logging
    - Aggregated statistics
    - Cost analytics
    - Usage trends
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__(session, organization_id, user_id)

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Normalize model name
        model_lower = model.lower()
        pricing = MODEL_PRICING.get(model_lower, MODEL_PRICING["default"])

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    async def record(
        self,
        model: str,
        provider: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        success: bool = True,
        error_message: Optional[str] = None,
        virtual_key_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """
        Record a usage event.

        Args:
            model: Model used
            provider: Provider (openai, anthropic, etc.)
            endpoint: API endpoint
            input_tokens: Input tokens
            output_tokens: Output tokens
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            error_message: Error message if failed
            virtual_key_id: Virtual key used (if any)
            metadata: Additional metadata

        Returns:
            Created UsageRecord
        """
        session = await self.get_session()

        record_id = str(uuid.uuid4())
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            id=record_id,
            organization_id=str(self._organization_id),
            user_id=str(self._user_id) if self._user_id else None,
            virtual_key_id=virtual_key_id,
            model=model,
            provider=provider,
            endpoint=endpoint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )

        # Store in database
        from backend.db.models import LLMUsageLog

        db_record = LLMUsageLog(
            id=uuid.UUID(record_id),
            user_id=self._user_id,
            model=model,
            provider_type=provider,
            operation_type=endpoint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            total_cost_usd=cost,
            request_duration_ms=latency_ms,
            success=success,
            error_message=error_message,
        )

        session.add(db_record)
        await session.commit()

        return record

    async def get_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        group_by: str = "day",  # day, week, month, model, user
    ) -> Dict[str, Any]:
        """
        Get aggregated usage statistics.

        Args:
            start_date: Start of period (default: 30 days ago)
            end_date: End of period (default: now)
            user_id: Filter by user
            model: Filter by model
            group_by: Aggregation grouping

        Returns:
            Usage statistics
        """
        session = await self.get_session()
        from backend.db.models import LLMUsageLog

        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

        # Base query
        query = select(
            func.count(LLMUsageLog.id).label("total_requests"),
            func.sum(LLMUsageLog.input_tokens).label("total_input_tokens"),
            func.sum(LLMUsageLog.output_tokens).label("total_output_tokens"),
            func.sum(LLMUsageLog.total_tokens).label("total_tokens"),
            func.sum(LLMUsageLog.total_cost_usd).label("total_cost"),
            func.avg(LLMUsageLog.request_duration_ms).label("avg_latency_ms"),
            func.sum(func.cast(LLMUsageLog.success, Integer)).label("successful_requests"),
        ).where(

            LLMUsageLog.created_at >= start_date,
            LLMUsageLog.created_at <= end_date,
        )

        if user_id:
            query = query.where(LLMUsageLog.user_id == uuid.UUID(user_id))

        if model:
            query = query.where(LLMUsageLog.model == model)

        result = await session.execute(query)
        row = result.one()

        total_requests = row.total_requests or 0
        successful = row.successful_requests or 0

        stats = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful,
                "failed_requests": total_requests - successful,
                "success_rate": round((successful / total_requests * 100) if total_requests > 0 else 0, 2),
                "total_input_tokens": row.total_input_tokens or 0,
                "total_output_tokens": row.total_output_tokens or 0,
                "total_tokens": row.total_tokens or 0,
                "total_cost_usd": round(float(row.total_cost or 0), 4),
                "avg_latency_ms": round(float(row.avg_latency_ms or 0), 2),
            },
        }

        # Get breakdown by model
        model_query = select(
            LLMUsageLog.model,
            func.count(LLMUsageLog.id).label("requests"),
            func.sum(LLMUsageLog.total_tokens).label("tokens"),
            func.sum(LLMUsageLog.total_cost_usd).label("cost"),
        ).where(

            LLMUsageLog.created_at >= start_date,
            LLMUsageLog.created_at <= end_date,
        ).group_by(LLMUsageLog.model)

        model_result = await session.execute(model_query)
        stats["by_model"] = [
            {
                "model": r.model,
                "requests": r.requests,
                "tokens": r.tokens or 0,
                "cost_usd": round(float(r.cost or 0), 4),
            }
            for r in model_result
        ]

        # Get breakdown by provider
        provider_query = select(
            LLMUsageLog.provider_type,
            func.count(LLMUsageLog.id).label("requests"),
            func.sum(LLMUsageLog.total_cost_usd).label("cost"),
        ).where(

            LLMUsageLog.created_at >= start_date,
            LLMUsageLog.created_at <= end_date,
        ).group_by(LLMUsageLog.provider_type)

        provider_result = await session.execute(provider_query)
        stats["by_provider"] = [
            {
                "provider": r.provider_type,
                "requests": r.requests,
                "cost_usd": round(float(r.cost or 0), 4),
            }
            for r in provider_result
        ]

        return stats

    async def get_daily_usage(
        self,
        days: int = 30,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get daily usage for the past N days.

        Args:
            days: Number of days to include
            user_id: Filter by user

        Returns:
            List of daily usage records
        """
        session = await self.get_session()
        from backend.db.models import LLMUsageLog

        start_date = datetime.utcnow() - timedelta(days=days)

        query = select(
            func.date(LLMUsageLog.created_at).label("date"),
            func.count(LLMUsageLog.id).label("requests"),
            func.sum(LLMUsageLog.total_tokens).label("tokens"),
            func.sum(LLMUsageLog.total_cost_usd).label("cost"),
        ).where(

            LLMUsageLog.created_at >= start_date,
        ).group_by(
            func.date(LLMUsageLog.created_at)
        ).order_by(
            func.date(LLMUsageLog.created_at)
        )

        if user_id:
            query = query.where(LLMUsageLog.user_id == uuid.UUID(user_id))

        result = await session.execute(query)

        return [
            {
                "date": str(r.date),
                "requests": r.requests,
                "tokens": r.tokens or 0,
                "cost_usd": round(float(r.cost or 0), 4),
            }
            for r in result
        ]

    async def get_top_users(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top users by usage.

        Args:
            days: Period in days
            limit: Number of users to return

        Returns:
            List of top users with usage stats
        """
        session = await self.get_session()
        from backend.db.models import LLMUsageLog, User

        start_date = datetime.utcnow() - timedelta(days=days)

        query = select(
            LLMUsageLog.user_id,
            func.count(LLMUsageLog.id).label("requests"),
            func.sum(LLMUsageLog.total_tokens).label("tokens"),
            func.sum(LLMUsageLog.total_cost_usd).label("cost"),
        ).where(

            LLMUsageLog.created_at >= start_date,
            LLMUsageLog.user_id != None,
        ).group_by(
            LLMUsageLog.user_id
        ).order_by(
            func.sum(LLMUsageLog.total_cost_usd).desc()
        ).limit(limit)

        result = await session.execute(query)

        users = []
        for r in result:
            # Get user info
            user_result = await session.execute(
                select(User).where(User.id == r.user_id)
            )
            user = user_result.scalar_one_or_none()

            users.append({
                "user_id": str(r.user_id),
                "email": user.email if user else "Unknown",
                "name": user.name if user else "Unknown",
                "requests": r.requests,
                "tokens": r.tokens or 0,
                "cost_usd": round(float(r.cost or 0), 4),
            })

        return users

    async def get_recent_requests(
        self,
        limit: int = 50,
        user_id: Optional[str] = None,
        success_only: bool = False,
    ) -> List[UsageRecord]:
        """
        Get recent requests.

        Args:
            limit: Number of records to return
            user_id: Filter by user
            success_only: Only return successful requests

        Returns:
            List of recent usage records
        """
        session = await self.get_session()
        from backend.db.models import LLMUsageLog

        query = select(LLMUsageLog)

        if user_id:
            query = query.where(LLMUsageLog.user_id == uuid.UUID(user_id))

        if success_only:
            query = query.where(LLMUsageLog.success == True)

        query = query.order_by(LLMUsageLog.created_at.desc()).limit(limit)

        result = await session.execute(query)
        db_records = result.scalars().all()

        return [
            UsageRecord(
                id=str(r.id),
                organization_id=str(self._organization_id) if self._organization_id else "",
                user_id=str(r.user_id) if r.user_id else None,
                virtual_key_id=None,
                model=r.model,
                provider=r.provider_type,
                endpoint=r.operation_type,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                total_tokens=r.total_tokens,
                cost_usd=float(r.total_cost_usd or 0),
                latency_ms=r.request_duration_ms or 0,
                success=r.success,
                error_message=r.error_message,
                metadata={},
                created_at=r.created_at,
            )
            for r in db_records
        ]

    async def export_usage(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
    ) -> Any:
        """
        Export usage data.

        Args:
            start_date: Start of period
            end_date: End of period
            format: Export format (json, csv)

        Returns:
            Exported data
        """
        session = await self.get_session()
        from backend.db.models import LLMUsageLog

        query = select(LLMUsageLog).where(

            LLMUsageLog.created_at >= start_date,
            LLMUsageLog.created_at <= end_date,
        ).order_by(LLMUsageLog.created_at)

        result = await session.execute(query)
        records = result.scalars().all()

        data = [
            {
                "id": str(r.id),
                "created_at": r.created_at.isoformat(),
                "model": r.model,
                "provider": r.provider_type,
                "endpoint": r.operation_type,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": float(r.total_cost_usd or 0),
                "latency_ms": r.request_duration_ms or 0,
                "success": r.success,
                "error_message": r.error_message,
                "user_id": str(r.user_id) if r.user_id else None,
            }
            for r in records
        ]

        if format == "csv":
            import csv
            import io

            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return output.getvalue()

        return data
