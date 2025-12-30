"""
OCR Metrics Service
===================

Service for recording and analyzing OCR performance metrics.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import Integer, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import OCRMetrics

logger = structlog.get_logger(__name__)


class OCRMetricsService:
    """Service for OCR performance tracking and analytics."""

    def __init__(self, session: AsyncSession):
        """Initialize OCR metrics service.

        Args:
            session: Database session
        """
        self.session = session

    async def record_ocr_operation(
        self,
        provider: str,
        language: str,
        processing_time_ms: int,
        success: bool,
        variant: Optional[str] = None,
        document_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None,
        page_count: int = 1,
        character_count: Optional[int] = None,
        confidence_score: Optional[float] = None,
        error_message: Optional[str] = None,
        fallback_used: bool = False,
        cost_usd: Optional[float] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> OCRMetrics:
        """Record an OCR operation.

        Args:
            provider: OCR provider (paddleocr, tesseract, auto)
            language: Language code (en, de, fr, etc.)
            processing_time_ms: Processing time in milliseconds
            success: Whether OCR was successful
            variant: Model variant (server, mobile)
            document_id: Associated document ID
            user_id: User who triggered OCR
            page_count: Number of pages processed
            character_count: Characters extracted
            confidence_score: Average confidence (0-1)
            error_message: Error message if failed
            fallback_used: Whether fallback was used
            cost_usd: Cost in USD
            extra_data: Additional metadata

        Returns:
            Created OCRMetrics record
        """
        metric = OCRMetrics(
            id=uuid.uuid4(),
            provider=provider,
            variant=variant,
            language=language,
            document_id=document_id,
            user_id=user_id,
            processing_time_ms=processing_time_ms,
            page_count=page_count,
            character_count=character_count,
            confidence_score=confidence_score,
            success=success,
            error_message=error_message,
            fallback_used=fallback_used,
            cost_usd=cost_usd,
            extra_data=extra_data,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.session.add(metric)
        await self.session.commit()
        await self.session.refresh(metric)

        logger.info(
            "OCR operation recorded",
            provider=provider,
            language=language,
            time_ms=processing_time_ms,
            success=success,
        )

        return metric

    async def get_metrics_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated OCR metrics.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            provider: Filter by provider

        Returns:
            Aggregated metrics dictionary
        """
        # Build filter conditions
        conditions = []
        if start_date:
            conditions.append(OCRMetrics.created_at >= start_date)
        if end_date:
            conditions.append(OCRMetrics.created_at <= end_date)
        if provider:
            conditions.append(OCRMetrics.provider == provider)

        where_clause = and_(*conditions) if conditions else True

        # Total operations
        total_count = await self.session.scalar(
            select(func.count(OCRMetrics.id)).where(where_clause)
        )

        # Success count
        success_count = await self.session.scalar(
            select(func.count(OCRMetrics.id)).where(
                and_(where_clause, OCRMetrics.success == True)
            )
        )

        # Average processing time
        avg_time = await self.session.scalar(
            select(func.avg(OCRMetrics.processing_time_ms)).where(where_clause)
        )

        # Total characters processed
        total_chars = await self.session.scalar(
            select(func.sum(OCRMetrics.character_count)).where(where_clause)
        )

        # Total cost
        total_cost = await self.session.scalar(
            select(func.sum(OCRMetrics.cost_usd)).where(where_clause)
        )

        # Fallback usage
        fallback_count = await self.session.scalar(
            select(func.count(OCRMetrics.id)).where(
                and_(where_clause, OCRMetrics.fallback_used == True)
            )
        )

        return {
            "total_operations": total_count or 0,
            "successful_operations": success_count or 0,
            "success_rate": (success_count / total_count * 100) if total_count else 0,
            "average_processing_time_ms": round(avg_time, 2) if avg_time else 0,
            "total_characters_processed": total_chars or 0,
            "total_cost_usd": round(total_cost, 4) if total_cost else 0,
            "fallback_used_count": fallback_count or 0,
            "fallback_usage_rate": (fallback_count / total_count * 100) if total_count else 0,
        }

    async def get_metrics_by_provider(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics grouped by provider.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            List of metrics per provider
        """
        conditions = []
        if start_date:
            conditions.append(OCRMetrics.created_at >= start_date)
        if end_date:
            conditions.append(OCRMetrics.created_at <= end_date)

        where_clause = and_(*conditions) if conditions else True

        # Group by provider
        result = await self.session.execute(
            select(
                OCRMetrics.provider,
                func.count(OCRMetrics.id).label("count"),
                func.avg(OCRMetrics.processing_time_ms).label("avg_time"),
                func.sum(
                    func.cast(OCRMetrics.success, Integer)
                ).label("success_count"),
            )
            .where(where_clause)
            .group_by(OCRMetrics.provider)
        )

        provider_metrics = []
        for row in result:
            total = row.count
            success = row.success_count or 0
            provider_metrics.append({
                "provider": row.provider,
                "total_operations": total,
                "success_rate": (success / total * 100) if total else 0,
                "average_processing_time_ms": round(row.avg_time, 2) if row.avg_time else 0,
            })

        return provider_metrics

    async def get_metrics_by_language(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics grouped by language.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            List of metrics per language
        """
        conditions = []
        if start_date:
            conditions.append(OCRMetrics.created_at >= start_date)
        if end_date:
            conditions.append(OCRMetrics.created_at <= end_date)

        where_clause = and_(*conditions) if conditions else True

        # Group by language
        result = await self.session.execute(
            select(
                OCRMetrics.language,
                func.count(OCRMetrics.id).label("count"),
                func.avg(OCRMetrics.processing_time_ms).label("avg_time"),
            )
            .where(where_clause)
            .group_by(OCRMetrics.language)
            .order_by(func.count(OCRMetrics.id).desc())
        )

        language_metrics = []
        for row in result:
            language_metrics.append({
                "language": row.language,
                "total_operations": row.count,
                "average_processing_time_ms": round(row.avg_time, 2) if row.avg_time else 0,
            })

        return language_metrics

    async def get_recent_metrics(
        self,
        limit: int = 100,
        provider: Optional[str] = None,
    ) -> List[OCRMetrics]:
        """Get recent OCR operations.

        Args:
            limit: Number of records to retrieve
            provider: Filter by provider

        Returns:
            List of recent OCRMetrics
        """
        query = select(OCRMetrics).order_by(OCRMetrics.created_at.desc()).limit(limit)

        if provider:
            query = query.where(OCRMetrics.provider == provider)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_performance_trend(
        self,
        days: int = 7,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get performance trend over time.

        Args:
            days: Number of days to analyze
            provider: Filter by provider

        Returns:
            Daily performance metrics
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        conditions = [OCRMetrics.created_at >= start_date]
        if provider:
            conditions.append(OCRMetrics.provider == provider)

        where_clause = and_(*conditions)

        # Group by date
        result = await self.session.execute(
            select(
                func.date(OCRMetrics.created_at).label("date"),
                func.count(OCRMetrics.id).label("count"),
                func.avg(OCRMetrics.processing_time_ms).label("avg_time"),
                func.sum(
                    func.cast(OCRMetrics.success, Integer)
                ).label("success_count"),
            )
            .where(where_clause)
            .group_by(func.date(OCRMetrics.created_at))
            .order_by(func.date(OCRMetrics.created_at))
        )

        trend = []
        for row in result:
            total = row.count
            success = row.success_count or 0
            trend.append({
                "date": row.date.isoformat() if row.date else None,
                "total_operations": total,
                "success_rate": (success / total * 100) if total else 0,
                "average_processing_time_ms": round(row.avg_time, 2) if row.avg_time else 0,
            })

        return trend
