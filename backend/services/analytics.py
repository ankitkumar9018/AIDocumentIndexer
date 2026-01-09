"""
AIDocumentIndexer - Analytics Dashboard Service
================================================

Comprehensive analytics for system monitoring and insights:
1. Query metrics - Performance, latency, cache hits
2. Document metrics - Index size, freshness, types
3. User metrics - Activity, engagement, feedback
4. System health - Errors, uptime, resource usage

Provides data for dashboards and operational monitoring.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class MetricPeriod(str, Enum):
    """Time periods for aggregation."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class QueryLogEntry:
    """Log entry for a query."""
    query_id: str
    query: str
    user_id: str
    latency_ms: float
    source_count: int
    intent: str
    complexity: str
    strategy: str
    cache_hit: bool
    success: bool
    error: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query[:100],  # Truncate for storage
            "user_id": self.user_id,
            "latency_ms": self.latency_ms,
            "source_count": self.source_count,
            "intent": self.intent,
            "complexity": self.complexity,
            "strategy": self.strategy,
            "cache_hit": self.cache_hit,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class QueryMetrics:
    """Aggregated query performance metrics."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_sources_per_query: float
    cache_hit_rate: float
    by_intent: Dict[str, int]
    by_complexity: Dict[str, int]
    by_strategy: Dict[str, int]
    queries_per_hour: Dict[str, int]  # ISO hour -> count


@dataclass
class DocumentMetrics:
    """Document and indexing metrics."""
    total_documents: int
    total_chunks: int
    total_embeddings: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    avg_chunk_size: float
    avg_chunks_per_doc: float
    index_last_updated: Optional[datetime]
    documents_added_today: int
    documents_added_week: int


@dataclass
class UserMetrics:
    """User activity metrics."""
    total_users: int
    active_users_today: int
    active_users_week: int
    active_users_month: int
    avg_queries_per_user: float
    avg_feedback_rating: Optional[float]
    feedback_count: int
    top_users: List[Dict[str, Any]]  # [{user_id, query_count}]


@dataclass
class SystemHealth:
    """System health metrics."""
    uptime_seconds: float
    error_rate_percent: float
    errors_last_hour: int
    cache_status: str  # "healthy", "degraded", "unavailable"
    database_status: str
    vector_store_status: str
    embedding_service_status: str
    memory_usage_mb: Optional[float]
    recent_errors: List[Dict[str, Any]]


@dataclass
class DashboardData:
    """Complete dashboard data."""
    query_metrics: QueryMetrics
    document_metrics: DocumentMetrics
    user_metrics: UserMetrics
    system_health: SystemHealth
    generated_at: datetime
    period: MetricPeriod


class AnalyticsService:
    """
    Track and report system analytics.

    Collects:
    - Query logs with performance data
    - Document indexing metrics
    - User activity tracking
    - System health monitoring
    """

    def __init__(
        self,
        cache=None,
        max_log_entries: int = 10000,
        flush_interval: int = 100,
    ):
        """
        Initialize analytics service.

        Args:
            cache: RedisCache or similar for persistence
            max_log_entries: Max in-memory log entries
            flush_interval: Flush to storage every N entries
        """
        self.cache = cache
        self.max_log_entries = max_log_entries
        self.flush_interval = flush_interval

        # In-memory storage
        self._query_log: List[QueryLogEntry] = []
        self._error_log: List[Dict[str, Any]] = []
        self._user_activity: Dict[str, List[datetime]] = {}  # user_id -> timestamps
        self._start_time = datetime.utcnow()

        # Counters
        self._total_queries = 0
        self._successful_queries = 0
        self._cache_hits = 0

    async def log_query(
        self,
        query_id: str,
        query: str,
        user_id: str,
        latency_ms: float,
        source_count: int,
        intent: str = "unknown",
        complexity: str = "unknown",
        strategy: str = "unknown",
        cache_hit: bool = False,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a query for analytics.

        Args:
            query_id: Unique query identifier
            query: The query text
            user_id: User who made query
            latency_ms: Response time in ms
            source_count: Number of sources retrieved
            intent: Classified intent
            complexity: Complexity level
            strategy: Retrieval strategy used
            cache_hit: Whether cache was hit
            success: Whether query succeeded
            error: Error message if failed
        """
        entry = QueryLogEntry(
            query_id=query_id,
            query=query,
            user_id=user_id,
            latency_ms=latency_ms,
            source_count=source_count,
            intent=intent,
            complexity=complexity,
            strategy=strategy,
            cache_hit=cache_hit,
            success=success,
            error=error,
        )

        self._query_log.append(entry)
        self._total_queries += 1

        if success:
            self._successful_queries += 1
        if cache_hit:
            self._cache_hits += 1

        # Track user activity
        if user_id not in self._user_activity:
            self._user_activity[user_id] = []
        self._user_activity[user_id].append(entry.timestamp)

        # Log errors separately
        if not success and error:
            self._error_log.append({
                "query_id": query_id,
                "error": error,
                "timestamp": entry.timestamp.isoformat(),
            })

        # Trim if too many entries
        if len(self._query_log) > self.max_log_entries:
            self._query_log = self._query_log[-self.max_log_entries:]

        if len(self._error_log) > 1000:
            self._error_log = self._error_log[-1000:]

        # Periodic flush
        if len(self._query_log) % self.flush_interval == 0:
            await self._flush_to_storage()

        logger.debug(
            "Query logged",
            query_id=query_id,
            latency_ms=latency_ms,
            success=success,
        )

    async def get_query_metrics(
        self,
        period: MetricPeriod = MetricPeriod.DAY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> QueryMetrics:
        """
        Get query performance metrics.

        Args:
            period: Aggregation period
            start_date: Start of range (default: based on period)
            end_date: End of range (default: now)

        Returns:
            QueryMetrics
        """
        if not end_date:
            end_date = datetime.utcnow()

        if not start_date:
            if period == MetricPeriod.HOUR:
                start_date = end_date - timedelta(hours=1)
            elif period == MetricPeriod.DAY:
                start_date = end_date - timedelta(days=1)
            elif period == MetricPeriod.WEEK:
                start_date = end_date - timedelta(weeks=1)
            else:
                start_date = end_date - timedelta(days=30)

        # Filter entries in range
        entries = [
            e for e in self._query_log
            if start_date <= e.timestamp <= end_date
        ]

        if not entries:
            return QueryMetrics(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                avg_sources_per_query=0,
                cache_hit_rate=0,
                by_intent={},
                by_complexity={},
                by_strategy={},
                queries_per_hour={},
            )

        # Calculate metrics
        latencies = sorted([e.latency_ms for e in entries])
        successful = [e for e in entries if e.success]
        cache_hits = [e for e in entries if e.cache_hit]

        # Percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        # Group by dimensions
        by_intent: Dict[str, int] = defaultdict(int)
        by_complexity: Dict[str, int] = defaultdict(int)
        by_strategy: Dict[str, int] = defaultdict(int)
        queries_per_hour: Dict[str, int] = defaultdict(int)

        for e in entries:
            by_intent[e.intent] += 1
            by_complexity[e.complexity] += 1
            by_strategy[e.strategy] += 1
            hour_key = e.timestamp.strftime("%Y-%m-%dT%H:00:00")
            queries_per_hour[hour_key] += 1

        return QueryMetrics(
            total_queries=len(entries),
            successful_queries=len(successful),
            failed_queries=len(entries) - len(successful),
            avg_latency_ms=sum(latencies) / len(latencies),
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
            avg_sources_per_query=sum(e.source_count for e in entries) / len(entries),
            cache_hit_rate=len(cache_hits) / len(entries) if entries else 0,
            by_intent=dict(by_intent),
            by_complexity=dict(by_complexity),
            by_strategy=dict(by_strategy),
            queries_per_hour=dict(queries_per_hour),
        )

    async def get_document_metrics(self, session=None) -> DocumentMetrics:
        """
        Get document and indexing metrics.

        Args:
            session: Optional database session

        Returns:
            DocumentMetrics
        """
        # These would typically come from database queries
        # For now, return placeholder that can be filled by integration
        return DocumentMetrics(
            total_documents=0,
            total_chunks=0,
            total_embeddings=0,
            by_type={},
            by_status={},
            avg_chunk_size=0,
            avg_chunks_per_doc=0,
            index_last_updated=None,
            documents_added_today=0,
            documents_added_week=0,
        )

    async def get_user_metrics(
        self,
        period: MetricPeriod = MetricPeriod.WEEK,
    ) -> UserMetrics:
        """
        Get user activity metrics.

        Args:
            period: Time period for active user calculation

        Returns:
            UserMetrics
        """
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=7)
        month_start = today_start - timedelta(days=30)

        active_today = set()
        active_week = set()
        active_month = set()
        query_counts: Dict[str, int] = defaultdict(int)

        for user_id, timestamps in self._user_activity.items():
            for ts in timestamps:
                if ts >= today_start:
                    active_today.add(user_id)
                if ts >= week_start:
                    active_week.add(user_id)
                if ts >= month_start:
                    active_month.add(user_id)
                query_counts[user_id] += 1

        # Top users by query count
        sorted_users = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        top_users = [
            {"user_id": uid[:8] + "...", "query_count": count}
            for uid, count in sorted_users
        ]

        total_users = len(self._user_activity)
        total_queries = sum(query_counts.values())

        return UserMetrics(
            total_users=total_users,
            active_users_today=len(active_today),
            active_users_week=len(active_week),
            active_users_month=len(active_month),
            avg_queries_per_user=total_queries / total_users if total_users else 0,
            avg_feedback_rating=None,  # Would come from personalization service
            feedback_count=0,
            top_users=top_users,
        )

    async def get_system_health(self) -> SystemHealth:
        """
        Get system health metrics.

        Returns:
            SystemHealth
        """
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)

        # Count recent errors
        recent_errors = [
            e for e in self._error_log
            if datetime.fromisoformat(e["timestamp"]) >= hour_ago
        ]

        # Calculate error rate
        recent_queries = [
            e for e in self._query_log
            if e.timestamp >= hour_ago
        ]
        error_rate = (
            len([q for q in recent_queries if not q.success]) / len(recent_queries) * 100
            if recent_queries else 0
        )

        # Check service statuses (would be actual health checks in production)
        cache_status = "healthy" if self.cache else "unavailable"

        uptime = (now - self._start_time).total_seconds()

        return SystemHealth(
            uptime_seconds=uptime,
            error_rate_percent=error_rate,
            errors_last_hour=len(recent_errors),
            cache_status=cache_status,
            database_status="healthy",  # Would check actual connection
            vector_store_status="healthy",
            embedding_service_status="healthy",
            memory_usage_mb=None,  # Would get from psutil
            recent_errors=recent_errors[-10:],  # Last 10 errors
        )

    async def get_dashboard_data(
        self,
        period: MetricPeriod = MetricPeriod.DAY,
        session=None,
    ) -> DashboardData:
        """
        Get complete dashboard data.

        Args:
            period: Aggregation period
            session: Optional database session

        Returns:
            DashboardData with all metrics
        """
        query_metrics = await self.get_query_metrics(period)
        document_metrics = await self.get_document_metrics(session)
        user_metrics = await self.get_user_metrics(period)
        system_health = await self.get_system_health()

        return DashboardData(
            query_metrics=query_metrics,
            document_metrics=document_metrics,
            user_metrics=user_metrics,
            system_health=system_health,
            generated_at=datetime.utcnow(),
            period=period,
        )

    async def get_popular_queries(
        self,
        limit: int = 10,
        period: MetricPeriod = MetricPeriod.WEEK,
    ) -> List[Dict[str, Any]]:
        """
        Get most popular queries.

        Args:
            limit: Max results
            period: Time period

        Returns:
            List of popular queries with counts
        """
        now = datetime.utcnow()
        if period == MetricPeriod.HOUR:
            start = now - timedelta(hours=1)
        elif period == MetricPeriod.DAY:
            start = now - timedelta(days=1)
        elif period == MetricPeriod.WEEK:
            start = now - timedelta(weeks=1)
        else:
            start = now - timedelta(days=30)

        # Count query occurrences (normalized)
        query_counts: Dict[str, int] = defaultdict(int)
        for entry in self._query_log:
            if entry.timestamp >= start:
                # Normalize query (lowercase, strip)
                normalized = entry.query.lower().strip()
                query_counts[normalized] += 1

        # Sort by count
        sorted_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return [
            {"query": q, "count": c}
            for q, c in sorted_queries
        ]

    async def get_slow_queries(
        self,
        threshold_ms: float = 2000,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get queries exceeding latency threshold.

        Args:
            threshold_ms: Latency threshold
            limit: Max results

        Returns:
            List of slow queries
        """
        slow = [
            e for e in self._query_log
            if e.latency_ms > threshold_ms
        ]

        # Sort by latency descending
        slow.sort(key=lambda x: x.latency_ms, reverse=True)

        return [
            {
                "query": e.query[:100],
                "latency_ms": e.latency_ms,
                "timestamp": e.timestamp.isoformat(),
                "strategy": e.strategy,
                "source_count": e.source_count,
            }
            for e in slow[:limit]
        ]

    async def get_error_summary(
        self,
        period: MetricPeriod = MetricPeriod.DAY,
    ) -> Dict[str, Any]:
        """
        Get error summary for period.

        Args:
            period: Time period

        Returns:
            Error summary with groupings
        """
        now = datetime.utcnow()
        if period == MetricPeriod.HOUR:
            start = now - timedelta(hours=1)
        elif period == MetricPeriod.DAY:
            start = now - timedelta(days=1)
        elif period == MetricPeriod.WEEK:
            start = now - timedelta(weeks=1)
        else:
            start = now - timedelta(days=30)

        recent_errors = [
            e for e in self._error_log
            if datetime.fromisoformat(e["timestamp"]) >= start
        ]

        # Group by error type
        by_type: Dict[str, int] = defaultdict(int)
        for e in recent_errors:
            # Extract error type (first line or class)
            error_type = e["error"].split(":")[0] if ":" in e["error"] else "Unknown"
            by_type[error_type] += 1

        return {
            "total_errors": len(recent_errors),
            "by_type": dict(by_type),
            "recent": recent_errors[-5:],
            "period": period.value,
        }

    async def get_latency_histogram(
        self,
        period: MetricPeriod = MetricPeriod.DAY,
        bucket_count: int = 10,
    ) -> Dict[str, Any]:
        """
        Get latency distribution histogram.

        Args:
            period: Time period
            bucket_count: Number of histogram buckets

        Returns:
            Histogram data
        """
        now = datetime.utcnow()
        if period == MetricPeriod.HOUR:
            start = now - timedelta(hours=1)
        elif period == MetricPeriod.DAY:
            start = now - timedelta(days=1)
        else:
            start = now - timedelta(weeks=1)

        latencies = [
            e.latency_ms for e in self._query_log
            if e.timestamp >= start and e.success
        ]

        if not latencies:
            return {"buckets": [], "counts": []}

        min_lat = min(latencies)
        max_lat = max(latencies)
        bucket_size = (max_lat - min_lat) / bucket_count if max_lat > min_lat else 100

        buckets = []
        counts = []
        for i in range(bucket_count):
            bucket_start = min_lat + i * bucket_size
            bucket_end = bucket_start + bucket_size
            count = sum(1 for lat in latencies if bucket_start <= lat < bucket_end)
            buckets.append(f"{bucket_start:.0f}-{bucket_end:.0f}ms")
            counts.append(count)

        return {
            "buckets": buckets,
            "counts": counts,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "avg_ms": sum(latencies) / len(latencies),
        }

    async def _flush_to_storage(self) -> None:
        """Flush analytics data to persistent storage."""
        if not self.cache:
            return

        try:
            # Store recent query log
            recent_entries = [e.to_dict() for e in self._query_log[-1000:]]
            await self.cache.set(
                "analytics:query_log",
                json.dumps(recent_entries),
                ttl=86400 * 7,  # 7 days
            )

            # Store counters
            await self.cache.set(
                "analytics:counters",
                json.dumps({
                    "total_queries": self._total_queries,
                    "successful_queries": self._successful_queries,
                    "cache_hits": self._cache_hits,
                }),
                ttl=86400 * 30,
            )

            logger.debug("Analytics flushed to storage")
        except Exception as e:
            logger.warning("Failed to flush analytics", error=str(e))

    async def load_from_storage(self) -> None:
        """Load analytics data from persistent storage."""
        if not self.cache:
            return

        try:
            # Load query log
            data = await self.cache.get("analytics:query_log")
            if data:
                entries = json.loads(data) if isinstance(data, str) else data
                self._query_log = [
                    QueryLogEntry(
                        query_id=e["query_id"],
                        query=e["query"],
                        user_id=e["user_id"],
                        latency_ms=e["latency_ms"],
                        source_count=e["source_count"],
                        intent=e.get("intent", "unknown"),
                        complexity=e.get("complexity", "unknown"),
                        strategy=e.get("strategy", "unknown"),
                        cache_hit=e.get("cache_hit", False),
                        success=e.get("success", True),
                        error=e.get("error"),
                        timestamp=datetime.fromisoformat(e["timestamp"]),
                    )
                    for e in entries
                ]

            # Load counters
            counters = await self.cache.get("analytics:counters")
            if counters:
                parsed = json.loads(counters) if isinstance(counters, str) else counters
                self._total_queries = parsed.get("total_queries", 0)
                self._successful_queries = parsed.get("successful_queries", 0)
                self._cache_hits = parsed.get("cache_hits", 0)

            logger.info("Analytics loaded from storage", entries=len(self._query_log))
        except Exception as e:
            logger.warning("Failed to load analytics", error=str(e))


# =============================================================================
# Convenience Functions
# =============================================================================

_service_instance: Optional[AnalyticsService] = None


def get_analytics_service(cache=None) -> AnalyticsService:
    """
    Get or create the analytics service singleton.

    Args:
        cache: Optional cache for persistence

    Returns:
        AnalyticsService instance
    """
    global _service_instance

    if _service_instance is None:
        _service_instance = AnalyticsService(cache=cache)

    return _service_instance


async def log_query_analytics(
    query_id: str,
    query: str,
    user_id: str,
    latency_ms: float,
    source_count: int,
    **kwargs,
) -> None:
    """
    Convenience function to log query analytics.

    Args:
        query_id: Query ID
        query: Query text
        user_id: User ID
        latency_ms: Response latency
        source_count: Number of sources
        **kwargs: Additional fields (intent, complexity, strategy, etc.)
    """
    service = get_analytics_service()
    await service.log_query(
        query_id=query_id,
        query=query,
        user_id=user_id,
        latency_ms=latency_ms,
        source_count=source_count,
        **kwargs,
    )


async def get_dashboard() -> Dict[str, Any]:
    """
    Convenience function to get dashboard data as dict.

    Returns:
        Dashboard data dictionary
    """
    service = get_analytics_service()
    data = await service.get_dashboard_data()

    return {
        "query_metrics": {
            "total_queries": data.query_metrics.total_queries,
            "successful_queries": data.query_metrics.successful_queries,
            "failed_queries": data.query_metrics.failed_queries,
            "avg_latency_ms": data.query_metrics.avg_latency_ms,
            "p50_latency_ms": data.query_metrics.p50_latency_ms,
            "p95_latency_ms": data.query_metrics.p95_latency_ms,
            "cache_hit_rate": data.query_metrics.cache_hit_rate,
            "by_intent": data.query_metrics.by_intent,
            "by_complexity": data.query_metrics.by_complexity,
            "by_strategy": data.query_metrics.by_strategy,
        },
        "document_metrics": {
            "total_documents": data.document_metrics.total_documents,
            "total_chunks": data.document_metrics.total_chunks,
            "by_type": data.document_metrics.by_type,
        },
        "user_metrics": {
            "total_users": data.user_metrics.total_users,
            "active_users_today": data.user_metrics.active_users_today,
            "active_users_week": data.user_metrics.active_users_week,
            "avg_queries_per_user": data.user_metrics.avg_queries_per_user,
            "top_users": data.user_metrics.top_users,
        },
        "system_health": {
            "uptime_seconds": data.system_health.uptime_seconds,
            "error_rate_percent": data.system_health.error_rate_percent,
            "cache_status": data.system_health.cache_status,
            "database_status": data.system_health.database_status,
        },
        "generated_at": data.generated_at.isoformat(),
    }
