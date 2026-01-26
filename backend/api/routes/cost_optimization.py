"""
AIDocumentIndexer - Cost Optimization API Routes
=================================================

Phase 68: Cost monitoring dashboard for 30-50% cost reduction.

Features:
- Track cost per query
- Token usage analytics
- Cache hit rate monitoring
- Model usage mix analysis
- Budget alerts and recommendations
- GPU utilization metrics
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/costs", tags=["cost-optimization"])


# =============================================================================
# Models
# =============================================================================

class TimeRange(str, Enum):
    """Time range for cost analysis."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class CostCategory(str, Enum):
    """Categories of costs."""
    LLM_INFERENCE = "llm_inference"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    OCR = "ocr"
    TTS = "tts"
    STORAGE = "storage"
    COMPUTE = "compute"


class UsageMetric(BaseModel):
    """A usage metric."""
    name: str
    value: float
    unit: str
    cost_usd: float
    timestamp: datetime


class CostBreakdown(BaseModel):
    """Cost breakdown by category."""
    category: CostCategory
    total_cost_usd: float
    percentage: float
    request_count: int
    avg_cost_per_request: float


class ModelUsage(BaseModel):
    """Model-level usage statistics."""
    model_id: str
    provider: str
    request_count: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    avg_latency_ms: float


class CacheStats(BaseModel):
    """Cache performance statistics."""
    cache_type: str
    hit_count: int
    miss_count: int
    hit_rate: float
    savings_usd: float  # Estimated cost saved by cache hits


class BudgetAlert(BaseModel):
    """Budget alert configuration."""
    id: str
    name: str
    threshold_usd: float
    period: TimeRange
    current_spend_usd: float
    triggered: bool
    triggered_at: Optional[datetime] = None


class CostRecommendation(BaseModel):
    """Cost optimization recommendation."""
    id: str
    title: str
    description: str
    potential_savings_usd: float
    priority: str  # "high", "medium", "low"
    action: str
    implemented: bool = False


class CostAnalysisResponse(BaseModel):
    """Full cost analysis response."""
    time_range: TimeRange
    start_date: datetime
    end_date: datetime
    total_cost_usd: float
    breakdown: List[CostBreakdown]
    model_usage: List[ModelUsage]
    cache_stats: List[CacheStats]
    daily_costs: List[Dict[str, Any]]


class BudgetAlertRequest(BaseModel):
    """Request to create a budget alert."""
    name: str
    threshold_usd: float = Field(..., gt=0)
    period: TimeRange = TimeRange.MONTH
    notify_email: Optional[str] = None
    notify_slack: bool = False


# =============================================================================
# In-Memory Storage (replace with DB in production)
# =============================================================================

# Cost tracking data
_cost_records: List[Dict[str, Any]] = []
_model_usage: Dict[str, Dict[str, Any]] = {}
_cache_stats: Dict[str, Dict[str, Any]] = {}
_budget_alerts: List[BudgetAlert] = []


def _record_cost(
    category: CostCategory,
    cost_usd: float,
    tokens: int = 0,
    model_id: Optional[str] = None,
    latency_ms: float = 0,
) -> None:
    """Record a cost event."""
    _cost_records.append({
        "category": category,
        "cost_usd": cost_usd,
        "tokens": tokens,
        "model_id": model_id,
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow(),
    })

    # Update model usage
    if model_id:
        if model_id not in _model_usage:
            _model_usage[model_id] = {
                "request_count": 0,
                "total_tokens": 0,
                "total_cost_usd": 0,
                "total_latency_ms": 0,
            }
        _model_usage[model_id]["request_count"] += 1
        _model_usage[model_id]["total_tokens"] += tokens
        _model_usage[model_id]["total_cost_usd"] += cost_usd
        _model_usage[model_id]["total_latency_ms"] += latency_ms


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/analysis", response_model=CostAnalysisResponse)
async def get_cost_analysis(
    time_range: TimeRange = Query(TimeRange.WEEK, description="Time range for analysis"),
) -> CostAnalysisResponse:
    """
    Get comprehensive cost analysis.

    Returns breakdown by category, model usage, cache performance,
    and daily cost trends.
    """
    # Calculate date range
    end_date = datetime.utcnow()
    if time_range == TimeRange.HOUR:
        start_date = end_date - timedelta(hours=1)
    elif time_range == TimeRange.DAY:
        start_date = end_date - timedelta(days=1)
    elif time_range == TimeRange.WEEK:
        start_date = end_date - timedelta(weeks=1)
    else:
        start_date = end_date - timedelta(days=30)

    # Filter records by date range
    filtered_records = [
        r for r in _cost_records
        if r["timestamp"] >= start_date
    ]

    # Calculate category breakdown
    category_totals: Dict[CostCategory, Dict[str, float]] = {}
    for record in filtered_records:
        cat = record["category"]
        if cat not in category_totals:
            category_totals[cat] = {"cost": 0, "count": 0}
        category_totals[cat]["cost"] += record["cost_usd"]
        category_totals[cat]["count"] += 1

    total_cost = sum(t["cost"] for t in category_totals.values())

    breakdown = [
        CostBreakdown(
            category=cat,
            total_cost_usd=round(data["cost"], 4),
            percentage=round(data["cost"] / total_cost * 100, 1) if total_cost > 0 else 0,
            request_count=int(data["count"]),
            avg_cost_per_request=round(data["cost"] / data["count"], 6) if data["count"] > 0 else 0,
        )
        for cat, data in category_totals.items()
    ]

    # Model usage
    model_usage = [
        ModelUsage(
            model_id=model_id,
            provider=model_id.split("/")[0] if "/" in model_id else "unknown",
            request_count=data["request_count"],
            total_tokens=data["total_tokens"],
            input_tokens=int(data["total_tokens"] * 0.7),  # Estimate
            output_tokens=int(data["total_tokens"] * 0.3),
            total_cost_usd=round(data["total_cost_usd"], 4),
            avg_latency_ms=round(data["total_latency_ms"] / data["request_count"], 2)
            if data["request_count"] > 0 else 0,
        )
        for model_id, data in _model_usage.items()
    ]

    # Cache stats
    cache_stats = [
        CacheStats(
            cache_type=cache_type,
            hit_count=data.get("hits", 0),
            miss_count=data.get("misses", 0),
            hit_rate=round(
                data.get("hits", 0) / (data.get("hits", 0) + data.get("misses", 0)),
                3
            ) if (data.get("hits", 0) + data.get("misses", 0)) > 0 else 0,
            savings_usd=round(data.get("hits", 0) * 0.001, 4),  # Estimated $0.001 per cache hit
        )
        for cache_type, data in _cache_stats.items()
    ]

    # Daily costs
    daily_costs = _aggregate_daily_costs(filtered_records)

    return CostAnalysisResponse(
        time_range=time_range,
        start_date=start_date,
        end_date=end_date,
        total_cost_usd=round(total_cost, 4),
        breakdown=breakdown,
        model_usage=model_usage,
        cache_stats=cache_stats,
        daily_costs=daily_costs,
    )


def _aggregate_daily_costs(records: List[Dict]) -> List[Dict[str, Any]]:
    """Aggregate costs by day."""
    daily: Dict[str, float] = {}
    for record in records:
        day = record["timestamp"].strftime("%Y-%m-%d")
        daily[day] = daily.get(day, 0) + record["cost_usd"]

    return [
        {"date": day, "cost_usd": round(cost, 4)}
        for day, cost in sorted(daily.items())
    ]


@router.get("/recommendations", response_model=List[CostRecommendation])
async def get_recommendations() -> List[CostRecommendation]:
    """
    Get cost optimization recommendations.

    Analyzes usage patterns and suggests ways to reduce costs.
    """
    recommendations = []

    # Check cache hit rate
    total_hits = sum(d.get("hits", 0) for d in _cache_stats.values())
    total_misses = sum(d.get("misses", 0) for d in _cache_stats.values())
    if total_hits + total_misses > 100:
        hit_rate = total_hits / (total_hits + total_misses)
        if hit_rate < 0.3:
            recommendations.append(CostRecommendation(
                id="enable-semantic-cache",
                title="Enable Semantic Caching",
                description="Your cache hit rate is low. Enable semantic caching to match similar queries.",
                potential_savings_usd=total_misses * 0.0005,
                priority="high",
                action="Enable 'rag.semantic_cache_enabled' in settings",
            ))

    # Check model usage
    expensive_models = [
        (mid, data) for mid, data in _model_usage.items()
        if data["total_cost_usd"] > 10
    ]
    if expensive_models:
        for model_id, data in expensive_models:
            if "gpt-4" in model_id.lower() or "claude-3-opus" in model_id.lower():
                recommendations.append(CostRecommendation(
                    id=f"downgrade-{model_id[:20]}",
                    title=f"Consider cheaper alternative for {model_id}",
                    description="High-cost model detected. Consider using GPT-4o-mini or Claude Haiku for simpler queries.",
                    potential_savings_usd=data["total_cost_usd"] * 0.7,
                    priority="medium",
                    action="Configure model routing in LLM settings",
                ))

    # Check if binary quantization could help
    if len(_cost_records) > 1000:
        recommendations.append(CostRecommendation(
            id="enable-binary-quant",
            title="Enable Binary Quantization",
            description="With 1000+ documents, binary quantization can reduce storage and search costs by 32x.",
            potential_savings_usd=5.0,
            priority="medium",
            action="Enable 'rag.binary_quantization_enabled' in settings",
        ))

    # Check if prefetch is enabled
    recommendations.append(CostRecommendation(
        id="enable-prefetch",
        title="Enable Query Prefetching",
        description="Prefetch likely follow-up queries to improve perceived latency and reduce redundant work.",
        potential_savings_usd=2.0,
        priority="low",
        action="Enable prefetching in RAG cache settings",
    ))

    return recommendations


@router.get("/budget-alerts", response_model=List[BudgetAlert])
async def list_budget_alerts() -> List[BudgetAlert]:
    """List all budget alerts."""
    # Update current spend for each alert
    for alert in _budget_alerts:
        alert.current_spend_usd = _calculate_spend_for_period(alert.period)
        alert.triggered = alert.current_spend_usd >= alert.threshold_usd

    return _budget_alerts


@router.post("/budget-alerts", response_model=BudgetAlert, status_code=status.HTTP_201_CREATED)
async def create_budget_alert(request: BudgetAlertRequest) -> BudgetAlert:
    """Create a new budget alert."""
    import uuid

    alert = BudgetAlert(
        id=str(uuid.uuid4()),
        name=request.name,
        threshold_usd=request.threshold_usd,
        period=request.period,
        current_spend_usd=_calculate_spend_for_period(request.period),
        triggered=False,
    )

    alert.triggered = alert.current_spend_usd >= alert.threshold_usd
    _budget_alerts.append(alert)

    logger.info(
        "Budget alert created",
        alert_id=alert.id,
        threshold=alert.threshold_usd,
    )

    return alert


@router.delete("/budget-alerts/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_budget_alert(alert_id: str) -> None:
    """Delete a budget alert."""
    global _budget_alerts
    _budget_alerts = [a for a in _budget_alerts if a.id != alert_id]


def _calculate_spend_for_period(period: TimeRange) -> float:
    """Calculate total spend for a time period."""
    if period == TimeRange.HOUR:
        cutoff = datetime.utcnow() - timedelta(hours=1)
    elif period == TimeRange.DAY:
        cutoff = datetime.utcnow() - timedelta(days=1)
    elif period == TimeRange.WEEK:
        cutoff = datetime.utcnow() - timedelta(weeks=1)
    else:
        cutoff = datetime.utcnow() - timedelta(days=30)

    return sum(
        r["cost_usd"] for r in _cost_records
        if r["timestamp"] >= cutoff
    )


@router.get("/tokens")
async def get_token_usage(
    time_range: TimeRange = Query(TimeRange.DAY),
) -> Dict[str, Any]:
    """
    Get token usage breakdown.

    Shows input vs output tokens, tokens per query, and cost per 1K tokens.
    """
    end_date = datetime.utcnow()
    if time_range == TimeRange.HOUR:
        start_date = end_date - timedelta(hours=1)
    elif time_range == TimeRange.DAY:
        start_date = end_date - timedelta(days=1)
    elif time_range == TimeRange.WEEK:
        start_date = end_date - timedelta(weeks=1)
    else:
        start_date = end_date - timedelta(days=30)

    filtered_records = [
        r for r in _cost_records
        if r["timestamp"] >= start_date and r.get("tokens", 0) > 0
    ]

    total_tokens = sum(r.get("tokens", 0) for r in filtered_records)
    total_cost = sum(r.get("cost_usd", 0) for r in filtered_records)
    query_count = len(filtered_records)

    return {
        "time_range": time_range.value,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_tokens": total_tokens,
        "input_tokens": int(total_tokens * 0.7),  # Estimate
        "output_tokens": int(total_tokens * 0.3),  # Estimate
        "total_cost_usd": round(total_cost, 4),
        "query_count": query_count,
        "avg_tokens_per_query": round(total_tokens / query_count, 1) if query_count > 0 else 0,
        "cost_per_1k_tokens": round(total_cost / total_tokens * 1000, 6) if total_tokens > 0 else 0,
    }


@router.get("/gpu")
async def get_gpu_utilization() -> Dict[str, Any]:
    """
    Get GPU utilization metrics.

    Returns current GPU usage, memory, and inference stats.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                mem_allocated = torch.cuda.memory_allocated(i)
                mem_reserved = torch.cuda.memory_reserved(i)

                devices.append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 2),
                    "allocated_memory_gb": round(mem_allocated / 1e9, 2),
                    "reserved_memory_gb": round(mem_reserved / 1e9, 2),
                    "utilization_percent": round(mem_allocated / props.total_memory * 100, 1),
                })

            return {
                "available": True,
                "device_count": device_count,
                "cuda_version": torch.version.cuda,
                "devices": devices,
            }

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to get GPU metrics: {e}")

    return {
        "available": False,
        "device_count": 0,
        "devices": [],
    }


@router.post("/record")
async def record_cost_event(
    category: CostCategory,
    cost_usd: float,
    tokens: int = 0,
    model_id: Optional[str] = None,
    latency_ms: float = 0,
) -> Dict[str, str]:
    """
    Record a cost event.

    Used internally by services to track costs.
    """
    _record_cost(category, cost_usd, tokens, model_id, latency_ms)
    return {"status": "recorded"}


@router.post("/cache-event")
async def record_cache_event(
    cache_type: str,
    hit: bool,
) -> Dict[str, str]:
    """
    Record a cache hit/miss event.

    Used internally by cache services to track performance.
    """
    if cache_type not in _cache_stats:
        _cache_stats[cache_type] = {"hits": 0, "misses": 0}

    if hit:
        _cache_stats[cache_type]["hits"] += 1
    else:
        _cache_stats[cache_type]["misses"] += 1

    return {"status": "recorded"}


@router.get("/summary")
async def get_cost_summary() -> Dict[str, Any]:
    """
    Get a quick cost summary.

    Returns key metrics for dashboard display.
    """
    # Calculate today's costs
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_records = [r for r in _cost_records if r["timestamp"] >= today]
    today_cost = sum(r["cost_usd"] for r in today_records)

    # Calculate this month's costs
    month_start = today.replace(day=1)
    month_records = [r for r in _cost_records if r["timestamp"] >= month_start]
    month_cost = sum(r["cost_usd"] for r in month_records)

    # Top model by cost
    top_model = None
    if _model_usage:
        top_model = max(_model_usage.items(), key=lambda x: x[1]["total_cost_usd"])

    # Cache savings
    total_hits = sum(d.get("hits", 0) for d in _cache_stats.values())
    estimated_savings = total_hits * 0.001  # $0.001 per cache hit

    return {
        "today_cost_usd": round(today_cost, 4),
        "month_cost_usd": round(month_cost, 4),
        "total_queries_today": len(today_records),
        "total_queries_month": len(month_records),
        "cache_hit_rate": round(
            total_hits / (total_hits + sum(d.get("misses", 0) for d in _cache_stats.values())),
            3
        ) if (total_hits + sum(d.get("misses", 0) for d in _cache_stats.values())) > 0 else 0,
        "estimated_cache_savings_usd": round(estimated_savings, 4),
        "top_model": {
            "model_id": top_model[0] if top_model else None,
            "cost_usd": round(top_model[1]["total_cost_usd"], 4) if top_model else 0,
        },
        "active_alerts": sum(1 for a in _budget_alerts if a.triggered),
    }


# =============================================================================
# Helper Functions for Integration
# =============================================================================

async def track_llm_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> None:
    """
    Track cost for an LLM inference call.

    Call this from LLM service after each inference.
    """
    # Cost estimation based on model (approximate 2024-2025 pricing)
    cost_per_1k_input = 0.0001  # Default
    cost_per_1k_output = 0.0002

    if "gpt-4o" in model_id.lower():
        cost_per_1k_input = 0.0025
        cost_per_1k_output = 0.01
    elif "gpt-4o-mini" in model_id.lower():
        cost_per_1k_input = 0.00015
        cost_per_1k_output = 0.0006
    elif "claude-3-opus" in model_id.lower():
        cost_per_1k_input = 0.015
        cost_per_1k_output = 0.075
    elif "claude-3-5-sonnet" in model_id.lower():
        cost_per_1k_input = 0.003
        cost_per_1k_output = 0.015
    elif "claude-3-5-haiku" in model_id.lower():
        cost_per_1k_input = 0.00025
        cost_per_1k_output = 0.00125

    cost = (input_tokens / 1000 * cost_per_1k_input) + (output_tokens / 1000 * cost_per_1k_output)

    _record_cost(
        CostCategory.LLM_INFERENCE,
        cost,
        input_tokens + output_tokens,
        model_id,
        latency_ms,
    )


async def track_embedding_cost(
    model_id: str,
    tokens: int,
) -> None:
    """Track cost for embedding generation."""
    # Embedding costs (approximate)
    cost_per_1k = 0.0001  # Default

    if "text-embedding-3-large" in model_id:
        cost_per_1k = 0.00013
    elif "text-embedding-3-small" in model_id:
        cost_per_1k = 0.00002
    elif "voyage" in model_id.lower():
        cost_per_1k = 0.0001
    elif "nova" in model_id.lower():
        cost_per_1k = 0.00008

    cost = tokens / 1000 * cost_per_1k

    _record_cost(CostCategory.EMBEDDING, cost, tokens, model_id)
