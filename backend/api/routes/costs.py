"""
AIDocumentIndexer - Cost Tracking API Routes
=============================================

Endpoints for monitoring and managing LLM usage costs.
"""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import AuthenticatedUser, AdminUser
from backend.services.cost_tracking import (
    CostTrackingService,
    CostPeriod,
    UsageType,
    UsageRecord,
    CostSummary,
    UserCostSummary,
    CostAlert,
    get_cost_service,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class UsageRecordResponse(BaseModel):
    """Usage record response."""
    id: str
    user_id: str
    model: str
    provider: str
    usage_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    request_id: Optional[str]
    timestamp: datetime


class UserCostSummaryResponse(BaseModel):
    """User cost summary response."""
    user_id: str
    period: str
    total_cost: float
    total_requests: int
    total_tokens: int
    by_model: dict
    by_usage_type: dict
    daily_costs: List[dict]


class SystemCostSummaryResponse(BaseModel):
    """System cost summary response."""
    period: str
    start_date: datetime
    end_date: datetime
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    by_model: dict
    by_provider: dict
    by_usage_type: dict
    by_user: dict


class CreateAlertRequest(BaseModel):
    """Request to create a cost alert."""
    threshold: float = Field(..., gt=0, description="Cost threshold in USD")
    period: str = Field(default="month", description="Time period: hour, day, week, month")
    user_id: Optional[str] = Field(None, description="User ID (admin only, None for system-wide)")


class AlertResponse(BaseModel):
    """Cost alert response."""
    id: str
    user_id: Optional[str]
    threshold: float
    period: str
    enabled: bool
    notification_sent: bool
    created_at: datetime


class EstimateCostRequest(BaseModel):
    """Request to estimate cost."""
    model: str = Field(..., description="Model name")
    prompt: str = Field(..., min_length=1, description="Input prompt")
    max_output_tokens: int = Field(default=1000, ge=1, le=8000)


class EstimateCostResponse(BaseModel):
    """Cost estimate response."""
    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    currency: str


# =============================================================================
# Helper Functions
# =============================================================================

def record_to_response(record: UsageRecord) -> UsageRecordResponse:
    """Convert UsageRecord to response model."""
    return UsageRecordResponse(
        id=record.id,
        user_id=record.user_id,
        model=record.model,
        provider=record.provider,
        usage_type=record.usage_type.value,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.total_tokens,
        cost=record.cost,
        request_id=record.request_id,
        timestamp=record.timestamp,
    )


def summary_to_response(summary: UserCostSummary) -> UserCostSummaryResponse:
    """Convert UserCostSummary to response model."""
    return UserCostSummaryResponse(
        user_id=summary.user_id,
        period=summary.period.value,
        total_cost=summary.total_cost,
        total_requests=summary.total_requests,
        total_tokens=summary.total_tokens,
        by_model=summary.by_model,
        by_usage_type=summary.by_usage_type,
        daily_costs=summary.daily_costs,
    )


def system_summary_to_response(summary: CostSummary) -> SystemCostSummaryResponse:
    """Convert CostSummary to response model."""
    return SystemCostSummaryResponse(
        period=summary.period.value,
        start_date=summary.start_date,
        end_date=summary.end_date,
        total_cost=summary.total_cost,
        total_requests=summary.total_requests,
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        by_model=summary.by_model,
        by_provider=summary.by_provider,
        by_usage_type=summary.by_usage_type,
        by_user=summary.by_user,
    )


def alert_to_response(alert: CostAlert) -> AlertResponse:
    """Convert CostAlert to response model."""
    return AlertResponse(
        id=alert.id,
        user_id=alert.user_id,
        threshold=alert.threshold,
        period=alert.period.value,
        enabled=alert.enabled,
        notification_sent=alert.notification_sent,
        created_at=alert.created_at,
    )


def parse_period(period_str: str) -> CostPeriod:
    """Parse period string to CostPeriod enum."""
    try:
        return CostPeriod(period_str.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period: {period_str}. Use: hour, day, week, month, all_time",
        )


# =============================================================================
# API Endpoints - User
# =============================================================================

@router.get("/my-usage", response_model=UserCostSummaryResponse)
async def get_my_usage(
    user: AuthenticatedUser,
    period: str = Query(default="month", description="Time period"),
):
    """
    Get your own usage summary.

    Returns cost breakdown by model and usage type.
    """
    service = get_cost_service()
    cost_period = parse_period(period)

    summary = await service.get_user_summary(
        user_id=user.user_id,
        period=cost_period,
    )

    return summary_to_response(summary)


@router.get("/my-history")
async def get_my_usage_history(
    user: AuthenticatedUser,
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    Get your recent usage history.
    """
    service = get_cost_service()

    records = await service.get_recent_usage(
        user_id=user.user_id,
        limit=limit,
    )

    return {
        "records": [record_to_response(r) for r in records],
        "total": len(records),
    }


@router.get("/my-cost")
async def get_my_current_cost(
    user: AuthenticatedUser,
    period: str = Query(default="month"),
):
    """
    Get your current cost for a period.
    """
    service = get_cost_service()
    cost_period = parse_period(period)

    cost = await service.get_user_cost(
        user_id=user.user_id,
        period=cost_period,
    )

    return {
        "user_id": user.user_id,
        "period": period,
        "cost": round(cost, 6),
        "currency": "USD",
    }


# =============================================================================
# API Endpoints - Admin
# =============================================================================

@router.get("/system", response_model=SystemCostSummaryResponse)
async def get_system_costs(
    user: AdminUser,
    period: str = Query(default="month"),
):
    """
    Get system-wide cost summary.

    Admin only.
    """
    service = get_cost_service()
    cost_period = parse_period(period)

    summary = await service.get_system_summary(period=cost_period)

    return system_summary_to_response(summary)


@router.get("/users/{user_id}", response_model=UserCostSummaryResponse)
async def get_user_costs(
    user_id: str,
    admin: AdminUser,
    period: str = Query(default="month"),
):
    """
    Get cost summary for a specific user.

    Admin only.
    """
    service = get_cost_service()
    cost_period = parse_period(period)

    summary = await service.get_user_summary(
        user_id=user_id,
        period=cost_period,
    )

    return summary_to_response(summary)


@router.get("/all-usage")
async def get_all_usage(
    admin: AdminUser,
    limit: int = Query(default=100, ge=1, le=1000),
    user_id: Optional[str] = Query(None),
):
    """
    Get all recent usage records.

    Admin only.
    """
    service = get_cost_service()

    records = await service.get_recent_usage(
        user_id=user_id,
        limit=limit,
    )

    return {
        "records": [record_to_response(r) for r in records],
        "total": len(records),
    }


# =============================================================================
# API Endpoints - Alerts
# =============================================================================

@router.get("/alerts")
async def list_alerts(
    user: AuthenticatedUser,
):
    """
    List cost alerts.

    Regular users see their own alerts.
    Admins see all alerts.
    """
    service = get_cost_service()

    if user.is_admin():
        alerts = service.list_alerts()
    else:
        alerts = service.list_alerts(user_id=user.user_id)

    return {
        "alerts": [alert_to_response(a) for a in alerts],
        "total": len(alerts),
    }


@router.post("/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    request: CreateAlertRequest,
    user: AuthenticatedUser,
):
    """
    Create a cost alert.

    Regular users can only create alerts for themselves.
    Admins can create system-wide alerts or for any user.
    """
    service = get_cost_service()

    # Non-admins can only create alerts for themselves
    alert_user_id = request.user_id
    if not user.is_admin():
        if alert_user_id and alert_user_id != user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only create alerts for yourself",
            )
        alert_user_id = user.user_id

    cost_period = parse_period(request.period)

    alert = service.create_alert(
        threshold=request.threshold,
        period=cost_period,
        user_id=alert_user_id,
    )

    return alert_to_response(alert)


@router.delete("/alerts/{alert_id}")
async def delete_alert(
    alert_id: str,
    user: AuthenticatedUser,
):
    """
    Delete a cost alert.
    """
    service = get_cost_service()

    # Check alert exists
    alerts = service.list_alerts()
    alert = next((a for a in alerts if a.id == alert_id), None)

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    # Non-admins can only delete their own alerts
    if not user.is_admin() and alert.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this alert",
        )

    service.delete_alert(alert_id)

    return {"message": "Alert deleted", "alert_id": alert_id}


@router.post("/alerts/{alert_id}/reset")
async def reset_alert(
    alert_id: str,
    user: AuthenticatedUser,
):
    """
    Reset an alert's notification status.
    """
    service = get_cost_service()

    # Check alert exists and access
    alerts = service.list_alerts()
    alert = next((a for a in alerts if a.id == alert_id), None)

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    if not user.is_admin() and alert.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to reset this alert",
        )

    service.reset_alert(alert_id)

    return {"message": "Alert reset", "alert_id": alert_id}


# =============================================================================
# API Endpoints - Utilities
# =============================================================================

@router.post("/estimate", response_model=EstimateCostResponse)
async def estimate_cost(
    request: EstimateCostRequest,
    user: AuthenticatedUser,
):
    """
    Estimate cost for a potential request.
    """
    service = get_cost_service()

    estimate = await service.estimate_request_cost(
        model=request.model,
        prompt=request.prompt,
        max_output_tokens=request.max_output_tokens,
    )

    return EstimateCostResponse(**estimate)


@router.get("/pricing")
async def get_pricing(
    user: AuthenticatedUser,
):
    """
    Get current model pricing.

    Returns cost per 1K tokens for each model.
    """
    service = get_cost_service()

    pricing = service.get_model_pricing()

    return {
        "pricing": pricing,
        "currency": "USD",
        "unit": "per 1K tokens",
    }


@router.get("/dashboard")
async def get_cost_dashboard(
    user: AuthenticatedUser,
):
    """
    Get dashboard data with costs and usage.

    Returns summary for multiple periods.
    """
    service = get_cost_service()

    # Get summaries for different periods
    hour_summary = await service.get_user_summary(user.user_id, CostPeriod.HOUR)
    day_summary = await service.get_user_summary(user.user_id, CostPeriod.DAY)
    week_summary = await service.get_user_summary(user.user_id, CostPeriod.WEEK)
    month_summary = await service.get_user_summary(user.user_id, CostPeriod.MONTH)

    # Get alerts
    alerts = service.list_alerts(user_id=user.user_id)
    triggered_alerts = [a for a in alerts if a.notification_sent]

    return {
        "user_id": user.user_id,
        "costs": {
            "last_hour": round(hour_summary.total_cost, 6),
            "last_day": round(day_summary.total_cost, 6),
            "last_week": round(week_summary.total_cost, 6),
            "last_month": round(month_summary.total_cost, 6),
        },
        "requests": {
            "last_hour": hour_summary.total_requests,
            "last_day": day_summary.total_requests,
            "last_week": week_summary.total_requests,
            "last_month": month_summary.total_requests,
        },
        "by_model": month_summary.by_model,
        "by_usage_type": month_summary.by_usage_type,
        "daily_costs": month_summary.daily_costs,
        "alerts": {
            "total": len(alerts),
            "triggered": len(triggered_alerts),
        },
        "currency": "USD",
    }
