"""
AIDocumentIndexer - OpenTelemetry Tracing API Routes
=====================================================

API endpoints for configuring and viewing OpenTelemetry traces.
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import require_admin
from backend.services.settings import get_settings_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/telemetry/tracing", tags=["OpenTelemetry Tracing"])


# =============================================================================
# Request/Response Models
# =============================================================================

class TracingConfigResponse(BaseModel):
    """OpenTelemetry tracing configuration."""
    enabled: bool = Field(..., description="Whether tracing is enabled")
    sample_rate: float = Field(..., description="Trace sampling rate (0.0-1.0)")
    otlp_endpoint: Optional[str] = Field(None, description="OTLP collector endpoint")
    service_name: str = Field(..., description="Service name in traces")
    otel_available: bool = Field(..., description="Whether OpenTelemetry SDK is installed")


class TracingConfigUpdate(BaseModel):
    """Update tracing configuration."""
    enabled: Optional[bool] = Field(None, description="Enable/disable tracing")
    sample_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sample rate (0-1)")
    otlp_endpoint: Optional[str] = Field(None, description="OTLP endpoint URL")


class TracingStatsResponse(BaseModel):
    """Tracing statistics."""
    spans_created: int
    spans_exported: int
    traces_sampled: int
    export_errors: int
    average_span_duration_ms: float


class SpanSummary(BaseModel):
    """Summary of a trace span."""
    trace_id: str
    span_id: str
    operation: str
    duration_ms: float
    status: str
    timestamp: datetime


class RecentTracesResponse(BaseModel):
    """Recent traces response."""
    traces: List[SpanSummary]
    total_count: int


# =============================================================================
# Routes
# =============================================================================

@router.get("/config", response_model=TracingConfigResponse)
async def get_tracing_config(
    current_user: dict = Depends(require_admin),
):
    """
    Get OpenTelemetry tracing configuration.

    Requires admin privileges.
    """
    from backend.services.otel_tracing import HAS_OPENTELEMETRY

    settings = get_settings_service()

    return TracingConfigResponse(
        enabled=settings.get("observability.tracing_enabled", False),
        sample_rate=settings.get("observability.tracing_sample_rate", 0.1),
        otlp_endpoint=settings.get("observability.otlp_endpoint"),
        service_name="aidocumentindexer",
        otel_available=HAS_OPENTELEMETRY,
    )


@router.put("/config", response_model=TracingConfigResponse)
async def update_tracing_config(
    request: TracingConfigUpdate,
    current_user: dict = Depends(require_admin),
):
    """
    Update OpenTelemetry tracing configuration.

    Note: Changes to OTLP endpoint require service restart to take effect.

    Requires admin privileges.
    """
    from backend.services.otel_tracing import HAS_OPENTELEMETRY

    settings = get_settings_service()

    if request.enabled is not None:
        if request.enabled and not HAS_OPENTELEMETRY:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OpenTelemetry SDK is not installed. Install with: pip install opentelemetry-api opentelemetry-sdk",
            )
        settings.set("observability.tracing_enabled", request.enabled)

    if request.sample_rate is not None:
        settings.set("observability.tracing_sample_rate", request.sample_rate)

    if request.otlp_endpoint is not None:
        settings.set("observability.otlp_endpoint", request.otlp_endpoint)

    logger.info(
        "otel_tracing.config_updated",
        enabled=request.enabled,
        sample_rate=request.sample_rate,
        user_id=current_user.get("id"),
    )

    return TracingConfigResponse(
        enabled=settings.get("observability.tracing_enabled", False),
        sample_rate=settings.get("observability.tracing_sample_rate", 0.1),
        otlp_endpoint=settings.get("observability.otlp_endpoint"),
        service_name="aidocumentindexer",
        otel_available=HAS_OPENTELEMETRY,
    )


@router.get("/stats", response_model=TracingStatsResponse)
async def get_tracing_stats(
    current_user: dict = Depends(require_admin),
):
    """
    Get tracing statistics.

    Requires admin privileges.
    """
    # In a full implementation, these would be tracked
    return TracingStatsResponse(
        spans_created=0,
        spans_exported=0,
        traces_sampled=0,
        export_errors=0,
        average_span_duration_ms=0.0,
    )


@router.get("/recent", response_model=RecentTracesResponse)
async def get_recent_traces(
    limit: int = 20,
    current_user: dict = Depends(require_admin),
):
    """
    Get recent traces.

    Note: This endpoint returns locally cached traces only.
    For production use, query your OTLP backend (Jaeger, Tempo, etc.)

    Requires admin privileges.
    """
    # Placeholder - in production this would query the tracing backend
    return RecentTracesResponse(
        traces=[],
        total_count=0,
    )


@router.post("/enable")
async def enable_tracing(
    current_user: dict = Depends(require_admin),
):
    """
    Enable OpenTelemetry tracing.

    Requires admin privileges.
    """
    from backend.services.otel_tracing import HAS_OPENTELEMETRY

    if not HAS_OPENTELEMETRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OpenTelemetry SDK is not installed",
        )

    settings = get_settings_service()
    settings.set("observability.tracing_enabled", True)

    logger.info(
        "otel_tracing.enabled",
        user_id=current_user.get("id"),
    )

    return {"message": "OpenTelemetry tracing enabled", "status": "active"}


@router.post("/disable")
async def disable_tracing(
    current_user: dict = Depends(require_admin),
):
    """
    Disable OpenTelemetry tracing.

    Requires admin privileges.
    """
    settings = get_settings_service()
    settings.set("observability.tracing_enabled", False)

    logger.info(
        "otel_tracing.disabled",
        user_id=current_user.get("id"),
    )

    return {"message": "OpenTelemetry tracing disabled", "status": "inactive"}


@router.get("/health")
async def check_tracing_health(
    current_user: dict = Depends(require_admin),
):
    """
    Check tracing system health.

    Requires admin privileges.
    """
    from backend.services.otel_tracing import HAS_OPENTELEMETRY, HAS_OTLP_EXPORTER

    settings = get_settings_service()
    enabled = settings.get("observability.tracing_enabled", False)
    otlp_endpoint = settings.get("observability.otlp_endpoint")

    health_status = "healthy" if enabled and HAS_OPENTELEMETRY else "inactive"

    return {
        "status": health_status,
        "otel_sdk_installed": HAS_OPENTELEMETRY,
        "otlp_exporter_installed": HAS_OTLP_EXPORTER,
        "tracing_enabled": enabled,
        "otlp_endpoint_configured": otlp_endpoint is not None,
        "components": {
            "tracer": "available" if HAS_OPENTELEMETRY else "unavailable",
            "exporter": "available" if HAS_OTLP_EXPORTER else "unavailable (using console)",
        },
    }
