"""
AIDocumentIndexer - Smart Model Router API Routes
==================================================

API endpoints for configuring cost-optimal model routing.
"""

from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import threading

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import require_admin
from backend.services.settings import get_settings_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/smart-router", tags=["Smart Model Router"])


# =============================================================================
# In-Memory Stats Tracking
# =============================================================================

class RouterStatsTracker:
    """Thread-safe tracker for router usage statistics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {
            "total_queries": 0,
            "simple_tier_count": 0,
            "moderate_tier_count": 0,
            "complex_tier_count": 0,
            "cost_without_routing": 0.0,  # Estimated cost if always using complex model
            "cost_with_routing": 0.0,  # Actual cost with smart routing
        }
        self._hourly_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_query(self, tier: str, estimated_tokens: int = 1000):
        """Record a routed query."""
        with self._lock:
            self._stats["total_queries"] += 1

            if tier == "simple":
                self._stats["simple_tier_count"] += 1
                # Estimate costs (rough approximations)
                self._stats["cost_with_routing"] += estimated_tokens * 0.0001  # Cheap model
            elif tier == "moderate":
                self._stats["moderate_tier_count"] += 1
                self._stats["cost_with_routing"] += estimated_tokens * 0.0005  # Mid model
            else:  # complex
                self._stats["complex_tier_count"] += 1
                self._stats["cost_with_routing"] += estimated_tokens * 0.002  # Expensive model

            # Always add complex model cost for comparison
            self._stats["cost_without_routing"] += estimated_tokens * 0.002

            # Track hourly stats
            hour_key = datetime.utcnow().strftime("%Y-%m-%d-%H")
            self._hourly_stats[hour_key][tier] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            total = self._stats["total_queries"]
            if total == 0:
                savings = 0.0
            else:
                cost_without = self._stats["cost_without_routing"]
                cost_with = self._stats["cost_with_routing"]
                if cost_without > 0:
                    savings = ((cost_without - cost_with) / cost_without) * 100
                else:
                    savings = 0.0

            return {
                "total_queries": self._stats["total_queries"],
                "simple_tier_count": self._stats["simple_tier_count"],
                "moderate_tier_count": self._stats["moderate_tier_count"],
                "complex_tier_count": self._stats["complex_tier_count"],
                "estimated_savings_percent": round(savings, 2),
            }

    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self._stats = {
                "total_queries": 0,
                "simple_tier_count": 0,
                "moderate_tier_count": 0,
                "complex_tier_count": 0,
                "cost_without_routing": 0.0,
                "cost_with_routing": 0.0,
            }
            self._hourly_stats.clear()


# Global stats tracker instance
_stats_tracker = RouterStatsTracker()


def get_stats_tracker() -> RouterStatsTracker:
    """Get the global stats tracker."""
    return _stats_tracker


# =============================================================================
# Request/Response Models
# =============================================================================

class TierModelConfig(BaseModel):
    """Model configuration for a query tier."""
    openai: Optional[str] = Field(None, description="OpenAI model for this tier")
    anthropic: Optional[str] = Field(None, description="Anthropic model for this tier")
    ollama: Optional[str] = Field(None, description="Ollama model for this tier")


class SmartRouterConfigResponse(BaseModel):
    """Smart router configuration."""
    enabled: bool = Field(..., description="Whether smart routing is enabled")
    simple_tier_model: Optional[str] = Field(None, description="Model for simple queries")
    complex_tier_model: Optional[str] = Field(None, description="Model for complex queries")
    tier_models: Dict[str, TierModelConfig] = Field(default_factory=dict)


class SmartRouterConfigUpdate(BaseModel):
    """Update smart router configuration."""
    enabled: Optional[bool] = Field(None, description="Enable/disable smart routing")
    simple_tier_model: Optional[str] = Field(None, description="Model for simple queries")
    complex_tier_model: Optional[str] = Field(None, description="Model for complex queries")


class RouteQueryRequest(BaseModel):
    """Request to route a query."""
    query: str = Field(..., description="Query to route")
    query_intent: Optional[str] = Field(None, description="Classified intent")
    query_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Intent confidence")
    context_length: int = Field(0, ge=0, description="Context length in chars")
    num_documents: int = Field(0, ge=0, description="Number of documents")


class RouteQueryResponse(BaseModel):
    """Response for query routing."""
    tier: str = Field(..., description="Assigned tier (simple/moderate/complex)")
    provider: Optional[str] = Field(None, description="Recommended provider")
    model: Optional[str] = Field(None, description="Recommended model")
    reason: str = Field(..., description="Routing reason")


class RouterStatsResponse(BaseModel):
    """Router usage statistics."""
    total_queries: int
    simple_tier_count: int
    moderate_tier_count: int
    complex_tier_count: int
    estimated_savings_percent: float


# =============================================================================
# Routes
# =============================================================================

@router.get("/config", response_model=SmartRouterConfigResponse)
async def get_router_config(
    current_user: dict = Depends(require_admin),
):
    """
    Get smart router configuration.

    Requires admin privileges.
    """
    settings = get_settings_service()

    return SmartRouterConfigResponse(
        enabled=settings.get("rag.smart_model_routing_enabled", False),
        simple_tier_model=settings.get("rag.smart_routing_simple_model"),
        complex_tier_model=settings.get("rag.smart_routing_complex_model"),
        tier_models={},
    )


@router.put("/config", response_model=SmartRouterConfigResponse)
async def update_router_config(
    request: SmartRouterConfigUpdate,
    current_user: dict = Depends(require_admin),
):
    """
    Update smart router configuration.

    Requires admin privileges.
    """
    settings = get_settings_service()

    if request.enabled is not None:
        settings.set("rag.smart_model_routing_enabled", request.enabled)

    if request.simple_tier_model is not None:
        settings.set("rag.smart_routing_simple_model", request.simple_tier_model)

    if request.complex_tier_model is not None:
        settings.set("rag.smart_routing_complex_model", request.complex_tier_model)

    logger.info(
        "smart_router.config_updated",
        enabled=request.enabled,
        user_id=current_user.get("id"),
    )

    return SmartRouterConfigResponse(
        enabled=settings.get("rag.smart_model_routing_enabled", False),
        simple_tier_model=settings.get("rag.smart_routing_simple_model"),
        complex_tier_model=settings.get("rag.smart_routing_complex_model"),
        tier_models={},
    )


@router.post("/route", response_model=RouteQueryResponse)
async def route_query(
    request: RouteQueryRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Route a query to the optimal model tier.

    This endpoint is primarily for testing/debugging the routing logic.
    In production, routing happens automatically in the RAG pipeline.

    Requires admin privileges.
    """
    from backend.services.smart_model_router import (
        classify_query_tier,
        get_model_for_tier,
    )

    settings = get_settings_service()

    if not settings.get("rag.smart_model_routing_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Smart model routing is not enabled",
        )

    tier = classify_query_tier(
        query_intent=request.query_intent,
        query_confidence=request.query_confidence,
        context_length=request.context_length,
        num_documents=request.num_documents,
    )

    route = get_model_for_tier(tier)

    # Track the routing decision
    get_stats_tracker().record_query(
        tier=route.tier.value,
        estimated_tokens=request.context_length // 4 + len(request.query),
    )

    return RouteQueryResponse(
        tier=route.tier.value,
        provider=route.provider,
        model=route.model,
        reason=route.reason,
    )


@router.get("/stats", response_model=RouterStatsResponse)
async def get_router_stats(
    current_user: dict = Depends(require_admin),
):
    """
    Get smart router usage statistics.

    Tracks queries routed to each tier and estimates cost savings
    compared to always using the complex model.

    Requires admin privileges.
    """
    stats = get_stats_tracker().get_stats()
    return RouterStatsResponse(**stats)


@router.post("/stats/reset")
async def reset_router_stats(
    current_user: dict = Depends(require_admin),
):
    """
    Reset router statistics.

    Requires admin privileges.
    """
    get_stats_tracker().reset_stats()
    logger.info("smart_router.stats_reset", user_id=current_user.get("id"))
    return {"status": "ok", "message": "Statistics reset successfully"}
