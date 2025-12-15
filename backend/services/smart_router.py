"""
AIDocumentIndexer - Smart Router Service
==========================================

Intelligent provider routing service for LLM requests.
Routes requests to optimal providers based on health, cost, latency, and priority.

Features:
- Priority-based routing
- Cost-optimized routing
- Latency-optimized routing
- Round-robin load balancing
- Health-aware provider selection
- Automatic failover
"""

import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.db.models import LLMProvider, ProviderHealthCache

logger = structlog.get_logger(__name__)


class RoutingStrategy(str, Enum):
    """Routing strategy for provider selection."""
    PRIORITY = "priority"  # Use provider priority order
    COST = "cost"  # Prefer cheapest providers
    LATENCY = "latency"  # Prefer fastest providers
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    RANDOM = "random"  # Random selection


@dataclass
class ProviderScore:
    """Scored provider for routing decisions."""
    provider_id: str
    provider_type: str
    name: str
    score: float
    is_healthy: bool
    latency_ms: Optional[int]
    cost_weight: float
    priority: int


class SmartRouter:
    """
    Service for intelligent provider routing.

    Selects the optimal provider based on the configured strategy,
    taking into account health status, costs, and latency.
    """

    # Default weights for scoring
    DEFAULT_PRIORITY_WEIGHT = 0.4
    DEFAULT_HEALTH_WEIGHT = 0.3
    DEFAULT_LATENCY_WEIGHT = 0.2
    DEFAULT_COST_WEIGHT = 0.1

    # Round-robin state
    _round_robin_index: int = 0

    async def select_provider(
        self,
        db: AsyncSession,
        strategy: RoutingStrategy = RoutingStrategy.PRIORITY,
        operation_type: Optional[str] = None,
        require_capability: Optional[str] = None,
        exclude_providers: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Select the optimal provider based on routing strategy.

        Args:
            db: Database session
            strategy: Routing strategy to use
            operation_type: Type of operation (chat, embeddings, etc.)
            require_capability: Required capability (e.g., "streaming", "functions")
            exclude_providers: Provider IDs to exclude

        Returns:
            Provider ID or None if no suitable provider found
        """
        # Get all active providers with health info
        providers = await self._get_providers_with_health(db)

        if not providers:
            logger.warning("No active providers found")
            return None

        # Filter excluded providers
        if exclude_providers:
            providers = [
                p for p in providers
                if str(p["id"]) not in exclude_providers
            ]

        if not providers:
            logger.warning("All providers excluded")
            return None

        # Apply routing strategy
        if strategy == RoutingStrategy.PRIORITY:
            return await self._route_by_priority(providers)
        elif strategy == RoutingStrategy.COST:
            return await self._route_by_cost(providers)
        elif strategy == RoutingStrategy.LATENCY:
            return await self._route_by_latency(providers)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._route_round_robin(providers)
        elif strategy == RoutingStrategy.RANDOM:
            return await self._route_random(providers)
        else:
            return await self._route_by_priority(providers)

    async def get_providers_by_priority(
        self,
        db: AsyncSession,
        healthy_only: bool = True,
    ) -> List[ProviderScore]:
        """
        Get providers ordered by priority.

        Args:
            db: Database session
            healthy_only: Only return healthy providers

        Returns:
            List of ProviderScore objects
        """
        providers = await self._get_providers_with_health(db)

        if healthy_only:
            providers = [p for p in providers if p.get("is_healthy", True)]

        # Sort by priority (lower = higher priority)
        providers.sort(key=lambda p: p.get("priority", 0))

        return [
            ProviderScore(
                provider_id=str(p["id"]),
                provider_type=p["provider_type"],
                name=p["name"],
                score=self._calculate_score(p),
                is_healthy=p.get("is_healthy", True),
                latency_ms=p.get("avg_latency_ms"),
                cost_weight=p.get("cost_weight", 1.0),
                priority=p.get("priority", 0),
            )
            for p in providers
        ]

    async def get_cheapest_provider(
        self,
        db: AsyncSession,
        model_capability: Optional[str] = None,
        healthy_only: bool = True,
    ) -> Optional[str]:
        """
        Get the cheapest provider.

        Args:
            db: Database session
            model_capability: Required capability
            healthy_only: Only consider healthy providers

        Returns:
            Provider ID or None
        """
        providers = await self._get_providers_with_health(db)

        if healthy_only:
            providers = [p for p in providers if p.get("is_healthy", True)]

        if not providers:
            return None

        # Sort by cost weight (lower = cheaper)
        providers.sort(key=lambda p: p.get("cost_weight", 1.0))

        return str(providers[0]["id"])

    async def get_fastest_provider(
        self,
        db: AsyncSession,
        model_capability: Optional[str] = None,
        healthy_only: bool = True,
    ) -> Optional[str]:
        """
        Get the fastest (lowest latency) provider.

        Args:
            db: Database session
            model_capability: Required capability
            healthy_only: Only consider healthy providers

        Returns:
            Provider ID or None
        """
        providers = await self._get_providers_with_health(db)

        if healthy_only:
            providers = [p for p in providers if p.get("is_healthy", True)]

        # Filter to providers with latency data
        providers_with_latency = [
            p for p in providers if p.get("avg_latency_ms") is not None
        ]

        if not providers_with_latency:
            # Fall back to priority if no latency data
            return await self._route_by_priority(providers) if providers else None

        # Sort by latency (lower = faster)
        providers_with_latency.sort(key=lambda p: p.get("avg_latency_ms", float("inf")))

        return str(providers_with_latency[0]["id"])

    async def update_provider_priority(
        self,
        db: AsyncSession,
        provider_id: str,
        priority: int,
    ) -> bool:
        """
        Update a provider's priority.

        Args:
            db: Database session
            provider_id: Provider UUID
            priority: New priority (lower = higher priority)

        Returns:
            True if successful
        """
        try:
            await db.execute(
                update(LLMProvider)
                .where(LLMProvider.id == uuid.UUID(provider_id))
                .values(priority=priority)
            )
            await db.commit()

            logger.info(
                "Updated provider priority",
                provider_id=provider_id,
                priority=priority,
            )
            return True

        except Exception as e:
            logger.error("Failed to update provider priority", error=str(e))
            await db.rollback()
            return False

    async def reorder_providers(
        self,
        db: AsyncSession,
        provider_order: List[str],
    ) -> bool:
        """
        Reorder all providers by setting their priorities.

        Args:
            db: Database session
            provider_order: List of provider IDs in desired order

        Returns:
            True if successful
        """
        try:
            for index, provider_id in enumerate(provider_order):
                await db.execute(
                    update(LLMProvider)
                    .where(LLMProvider.id == uuid.UUID(provider_id))
                    .values(priority=index)
                )

            await db.commit()

            logger.info(
                "Reordered providers",
                count=len(provider_order),
            )
            return True

        except Exception as e:
            logger.error("Failed to reorder providers", error=str(e))
            await db.rollback()
            return False

    async def get_routing_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """
        Get routing statistics.

        Args:
            db: Database session

        Returns:
            Dictionary with routing statistics
        """
        providers = await self._get_providers_with_health(db)

        healthy_count = sum(1 for p in providers if p.get("is_healthy", True))
        unhealthy_count = len(providers) - healthy_count

        avg_latency = None
        latencies = [p.get("avg_latency_ms") for p in providers if p.get("avg_latency_ms")]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)

        return {
            "total_providers": len(providers),
            "healthy_providers": healthy_count,
            "unhealthy_providers": unhealthy_count,
            "average_latency_ms": avg_latency,
            "providers": [
                {
                    "id": str(p["id"]),
                    "name": p["name"],
                    "type": p["provider_type"],
                    "priority": p.get("priority", 0),
                    "is_healthy": p.get("is_healthy", True),
                    "latency_ms": p.get("avg_latency_ms"),
                    "cost_weight": p.get("cost_weight", 1.0),
                }
                for p in providers
            ],
        }

    async def _get_providers_with_health(
        self,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """Get all active providers with their health status."""
        try:
            # Query providers with health cache
            result = await db.execute(
                select(LLMProvider)
                .where(LLMProvider.is_active == True)
                .order_by(LLMProvider.name)
            )
            providers = result.scalars().all()

            # Get health cache for all providers
            health_result = await db.execute(select(ProviderHealthCache))
            health_cache = {str(h.provider_id): h for h in health_result.scalars().all()}

            provider_data = []
            for provider in providers:
                health = health_cache.get(str(provider.id))
                settings = provider.settings or {}

                provider_data.append({
                    "id": provider.id,
                    "name": provider.name,
                    "provider_type": provider.provider_type,
                    "priority": settings.get("priority", 0),
                    "cost_weight": settings.get("cost_weight", 1.0),
                    "latency_weight": settings.get("latency_weight", 1.0),
                    "is_healthy": health.is_healthy if health else True,
                    "circuit_open": health.circuit_open if health else False,
                    "avg_latency_ms": health.avg_latency_ms if health else None,
                    "last_latency_ms": health.last_latency_ms if health else None,
                    "consecutive_failures": health.consecutive_failures if health else 0,
                })

            return provider_data

        except Exception as e:
            logger.error("Failed to get providers with health", error=str(e))
            return []

    async def _route_by_priority(
        self,
        providers: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Route by priority, preferring healthy providers."""
        # Filter healthy providers
        healthy = [p for p in providers if p.get("is_healthy", True) and not p.get("circuit_open", False)]

        if not healthy:
            logger.warning("No healthy providers, using any available")
            healthy = providers

        if not healthy:
            return None

        # Sort by priority
        healthy.sort(key=lambda p: p.get("priority", 0))

        return str(healthy[0]["id"])

    async def _route_by_cost(
        self,
        providers: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Route by cost, preferring cheaper providers."""
        healthy = [p for p in providers if p.get("is_healthy", True) and not p.get("circuit_open", False)]

        if not healthy:
            healthy = providers

        if not healthy:
            return None

        # Sort by cost weight
        healthy.sort(key=lambda p: p.get("cost_weight", 1.0))

        return str(healthy[0]["id"])

    async def _route_by_latency(
        self,
        providers: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Route by latency, preferring faster providers."""
        healthy = [p for p in providers if p.get("is_healthy", True) and not p.get("circuit_open", False)]

        if not healthy:
            healthy = providers

        if not healthy:
            return None

        # Filter to those with latency data
        with_latency = [p for p in healthy if p.get("avg_latency_ms") is not None]

        if not with_latency:
            # Fall back to priority
            healthy.sort(key=lambda p: p.get("priority", 0))
            return str(healthy[0]["id"])

        # Sort by latency
        with_latency.sort(key=lambda p: p.get("avg_latency_ms", float("inf")))

        return str(with_latency[0]["id"])

    async def _route_round_robin(
        self,
        providers: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Round-robin routing among healthy providers."""
        healthy = [p for p in providers if p.get("is_healthy", True) and not p.get("circuit_open", False)]

        if not healthy:
            healthy = providers

        if not healthy:
            return None

        # Get next provider in rotation
        index = self._round_robin_index % len(healthy)
        self._round_robin_index = (self._round_robin_index + 1) % len(healthy)

        return str(healthy[index]["id"])

    async def _route_random(
        self,
        providers: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Random routing among healthy providers."""
        healthy = [p for p in providers if p.get("is_healthy", True) and not p.get("circuit_open", False)]

        if not healthy:
            healthy = providers

        if not healthy:
            return None

        return str(random.choice(healthy)["id"])

    def _calculate_score(self, provider: Dict[str, Any]) -> float:
        """Calculate a composite score for a provider."""
        score = 0.0

        # Priority score (lower priority = higher score)
        priority = provider.get("priority", 0)
        priority_score = 1.0 / (1 + priority)
        score += priority_score * self.DEFAULT_PRIORITY_WEIGHT

        # Health score
        health_score = 1.0 if provider.get("is_healthy", True) else 0.0
        if provider.get("circuit_open", False):
            health_score = 0.0
        score += health_score * self.DEFAULT_HEALTH_WEIGHT

        # Latency score (lower latency = higher score)
        latency = provider.get("avg_latency_ms")
        if latency:
            latency_score = 1000.0 / (1000.0 + latency)  # Normalize
        else:
            latency_score = 0.5  # Unknown latency
        score += latency_score * self.DEFAULT_LATENCY_WEIGHT

        # Cost score (lower cost = higher score)
        cost = provider.get("cost_weight", 1.0)
        cost_score = 1.0 / cost
        score += cost_score * self.DEFAULT_COST_WEIGHT

        return score


# Singleton instance
_smart_router: Optional[SmartRouter] = None


def get_smart_router() -> SmartRouter:
    """Get the smart router singleton."""
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartRouter()
    return _smart_router
