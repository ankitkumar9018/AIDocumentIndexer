"""
AIDocumentIndexer - Provider Health Monitoring Service
=======================================================

Monitors LLM provider health, manages failover, and implements circuit breaker pattern.

Features:
- Periodic health checks for all active providers
- Automatic failover to backup providers
- Circuit breaker to prevent cascading failures
- Health history for analytics
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context
from backend.db.models import (
    LLMProvider,
    ProviderHealthCache,
    ProviderHealthLog,
    ProviderHealthStatus,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 3  # Consecutive failures before opening circuit
CIRCUIT_BREAKER_TIMEOUT_SECONDS = 300  # 5 minutes before retry
HEALTH_CHECK_TIMEOUT_SECONDS = 10  # Timeout for health check requests

# Health status thresholds
DEGRADED_LATENCY_MS = 2000  # Latency above this = degraded
UNHEALTHY_LATENCY_MS = 5000  # Latency above this = unhealthy


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a provider health check."""
    provider_id: str
    provider_type: str
    provider_name: str
    is_healthy: bool
    status: str
    latency_ms: Optional[int] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    checked_at: datetime = None

    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.utcnow()


# =============================================================================
# Provider Health Checker Service
# =============================================================================

class ProviderHealthChecker:
    """
    Service for monitoring LLM provider health.

    Provides methods for:
    - Individual provider health checks
    - Batch health checks for all providers
    - Health status caching
    - Circuit breaker management
    """

    @classmethod
    async def check_provider_health(
        cls,
        provider_id: str,
        check_type: str = "ping",
    ) -> HealthCheckResult:
        """
        Perform a health check on a specific provider.

        Args:
            provider_id: UUID of the provider to check
            check_type: Type of check ("ping", "completion", "embedding")

        Returns:
            HealthCheckResult with status and metrics
        """
        start_time = time.time()

        try:
            async with async_session_context() as db:
                # Get provider details
                result = await db.execute(
                    select(LLMProvider).where(LLMProvider.id == provider_id)
                )
                provider = result.scalar_one_or_none()

                if not provider:
                    return HealthCheckResult(
                        provider_id=provider_id,
                        provider_type="unknown",
                        provider_name="Unknown",
                        is_healthy=False,
                        status=ProviderHealthStatus.UNKNOWN.value,
                        error_message="Provider not found",
                    )

                # Perform the actual health check based on provider type
                check_result = await cls._perform_health_check(
                    provider=provider,
                    check_type=check_type,
                )

                latency_ms = int((time.time() - start_time) * 1000)
                check_result.latency_ms = latency_ms

                # Determine status based on latency and result
                if not check_result.is_healthy:
                    check_result.status = ProviderHealthStatus.UNHEALTHY.value
                elif latency_ms > UNHEALTHY_LATENCY_MS:
                    check_result.status = ProviderHealthStatus.UNHEALTHY.value
                    check_result.is_healthy = False
                elif latency_ms > DEGRADED_LATENCY_MS:
                    check_result.status = ProviderHealthStatus.DEGRADED.value
                else:
                    check_result.status = ProviderHealthStatus.HEALTHY.value

                # Log and cache the result
                await cls._log_health_check(db, check_result, check_type)
                await cls._update_health_cache(db, check_result)

                return check_result

        except Exception as e:
            logger.error("Health check failed", provider_id=provider_id, error=str(e))
            latency_ms = int((time.time() - start_time) * 1000)

            return HealthCheckResult(
                provider_id=provider_id,
                provider_type="unknown",
                provider_name="Unknown",
                is_healthy=False,
                status=ProviderHealthStatus.UNHEALTHY.value,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    @classmethod
    async def _perform_health_check(
        cls,
        provider: LLMProvider,
        check_type: str,
    ) -> HealthCheckResult:
        """
        Perform the actual health check based on provider type.

        This tries to make a minimal API call to verify the provider is responsive.
        """
        from backend.services.llm import LLMFactory, llm_config
        from backend.services.encryption import decrypt_value

        try:
            # Get API key
            api_key = None
            if provider.api_key_encrypted:
                try:
                    api_key = decrypt_value(provider.api_key_encrypted)
                except Exception:
                    pass

            # Fall back to environment variable
            if not api_key:
                env_key_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "google": "GOOGLE_API_KEY",
                    "groq": "GROQ_API_KEY",
                }
                import os
                api_key = os.getenv(env_key_map.get(provider.provider_type, ""), "")

            # For Ollama, we just need to check if the server is running
            if provider.provider_type == "ollama":
                import httpx
                base_url = provider.api_base_url or llm_config.ollama_host
                async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT_SECONDS) as client:
                    response = await client.get(f"{base_url}/api/tags")
                    if response.status_code == 200:
                        return HealthCheckResult(
                            provider_id=str(provider.id),
                            provider_type=provider.provider_type,
                            provider_name=provider.name,
                            is_healthy=True,
                            status=ProviderHealthStatus.HEALTHY.value,
                        )
                    else:
                        return HealthCheckResult(
                            provider_id=str(provider.id),
                            provider_type=provider.provider_type,
                            provider_name=provider.name,
                            is_healthy=False,
                            status=ProviderHealthStatus.UNHEALTHY.value,
                            error_message=f"HTTP {response.status_code}",
                            error_code=str(response.status_code),
                        )

            # For API-based providers, try a minimal models list call
            elif provider.provider_type == "openai":
                import httpx
                async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT_SECONDS) as client:
                    response = await client.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if response.status_code == 200:
                        return HealthCheckResult(
                            provider_id=str(provider.id),
                            provider_type=provider.provider_type,
                            provider_name=provider.name,
                            is_healthy=True,
                            status=ProviderHealthStatus.HEALTHY.value,
                        )
                    else:
                        return HealthCheckResult(
                            provider_id=str(provider.id),
                            provider_type=provider.provider_type,
                            provider_name=provider.name,
                            is_healthy=False,
                            status=ProviderHealthStatus.UNHEALTHY.value,
                            error_message=f"HTTP {response.status_code}",
                            error_code=str(response.status_code),
                        )

            elif provider.provider_type == "anthropic":
                import httpx
                # Anthropic doesn't have a models endpoint, so we'll just check auth
                async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT_SECONDS) as client:
                    # Use a minimal request to check if API key is valid
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": provider.default_chat_model or "claude-3-haiku-20240307",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                    # Any 2xx or 4xx (except 401) indicates the API is reachable
                    if response.status_code in [200, 400, 429]:
                        return HealthCheckResult(
                            provider_id=str(provider.id),
                            provider_type=provider.provider_type,
                            provider_name=provider.name,
                            is_healthy=True,
                            status=ProviderHealthStatus.HEALTHY.value,
                        )
                    elif response.status_code == 401:
                        return HealthCheckResult(
                            provider_id=str(provider.id),
                            provider_type=provider.provider_type,
                            provider_name=provider.name,
                            is_healthy=False,
                            status=ProviderHealthStatus.UNHEALTHY.value,
                            error_message="Invalid API key",
                            error_code="401",
                        )
                    else:
                        return HealthCheckResult(
                            provider_id=str(provider.id),
                            provider_type=provider.provider_type,
                            provider_name=provider.name,
                            is_healthy=False,
                            status=ProviderHealthStatus.UNHEALTHY.value,
                            error_message=f"HTTP {response.status_code}",
                            error_code=str(response.status_code),
                        )

            else:
                # For other providers, assume healthy if they're configured
                return HealthCheckResult(
                    provider_id=str(provider.id),
                    provider_type=provider.provider_type,
                    provider_name=provider.name,
                    is_healthy=True,
                    status=ProviderHealthStatus.HEALTHY.value,
                )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                provider_id=str(provider.id),
                provider_type=provider.provider_type,
                provider_name=provider.name,
                is_healthy=False,
                status=ProviderHealthStatus.UNHEALTHY.value,
                error_message="Health check timed out",
                error_code="TIMEOUT",
            )
        except Exception as e:
            return HealthCheckResult(
                provider_id=str(provider.id),
                provider_type=provider.provider_type,
                provider_name=provider.name,
                is_healthy=False,
                status=ProviderHealthStatus.UNHEALTHY.value,
                error_message=str(e),
            )

    @classmethod
    async def _log_health_check(
        cls,
        db: AsyncSession,
        result: HealthCheckResult,
        check_type: str,
    ) -> None:
        """Log the health check result to the database."""
        try:
            # Get current consecutive failures
            cache_result = await db.execute(
                select(ProviderHealthCache).where(
                    ProviderHealthCache.provider_id == result.provider_id
                )
            )
            cache = cache_result.scalar_one_or_none()
            consecutive_failures = 0

            if cache:
                if result.is_healthy:
                    consecutive_failures = 0
                else:
                    consecutive_failures = cache.consecutive_failures + 1

            log_entry = ProviderHealthLog(
                provider_id=uuid.UUID(result.provider_id),
                status=result.status,
                is_healthy=result.is_healthy,
                latency_ms=result.latency_ms,
                error_message=result.error_message,
                error_code=result.error_code,
                consecutive_failures=consecutive_failures,
                check_type=check_type,
                checked_at=result.checked_at,
            )
            db.add(log_entry)
            await db.commit()

        except Exception as e:
            logger.error("Failed to log health check", error=str(e))

    @classmethod
    async def _update_health_cache(
        cls,
        db: AsyncSession,
        result: HealthCheckResult,
    ) -> None:
        """Update the health cache with the latest result."""
        try:
            cache_result = await db.execute(
                select(ProviderHealthCache).where(
                    ProviderHealthCache.provider_id == result.provider_id
                )
            )
            cache = cache_result.scalar_one_or_none()

            now = datetime.utcnow()

            if cache:
                # Update existing cache
                if result.is_healthy:
                    cache.consecutive_failures = 0
                    cache.last_success_at = now
                    # Close circuit if it was open
                    if cache.circuit_open:
                        cache.circuit_open = False
                        cache.circuit_open_until = None
                        logger.info("Circuit closed", provider_id=result.provider_id)
                else:
                    cache.consecutive_failures += 1
                    cache.last_failure_at = now

                    # Check if we should open the circuit
                    if cache.consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                        cache.circuit_open = True
                        cache.circuit_open_until = now + timedelta(
                            seconds=CIRCUIT_BREAKER_TIMEOUT_SECONDS
                        )
                        logger.warning(
                            "Circuit opened",
                            provider_id=result.provider_id,
                            failures=cache.consecutive_failures,
                        )

                cache.status = result.status
                cache.is_healthy = result.is_healthy
                cache.last_latency_ms = result.latency_ms
                cache.last_check_at = now

                # Calculate rolling average latency
                if result.latency_ms:
                    if cache.avg_latency_ms:
                        # Simple exponential moving average
                        cache.avg_latency_ms = int(
                            cache.avg_latency_ms * 0.7 + result.latency_ms * 0.3
                        )
                    else:
                        cache.avg_latency_ms = result.latency_ms

            else:
                # Create new cache entry
                cache = ProviderHealthCache(
                    provider_id=uuid.UUID(result.provider_id),
                    status=result.status,
                    is_healthy=result.is_healthy,
                    last_latency_ms=result.latency_ms,
                    avg_latency_ms=result.latency_ms,
                    consecutive_failures=0 if result.is_healthy else 1,
                    last_success_at=now if result.is_healthy else None,
                    last_failure_at=None if result.is_healthy else now,
                    circuit_open=False,
                    last_check_at=now,
                )
                db.add(cache)

            await db.commit()

        except Exception as e:
            logger.error("Failed to update health cache", error=str(e))

    @classmethod
    async def run_all_health_checks(cls) -> List[HealthCheckResult]:
        """
        Run health checks for all active providers.

        Returns:
            List of HealthCheckResult for all providers
        """
        results = []

        try:
            async with async_session_context() as db:
                # Get all active providers
                provider_result = await db.execute(
                    select(LLMProvider).where(LLMProvider.is_active == True)
                )
                providers = provider_result.scalars().all()

                logger.info(f"Running health checks for {len(providers)} providers")

                # Run checks concurrently
                tasks = [
                    cls.check_provider_health(str(p.id))
                    for p in providers
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions and convert to results
                valid_results = []
                for r in results:
                    if isinstance(r, Exception):
                        logger.error("Health check exception", error=str(r))
                    elif isinstance(r, HealthCheckResult):
                        valid_results.append(r)

                return valid_results

        except Exception as e:
            logger.error("Failed to run health checks", error=str(e))
            return []

    @classmethod
    async def get_provider_health_status(
        cls,
        provider_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current health status for a provider from cache.

        Args:
            provider_id: UUID of the provider

        Returns:
            Dictionary with health status or None if not found
        """
        try:
            async with async_session_context() as db:
                result = await db.execute(
                    select(ProviderHealthCache).where(
                        ProviderHealthCache.provider_id == provider_id
                    )
                )
                cache = result.scalar_one_or_none()

                if not cache:
                    return None

                return {
                    "provider_id": str(cache.provider_id),
                    "status": cache.status,
                    "is_healthy": cache.is_healthy,
                    "last_latency_ms": cache.last_latency_ms,
                    "avg_latency_ms": cache.avg_latency_ms,
                    "consecutive_failures": cache.consecutive_failures,
                    "circuit_open": cache.circuit_open,
                    "circuit_open_until": cache.circuit_open_until.isoformat() if cache.circuit_open_until else None,
                    "last_check_at": cache.last_check_at.isoformat() if cache.last_check_at else None,
                    "last_success_at": cache.last_success_at.isoformat() if cache.last_success_at else None,
                    "last_failure_at": cache.last_failure_at.isoformat() if cache.last_failure_at else None,
                }

        except Exception as e:
            logger.error("Failed to get provider health", error=str(e))
            return None

    @classmethod
    async def get_all_provider_health(cls) -> List[Dict[str, Any]]:
        """
        Get health status for all providers.

        Returns:
            List of health status dictionaries
        """
        try:
            async with async_session_context() as db:
                # Join with providers to get names
                result = await db.execute(
                    select(ProviderHealthCache, LLMProvider)
                    .join(LLMProvider, ProviderHealthCache.provider_id == LLMProvider.id)
                    .where(LLMProvider.is_active == True)
                )
                rows = result.all()

                return [
                    {
                        "provider_id": str(cache.provider_id),
                        "provider_name": provider.name,
                        "provider_type": provider.provider_type,
                        "status": cache.status,
                        "is_healthy": cache.is_healthy,
                        "last_latency_ms": cache.last_latency_ms,
                        "avg_latency_ms": cache.avg_latency_ms,
                        "consecutive_failures": cache.consecutive_failures,
                        "circuit_open": cache.circuit_open,
                        "last_check_at": cache.last_check_at.isoformat() if cache.last_check_at else None,
                    }
                    for cache, provider in rows
                ]

        except Exception as e:
            logger.error("Failed to get all provider health", error=str(e))
            return []

    @classmethod
    async def get_health_history(
        cls,
        provider_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get health check history for a provider.

        Args:
            provider_id: UUID of the provider
            limit: Maximum number of records to return

        Returns:
            List of health log dictionaries
        """
        try:
            async with async_session_context() as db:
                result = await db.execute(
                    select(ProviderHealthLog)
                    .where(ProviderHealthLog.provider_id == provider_id)
                    .order_by(ProviderHealthLog.checked_at.desc())
                    .limit(limit)
                )
                logs = result.scalars().all()

                return [
                    {
                        "id": str(log.id),
                        "status": log.status,
                        "is_healthy": log.is_healthy,
                        "latency_ms": log.latency_ms,
                        "error_message": log.error_message,
                        "error_code": log.error_code,
                        "check_type": log.check_type,
                        "checked_at": log.checked_at.isoformat() if log.checked_at else None,
                    }
                    for log in logs
                ]

        except Exception as e:
            logger.error("Failed to get health history", error=str(e))
            return []

    @classmethod
    async def is_provider_available(cls, provider_id: str) -> bool:
        """
        Check if a provider is available (healthy and circuit closed).

        Args:
            provider_id: UUID of the provider

        Returns:
            True if provider is available for use
        """
        try:
            async with async_session_context() as db:
                result = await db.execute(
                    select(ProviderHealthCache).where(
                        ProviderHealthCache.provider_id == provider_id
                    )
                )
                cache = result.scalar_one_or_none()

                if not cache:
                    # No health data, assume available
                    return True

                # Check if circuit is open
                if cache.circuit_open:
                    if cache.circuit_open_until and datetime.utcnow() < cache.circuit_open_until:
                        return False
                    # Circuit timeout expired, allow retry

                return cache.is_healthy

        except Exception as e:
            logger.error("Failed to check provider availability", error=str(e))
            return True  # Fail open

    @classmethod
    async def get_healthy_provider_for_failover(
        cls,
        exclude_provider_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a healthy provider for failover.

        Args:
            exclude_provider_id: Provider to exclude (the one that failed)

        Returns:
            Provider ID of a healthy provider or None
        """
        try:
            async with async_session_context() as db:
                query = (
                    select(ProviderHealthCache, LLMProvider)
                    .join(LLMProvider, ProviderHealthCache.provider_id == LLMProvider.id)
                    .where(ProviderHealthCache.is_healthy == True)
                    .where(ProviderHealthCache.circuit_open == False)
                    .where(LLMProvider.is_active == True)
                )

                if exclude_provider_id:
                    query = query.where(
                        ProviderHealthCache.provider_id != exclude_provider_id
                    )

                # Order by latency and get the best one
                query = query.order_by(ProviderHealthCache.avg_latency_ms.asc())

                result = await db.execute(query)
                row = result.first()

                if row:
                    return str(row[0].provider_id)

                return None

        except Exception as e:
            logger.error("Failed to get failover provider", error=str(e))
            return None


# =============================================================================
# Background Health Check Task
# =============================================================================

async def run_periodic_health_checks(interval_seconds: int = 60):
    """
    Background task to run periodic health checks.

    Args:
        interval_seconds: Time between health check runs
    """
    logger.info("Starting periodic health check task", interval=interval_seconds)

    while True:
        try:
            results = await ProviderHealthChecker.run_all_health_checks()

            healthy_count = sum(1 for r in results if r.is_healthy)
            total_count = len(results)

            logger.info(
                "Health check completed",
                healthy=healthy_count,
                total=total_count,
            )

        except Exception as e:
            logger.error("Periodic health check failed", error=str(e))

        await asyncio.sleep(interval_seconds)


# =============================================================================
# Convenience Functions
# =============================================================================

async def get_provider_health_service() -> ProviderHealthChecker:
    """Get the provider health checker service."""
    return ProviderHealthChecker()


def get_provider_health_checker() -> ProviderHealthChecker:
    """Get the provider health checker (sync version for backward compatibility)."""
    return ProviderHealthChecker()
