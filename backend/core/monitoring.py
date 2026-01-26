"""
AIDocumentIndexer - Monitoring & Observability
===============================================

Sentry integration for error tracking and performance monitoring.
Prometheus metrics for system observability.

Usage:
    from backend.core.monitoring import init_monitoring

    # In main.py startup
    init_monitoring(app)
"""

import structlog
from typing import Optional

logger = structlog.get_logger(__name__)


def init_sentry() -> bool:
    """
    Initialize Sentry for error tracking and performance monitoring.

    Returns:
        True if Sentry was initialized, False otherwise.
    """
    from backend.core.config import settings

    if not settings.SENTRY_DSN:
        logger.info("Sentry DSN not configured, skipping initialization")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.redis import RedisIntegration
        from sentry_sdk.integrations.httpx import HttpxIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.ENVIRONMENT,
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
            profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
                RedisIntegration(),
                HttpxIntegration(),
                LoggingIntegration(
                    level=None,  # Capture all levels
                    event_level=None,  # Don't send as events by default
                ),
            ],
            # Don't send PII by default
            send_default_pii=False,
            # Attach stack traces to all messages
            attach_stacktrace=True,
            # Set release version if available
            release=f"aidocindexer@{_get_version()}",
            # Custom before_send to filter sensitive data
            before_send=_filter_sensitive_data,
        )

        logger.info(
            "Sentry initialized",
            environment=settings.ENVIRONMENT,
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        )
        return True

    except ImportError:
        logger.warning("sentry-sdk not installed, skipping Sentry initialization")
        return False
    except Exception as e:
        logger.error("Failed to initialize Sentry", error=str(e))
        return False


def _get_version() -> str:
    """Get application version."""
    try:
        from importlib.metadata import version
        return version("aidocindexer")
    except Exception as e:
        logger.debug("Failed to get application version", error=str(e))
        return "unknown"


def _filter_sensitive_data(event: dict, hint: dict) -> Optional[dict]:
    """
    Filter sensitive data from Sentry events before sending.

    This removes:
    - Authorization headers
    - API keys
    - Passwords
    - Other sensitive fields
    """
    # Filter request headers
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        sensitive_headers = ["authorization", "x-api-key", "cookie", "x-auth-token"]
        for header in sensitive_headers:
            if header in headers:
                headers[header] = "[Filtered]"

    # Filter query strings
    if "request" in event and "query_string" in event["request"]:
        qs = event["request"]["query_string"]
        if isinstance(qs, str):
            # Filter common sensitive params
            for param in ["api_key", "token", "password", "secret"]:
                if param in qs.lower():
                    event["request"]["query_string"] = "[Filtered]"
                    break

    # Filter exception messages that might contain sensitive data
    if "exception" in event and "values" in event["exception"]:
        for exc in event["exception"]["values"]:
            if "value" in exc:
                value = exc["value"]
                # Check for patterns that might indicate sensitive data
                if any(p in value.lower() for p in ["password", "secret", "api_key", "token"]):
                    exc["value"] = "[Filtered - may contain sensitive data]"

    return event


def capture_exception(error: Exception, **extra_context) -> Optional[str]:
    """
    Capture an exception to Sentry with additional context.

    Args:
        error: The exception to capture
        **extra_context: Additional context to attach

    Returns:
        Sentry event ID if captured, None otherwise
    """
    from backend.core.config import settings

    if not settings.SENTRY_DSN:
        return None

    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in extra_context.items():
                scope.set_extra(key, value)
            return sentry_sdk.capture_exception(error)
    except Exception as e:
        logger.debug("Sentry capture_exception failed", error=str(e))
        return None


def capture_message(message: str, level: str = "info", **extra_context) -> Optional[str]:
    """
    Capture a message to Sentry.

    Args:
        message: The message to capture
        level: Severity level (debug, info, warning, error, fatal)
        **extra_context: Additional context to attach

    Returns:
        Sentry event ID if captured, None otherwise
    """
    from backend.core.config import settings

    if not settings.SENTRY_DSN:
        return None

    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in extra_context.items():
                scope.set_extra(key, value)
            return sentry_sdk.capture_message(message, level=level)
    except Exception as e:
        logger.debug("Sentry capture_message failed", error=str(e))
        return None


def set_user_context(user_id: str, email: Optional[str] = None, username: Optional[str] = None):
    """
    Set user context for Sentry events.

    Args:
        user_id: User ID
        email: User email (optional)
        username: Username (optional)
    """
    from backend.core.config import settings

    if not settings.SENTRY_DSN:
        return

    try:
        import sentry_sdk

        sentry_sdk.set_user({
            "id": user_id,
            "email": email,
            "username": username,
        })
    except Exception as e:
        logger.debug("Sentry set_user failed", error=str(e))


def init_prometheus_metrics(app):
    """
    Initialize Prometheus metrics endpoint.

    Exposes /metrics endpoint for Prometheus scraping.
    """
    from backend.core.config import settings

    if not settings.ENABLE_METRICS:
        logger.info("Prometheus metrics disabled")
        return

    try:
        from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
        from fastapi import Response
        from starlette.middleware.base import BaseHTTPMiddleware
        import time

        # Define metrics
        REQUEST_COUNT = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )

        REQUEST_LATENCY = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        ACTIVE_REQUESTS = Gauge(
            "http_requests_active",
            "Active HTTP requests",
            ["method"]
        )

        # LLM-specific metrics
        LLM_REQUESTS = Counter(
            "llm_requests_total",
            "Total LLM API requests",
            ["provider", "model", "status"]
        )

        LLM_TOKENS = Counter(
            "llm_tokens_total",
            "Total LLM tokens used",
            ["provider", "model", "type"]  # type: prompt, completion
        )

        LLM_LATENCY = Histogram(
            "llm_request_duration_seconds",
            "LLM request latency",
            ["provider", "model"],
            buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )

        # Document processing metrics
        DOCUMENTS_PROCESSED = Counter(
            "documents_processed_total",
            "Total documents processed",
            ["status", "type"]
        )

        EMBEDDINGS_GENERATED = Counter(
            "embeddings_generated_total",
            "Total embeddings generated",
            ["provider"]
        )

        # Middleware to track request metrics
        class MetricsMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                method = request.method
                # Normalize path to avoid high cardinality
                path = request.url.path
                # Remove IDs from paths
                import re
                path = re.sub(r"/[0-9a-f-]{36}", "/{id}", path)
                path = re.sub(r"/\d+", "/{id}", path)

                ACTIVE_REQUESTS.labels(method=method).inc()
                start_time = time.time()

                try:
                    response = await call_next(request)
                    REQUEST_COUNT.labels(
                        method=method,
                        endpoint=path,
                        status=response.status_code
                    ).inc()
                    return response
                finally:
                    REQUEST_LATENCY.labels(method=method, endpoint=path).observe(
                        time.time() - start_time
                    )
                    ACTIVE_REQUESTS.labels(method=method).dec()

        # Add middleware
        app.add_middleware(MetricsMiddleware)

        # Add metrics endpoint
        @app.get("/metrics", include_in_schema=False)
        async def metrics():
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )

        logger.info("Prometheus metrics initialized at /metrics")

        # Export metric objects for use elsewhere
        app.state.metrics = {
            "request_count": REQUEST_COUNT,
            "request_latency": REQUEST_LATENCY,
            "active_requests": ACTIVE_REQUESTS,
            "llm_requests": LLM_REQUESTS,
            "llm_tokens": LLM_TOKENS,
            "llm_latency": LLM_LATENCY,
            "documents_processed": DOCUMENTS_PROCESSED,
            "embeddings_generated": EMBEDDINGS_GENERATED,
        }

    except ImportError:
        logger.warning("prometheus-client not installed, skipping metrics initialization")
    except Exception as e:
        logger.error("Failed to initialize Prometheus metrics", error=str(e))


def init_monitoring(app):
    """
    Initialize all monitoring systems.

    Args:
        app: FastAPI application instance
    """
    init_sentry()
    init_prometheus_metrics(app)
