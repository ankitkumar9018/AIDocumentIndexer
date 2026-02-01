"""
AIDocumentIndexer - OpenTelemetry Distributed Tracing for RAG Pipeline
======================================================================

Provides structured tracing for RAG operations (query, retrieval, reranking,
generation) using OpenTelemetry.  Falls back to no-ops when the
opentelemetry-api / opentelemetry-sdk packages are not installed so that the
rest of the application can import this module unconditionally.

Configuration is driven by the settings service:
    observability.tracing_enabled       - master switch (default False)
    observability.tracing_sample_rate   - head-based sampling rate (0.0-1.0)
    observability.otlp_endpoint         - OTLP/gRPC collector endpoint

Environment variable overrides (take precedence over DB settings):
    OTEL_EXPORTER_OTLP_ENDPOINT   - OTLP endpoint URL
    OTEL_SERVICE_NAME             - logical service name (default "aidocumentindexer")
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Graceful degradation: try to import OpenTelemetry, flag availability
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        SpanKind,
        StatusCode,
        Tracer,
        Span,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.sampling import (
        TraceIdRatioBased,
        ALWAYS_OFF,
    )

    HAS_OPENTELEMETRY = True
except ImportError:  # pragma: no cover
    HAS_OPENTELEMETRY = False
    trace = None  # type: ignore[assignment]
    SpanKind = None  # type: ignore[assignment,misc]
    StatusCode = None  # type: ignore[assignment,misc]
    Tracer = None  # type: ignore[assignment,misc]
    Span = None  # type: ignore[assignment,misc]
    TracerProvider = None  # type: ignore[assignment,misc]
    BatchSpanProcessor = None  # type: ignore[assignment,misc]
    ConsoleSpanExporter = None  # type: ignore[assignment,misc]
    Resource = None  # type: ignore[assignment,misc]
    TraceIdRatioBased = None  # type: ignore[assignment,misc]
    ALWAYS_OFF = None  # type: ignore[assignment,misc]

# Optional OTLP exporter (installed separately)
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    HAS_OTLP_EXPORTER = True
except ImportError:
    HAS_OTLP_EXPORTER = False
    OTLPSpanExporter = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "aidocumentindexer")
_TRACER_INSTRUMENTATION_NAME = "aidocumentindexer.rag"
_TRACER_VERSION = "1.0.0"


# ============================================================================
# No-Op helpers (used when OpenTelemetry is unavailable or tracing disabled)
# ============================================================================

class _NoOpSpan:
    """Minimal stand-in for ``opentelemetry.trace.Span``."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def set_status(self, status: Any, description: Optional[str] = None) -> None:
        pass

    def record_exception(self, exception: BaseException, **kwargs: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


# ============================================================================
# RAGTracer
# ============================================================================

class RAGTracer:
    """High-level tracing wrapper for RAG pipeline stages.

    Each ``start_*_span`` method returns a context manager that automatically
    records latency, captures key attributes, and marks the span as failed
    when an unhandled exception propagates.

    Usage::

        tracer = get_rag_tracer()
        with tracer.start_query_span(query="What is X?") as span:
            results = await rag_service.query(...)
            span.set_attribute("rag.result_count", len(results))
    """

    def __init__(self, tracer: Optional[Any] = None, enabled: bool = True) -> None:
        self._tracer = tracer
        self._enabled = enabled and HAS_OPENTELEMETRY and tracer is not None
        self._logger = structlog.get_logger("rag_tracer")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    @contextmanager
    def _span(
        self,
        operation: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Any = None,
    ) -> Generator[Any, None, None]:
        """Create and yield an OpenTelemetry span (or a no-op)."""
        if not self._enabled:
            yield _NOOP_SPAN
            return

        if kind is None:
            kind = SpanKind.INTERNAL

        span = self._tracer.start_span(
            name=operation,
            kind=kind,
            attributes=attributes or {},
        )
        start = time.monotonic()
        try:
            yield span
            elapsed_ms = (time.monotonic() - start) * 1000
            span.set_attribute("rag.latency_ms", round(elapsed_ms, 2))
            span.set_status(StatusCode.OK)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            span.set_attribute("rag.latency_ms", round(elapsed_ms, 2))
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, description=str(exc))
            self._logger.error(
                "rag_span_error",
                operation=operation,
                error=str(exc),
                latency_ms=round(elapsed_ms, 2),
            )
            raise
        finally:
            span.end()

    # ------------------------------------------------------------------
    # Public span methods
    # ------------------------------------------------------------------

    @contextmanager
    def start_query_span(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        search_type: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """Trace the full RAG query lifecycle.

        Attributes captured:
            rag.query.length, rag.query.search_type, rag.session_id, rag.user_id
        """
        attrs: Dict[str, Any] = {
            "rag.operation": "query",
            "rag.query.length": len(query),
        }
        if session_id:
            attrs["rag.session_id"] = session_id
        if user_id:
            attrs["rag.user_id"] = user_id
        if search_type:
            attrs["rag.query.search_type"] = search_type

        with self._span("rag.query", attributes=attrs) as span:
            yield span

    @contextmanager
    def start_retrieval_span(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        search_type: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """Trace the document retrieval stage.

        Attributes captured:
            rag.retrieval.query_length, rag.retrieval.top_k,
            rag.retrieval.search_type, rag.retrieval.collection
        """
        attrs: Dict[str, Any] = {
            "rag.operation": "retrieval",
            "rag.retrieval.query_length": len(query),
        }
        if top_k is not None:
            attrs["rag.retrieval.top_k"] = top_k
        if search_type:
            attrs["rag.retrieval.search_type"] = search_type
        if collection:
            attrs["rag.retrieval.collection"] = collection

        with self._span("rag.retrieval", attributes=attrs) as span:
            yield span

    @contextmanager
    def start_reranking_span(
        self,
        *,
        doc_count: Optional[int] = None,
        reranker_model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> Generator[Any, None, None]:
        """Trace the reranking stage.

        Attributes captured:
            rag.reranking.doc_count, rag.reranking.model, rag.reranking.top_n
        """
        attrs: Dict[str, Any] = {
            "rag.operation": "reranking",
        }
        if doc_count is not None:
            attrs["rag.reranking.doc_count"] = doc_count
        if reranker_model:
            attrs["rag.reranking.model"] = reranker_model
        if top_n is not None:
            attrs["rag.reranking.top_n"] = top_n

        with self._span("rag.reranking", attributes=attrs) as span:
            yield span

    @contextmanager
    def start_generation_span(
        self,
        *,
        model_name: Optional[str] = None,
        doc_count: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        streaming: bool = False,
    ) -> Generator[Any, None, None]:
        """Trace the LLM generation stage.

        Attributes captured:
            rag.generation.model, rag.generation.doc_count,
            rag.generation.prompt_tokens, rag.generation.max_tokens,
            rag.generation.temperature, rag.generation.streaming
        """
        attrs: Dict[str, Any] = {
            "rag.operation": "generation",
            "rag.generation.streaming": streaming,
        }
        if model_name:
            attrs["rag.generation.model"] = model_name
        if doc_count is not None:
            attrs["rag.generation.doc_count"] = doc_count
        if prompt_tokens is not None:
            attrs["rag.generation.prompt_tokens"] = prompt_tokens
        if max_tokens is not None:
            attrs["rag.generation.max_tokens"] = max_tokens
        if temperature is not None:
            attrs["rag.generation.temperature"] = temperature

        with self._span("rag.generation", attributes=attrs) as span:
            yield span

    # ------------------------------------------------------------------
    # Metrics helper
    # ------------------------------------------------------------------

    def record_metric(
        self,
        name: str,
        value: Any,
        *,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a one-off metric as a span event.

        This is a lightweight mechanism for recording point-in-time
        measurements (e.g. cache hit rate, token counts) without requiring a
        full OpenTelemetry Metrics SDK setup.

        Args:
            name: Metric / event name (e.g. ``"rag.cache_hit"``).
            value: The metric value.
            attributes: Additional key-value pairs to attach.
        """
        if not self._enabled:
            return

        current_span = trace.get_current_span()
        if current_span is None or current_span is _NOOP_SPAN:
            return

        event_attrs: Dict[str, Any] = {"metric.value": value}
        if attributes:
            event_attrs.update(attributes)

        current_span.add_event(name, attributes=event_attrs)
        self._logger.debug("rag_metric_recorded", metric=name, value=value)


# ============================================================================
# Setup & singleton
# ============================================================================

_rag_tracer: Optional[RAGTracer] = None
_provider_initialized: bool = False


def setup_tracing(
    *,
    enabled: bool = False,
    sample_rate: float = 0.1,
    otlp_endpoint: str = "",
    service_name: Optional[str] = None,
) -> None:
    """Initialize the OpenTelemetry tracer provider.

    Call this once during application startup.  The function is idempotent;
    subsequent calls are ignored.

    Args:
        enabled: Master switch.  When ``False`` all tracing becomes no-ops.
        sample_rate: Head-based sampling ratio (0.0 = drop all, 1.0 = keep all).
        otlp_endpoint: OTLP/gRPC collector endpoint.  When empty, spans are
            exported to the console (useful for local development).
        service_name: Logical service name embedded in every span.
    """
    global _provider_initialized, _rag_tracer

    if _provider_initialized:
        logger.debug("tracing_already_initialized")
        return

    if not HAS_OPENTELEMETRY:
        logger.info(
            "opentelemetry_not_installed",
            msg="OpenTelemetry packages not found; tracing disabled. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk",
        )
        _rag_tracer = RAGTracer(tracer=None, enabled=False)
        _provider_initialized = True
        return

    if not enabled:
        logger.info("tracing_disabled", msg="Tracing is disabled via settings.")
        _rag_tracer = RAGTracer(tracer=None, enabled=False)
        _provider_initialized = True
        return

    svc_name = service_name or _SERVICE_NAME
    resource = Resource.create({"service.name": svc_name})

    sampler = TraceIdRatioBased(max(0.0, min(1.0, sample_rate)))

    provider = TracerProvider(resource=resource, sampler=sampler)

    # Determine exporter: prefer OTLP if an endpoint is configured,
    # fall back to console.
    effective_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "") or otlp_endpoint
    if effective_endpoint and HAS_OTLP_EXPORTER:
        exporter = OTLPSpanExporter(endpoint=effective_endpoint, insecure=True)
        logger.info(
            "tracing_otlp_exporter",
            endpoint=effective_endpoint,
            service=svc_name,
        )
    elif effective_endpoint and not HAS_OTLP_EXPORTER:
        logger.warning(
            "otlp_exporter_unavailable",
            msg="OTLP endpoint configured but opentelemetry-exporter-otlp-proto-grpc "
            "is not installed. Falling back to console exporter. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-grpc",
            endpoint=effective_endpoint,
        )
        exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
        logger.info(
            "tracing_console_exporter",
            msg="No OTLP endpoint configured; using console exporter.",
            service=svc_name,
        )

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(
        _TRACER_INSTRUMENTATION_NAME,
        _TRACER_VERSION,
    )

    _rag_tracer = RAGTracer(tracer=tracer, enabled=True)
    _provider_initialized = True

    logger.info(
        "tracing_initialized",
        service=svc_name,
        sample_rate=sample_rate,
        exporter_type="otlp" if (effective_endpoint and HAS_OTLP_EXPORTER) else "console",
    )


def get_rag_tracer() -> RAGTracer:
    """Return the singleton ``RAGTracer`` instance.

    If ``setup_tracing()`` has not been called yet, returns a no-op tracer so
    callers never have to worry about ``None`` checks.
    """
    global _rag_tracer
    if _rag_tracer is None:
        _rag_tracer = RAGTracer(tracer=None, enabled=False)
    return _rag_tracer


def reset_tracing() -> None:
    """Reset tracing state.  Primarily intended for testing."""
    global _rag_tracer, _provider_initialized
    _rag_tracer = None
    _provider_initialized = False
