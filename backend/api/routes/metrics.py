"""
AIDocumentIndexer - Prometheus Metrics Endpoint
================================================

Provides Prometheus-compatible metrics for monitoring system health,
performance, and usage patterns.

Metrics exposed:
- Request latency (p50, p95, p99)
- LLM token usage and costs
- Embedding generation time
- Cache hit rates
- Document processing counts
- Active connections
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import structlog

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from backend.db.database import async_session_context
from backend.db.models import Document, Chunk, ChatSession, ChatMessage, LLMUsageLog
from sqlalchemy import select, func

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@dataclass
class MetricsCollector:
    """
    Collects and tracks application metrics.

    Thread-safe singleton for collecting metrics across the application.
    """
    # Request latency tracking (in seconds)
    request_latencies: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Counter metrics
    request_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # LLM metrics
    llm_requests: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    llm_tokens_input: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    llm_tokens_output: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    # Embedding metrics
    embedding_latencies: List[float] = field(default_factory=list)
    embeddings_generated: int = 0

    # Max samples to keep for latency calculations
    max_samples: int = 1000

    def record_request_latency(self, endpoint: str, latency_seconds: float):
        """Record request latency for an endpoint."""
        latencies = self.request_latencies[endpoint]
        latencies.append(latency_seconds)
        # Keep only last max_samples
        if len(latencies) > self.max_samples:
            self.request_latencies[endpoint] = latencies[-self.max_samples:]
        self.request_counts[endpoint] += 1

    def record_error(self, endpoint: str):
        """Record an error for an endpoint."""
        self.error_counts[endpoint] += 1

    def record_llm_usage(self, provider: str, input_tokens: int, output_tokens: int):
        """Record LLM token usage."""
        self.llm_requests[provider] += 1
        self.llm_tokens_input[provider] += input_tokens
        self.llm_tokens_output[provider] += output_tokens

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    def record_embedding_latency(self, latency_seconds: float):
        """Record embedding generation latency."""
        self.embedding_latencies.append(latency_seconds)
        if len(self.embedding_latencies) > self.max_samples:
            self.embedding_latencies = self.embedding_latencies[-self.max_samples:]
        self.embeddings_generated += 1

    def get_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate a percentile from a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def reset(self):
        """Reset all metrics (for testing)."""
        self.request_latencies = defaultdict(list)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.llm_requests = defaultdict(int)
        self.llm_tokens_input = defaultdict(int)
        self.llm_tokens_output = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.embedding_latencies = []
        self.embeddings_generated = 0


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


async def get_database_metrics() -> Dict[str, Any]:
    """Fetch database-level metrics."""
    try:
        async with async_session_context() as db:
            # Document counts
            total_docs = await db.scalar(select(func.count(Document.id))) or 0

            # Chunk counts
            total_chunks = await db.scalar(select(func.count(Chunk.id))) or 0
            chunks_with_embeddings = await db.scalar(
                select(func.count(Chunk.id)).where(Chunk.has_embedding == True)
            ) or 0

            # Chat metrics
            total_sessions = await db.scalar(select(func.count(ChatSession.id))) or 0
            total_messages = await db.scalar(select(func.count(ChatMessage.id))) or 0

            # Recent LLM usage (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_llm_usage = await db.execute(
                select(
                    LLMUsageLog.provider,
                    func.sum(LLMUsageLog.input_tokens).label("input_tokens"),
                    func.sum(LLMUsageLog.output_tokens).label("output_tokens"),
                    func.sum(LLMUsageLog.cost).label("total_cost"),
                    func.count(LLMUsageLog.id).label("request_count"),
                )
                .where(LLMUsageLog.created_at >= yesterday)
                .group_by(LLMUsageLog.provider)
            )
            llm_usage = [
                {
                    "provider": row.provider,
                    "input_tokens": row.input_tokens or 0,
                    "output_tokens": row.output_tokens or 0,
                    "cost": float(row.total_cost or 0),
                    "requests": row.request_count,
                }
                for row in recent_llm_usage.fetchall()
            ]

            return {
                "documents_total": total_docs,
                "chunks_total": total_chunks,
                "chunks_embedded": chunks_with_embeddings,
                "embedding_coverage_percent": (
                    (chunks_with_embeddings / total_chunks * 100) if total_chunks > 0 else 0
                ),
                "chat_sessions_total": total_sessions,
                "chat_messages_total": total_messages,
                "llm_usage_24h": llm_usage,
            }
    except Exception as e:
        logger.warning("Failed to fetch database metrics", error=str(e))
        return {}


def format_prometheus_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics in Prometheus text exposition format."""
    lines = []

    # Helper to add metric with optional labels
    def add_metric(name: str, value: float, help_text: str = "", labels: Dict[str, str] = None):
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")

        if labels:
            label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
            lines.append(f"{name}{{{label_str}}} {value}")
        else:
            lines.append(f"{name} {value}")

    # Database metrics
    db_metrics = metrics.get("database", {})
    add_metric("aidoc_documents_total", db_metrics.get("documents_total", 0),
               "Total number of documents indexed")
    add_metric("aidoc_chunks_total", db_metrics.get("chunks_total", 0),
               "Total number of document chunks")
    add_metric("aidoc_chunks_embedded", db_metrics.get("chunks_embedded", 0),
               "Number of chunks with embeddings")
    add_metric("aidoc_embedding_coverage_percent", db_metrics.get("embedding_coverage_percent", 0),
               "Percentage of chunks with embeddings")
    add_metric("aidoc_chat_sessions_total", db_metrics.get("chat_sessions_total", 0),
               "Total number of chat sessions")
    add_metric("aidoc_chat_messages_total", db_metrics.get("chat_messages_total", 0),
               "Total number of chat messages")

    # LLM usage metrics (last 24h)
    for usage in db_metrics.get("llm_usage_24h", []):
        provider = usage["provider"]
        add_metric("aidoc_llm_input_tokens_24h", usage["input_tokens"],
                   "LLM input tokens in last 24 hours", {"provider": provider})
        add_metric("aidoc_llm_output_tokens_24h", usage["output_tokens"],
                   "LLM output tokens in last 24 hours", {"provider": provider})
        add_metric("aidoc_llm_cost_24h", usage["cost"],
                   "LLM cost in last 24 hours (USD)", {"provider": provider})
        add_metric("aidoc_llm_requests_24h", usage["requests"],
                   "LLM requests in last 24 hours", {"provider": provider})

    # In-memory metrics from collector
    collector = metrics.get("collector", {})

    # Request latencies
    for endpoint, latencies in collector.get("request_latencies", {}).items():
        if latencies:
            add_metric("aidoc_request_latency_p50_seconds", latencies.get("p50", 0),
                       "Request latency 50th percentile", {"endpoint": endpoint})
            add_metric("aidoc_request_latency_p95_seconds", latencies.get("p95", 0),
                       "Request latency 95th percentile", {"endpoint": endpoint})
            add_metric("aidoc_request_latency_p99_seconds", latencies.get("p99", 0),
                       "Request latency 99th percentile", {"endpoint": endpoint})

    # Request counts
    for endpoint, count in collector.get("request_counts", {}).items():
        add_metric("aidoc_requests_total", count,
                   "Total requests by endpoint", {"endpoint": endpoint})

    # Error counts
    for endpoint, count in collector.get("error_counts", {}).items():
        add_metric("aidoc_errors_total", count,
                   "Total errors by endpoint", {"endpoint": endpoint})

    # Cache metrics
    cache_hits = collector.get("cache_hits", 0)
    cache_misses = collector.get("cache_misses", 0)
    total_cache = cache_hits + cache_misses
    add_metric("aidoc_cache_hits_total", cache_hits, "Total cache hits")
    add_metric("aidoc_cache_misses_total", cache_misses, "Total cache misses")
    if total_cache > 0:
        add_metric("aidoc_cache_hit_ratio", cache_hits / total_cache, "Cache hit ratio")

    # Embedding metrics
    add_metric("aidoc_embeddings_generated_total", collector.get("embeddings_generated", 0),
               "Total embeddings generated")
    embedding_latencies = collector.get("embedding_latencies", {})
    if embedding_latencies:
        add_metric("aidoc_embedding_latency_p50_seconds", embedding_latencies.get("p50", 0),
                   "Embedding generation latency 50th percentile")
        add_metric("aidoc_embedding_latency_p95_seconds", embedding_latencies.get("p95", 0),
                   "Embedding generation latency 95th percentile")

    # Performance optimization metrics
    perf_metrics = metrics.get("performance", {})
    add_metric("aidoc_cython_enabled", 1 if perf_metrics.get("cython_enabled") else 0,
               "Cython optimizations enabled (1=yes, 0=no)")
    add_metric("aidoc_gpu_enabled", 1 if perf_metrics.get("gpu_enabled") else 0,
               "GPU acceleration enabled (1=yes, 0=no)")
    add_metric("aidoc_minhash_enabled", 1 if perf_metrics.get("minhash_enabled") else 0,
               "MinHash deduplication enabled (1=yes, 0=no)")
    add_metric("aidoc_uvloop_enabled", 1 if perf_metrics.get("uvloop_enabled") else 0,
               "uvloop async optimization enabled (1=yes, 0=no)")
    add_metric("aidoc_orjson_enabled", 1 if perf_metrics.get("orjson_enabled") else 0,
               "ORJSON fast serialization enabled (1=yes, 0=no)")

    # Memory metrics
    memory_metrics = metrics.get("memory", {})
    add_metric("aidoc_memory_rss_bytes", memory_metrics.get("rss_bytes", 0),
               "Resident set size in bytes")
    add_metric("aidoc_memory_vms_bytes", memory_metrics.get("vms_bytes", 0),
               "Virtual memory size in bytes")

    return "\n".join(lines)


@router.get("", response_class=PlainTextResponse)
async def get_metrics():
    """
    Get Prometheus-compatible metrics.

    Returns metrics in Prometheus text exposition format for scraping.
    Includes:
    - Database statistics (documents, chunks, embeddings)
    - LLM usage (tokens, costs, requests)
    - Request latencies (p50, p95, p99)
    - Cache hit rates
    - Error counts
    """
    try:
        # Fetch database metrics
        db_metrics = await get_database_metrics()

        # Get in-memory metrics from collector
        collector = get_metrics_collector()

        # Calculate percentiles for latencies
        collector_metrics = {
            "request_latencies": {},
            "request_counts": dict(collector.request_counts),
            "error_counts": dict(collector.error_counts),
            "cache_hits": collector.cache_hits,
            "cache_misses": collector.cache_misses,
            "embeddings_generated": collector.embeddings_generated,
            "embedding_latencies": {},
        }

        for endpoint, latencies in collector.request_latencies.items():
            if latencies:
                collector_metrics["request_latencies"][endpoint] = {
                    "p50": collector.get_percentile(latencies, 50),
                    "p95": collector.get_percentile(latencies, 95),
                    "p99": collector.get_percentile(latencies, 99),
                }

        if collector.embedding_latencies:
            collector_metrics["embedding_latencies"] = {
                "p50": collector.get_percentile(collector.embedding_latencies, 50),
                "p95": collector.get_percentile(collector.embedding_latencies, 95),
            }

        # Get performance optimization status
        perf_metrics = {}
        try:
            from backend.services.performance_init import get_performance_status
            perf_status = get_performance_status()
            perf_metrics = {
                "cython_enabled": perf_status.get("cython", {}).get("using_cython", False),
                "gpu_enabled": perf_status.get("gpu", {}).get("has_gpu", False),
                "minhash_enabled": perf_status.get("minhash", {}).get("using_minhash", False),
                "uvloop_enabled": True,  # Set at module import
                "orjson_enabled": True,  # Set at module import
            }
        except Exception:
            pass

        # Get memory metrics
        memory_metrics = {}
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            memory_metrics = {
                "rss_bytes": mem_info.rss,
                "vms_bytes": mem_info.vms,
            }
        except Exception:
            pass

        # Combine all metrics
        all_metrics = {
            "database": db_metrics,
            "collector": collector_metrics,
            "performance": perf_metrics,
            "memory": memory_metrics,
        }

        # Format for Prometheus
        prometheus_output = format_prometheus_metrics(all_metrics)

        return Response(
            content=prometheus_output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain",
            status_code=500,
        )


@router.get("/json")
async def get_metrics_json():
    """
    Get metrics in JSON format.

    Alternative to Prometheus format for easier debugging and integration
    with other monitoring systems.
    """
    try:
        db_metrics = await get_database_metrics()
        collector = get_metrics_collector()

        # Build response
        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_metrics,
            "requests": {
                "counts": dict(collector.request_counts),
                "errors": dict(collector.error_counts),
                "latencies": {},
            },
            "cache": {
                "hits": collector.cache_hits,
                "misses": collector.cache_misses,
                "hit_ratio": (
                    collector.cache_hits / (collector.cache_hits + collector.cache_misses)
                    if (collector.cache_hits + collector.cache_misses) > 0 else 0
                ),
            },
            "embeddings": {
                "generated": collector.embeddings_generated,
                "latency_p50": collector.get_percentile(collector.embedding_latencies, 50),
                "latency_p95": collector.get_percentile(collector.embedding_latencies, 95),
            },
            "llm": {
                "requests": dict(collector.llm_requests),
                "tokens_input": dict(collector.llm_tokens_input),
                "tokens_output": dict(collector.llm_tokens_output),
            },
        }

        # Add latency percentiles
        for endpoint, latencies in collector.request_latencies.items():
            if latencies:
                response["requests"]["latencies"][endpoint] = {
                    "p50": collector.get_percentile(latencies, 50),
                    "p95": collector.get_percentile(latencies, 95),
                    "p99": collector.get_percentile(latencies, 99),
                    "count": len(latencies),
                }

        return response

    except Exception as e:
        logger.error("Failed to generate JSON metrics", error=str(e))
        return {"error": str(e)}


@router.get("/health")
async def health_check():
    """
    Quick health check endpoint.

    Returns basic health status for load balancers and monitoring.
    """
    try:
        # Quick database connectivity check
        async with async_session_context() as db:
            await db.execute(select(func.count(Document.id)).limit(1))

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
