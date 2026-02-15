"""
AIDocumentIndexer - Diagnostics & Monitoring API Routes
========================================================

API endpoints for system monitoring, diagnostics, and health checks.

Endpoints:
- GET /diagnostics/health/all - Comprehensive health check
- GET /diagnostics/task-queue/status - Task queue status
- GET /diagnostics/distributed-processor/status - Distributed processor status
- GET /diagnostics/analytics/usage - Usage analytics
- GET /diagnostics/services/status - All service statuses
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import AdminUser

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


# =============================================================================
# Response Models
# =============================================================================

class ServiceStatus(BaseModel):
    """Individual service status."""
    name: str
    status: str  # healthy, degraded, unhealthy, unknown
    latency_ms: Optional[float] = None
    last_check: Optional[str] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Comprehensive health check response."""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    version: Optional[str] = None
    uptime_seconds: Optional[float] = None
    services: List[ServiceStatus]
    summary: Dict[str, int]  # {healthy: n, degraded: n, unhealthy: n}


class TaskQueueStatus(BaseModel):
    """Task queue status response."""
    status: str
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    workers: int = 0
    queue_size: int = 0
    oldest_pending_age_seconds: Optional[float] = None
    avg_task_duration_seconds: Optional[float] = None
    tasks_per_minute: Optional[float] = None
    memory_usage_mb: Optional[float] = None


class DistributedProcessorStatus(BaseModel):
    """Distributed processor status response."""
    status: str
    ray_available: bool = False
    ray_connected: bool = False
    cluster_nodes: int = 0
    cluster_cpus: int = 0
    cluster_gpus: int = 0
    active_jobs: int = 0
    pending_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    memory_usage_gb: Optional[float] = None
    object_store_usage_gb: Optional[float] = None


class UsageAnalytics(BaseModel):
    """Usage analytics response."""
    period: str  # last_hour, last_24h, last_7d, last_30d
    total_queries: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    total_users: int = 0
    active_users: int = 0
    llm_calls: int = 0
    llm_tokens_used: int = 0
    llm_estimated_cost: float = 0.0
    embedding_calls: int = 0
    cache_hit_rate: float = 0.0
    avg_query_latency_ms: float = 0.0
    avg_response_quality: Optional[float] = None
    documents_processed: int = 0
    errors_count: int = 0


class AllServicesStatus(BaseModel):
    """Status of all services."""
    overall_status: str
    timestamp: str
    services: Dict[str, ServiceStatus]


# =============================================================================
# Helper Functions
# =============================================================================

async def check_database_health() -> ServiceStatus:
    """Check database connectivity and health."""
    import time
    start = time.time()

    try:
        from backend.db.database import async_session_context
        from sqlalchemy import text

        async with async_session_context() as session:
            await session.execute(text("SELECT 1"))

        latency = (time.time() - start) * 1000

        return ServiceStatus(
            name="database",
            status="healthy",
            latency_ms=round(latency, 2),
            last_check=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        return ServiceStatus(
            name="database",
            status="unhealthy",
            last_check=datetime.utcnow().isoformat(),
            message=str(e),
        )


async def check_redis_health() -> ServiceStatus:
    """Check Redis connectivity."""
    import time
    start = time.time()

    try:
        import redis.asyncio as redis
        from backend.core.config import settings

        if not settings.REDIS_URL:
            return ServiceStatus(
                name="redis",
                status="unknown",
                message="Redis not configured",
            )

        client = redis.from_url(settings.REDIS_URL)
        await client.ping()
        await client.close()

        latency = (time.time() - start) * 1000

        return ServiceStatus(
            name="redis",
            status="healthy",
            latency_ms=round(latency, 2),
            last_check=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        return ServiceStatus(
            name="redis",
            status="unhealthy",
            last_check=datetime.utcnow().isoformat(),
            message=str(e),
        )


async def check_vectorstore_health() -> ServiceStatus:
    """Check vector store health."""
    import time
    start = time.time()

    try:
        from backend.services.vectorstore import get_vector_store

        vector_store = get_vector_store()
        if not vector_store:
            return ServiceStatus(
                name="vectorstore",
                status="unknown",
                message="Vector store not initialized",
            )

        # Try a simple health check
        health = await vector_store.health_check() if hasattr(vector_store, 'health_check') else True
        latency = (time.time() - start) * 1000

        return ServiceStatus(
            name="vectorstore",
            status="healthy" if health else "degraded",
            latency_ms=round(latency, 2),
            last_check=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        return ServiceStatus(
            name="vectorstore",
            status="unhealthy",
            last_check=datetime.utcnow().isoformat(),
            message=str(e),
        )


async def check_llm_health() -> ServiceStatus:
    """Check LLM provider health."""
    try:
        from backend.services.llm_router import get_llm_router

        router = get_llm_router()
        models = router.get_available_models()

        return ServiceStatus(
            name="llm",
            status="healthy" if models else "degraded",
            last_check=datetime.utcnow().isoformat(),
            details={"available_models": len(models)},
        )
    except Exception as e:
        return ServiceStatus(
            name="llm",
            status="unhealthy",
            last_check=datetime.utcnow().isoformat(),
            message=str(e),
        )


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/health/all", response_model=HealthCheckResponse)
async def comprehensive_health_check(
    _user: AdminUser,
):
    """
    Perform comprehensive health check across all services.

    Returns status of:
    - Database
    - Redis (if configured)
    - Vector store
    - LLM provider
    - Task queue
    - Distributed processor
    """
    import time
    from backend.core.config import settings

    start_time = datetime.utcnow()

    # Check all services in parallel
    services = []

    # Database health
    services.append(await check_database_health())

    # Redis health
    services.append(await check_redis_health())

    # Vector store health
    services.append(await check_vectorstore_health())

    # LLM health
    services.append(await check_llm_health())

    # Calculate summary
    summary = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
    for service in services:
        summary[service.status] = summary.get(service.status, 0) + 1

    # Determine overall status
    if summary["unhealthy"] > 0:
        overall_status = "unhealthy"
    elif summary["degraded"] > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthCheckResponse(
        status=overall_status,
        timestamp=start_time.isoformat(),
        version=getattr(settings, "VERSION", "unknown"),
        services=services,
        summary=summary,
    )


@router.get("/task-queue/status", response_model=TaskQueueStatus)
async def get_task_queue_status(
    _user: AdminUser,
):
    """
    Get status of the background task queue.

    Returns information about:
    - Pending, running, and completed tasks
    - Worker status
    - Queue performance metrics
    """
    try:
        from backend.services.task_queue import get_task_queue

        queue = get_task_queue()

        if not queue:
            return TaskQueueStatus(
                status="not_initialized",
                message="Task queue not available",
            )

        # Get queue stats
        stats = await queue.get_stats() if hasattr(queue, 'get_stats') else {}

        return TaskQueueStatus(
            status="healthy",
            pending_tasks=stats.get("pending", 0),
            running_tasks=stats.get("running", 0),
            completed_tasks=stats.get("completed", 0),
            failed_tasks=stats.get("failed", 0),
            workers=stats.get("workers", 0),
            queue_size=stats.get("queue_size", 0),
            oldest_pending_age_seconds=stats.get("oldest_pending_age"),
            avg_task_duration_seconds=stats.get("avg_duration"),
            tasks_per_minute=stats.get("throughput"),
            memory_usage_mb=stats.get("memory_mb"),
        )

    except Exception as e:
        logger.error("Failed to get task queue status", error=str(e))
        return TaskQueueStatus(
            status="error",
            message=str(e),
        )


@router.get("/distributed-processor/status", response_model=DistributedProcessorStatus)
async def get_distributed_processor_status(
    _user: AdminUser,
):
    """
    Get status of the distributed processing system (Ray).

    Returns information about:
    - Ray cluster availability
    - Node and resource counts
    - Active and pending jobs
    """
    try:
        from backend.services.distributed_processor import get_distributed_processor

        processor = get_distributed_processor()

        if not processor:
            return DistributedProcessorStatus(
                status="not_initialized",
                ray_available=False,
            )

        # Check Ray availability
        try:
            import ray
            ray_available = True
            ray_connected = ray.is_initialized()
        except ImportError:
            ray_available = False
            ray_connected = False

        # Get cluster info if connected
        cluster_info = {}
        if ray_connected:
            try:
                resources = ray.cluster_resources()
                cluster_info = {
                    "nodes": int(resources.get("node:__internal_head__", 0)) + len(ray.nodes()),
                    "cpus": int(resources.get("CPU", 0)),
                    "gpus": int(resources.get("GPU", 0)),
                    "memory_gb": resources.get("memory", 0) / (1024 ** 3),
                    "object_store_gb": resources.get("object_store_memory", 0) / (1024 ** 3),
                }
            except Exception:
                pass

        # Get job stats
        stats = await processor.get_stats() if hasattr(processor, 'get_stats') else {}

        return DistributedProcessorStatus(
            status="healthy" if ray_connected else ("degraded" if ray_available else "unavailable"),
            ray_available=ray_available,
            ray_connected=ray_connected,
            cluster_nodes=cluster_info.get("nodes", 0),
            cluster_cpus=cluster_info.get("cpus", 0),
            cluster_gpus=cluster_info.get("gpus", 0),
            active_jobs=stats.get("active", 0),
            pending_jobs=stats.get("pending", 0),
            completed_jobs=stats.get("completed", 0),
            failed_jobs=stats.get("failed", 0),
            memory_usage_gb=cluster_info.get("memory_gb"),
            object_store_usage_gb=cluster_info.get("object_store_gb"),
        )

    except Exception as e:
        logger.error("Failed to get distributed processor status", error=str(e))
        return DistributedProcessorStatus(
            status="error",
            ray_available=False,
        )


# =============================================================================
# Ray Cluster Control Endpoints
# =============================================================================

class RayControlResponse(BaseModel):
    """Response for Ray control operations."""
    success: bool
    message: str
    ray_connected: bool = False
    cluster_info: Optional[Dict[str, Any]] = None


@router.get("/ray/status")
async def get_ray_status(_user: AdminUser) -> Dict[str, Any]:
    """
    Get detailed Ray cluster status.

    Returns comprehensive information about the Ray cluster including:
    - Connection status
    - Cluster resources (CPUs, GPUs, memory)
    - Node information
    - Active workers
    """
    try:
        import ray

        ray_available = True
        ray_connected = ray.is_initialized()

        if not ray_connected:
            return {
                "status": "disconnected",
                "ray_available": ray_available,
                "ray_connected": False,
                "message": "Ray is installed but not initialized",
            }

        # Get cluster resources
        resources = ray.cluster_resources()
        available = ray.available_resources()
        nodes = ray.nodes()

        # Count active nodes
        active_nodes = sum(1 for n in nodes if n.get("Alive", False))

        return {
            "status": "connected",
            "ray_available": ray_available,
            "ray_connected": True,
            "cluster_resources": {
                "total_cpus": int(resources.get("CPU", 0)),
                "available_cpus": int(available.get("CPU", 0)),
                "total_gpus": int(resources.get("GPU", 0)),
                "available_gpus": int(available.get("GPU", 0)),
                "total_memory_gb": round(resources.get("memory", 0) / (1024 ** 3), 2),
                "available_memory_gb": round(available.get("memory", 0) / (1024 ** 3), 2),
                "object_store_memory_gb": round(resources.get("object_store_memory", 0) / (1024 ** 3), 2),
            },
            "nodes": {
                "total": len(nodes),
                "active": active_nodes,
            },
            "node_details": [
                {
                    "node_id": n.get("NodeID", "")[:8],
                    "alive": n.get("Alive", False),
                    "resources": n.get("Resources", {}),
                }
                for n in nodes
            ],
        }

    except ImportError:
        return {
            "status": "unavailable",
            "ray_available": False,
            "ray_connected": False,
            "message": "Ray is not installed",
        }
    except Exception as e:
        logger.error("Failed to get Ray status", error=str(e))
        return {
            "status": "error",
            "ray_available": True,
            "ray_connected": False,
            "message": str(e),
        }


@router.post("/ray/start", response_model=RayControlResponse)
async def start_ray(_user: AdminUser) -> RayControlResponse:
    """
    Start Ray cluster.

    Initializes Ray with settings from the Admin UI configuration.
    If Ray is already running, returns success without reinitializing.
    """
    try:
        import ray

        if ray.is_initialized():
            resources = ray.cluster_resources()
            return RayControlResponse(
                success=True,
                message="Ray is already running",
                ray_connected=True,
                cluster_info={
                    "cpus": int(resources.get("CPU", 0)),
                    "gpus": int(resources.get("GPU", 0)),
                    "memory_gb": round(resources.get("memory", 0) / (1024 ** 3), 2),
                },
            )

        # Initialize Ray with settings
        from backend.ray_workers.config import init_ray_async
        success = await init_ray_async(cleanup_stale=True, init_timeout=30.0)

        if success:
            resources = ray.cluster_resources()
            return RayControlResponse(
                success=True,
                message="Ray cluster started successfully",
                ray_connected=True,
                cluster_info={
                    "cpus": int(resources.get("CPU", 0)),
                    "gpus": int(resources.get("GPU", 0)),
                    "memory_gb": round(resources.get("memory", 0) / (1024 ** 3), 2),
                },
            )
        else:
            return RayControlResponse(
                success=False,
                message="Failed to initialize Ray cluster",
                ray_connected=False,
            )

    except ImportError:
        return RayControlResponse(
            success=False,
            message="Ray is not installed",
            ray_connected=False,
        )
    except Exception as e:
        logger.error("Failed to start Ray", error=str(e))
        return RayControlResponse(
            success=False,
            message=f"Error starting Ray: {str(e)}",
            ray_connected=False,
        )


@router.post("/ray/stop", response_model=RayControlResponse)
async def stop_ray(_user: AdminUser) -> RayControlResponse:
    """
    Stop Ray cluster.

    Gracefully shuts down the Ray cluster connection.
    Running tasks will be cancelled.
    """
    try:
        import ray

        if not ray.is_initialized():
            return RayControlResponse(
                success=True,
                message="Ray is not running",
                ray_connected=False,
            )

        from backend.ray_workers.config import shutdown_ray
        shutdown_ray(timeout=10.0)

        return RayControlResponse(
            success=True,
            message="Ray cluster stopped successfully",
            ray_connected=False,
        )

    except ImportError:
        return RayControlResponse(
            success=False,
            message="Ray is not installed",
            ray_connected=False,
        )
    except Exception as e:
        logger.error("Failed to stop Ray", error=str(e))
        return RayControlResponse(
            success=False,
            message=f"Error stopping Ray: {str(e)}",
            ray_connected=False,
        )


@router.post("/ray/restart", response_model=RayControlResponse)
async def restart_ray(_user: AdminUser) -> RayControlResponse:
    """
    Restart Ray cluster.

    Stops and then starts Ray with fresh configuration.
    Useful after changing Ray settings.
    """
    try:
        import ray

        # Stop if running
        if ray.is_initialized():
            from backend.ray_workers.config import shutdown_ray
            shutdown_ray(timeout=10.0)
            # Wait a moment for cleanup
            import asyncio
            await asyncio.sleep(2)

        # Start fresh
        from backend.ray_workers.config import init_ray_async
        success = await init_ray_async(cleanup_stale=True, init_timeout=30.0)

        if success:
            resources = ray.cluster_resources()
            return RayControlResponse(
                success=True,
                message="Ray cluster restarted successfully",
                ray_connected=True,
                cluster_info={
                    "cpus": int(resources.get("CPU", 0)),
                    "gpus": int(resources.get("GPU", 0)),
                    "memory_gb": round(resources.get("memory", 0) / (1024 ** 3), 2),
                },
            )
        else:
            return RayControlResponse(
                success=False,
                message="Failed to restart Ray cluster",
                ray_connected=False,
            )

    except ImportError:
        return RayControlResponse(
            success=False,
            message="Ray is not installed",
            ray_connected=False,
        )
    except Exception as e:
        logger.error("Failed to restart Ray", error=str(e))
        return RayControlResponse(
            success=False,
            message=f"Error restarting Ray: {str(e)}",
            ray_connected=False,
        )


@router.get("/analytics/usage", response_model=UsageAnalytics)
async def get_usage_analytics(
    _user: AdminUser,
    period: str = "last_24h",
):
    """
    Get usage analytics for the system.

    Periods: last_hour, last_24h, last_7d, last_30d
    """
    from backend.db.database import async_session_context
    from backend.db.models import Document, ChatSession, ChatMessage, LLMUsageLog, User
    from sqlalchemy import select, func

    # Calculate time window
    now = datetime.utcnow()
    if period == "last_hour":
        start_time = now - timedelta(hours=1)
    elif period == "last_24h":
        start_time = now - timedelta(hours=24)
    elif period == "last_7d":
        start_time = now - timedelta(days=7)
    elif period == "last_30d":
        start_time = now - timedelta(days=30)
    else:
        start_time = now - timedelta(hours=24)
        period = "last_24h"

    try:
        async with async_session_context() as session:
            # Total documents
            total_docs = await session.scalar(
                select(func.count(Document.id))
            ) or 0

            # Documents in period
            docs_in_period = await session.scalar(
                select(func.count(Document.id))
                .where(Document.created_at >= start_time)
            ) or 0

            # Total users
            total_users = await session.scalar(
                select(func.count(User.id))
            ) or 0

            # Active users in period
            active_users = await session.scalar(
                select(func.count(func.distinct(ChatSession.user_id)))
                .where(ChatSession.created_at >= start_time)
            ) or 0

            # Total queries (messages) in period
            total_queries = await session.scalar(
                select(func.count(ChatMessage.id))
                .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                .where(ChatSession.created_at >= start_time)
            ) or 0

            # LLM usage in period
            llm_usage = await session.execute(
                select(
                    func.count(LLMUsageLog.id),
                    func.sum(LLMUsageLog.total_tokens),
                    func.sum(LLMUsageLog.estimated_cost),
                )
                .where(LLMUsageLog.created_at >= start_time)
            )
            llm_row = llm_usage.one()
            llm_calls = llm_row[0] or 0
            llm_tokens = llm_row[1] or 0
            llm_cost = float(llm_row[2] or 0)

            return UsageAnalytics(
                period=period,
                total_queries=total_queries,
                total_documents=total_docs,
                total_users=total_users,
                active_users=active_users,
                llm_calls=llm_calls,
                llm_tokens_used=llm_tokens,
                llm_estimated_cost=round(llm_cost, 4),
                documents_processed=docs_in_period,
            )

    except Exception as e:
        logger.error("Failed to get usage analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics",
        )


@router.get("/services/status", response_model=AllServicesStatus)
async def get_all_services_status(
    _user: AdminUser,
):
    """
    Get status of all background services.

    Returns a comprehensive view of all system services including:
    - Database, Redis, Vector Store, LLM
    - Task Queue, Distributed Processor
    - Cache, Search, etc.
    """
    services = {}

    # Core services
    services["database"] = await check_database_health()
    services["redis"] = await check_redis_health()
    services["vectorstore"] = await check_vectorstore_health()
    services["llm"] = await check_llm_health()

    # Additional services
    try:
        from backend.services.search_cache import get_search_cache
        cache = get_search_cache()
        services["search_cache"] = ServiceStatus(
            name="search_cache",
            status="healthy" if cache else "unknown",
            last_check=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        services["search_cache"] = ServiceStatus(
            name="search_cache",
            status="unknown",
            message=str(e),
        )

    # Determine overall status
    statuses = [s.status for s in services.values()]
    if "unhealthy" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    return AllServicesStatus(
        overall_status=overall,
        timestamp=datetime.utcnow().isoformat(),
        services=services,
    )


@router.get("/memory")
async def get_memory_usage(
    _user: AdminUser,
):
    """
    Get current memory usage of the application.
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss_mb": round(memory_info.rss / (1024 * 1024), 2),
        "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
        "percent": round(process.memory_percent(), 2),
        "system_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
        "system_available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
    }


@router.get("/cpu")
async def get_cpu_usage(
    _user: AdminUser,
):
    """
    Get current CPU usage of the application.
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())

    return {
        "process_cpu_percent": process.cpu_percent(),
        "system_cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_count": psutil.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
    }


# =============================================================================
# Hardware Auto-Detection
# =============================================================================


class HardwareInfo(BaseModel):
    """Detected hardware information."""
    cpu_count: int
    total_memory_gb: float
    usable_memory_gb: float
    platform: str
    gpu_type: str
    has_cuda: bool
    has_mps: bool


class RecommendedSettingsResponse(BaseModel):
    """Hardware-based recommended settings."""
    hardware: HardwareInfo
    recommended: Dict[str, Any]
    current: Dict[str, Any]


@router.get("/hardware/recommended-settings")
async def get_hardware_recommended_settings(
    _user: AdminUser,
) -> RecommendedSettingsResponse:
    """
    Detect system hardware and return recommended settings.

    Analyzes CPU, memory, and GPU capabilities to compute optimal values
    for Ray, parallel processing, and server configuration settings.
    """
    from backend.utils.platform import get_recommended_settings
    from backend.services.settings import get_settings_service

    result = get_recommended_settings()

    # Fetch current values for comparison
    settings_service = get_settings_service()
    current_keys = list(result["recommended"].keys())
    current_values = await settings_service.get_settings_batch(current_keys)

    return RecommendedSettingsResponse(
        hardware=HardwareInfo(**result["hardware"]),
        recommended=result["recommended"],
        current=current_values,
    )
