"""
AIDocumentIndexer - Celery Task Queue
======================================

Provides asynchronous task processing for:
- Document ingestion and processing
- Batch document uploads
- OCR processing
- Embedding generation
- Background reprocessing

Settings-aware: Respects queue.celery_enabled setting.
When disabled, falls back to synchronous processing.
"""

import os
from typing import Any, Dict, Optional

from celery import Celery
from celery.result import AsyncResult
import structlog

from backend.services.redis_client import is_redis_enabled_sync

logger = structlog.get_logger(__name__)

# Celery configuration from environment
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/0"))

# Create Celery app (always created, but may not be used if disabled)
celery_app = Celery(
    "aidocumentindexer",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["backend.tasks.document_tasks"],
)

# =============================================================================
# Priority Queue Configuration
# =============================================================================
# Queue Priority Levels:
#   - critical: User-facing requests (chat, search) - MUST be fast
#   - high: Interactive features (audio preview, quick queries)
#   - default: Standard document processing
#   - batch: Bulk uploads and batch operations
#   - background: KG extraction, analytics, non-urgent tasks

QUEUE_PRIORITIES = {
    "critical": 10,   # Highest priority - user chat/search
    "high": 7,        # High priority - interactive features
    "default": 5,     # Normal priority - document processing
    "batch": 3,       # Low priority - bulk uploads
    "background": 1,  # Lowest priority - KG extraction, analytics
}

# Define all queues with their priorities
CELERY_QUEUES = {
    "critical": {"exchange": "critical", "routing_key": "critical"},
    "high": {"exchange": "high", "routing_key": "high"},
    "default": {"exchange": "default", "routing_key": "default"},
    "batch": {"exchange": "batch", "routing_key": "batch"},
    "background": {"exchange": "background", "routing_key": "background"},
    # Legacy queues for backwards compatibility
    "documents": {"exchange": "default", "routing_key": "documents"},
    "ocr": {"exchange": "default", "routing_key": "ocr"},
    "embeddings": {"exchange": "high", "routing_key": "embeddings"},
}

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,  # Re-queue if worker dies
    worker_prefetch_multiplier=1,  # One task at a time per worker

    # Result backend
    result_expires=60 * 60 * 24,  # Results expire after 24 hours

    # Priority queue configuration
    task_default_queue="default",
    task_default_priority=5,
    broker_transport_options={
        "priority_steps": list(range(11)),  # 0-10 priority levels
        "sep": ":",
        "queue_order_strategy": "priority",
    },

    # Task routes with priority queues
    task_routes={
        # Critical priority - user-facing, must be instant
        "backend.tasks.chat_tasks.*": {"queue": "critical"},
        "backend.tasks.search_tasks.*": {"queue": "critical"},

        # High priority - interactive features
        "backend.tasks.document_tasks.embedding_task": {"queue": "high"},
        "backend.tasks.audio_tasks.generate_preview_task": {"queue": "high"},

        # Default priority - standard processing
        "backend.tasks.document_tasks.process_document_task": {"queue": "default"},
        "backend.tasks.document_tasks.ocr_task": {"queue": "default"},
        "backend.tasks.document_tasks.reprocess_document_task": {"queue": "default"},

        # Batch priority - bulk operations
        "backend.tasks.document_tasks.process_batch_task": {"queue": "batch"},
        "backend.tasks.document_tasks.process_bulk_upload_task": {"queue": "batch"},

        # Background priority - non-urgent tasks
        "backend.tasks.kg_tasks.*": {"queue": "background"},
        "backend.tasks.analytics_tasks.*": {"queue": "background"},
        "backend.tasks.document_tasks.extract_kg_task": {"queue": "background"},
    },

    # Rate limiting per queue
    task_annotations={
        # Critical tasks - no rate limit, must be fast
        "backend.tasks.chat_tasks.*": {
            "rate_limit": None,
        },
        # Document processing - moderate rate
        "backend.tasks.document_tasks.process_document_task": {
            "rate_limit": "30/m",  # 30 documents per minute (increased)
        },
        # Embedding tasks - higher rate for parallel processing
        "backend.tasks.document_tasks.embedding_task": {
            "rate_limit": "200/m",  # 200 embedding batches per minute
        },
        # Batch tasks - unlimited, managed by concurrency
        "backend.tasks.document_tasks.process_batch_task": {
            "rate_limit": None,
        },
        # Background tasks - lower rate to not impact user requests
        "backend.tasks.kg_tasks.*": {
            "rate_limit": "10/m",
        },
    },

    # Retry policy
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,

    # Worker concurrency per queue (for autoscaling)
    worker_concurrency=4,  # Default workers per process
)


# =============================================================================
# Task Status Management
# =============================================================================

class TaskStatus:
    """Task status constants."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    RETRY = "RETRY"


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a Celery task."""
    result = AsyncResult(task_id, app=celery_app)

    status = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
    }

    if result.ready():
        if result.successful():
            status["result"] = result.result
        else:
            status["error"] = str(result.result) if result.result else "Unknown error"

    # Get task metadata if available
    if hasattr(result, "info") and result.info:
        if isinstance(result.info, dict):
            status["progress"] = result.info.get("progress", 0)
            status["current"] = result.info.get("current", 0)
            status["total"] = result.info.get("total", 0)
            status["message"] = result.info.get("message", "")

    return status


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """Revoke/cancel a pending task."""
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        logger.info("Task revoked", task_id=task_id, terminate=terminate)
        return True
    except Exception as e:
        logger.error("Failed to revoke task", task_id=task_id, error=str(e))
        return False


def get_active_tasks() -> Dict[str, Any]:
    """Get list of active tasks across all workers."""
    try:
        inspect = celery_app.control.inspect()
        return {
            "active": inspect.active() or {},
            "scheduled": inspect.scheduled() or {},
            "reserved": inspect.reserved() or {},
        }
    except Exception as e:
        logger.error("Failed to inspect tasks", error=str(e))
        return {"active": {}, "scheduled": {}, "reserved": {}}


def get_queue_length(queue_name: str = "celery") -> int:
    """Get the number of tasks in a queue."""
    try:
        with celery_app.connection_or_acquire() as conn:
            return conn.default_channel.queue_declare(
                queue=queue_name, passive=True
            ).message_count
    except Exception as e:
        logger.debug("Failed to get queue length", queue=queue_name, error=str(e))
        return 0


# =============================================================================
# Helper Functions
# =============================================================================

def is_celery_enabled() -> bool:
    """
    Check if Celery is enabled in settings.

    Uses synchronous check - suitable for task submission decisions.
    """
    return is_redis_enabled_sync()


async def is_celery_enabled_async() -> bool:
    """
    Async check if Celery is enabled in settings.

    For use in async contexts where we can access the database.
    """
    from backend.services.redis_client import is_redis_enabled
    return await is_redis_enabled()


def is_celery_available() -> bool:
    """Check if Celery/Redis is available and enabled."""
    # First check if enabled in settings
    if not is_celery_enabled():
        return False

    try:
        celery_app.control.ping(timeout=1)
        return True
    except Exception as e:
        logger.debug("Celery ping failed", error=str(e))
        return False


def get_worker_stats() -> Dict[str, Any]:
    """Get statistics about Celery workers."""
    # Check if enabled first
    if not is_celery_enabled():
        return {
            "workers": [],
            "count": 0,
            "stats": {},
            "enabled": False,
            "message": "Celery is disabled in settings"
        }

    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats() or {}
        return {
            "workers": list(stats.keys()),
            "count": len(stats),
            "stats": stats,
            "enabled": True,
        }
    except Exception as e:
        logger.error("Failed to get worker stats", error=str(e))
        return {"workers": [], "count": 0, "stats": {}, "enabled": True}


def submit_task_or_run_sync(task_func, *args, **kwargs):
    """
    Submit a Celery task if enabled, otherwise run synchronously.

    Args:
        task_func: The Celery task to run
        *args: Task arguments
        **kwargs: Task keyword arguments

    Returns:
        AsyncResult if Celery enabled, or direct result if sync
    """
    if is_celery_enabled():
        return task_func.delay(*args, **kwargs)
    else:
        # Run synchronously
        logger.debug("Celery disabled, running task synchronously")
        return task_func(*args, **kwargs)


# =============================================================================
# Priority Queue Submission Helpers
# =============================================================================

def submit_critical_task(task_func, *args, **kwargs):
    """
    Submit a task to the critical (highest priority) queue.
    Use for user-facing requests like chat and search.
    """
    if is_celery_enabled():
        return task_func.apply_async(
            args=args,
            kwargs=kwargs,
            queue="critical",
            priority=QUEUE_PRIORITIES["critical"],
        )
    else:
        return task_func(*args, **kwargs)


def submit_high_priority_task(task_func, *args, **kwargs):
    """
    Submit a task to the high priority queue.
    Use for interactive features like audio preview.
    """
    if is_celery_enabled():
        return task_func.apply_async(
            args=args,
            kwargs=kwargs,
            queue="high",
            priority=QUEUE_PRIORITIES["high"],
        )
    else:
        return task_func(*args, **kwargs)


def submit_default_task(task_func, *args, **kwargs):
    """
    Submit a task to the default priority queue.
    Use for standard document processing.
    """
    if is_celery_enabled():
        return task_func.apply_async(
            args=args,
            kwargs=kwargs,
            queue="default",
            priority=QUEUE_PRIORITIES["default"],
        )
    else:
        return task_func(*args, **kwargs)


def submit_batch_task(task_func, *args, **kwargs):
    """
    Submit a task to the batch (low priority) queue.
    Use for bulk uploads and batch operations.
    """
    if is_celery_enabled():
        return task_func.apply_async(
            args=args,
            kwargs=kwargs,
            queue="batch",
            priority=QUEUE_PRIORITIES["batch"],
        )
    else:
        return task_func(*args, **kwargs)


def submit_background_task(task_func, *args, **kwargs):
    """
    Submit a task to the background (lowest priority) queue.
    Use for KG extraction, analytics, and non-urgent tasks.
    """
    if is_celery_enabled():
        return task_func.apply_async(
            args=args,
            kwargs=kwargs,
            queue="background",
            priority=QUEUE_PRIORITIES["background"],
        )
    else:
        return task_func(*args, **kwargs)


def submit_task_with_priority(task_func, priority: str = "default", *args, **kwargs):
    """
    Submit a task with a specific priority level.

    Args:
        task_func: The Celery task to run
        priority: Priority level ("critical", "high", "default", "batch", "background")
        *args: Task arguments
        **kwargs: Task keyword arguments

    Returns:
        AsyncResult if Celery enabled, or direct result if sync
    """
    if priority not in QUEUE_PRIORITIES:
        priority = "default"

    if is_celery_enabled():
        return task_func.apply_async(
            args=args,
            kwargs=kwargs,
            queue=priority,
            priority=QUEUE_PRIORITIES[priority],
        )
    else:
        return task_func(*args, **kwargs)


def get_all_queue_lengths() -> Dict[str, int]:
    """Get the number of tasks in all priority queues."""
    lengths = {}
    for queue_name in QUEUE_PRIORITIES.keys():
        lengths[queue_name] = get_queue_length(queue_name)
    return lengths


def get_queue_health() -> Dict[str, Any]:
    """
    Get health status of all queues.

    Returns queue lengths, worker status, and recommendations.
    """
    if not is_celery_enabled():
        return {
            "enabled": False,
            "message": "Celery is disabled in settings",
            "queues": {},
            "workers": {},
        }

    queue_lengths = get_all_queue_lengths()
    worker_stats = get_worker_stats()

    # Calculate health status
    health = "healthy"
    warnings = []

    # Check for queue backlogs
    if queue_lengths.get("critical", 0) > 10:
        health = "degraded"
        warnings.append("Critical queue has backlog - user requests may be delayed")
    if queue_lengths.get("high", 0) > 50:
        health = "degraded"
        warnings.append("High priority queue has backlog")
    if queue_lengths.get("default", 0) > 100:
        warnings.append("Default queue has significant backlog")
    if queue_lengths.get("batch", 0) > 1000:
        warnings.append("Batch queue has large backlog - bulk uploads may be slow")

    # Check worker availability
    if worker_stats.get("count", 0) == 0:
        health = "unhealthy"
        warnings.append("No Celery workers available")

    return {
        "enabled": True,
        "health": health,
        "warnings": warnings,
        "queues": queue_lengths,
        "workers": worker_stats,
        "priorities": QUEUE_PRIORITIES,
    }
