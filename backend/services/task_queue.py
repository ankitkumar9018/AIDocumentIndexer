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

    # Task routes (optional - for scaling)
    task_routes={
        "backend.tasks.document_tasks.process_document_task": {"queue": "documents"},
        "backend.tasks.document_tasks.process_batch_task": {"queue": "batch"},
        "backend.tasks.document_tasks.ocr_task": {"queue": "ocr"},
        "backend.tasks.document_tasks.embedding_task": {"queue": "embeddings"},
    },

    # Rate limiting
    task_annotations={
        "backend.tasks.document_tasks.process_document_task": {
            "rate_limit": "10/m",  # 10 documents per minute
        },
        "backend.tasks.document_tasks.embedding_task": {
            "rate_limit": "100/m",  # 100 embedding batches per minute
        },
    },

    # Retry policy
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
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
    except Exception:
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
    except Exception:
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
