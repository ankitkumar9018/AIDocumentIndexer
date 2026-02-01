"""
AIDocumentIndexer - Ray Module
==============================

Distributed processing with Ray for document processing,
embedding generation, and batch operations.
"""

from backend.ray_workers.config import (
    init_ray,
    init_ray_async,
    shutdown_ray,
    get_ray_resources,
    get_ray_nodes,
    ray_task,
    wait_for_tasks,
    get_results,
    ray_config,
)

# VLM Worker for distributed vision-language model processing
try:
    from backend.services.distributed_processor import VLMWorker
except ImportError:
    VLMWorker = None  # type: ignore

__all__ = [
    "init_ray",
    "init_ray_async",
    "shutdown_ray",
    "get_ray_resources",
    "get_ray_nodes",
    "ray_task",
    "wait_for_tasks",
    "get_results",
    "ray_config",
    "VLMWorker",
]
