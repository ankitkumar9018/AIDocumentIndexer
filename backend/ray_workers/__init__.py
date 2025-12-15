"""
AIDocumentIndexer - Ray Module
==============================

Distributed processing with Ray for document processing,
embedding generation, and batch operations.
"""

from backend.ray_workers.config import (
    init_ray,
    shutdown_ray,
    get_ray_resources,
    get_ray_nodes,
    ray_task,
    wait_for_tasks,
    get_results,
)

__all__ = [
    "init_ray",
    "shutdown_ray",
    "get_ray_resources",
    "get_ray_nodes",
    "ray_task",
    "wait_for_tasks",
    "get_results",
]
