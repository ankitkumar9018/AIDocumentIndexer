"""
AIDocumentIndexer - Ray Cluster Configuration
==============================================

This module handles Ray cluster initialization and configuration
for distributed document processing.
"""

import os
from typing import Optional

import ray
import structlog

logger = structlog.get_logger(__name__)


class RayConfig:
    """Ray cluster configuration."""

    def __init__(self):
        self.address = os.getenv("RAY_ADDRESS", "auto")
        self.num_cpus = self._get_int_env("RAY_NUM_CPUS")
        self.num_gpus = self._get_int_env("RAY_NUM_GPUS", 0)
        self.object_store_memory = self._get_int_env("RAY_OBJECT_STORE_MEMORY")
        self.dashboard_port = self._get_int_env("RAY_DASHBOARD_PORT", 8265)

    @staticmethod
    def _get_int_env(key: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default


# Global Ray configuration
ray_config = RayConfig()


def init_ray() -> None:
    """
    Initialize Ray cluster connection.

    This connects to an existing Ray cluster or starts a local one
    based on the RAY_ADDRESS environment variable.
    """
    if ray.is_initialized():
        logger.info("Ray already initialized")
        return

    try:
        init_kwargs = {
            "address": ray_config.address if ray_config.address != "auto" else None,
            "ignore_reinit_error": True,
            "logging_level": "warning",
        }

        # Add resource constraints if specified
        if ray_config.num_cpus is not None:
            init_kwargs["num_cpus"] = ray_config.num_cpus

        if ray_config.num_gpus is not None:
            init_kwargs["num_gpus"] = ray_config.num_gpus

        if ray_config.object_store_memory is not None:
            init_kwargs["object_store_memory"] = ray_config.object_store_memory

        ray.init(**init_kwargs)

        logger.info(
            "Ray initialized",
            address=ray_config.address,
            num_cpus=ray_config.num_cpus,
            num_gpus=ray_config.num_gpus,
            dashboard_port=ray_config.dashboard_port,
        )

    except Exception as e:
        logger.error("Failed to initialize Ray", error=str(e))
        raise


def shutdown_ray() -> None:
    """Shutdown Ray cluster connection."""
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")


def get_ray_resources() -> dict:
    """
    Get current Ray cluster resources.

    Returns:
        dict: Available resources in the cluster
    """
    if not ray.is_initialized():
        return {}

    return ray.cluster_resources()


def get_ray_nodes() -> list:
    """
    Get information about Ray cluster nodes.

    Returns:
        list: Information about each node in the cluster
    """
    if not ray.is_initialized():
        return []

    return ray.nodes()


# =============================================================================
# Ray Remote Task Decorators
# =============================================================================

def ray_task(
    num_cpus: int = 1,
    num_gpus: int = 0,
    max_retries: int = 3,
    retry_exceptions: bool = True,
):
    """
    Decorator to create a Ray remote task with common settings.

    Args:
        num_cpus: Number of CPUs required
        num_gpus: Number of GPUs required
        max_retries: Maximum number of retries on failure
        retry_exceptions: Whether to retry on exceptions

    Returns:
        Decorated function as Ray remote task
    """
    def decorator(func):
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )(func)

    return decorator


# =============================================================================
# Utility Functions
# =============================================================================

def wait_for_tasks(
    task_refs: list,
    num_returns: Optional[int] = None,
    timeout: Optional[float] = None,
) -> tuple:
    """
    Wait for Ray tasks to complete.

    Args:
        task_refs: List of Ray object references
        num_returns: Number of tasks to wait for (default: all)
        timeout: Maximum time to wait in seconds

    Returns:
        tuple: (ready_refs, not_ready_refs)
    """
    if num_returns is None:
        num_returns = len(task_refs)

    return ray.wait(
        task_refs,
        num_returns=num_returns,
        timeout=timeout,
    )


def get_results(task_refs: list) -> list:
    """
    Get results from Ray tasks.

    Args:
        task_refs: List of Ray object references

    Returns:
        list: Results from the tasks
    """
    return ray.get(task_refs)
