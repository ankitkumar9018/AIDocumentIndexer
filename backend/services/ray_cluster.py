"""
AIDocumentIndexer - Ray Cluster Management
============================================

Provides Ray cluster initialization, health monitoring, and resource management.

Ray is used alongside Celery for heavy ML workloads:
- Celery: Simple async tasks (uploads, notifications, cleanup)
- Ray: Heavy ML workloads (embeddings, KG extraction, VLM processing)

Features:
- Auto-detection of Ray cluster
- Graceful fallback when Ray unavailable
- Health checks and cluster metrics
- Resource-aware actor pool management

Based on:
- Ray 2.x patterns (https://docs.ray.io/en/latest/)
- 10 Ray Cluster Patterns for 2025

Usage:
    from backend.services.ray_cluster import get_ray_manager

    manager = await get_ray_manager()
    if manager.is_available:
        result = await manager.submit_task(my_function, *args)
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Type variable for generic return types
T = TypeVar('T')


# =============================================================================
# Configuration
# =============================================================================

class RayStatus(str, Enum):
    """Ray cluster connection status."""
    NOT_INSTALLED = "not_installed"
    NOT_CONNECTED = "not_connected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class RayConfig:
    """Configuration for Ray cluster."""
    # Connection
    address: str = "auto"  # 'auto' for local, or 'ray://host:port'
    namespace: str = "aidocindexer"

    # Resources
    num_cpus: Optional[int] = None  # None = auto-detect
    num_gpus: Optional[int] = None  # None = auto-detect
    object_store_memory: Optional[int] = None  # bytes

    # Worker pools
    embedding_workers: int = 4
    kg_workers: int = 2
    vlm_workers: int = 2

    # Timeouts
    init_timeout_seconds: float = 30.0
    task_timeout_seconds: float = 300.0

    # Fault tolerance
    max_restarts: int = 3
    max_task_retries: int = 2

    # Performance
    enable_task_events: bool = False  # Disable for performance


@dataclass
class ClusterMetrics:
    """Ray cluster metrics."""
    status: RayStatus
    num_nodes: int = 0
    num_cpus: int = 0
    num_gpus: int = 0
    available_cpus: float = 0.0
    available_gpus: float = 0.0
    object_store_memory_bytes: int = 0
    object_store_available_bytes: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    error: Optional[str] = None
    last_updated: float = field(default_factory=time.time)


# =============================================================================
# Ray Manager
# =============================================================================

class RayManager:
    """
    Manages Ray cluster connection and provides distributed computing utilities.

    Features:
    - Lazy initialization (only connects when needed)
    - Health monitoring
    - Actor pool management
    - Graceful degradation when Ray unavailable
    """

    def __init__(self, config: Optional[RayConfig] = None):
        self.config = config or RayConfig()
        self._ray = None  # Lazy import
        self._status = RayStatus.NOT_INSTALLED
        self._metrics = ClusterMetrics(status=RayStatus.NOT_INSTALLED)
        self._initialized = False
        self._actor_pools: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def is_available(self) -> bool:
        """Check if Ray is available and connected."""
        return self._status == RayStatus.CONNECTED

    @property
    def status(self) -> RayStatus:
        """Get current Ray status."""
        return self._status

    @property
    def metrics(self) -> ClusterMetrics:
        """Get current cluster metrics."""
        return self._metrics

    async def initialize(self) -> bool:
        """
        Initialize Ray connection.

        Returns:
            True if Ray is available and connected
        """
        if self._initialized:
            return self.is_available

        async with self._lock:
            if self._initialized:
                return self.is_available

            # Check if Ray is installed
            try:
                import ray
                self._ray = ray
                self._status = RayStatus.NOT_CONNECTED
            except ImportError:
                logger.warning("Ray not installed. Install with: pip install ray[default]")
                self._status = RayStatus.NOT_INSTALLED
                self._metrics = ClusterMetrics(status=RayStatus.NOT_INSTALLED)
                self._initialized = True
                return False

            # Try to connect
            try:
                self._status = RayStatus.CONNECTING
                logger.info("Connecting to Ray cluster", address=self.config.address)

                # Initialize Ray
                if not ray.is_initialized():
                    init_kwargs = {
                        "namespace": self.config.namespace,
                        "logging_level": "warning",
                        "configure_logging": False,
                    }

                    if self.config.address != "auto":
                        init_kwargs["address"] = self.config.address
                    if self.config.num_cpus is not None:
                        init_kwargs["num_cpus"] = self.config.num_cpus
                    if self.config.num_gpus is not None:
                        init_kwargs["num_gpus"] = self.config.num_gpus
                    if self.config.object_store_memory is not None:
                        init_kwargs["object_store_memory"] = self.config.object_store_memory

                    ray.init(**init_kwargs)

                self._status = RayStatus.CONNECTED
                await self._update_metrics()

                logger.info(
                    "Ray cluster connected",
                    nodes=self._metrics.num_nodes,
                    cpus=self._metrics.num_cpus,
                    gpus=self._metrics.num_gpus,
                )

                self._initialized = True
                return True

            except Exception as e:
                self._status = RayStatus.ERROR
                self._metrics = ClusterMetrics(
                    status=RayStatus.ERROR,
                    error=str(e),
                )
                logger.error("Failed to connect to Ray", error=str(e))
                self._initialized = True
                return False

    async def _update_metrics(self) -> None:
        """Update cluster metrics."""
        if not self.is_available or self._ray is None:
            return

        try:
            # Get cluster resources
            resources = self._ray.cluster_resources()
            available = self._ray.available_resources()

            # Get node info
            nodes = self._ray.nodes()
            alive_nodes = [n for n in nodes if n.get("Alive", False)]

            self._metrics = ClusterMetrics(
                status=RayStatus.CONNECTED,
                num_nodes=len(alive_nodes),
                num_cpus=int(resources.get("CPU", 0)),
                num_gpus=int(resources.get("GPU", 0)),
                available_cpus=available.get("CPU", 0),
                available_gpus=available.get("GPU", 0),
                object_store_memory_bytes=int(resources.get("object_store_memory", 0)),
            )

        except Exception as e:
            logger.warning("Failed to update Ray metrics", error=str(e))

    async def submit_task(
        self,
        func: Callable[..., T],
        *args,
        num_cpus: int = 1,
        num_gpus: float = 0,
        max_retries: int = 2,
        **kwargs,
    ) -> T:
        """
        Submit a task to Ray for distributed execution.

        Args:
            func: Function to execute
            *args: Function arguments
            num_cpus: CPU requirement
            num_gpus: GPU requirement
            max_retries: Max retry count
            **kwargs: Function keyword arguments

        Returns:
            Task result
        """
        if not await self.initialize():
            # Fallback: execute locally
            logger.debug("Ray unavailable, executing locally")
            return func(*args, **kwargs)

        try:
            # Create remote function
            remote_func = self._ray.remote(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_retries=max_retries,
            )(func)

            # Submit and wait
            result_ref = remote_func.remote(*args, **kwargs)
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                self._ray.get,
                result_ref,
            )

            return result

        except Exception as e:
            logger.error("Ray task failed", error=str(e))
            # Fallback to local execution
            return func(*args, **kwargs)

    async def submit_batch(
        self,
        func: Callable[..., T],
        batch_args: List[tuple],
        num_cpus: int = 1,
        num_gpus: float = 0,
    ) -> List[T]:
        """
        Submit a batch of tasks in parallel.

        Args:
            func: Function to execute
            batch_args: List of (args, kwargs) tuples
            num_cpus: CPU requirement per task
            num_gpus: GPU requirement per task

        Returns:
            List of results
        """
        if not await self.initialize():
            # Fallback: execute sequentially
            return [func(*args, **kwargs) for args, kwargs in batch_args]

        try:
            # Create remote function
            remote_func = self._ray.remote(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )(func)

            # Submit all tasks
            refs = [remote_func.remote(*args, **kwargs) for args, kwargs in batch_args]

            # Wait for all
            results = await asyncio.get_running_loop().run_in_executor(
                None,
                self._ray.get,
                refs,
            )

            return results

        except Exception as e:
            logger.error("Ray batch failed", error=str(e))
            return [func(*args, **kwargs) for args, kwargs in batch_args]

    async def get_actor_pool(
        self,
        name: str,
        actor_class: type,
        pool_size: int = 4,
        *init_args,
        **init_kwargs,
    ) -> Optional[Any]:
        """
        Get or create an actor pool.

        Args:
            name: Pool identifier
            actor_class: Actor class to instantiate
            pool_size: Number of actors in pool
            *init_args: Actor constructor args
            **init_kwargs: Actor constructor kwargs

        Returns:
            Actor pool or None if unavailable
        """
        if not await self.initialize():
            return None

        if name in self._actor_pools:
            return self._actor_pools[name]

        try:
            from ray.util.actor_pool import ActorPool

            # Create remote actor class
            remote_class = self._ray.remote(
                max_restarts=self.config.max_restarts,
                max_task_retries=self.config.max_task_retries,
            )(actor_class)

            # Create pool
            actors = [remote_class.remote(*init_args, **init_kwargs) for _ in range(pool_size)]
            pool = ActorPool(actors)

            self._actor_pools[name] = pool
            logger.info(f"Created actor pool: {name}", size=pool_size)

            return pool

        except Exception as e:
            logger.error(f"Failed to create actor pool: {name}", error=str(e))
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Ray cluster.

        Returns:
            Health status dictionary
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "ray_status": self._status.value,
                "error": self._metrics.error,
            }

        await self._update_metrics()

        return {
            "status": "healthy" if self._metrics.num_nodes > 0 else "degraded",
            "ray_status": self._status.value,
            "nodes": self._metrics.num_nodes,
            "cpus": {
                "total": self._metrics.num_cpus,
                "available": self._metrics.available_cpus,
            },
            "gpus": {
                "total": self._metrics.num_gpus,
                "available": self._metrics.available_gpus,
            },
            "actor_pools": list(self._actor_pools.keys()),
        }

    async def shutdown(self) -> None:
        """Shutdown Ray connection."""
        if self._ray is not None and self._ray.is_initialized():
            try:
                self._ray.shutdown()
                logger.info("Ray shutdown complete")
            except Exception as e:
                logger.warning("Error during Ray shutdown", error=str(e))

        self._status = RayStatus.NOT_CONNECTED
        self._initialized = False
        self._actor_pools.clear()


# =============================================================================
# Global Manager
# =============================================================================

_ray_manager: Optional[RayManager] = None
_manager_lock = asyncio.Lock()


async def get_ray_manager(config: Optional[RayConfig] = None) -> RayManager:
    """Get or create Ray manager singleton."""
    global _ray_manager

    if _ray_manager is not None:
        return _ray_manager

    async with _manager_lock:
        if _ray_manager is not None:
            return _ray_manager

        # Build config from settings
        if config is None:
            config = RayConfig(
                address=getattr(settings, 'RAY_ADDRESS', 'auto'),
                embedding_workers=getattr(settings, 'RAY_NUM_WORKERS', 8),
            )

        _ray_manager = RayManager(config)
        return _ray_manager


def get_ray_manager_sync(config: Optional[RayConfig] = None) -> RayManager:
    """Get Ray manager synchronously (doesn't initialize)."""
    global _ray_manager

    if _ray_manager is not None:
        return _ray_manager

    if config is None:
        config = RayConfig(
            address=getattr(settings, 'RAY_ADDRESS', 'auto'),
            embedding_workers=getattr(settings, 'RAY_NUM_WORKERS', 8),
        )

    _ray_manager = RayManager(config)
    return _ray_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RayStatus",
    "RayConfig",
    "ClusterMetrics",
    "RayManager",
    "get_ray_manager",
    "get_ray_manager_sync",
]
