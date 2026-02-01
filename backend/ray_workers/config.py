"""
AIDocumentIndexer - Ray Cluster Configuration
==============================================

This module handles Ray cluster initialization and configuration
for distributed document processing.

Includes safeguards to prevent stuck processes:
- Graceful shutdown with timeout
- Worker cleanup on initialization
- Signal handlers for clean termination
"""

import atexit
import os
import signal
import sys
from typing import Optional

import ray
import structlog

logger = structlog.get_logger(__name__)

# Track if we've registered cleanup handlers
_cleanup_registered = False


class RayConfig:
    """
    Ray cluster configuration.

    Settings are loaded from:
    1. Settings service (Admin UI configurable) - preferred
    2. Environment variables - fallback
    3. Defaults - last resort

    Settings keys:
    - processing.ray_enabled: Enable Ray (default: False)
    - processing.ray_address: Cluster address (default: auto)
    - processing.ray_num_cpus: Max CPUs (default: 4)
    - processing.ray_num_gpus: Max GPUs (default: 0)
    - processing.ray_memory_limit_gb: Memory limit (default: 8)
    - processing.ray_num_workers: Actor pool size (default: 8)
    """

    def __init__(self):
        # Initialize with env var defaults, updated async later
        self.enabled = os.getenv("RAY_ENABLED", "true").lower() in ("true", "1", "yes")
        # Empty or unset RAY_ADDRESS means start local cluster
        # "auto" means try to connect first, then start local if not found
        # Explicit address means connect to that cluster
        self.address = os.getenv("RAY_ADDRESS", "") or ""
        self.num_cpus = self._get_int_env("RAY_NUM_CPUS")
        self.num_gpus = self._get_int_env("RAY_NUM_GPUS", 0)
        self.object_store_memory = self._get_int_env("RAY_OBJECT_STORE_MEMORY")
        self.dashboard_port = self._get_int_env("RAY_DASHBOARD_PORT", 8265)
        self.memory_limit_gb = self._get_int_env("RAY_MEMORY_LIMIT_GB", 8)
        self.num_workers = self._get_int_env("RAY_NUM_WORKERS", 8)
        self._settings_loaded = False

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

    async def load_from_settings(self) -> None:
        """
        Load configuration from settings service.

        Call this before init_ray() to use Admin UI settings.
        Falls back to env vars if settings unavailable.
        """
        if self._settings_loaded:
            return

        try:
            from backend.services.settings import get_settings_service
            settings = get_settings_service()

            # Load each setting, keep env var value as fallback
            enabled = await settings.get_setting("processing.ray_enabled")
            if enabled is not None:
                self.enabled = enabled

            address = await settings.get_setting("processing.ray_address")
            if address:
                self.address = address

            num_cpus = await settings.get_setting("processing.ray_num_cpus")
            if num_cpus is not None:
                self.num_cpus = num_cpus if num_cpus > 0 else None

            num_gpus = await settings.get_setting("processing.ray_num_gpus")
            if num_gpus is not None:
                self.num_gpus = num_gpus

            memory_limit = await settings.get_setting("processing.ray_memory_limit_gb")
            if memory_limit is not None:
                self.memory_limit_gb = memory_limit

            num_workers = await settings.get_setting("processing.ray_num_workers")
            if num_workers is not None:
                self.num_workers = num_workers

            self._settings_loaded = True
            logger.info(
                "Ray config loaded from settings",
                enabled=self.enabled,
                address=self.address,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )

        except Exception as e:
            logger.debug("Could not load Ray settings, using env vars", error=str(e))
            self._settings_loaded = True


# Global Ray configuration
ray_config = RayConfig()


def _cleanup_stale_ray() -> None:
    """
    Clean up any stale Ray processes before initialization.

    This helps prevent stuck workers from previous runs.
    """
    try:
        # Check for stale Ray processes and clean up
        if ray.is_initialized():
            logger.info("Found existing Ray instance, shutting down first")
            ray.shutdown()
            import time
            time.sleep(1)  # Give processes time to clean up
    except Exception as e:
        logger.debug("Stale Ray cleanup error (safe to ignore)", error=str(e))


def _register_cleanup_handlers() -> None:
    """Register signal handlers and atexit for graceful cleanup."""
    global _cleanup_registered

    if _cleanup_registered:
        return

    def signal_handler(signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, initiating Ray shutdown")
        shutdown_ray(timeout=5.0)
        sys.exit(0)

    # Register signal handlers (only in main process)
    try:
        if os.getpid() == os.getppid() or True:  # Always register for safety
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
    except (ValueError, OSError) as e:
        # Signal handlers can't be set in some contexts (e.g., threads)
        logger.debug("Could not register signal handlers", error=str(e))

    # Register atexit handler for cleanup
    atexit.register(lambda: shutdown_ray(timeout=5.0))

    _cleanup_registered = True
    logger.debug("Ray cleanup handlers registered")


async def init_ray_async(
    cleanup_stale: bool = True,
    init_timeout: float = 30.0,
) -> bool:
    """
    Initialize Ray cluster connection with settings from Admin UI.

    Async version that loads configuration from settings service first.
    Use this version when calling from an async context.

    Args:
        cleanup_stale: Clean up any stale Ray processes first
        init_timeout: Timeout for initialization in seconds

    Returns:
        True if initialization succeeded, False otherwise
    """
    # Load settings from Admin UI
    await ray_config.load_from_settings()

    # Check if Ray is enabled
    if not ray_config.enabled:
        logger.info("Ray is disabled in settings, skipping initialization")
        return False

    # Call sync init
    return init_ray(cleanup_stale=cleanup_stale, init_timeout=init_timeout)


def init_ray(
    cleanup_stale: bool = True,
    init_timeout: float = 30.0,
) -> bool:
    """
    Initialize Ray cluster connection with safety features.

    This connects to an existing Ray cluster or starts a local one
    based on configuration from settings/environment variables.

    Args:
        cleanup_stale: Clean up any stale Ray processes first
        init_timeout: Timeout for initialization in seconds

    Returns:
        True if initialization succeeded, False otherwise
    """
    if ray.is_initialized():
        logger.info("Ray already initialized")
        return True

    # Check if Ray is enabled (if settings have been loaded)
    if ray_config._settings_loaded and not ray_config.enabled:
        logger.info("Ray is disabled in settings, skipping initialization")
        return False

    try:
        # Clean up stale processes first
        if cleanup_stale:
            _cleanup_stale_ray()

        init_kwargs = {
            "ignore_reinit_error": True,
            "logging_level": "warning",
            # Configure for better cleanup behavior
            "configure_logging": False,  # Prevent Ray from messing with logging
            "_temp_dir": os.path.join(os.path.expanduser("~"), ".ray_temp"),
            # Exclude large directories from runtime environment scanning
            "runtime_env": {
                "excludes": [
                    ".git/",
                    ".git/objects/",
                    "node_modules/",
                    "__pycache__/",
                    "*.pack",
                    "*.idx",
                    ".venv/",
                    "venv/",
                    "*.pyc",
                    "*.pyo",
                    "logs/",
                    "data/",
                    "*.db",
                    "*.sqlite",
                ],
            },
        }

        # Set address only if explicitly configured (not "auto")
        # When address is "auto" or not set, Ray will start a local cluster
        if ray_config.address and ray_config.address != "auto":
            init_kwargs["address"] = ray_config.address

        # Add resource constraints if specified
        if ray_config.num_cpus is not None:
            init_kwargs["num_cpus"] = ray_config.num_cpus

        if ray_config.num_gpus is not None:
            init_kwargs["num_gpus"] = ray_config.num_gpus

        if ray_config.object_store_memory is not None:
            init_kwargs["object_store_memory"] = ray_config.object_store_memory

        # Initialize with timeout protection
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ray.init, **init_kwargs)
            try:
                future.result(timeout=init_timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Ray initialization timed out after {init_timeout}s")
                return False

        # Register cleanup handlers after successful init
        _register_cleanup_handlers()

        logger.info(
            "Ray initialized",
            address=ray_config.address,
            num_cpus=ray_config.num_cpus,
            num_gpus=ray_config.num_gpus,
            dashboard_port=ray_config.dashboard_port,
        )
        return True

    except Exception as e:
        logger.error("Failed to initialize Ray", error=str(e))
        return False


def shutdown_ray(timeout: float = 10.0) -> None:
    """
    Shutdown Ray cluster connection gracefully.

    Args:
        timeout: Maximum time to wait for shutdown in seconds
    """
    if not ray.is_initialized():
        logger.debug("Ray not initialized, nothing to shutdown")
        return

    try:
        logger.info("Initiating Ray shutdown...")

        # Cancel any running tasks first
        try:
            # Get all running tasks and cancel them
            pass  # Ray doesn't have a direct "cancel all" but shutdown handles it
        except Exception:
            pass

        # Shutdown with timeout protection
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ray.shutdown)
            try:
                future.result(timeout=timeout)
                logger.info("Ray shutdown complete")
            except concurrent.futures.TimeoutError:
                logger.warning(f"Ray shutdown timed out after {timeout}s, forcing...")
                # Force kill any remaining Ray processes
                _force_kill_ray_processes()

    except Exception as e:
        logger.error("Ray shutdown error", error=str(e))
        _force_kill_ray_processes()


def _force_kill_ray_processes() -> None:
    """Force kill any remaining Ray processes (last resort)."""
    import subprocess

    try:
        # Kill Ray processes by name pattern
        if sys.platform == "darwin":  # macOS
            subprocess.run(
                ["pkill", "-9", "-f", "ray::"],
                capture_output=True,
                timeout=5,
            )
            subprocess.run(
                ["pkill", "-9", "-f", "raylet"],
                capture_output=True,
                timeout=5,
            )
        else:  # Linux
            subprocess.run(
                ["pkill", "-9", "-f", "ray::"],
                capture_output=True,
                timeout=5,
            )
        logger.warning("Force killed remaining Ray processes")
    except Exception as e:
        logger.debug("Force kill failed (may be nothing to kill)", error=str(e))


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


def get_results(task_refs: list, timeout: Optional[float] = None) -> list:
    """
    Get results from Ray tasks with timeout protection.

    Args:
        task_refs: List of Ray object references
        timeout: Timeout in seconds (default: RAY_TASK_TIMEOUT env var or 300s)

    Returns:
        list: Results from the tasks

    Raises:
        TimeoutError: If tasks exceed timeout
    """
    if timeout is None:
        timeout = float(os.getenv("RAY_TASK_TIMEOUT", "300"))

    try:
        return ray.get(task_refs, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        logger.warning(
            "Ray tasks timed out, cancelling",
            timeout=timeout,
            task_count=len(task_refs),
        )
        # Cancel pending tasks
        for ref in task_refs:
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass
        raise TimeoutError(f"Ray tasks exceeded {timeout}s timeout")
