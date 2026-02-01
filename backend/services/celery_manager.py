"""
AIDocumentIndexer - Celery Worker Manager
==========================================

Manages Celery worker lifecycle:
- Auto-start on backend startup (if enabled in settings)
- Auto-stop on backend shutdown
- Manual start/stop via admin API

The worker status is controlled by the `queue.celery_enabled` database setting.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Track the Celery worker process
_celery_process: Optional[subprocess.Popen] = None
_celery_pid: Optional[int] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[2]


async def is_celery_enabled_in_settings() -> bool:
    """Check if Celery is enabled in database settings."""
    try:
        from backend.services.settings import get_settings_service
        settings = get_settings_service()
        enabled = await settings.get_setting("queue.celery_enabled")
        return bool(enabled)
    except Exception as e:
        logger.debug("Could not check Celery settings", error=str(e))
        return False


async def is_redis_available() -> bool:
    """Check if Redis is available and reachable."""
    try:
        from backend.services.redis_client import check_redis_connection
        status = await check_redis_connection()
        return status.get("connected", False)
    except Exception as e:
        logger.debug("Redis availability check failed", error=str(e))
        return False


def is_worker_running() -> bool:
    """Check if our managed Celery worker is running."""
    global _celery_process, _celery_pid

    if _celery_process is None:
        return False

    # Check if process is still alive
    if _celery_process.poll() is None:
        return True

    # Process has ended, clean up
    _celery_process = None
    _celery_pid = None
    return False


def get_worker_pid() -> Optional[int]:
    """Get the PID of the managed Celery worker, if running."""
    if is_worker_running():
        return _celery_pid
    return None


async def start_celery_worker_auto() -> bool:
    """
    Auto-start Celery worker on backend startup.

    Only starts if:
    1. queue.celery_enabled is True in settings
    2. Redis is available
    3. No worker is already running

    Returns:
        True if worker was started, False otherwise
    """
    global _celery_process, _celery_pid

    # Check if already running
    if is_worker_running():
        logger.info("Celery worker already running", pid=_celery_pid)
        return True

    # Check settings
    if not await is_celery_enabled_in_settings():
        logger.info("Celery disabled in settings (queue.celery_enabled=false)")
        return False

    # Check Redis
    if not await is_redis_available():
        logger.warning("Celery enabled but Redis not available - skipping worker start")
        return False

    # Start the worker
    return await _start_worker()


async def _start_worker() -> bool:
    """Internal: Start the Celery worker process."""
    global _celery_process, _celery_pid

    project_root = get_project_root()

    # Build the celery command
    # The Celery app is defined in backend.services.task_queue:celery_app
    # Use --pool=threads to avoid macOS fork crashes (CoreFoundation/NSUnarchiver)
    # See: https://github.com/celery/celery/discussions/9893
    celery_cmd = [
        sys.executable, "-m", "celery",
        "-A", "backend.services.task_queue:celery_app",
        "worker",
        "--loglevel=info",
        "--pool=threads",
        "--concurrency=4",
    ]

    # Create logs directory
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    celery_log = log_dir / "celery_worker.log"

    try:
        # Set environment for subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)

        # Ensure dev mode settings are passed through
        env.setdefault("DEV_MODE", "true")
        env.setdefault("LOCAL_MODE", "true")
        env.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
        env.setdefault("FASTEMBED_DISABLE_MODEL_SOURCE_CHECK", "1")

        # Start Celery worker as subprocess
        with open(celery_log, "a") as log_file:
            _celery_process = subprocess.Popen(
                celery_cmd,
                cwd=str(project_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )
        _celery_pid = _celery_process.pid

        # Wait a moment for worker to initialize
        await asyncio.sleep(2)

        # Verify it's still running
        if is_worker_running():
            logger.info("Celery worker started", pid=_celery_pid, log=str(celery_log))
            return True
        else:
            logger.error("Celery worker exited immediately after start - check logs", log=str(celery_log))
            return False

    except Exception as e:
        logger.error("Failed to start Celery worker", error=str(e))
        _celery_process = None
        _celery_pid = None
        return False


async def stop_celery_worker_auto() -> bool:
    """
    Stop the managed Celery worker.

    Called during backend shutdown.

    Returns:
        True if worker was stopped, False if no worker was running
    """
    global _celery_process, _celery_pid

    if not is_worker_running():
        logger.debug("No managed Celery worker to stop")
        return False

    try:
        # Graceful termination
        _celery_process.terminate()

        # Wait for graceful shutdown
        try:
            _celery_process.wait(timeout=10)
            logger.info("Celery worker terminated gracefully", pid=_celery_pid)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown takes too long
            _celery_process.kill()
            _celery_process.wait(timeout=5)
            logger.warning("Celery worker force-killed after timeout", pid=_celery_pid)

        _celery_process = None
        _celery_pid = None
        return True

    except Exception as e:
        logger.error("Error stopping Celery worker", error=str(e))
        _celery_process = None
        _celery_pid = None
        return False


async def restart_celery_worker() -> bool:
    """
    Restart the Celery worker.

    Useful after changing settings.

    Returns:
        True if worker was restarted successfully
    """
    await stop_celery_worker_auto()
    await asyncio.sleep(1)  # Brief pause between stop and start
    return await start_celery_worker_auto()


def get_worker_status() -> dict:
    """
    Get the current Celery worker status.

    Returns:
        Dict with running status, pid, etc.
    """
    running = is_worker_running()
    return {
        "running": running,
        "pid": _celery_pid if running else None,
        "managed": True,  # This is a managed (auto-started) worker
    }


# Also try to clean up any orphaned workers (started by previous server run)
async def cleanup_orphaned_workers() -> int:
    """
    Kill any orphaned Celery workers from previous runs.

    Returns:
        Number of workers killed
    """
    import platform

    killed = 0

    if platform.system() != "Windows":
        try:
            result = subprocess.run(
                ["pkill", "-f", "celery.*worker"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                killed = 1  # pkill doesn't tell us how many
                logger.info("Cleaned up orphaned Celery workers")
        except Exception as e:
            logger.debug("pkill not available for cleanup", error=str(e))

    return killed
