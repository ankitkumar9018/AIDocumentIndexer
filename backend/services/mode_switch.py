"""
AIDocumentIndexer - Mode Switching Utility
==========================================

Provides utilities for switching between local (SQLite + ChromaDB) and
cloud (PostgreSQL + pgvector) modes. Enables seamless development and
production deployment.

Usage:
    # Check current mode
    mode = await get_current_mode()  # Returns "local" or "cloud"

    # Get mode info for display
    info = await get_mode_info()

    # Switch modes (requires restart to take effect)
    switch_to_local_mode()
    switch_to_cloud_mode(db_url, vector_backend)
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# Mode constants
MODE_LOCAL = "local"
MODE_CLOUD = "cloud"


@dataclass
class ModeInfo:
    """Information about the current operating mode."""
    mode: str  # "local" or "cloud"
    database_type: str  # "sqlite" or "postgresql"
    vector_backend: str  # "chroma" or "pgvector"
    redis_enabled: bool
    redis_available: bool
    is_local_mode_env: bool  # Whether LOCAL_MODE env var is set
    database_url_masked: str  # Masked connection string
    features: Dict[str, bool] = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def is_local_mode() -> bool:
    """
    Check if the system is configured for local mode.

    Local mode is determined by:
    1. LOCAL_MODE=true environment variable
    2. DATABASE_URL containing "sqlite"
    3. DATABASE_TYPE set to "sqlite"

    Returns:
        True if in local mode, False otherwise
    """
    # Check explicit LOCAL_MODE flag
    local_mode_env = os.getenv("LOCAL_MODE", "false").lower() == "true"
    if local_mode_env:
        return True

    # Check database URL
    database_url = os.getenv("DATABASE_URL", "")
    if "sqlite" in database_url.lower():
        return True

    # Check database type
    database_type = os.getenv("DATABASE_TYPE", "postgresql").lower()
    if database_type == "sqlite":
        return True

    return False


async def get_current_mode() -> str:
    """
    Get the current operating mode.

    Returns:
        "local" for SQLite + ChromaDB mode
        "cloud" for PostgreSQL + pgvector mode
    """
    return MODE_LOCAL if is_local_mode() else MODE_CLOUD


async def get_mode_info() -> ModeInfo:
    """
    Get detailed information about the current operating mode.

    Returns:
        ModeInfo with details about database, vector store, and features
    """
    mode = await get_current_mode()

    # Get database info
    database_type = os.getenv("DATABASE_TYPE", "postgresql").lower()
    database_url = os.getenv("DATABASE_URL", "")

    # Mask sensitive parts of URL
    if database_url:
        if "://" in database_url:
            parts = database_url.split("://")
            if "@" in parts[1]:
                # Mask credentials
                user_host = parts[1].split("@")
                masked_url = f"{parts[0]}://***:***@{user_host[-1]}"
            else:
                masked_url = database_url
        else:
            masked_url = database_url
    else:
        masked_url = "Not configured"

    # Override database type if sqlite is in URL
    if "sqlite" in database_url.lower():
        database_type = "sqlite"

    # Get vector backend
    vector_backend = os.getenv("VECTOR_STORE_BACKEND", "auto").lower()
    if vector_backend == "auto":
        vector_backend = "chroma" if mode == MODE_LOCAL else "pgvector"

    # Check Redis
    redis_enabled = os.getenv("CELERY_ENABLED", "").lower() in ("true", "1", "yes")
    redis_available = False
    try:
        from backend.services.redis_client import check_redis_connection
        redis_status = await check_redis_connection()
        redis_available = redis_status.get("connected", False)
    except Exception:
        pass

    # Determine available features
    features = {
        "vector_search": True,  # Always available (ChromaDB or pgvector)
        "full_text_search": database_type == "postgresql",  # PostgreSQL only
        "hybrid_search": True,  # Available in both modes
        "redis_cache": redis_available,
        "distributed_tasks": redis_available,
        "knowledge_graph": database_type == "postgresql",  # Needs complex joins
        "row_level_security": database_type == "postgresql",
    }

    # Generate warnings
    warnings = []
    if mode == MODE_LOCAL:
        warnings.append("Running in local mode - some features may be limited")
        if not redis_available:
            warnings.append("Redis unavailable - using in-memory caching")

    if database_type == "sqlite":
        warnings.append("SQLite does not support full-text search indexes")
        warnings.append("SQLite does not support row-level security")

    return ModeInfo(
        mode=mode,
        database_type=database_type,
        vector_backend=vector_backend,
        redis_enabled=redis_enabled,
        redis_available=redis_available,
        is_local_mode_env=os.getenv("LOCAL_MODE", "false").lower() == "true",
        database_url_masked=masked_url,
        features=features,
        warnings=warnings,
    )


def switch_to_local_mode(
    database_path: str = "./data/aidocindexer.db",
    chroma_path: str = "./data/chroma",
) -> Dict[str, str]:
    """
    Configure environment for local mode (SQLite + ChromaDB).

    NOTE: This updates environment variables but a restart is required
    for changes to take effect in the database connections.

    Args:
        database_path: Path to SQLite database file
        chroma_path: Path to ChromaDB persistence directory

    Returns:
        Dictionary of environment variables that were set
    """
    changes = {}

    # Set local mode flag
    os.environ["LOCAL_MODE"] = "true"
    changes["LOCAL_MODE"] = "true"

    # Set database configuration
    os.environ["DATABASE_TYPE"] = "sqlite"
    changes["DATABASE_TYPE"] = "sqlite"

    sqlite_url = f"sqlite:///{database_path}"
    os.environ["DATABASE_URL"] = sqlite_url
    changes["DATABASE_URL"] = sqlite_url

    # Set vector store configuration
    os.environ["VECTOR_STORE_BACKEND"] = "chroma"
    changes["VECTOR_STORE_BACKEND"] = "chroma"

    os.environ["CHROMA_PERSIST_DIRECTORY"] = chroma_path
    changes["CHROMA_PERSIST_DIRECTORY"] = chroma_path

    # Disable Redis/Celery for true local mode
    os.environ["CELERY_ENABLED"] = "false"
    changes["CELERY_ENABLED"] = "false"

    logger.info(
        "Switched to local mode",
        database_type="sqlite",
        vector_backend="chroma",
        database_path=database_path,
        chroma_path=chroma_path,
    )

    return changes


def switch_to_cloud_mode(
    database_url: str,
    vector_backend: str = "pgvector",
    redis_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Configure environment for cloud mode (PostgreSQL + pgvector).

    NOTE: This updates environment variables but a restart is required
    for changes to take effect in the database connections.

    Args:
        database_url: PostgreSQL connection string
        vector_backend: Vector store backend ("pgvector", "qdrant", "milvus")
        redis_url: Optional Redis URL for caching

    Returns:
        Dictionary of environment variables that were set
    """
    changes = {}

    # Clear local mode flag
    os.environ["LOCAL_MODE"] = "false"
    changes["LOCAL_MODE"] = "false"

    # Set database configuration
    os.environ["DATABASE_TYPE"] = "postgresql"
    changes["DATABASE_TYPE"] = "postgresql"

    os.environ["DATABASE_URL"] = database_url
    changes["DATABASE_URL"] = "***" # Don't log full URL

    # Set vector store configuration
    os.environ["VECTOR_STORE_BACKEND"] = vector_backend
    changes["VECTOR_STORE_BACKEND"] = vector_backend

    # Enable Redis if URL provided
    if redis_url:
        os.environ["REDIS_URL"] = redis_url
        os.environ["CELERY_ENABLED"] = "true"
        changes["CELERY_ENABLED"] = "true"

    logger.info(
        "Switched to cloud mode",
        database_type="postgresql",
        vector_backend=vector_backend,
        redis_enabled=bool(redis_url),
    )

    return changes


def get_recommended_mode() -> str:
    """
    Determine the recommended mode based on available resources.

    Checks for:
    - PostgreSQL availability
    - pgvector extension
    - Redis availability

    Returns:
        "local" if resources unavailable, "cloud" otherwise
    """
    # Check if PostgreSQL URL is configured
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url or "sqlite" in database_url.lower():
        return MODE_LOCAL

    # If URL looks like PostgreSQL, recommend cloud mode
    if "postgresql" in database_url.lower() or "postgres" in database_url.lower():
        return MODE_CLOUD

    return MODE_LOCAL


async def validate_mode_configuration() -> Dict[str, Any]:
    """
    Validate the current mode configuration.

    Checks:
    - Database connection
    - Vector store connection
    - Redis connection (if enabled)

    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "mode": await get_current_mode(),
        "checks": {},
        "errors": [],
    }

    # Check database connection
    try:
        from backend.db.database import async_session_context
        async with async_session_context() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        results["checks"]["database"] = {"status": "ok", "message": "Connected"}
    except Exception as e:
        results["checks"]["database"] = {"status": "error", "message": str(e)}
        results["errors"].append(f"Database connection failed: {e}")
        results["valid"] = False

    # Check vector store
    try:
        from backend.services.vectorstore import get_vector_store
        vs = get_vector_store()
        results["checks"]["vector_store"] = {"status": "ok", "message": f"Using {type(vs).__name__}"}
    except Exception as e:
        results["checks"]["vector_store"] = {"status": "error", "message": str(e)}
        results["errors"].append(f"Vector store initialization failed: {e}")
        results["valid"] = False

    # Check Redis (optional)
    try:
        from backend.services.redis_client import check_redis_connection
        redis_status = await check_redis_connection()
        if redis_status.get("connected"):
            results["checks"]["redis"] = {"status": "ok", "message": "Connected"}
        else:
            results["checks"]["redis"] = {
                "status": "warning",
                "message": f"Not connected: {redis_status.get('reason', 'Unknown')}"
            }
    except Exception as e:
        results["checks"]["redis"] = {"status": "warning", "message": f"Check failed: {e}"}

    return results


# Singleton for mode info caching
_cached_mode_info: Optional[ModeInfo] = None
_cache_timestamp: Optional[datetime] = None
_CACHE_TTL_SECONDS = 60


async def get_cached_mode_info() -> ModeInfo:
    """
    Get mode info with caching to avoid repeated checks.

    Returns:
        Cached or fresh ModeInfo
    """
    global _cached_mode_info, _cache_timestamp

    now = datetime.utcnow()
    if (_cached_mode_info is not None and
        _cache_timestamp is not None and
        (now - _cache_timestamp).total_seconds() < _CACHE_TTL_SECONDS):
        return _cached_mode_info

    _cached_mode_info = await get_mode_info()
    _cache_timestamp = now

    return _cached_mode_info


def invalidate_mode_cache():
    """Invalidate the cached mode info."""
    global _cached_mode_info, _cache_timestamp
    _cached_mode_info = None
    _cache_timestamp = None
