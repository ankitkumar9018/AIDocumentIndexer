"""
AIDocumentIndexer - Memory Manager Service
===========================================

Centralized memory management with dirty tracking and debounced sync.

Inspired by OpenClaw/Moltbot patterns:
- Dirty tracking with configurable debounce (default 1.5s)
- Atomic reindexing with safe temp file swap
- Automatic sync triggers on settings/model changes
- LRU cache pruning for memory control

Usage:
    from backend.services.memory_manager import get_memory_manager

    manager = get_memory_manager()

    # Mark data as dirty (triggers debounced sync)
    await manager.mark_dirty("embeddings")

    # Force immediate sync
    await manager.sync_now()

    # Check if sync needed
    if manager.is_dirty:
        await manager.sync_now()
"""

import asyncio
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class DirtyReason(str, Enum):
    """Reasons for marking data as dirty."""
    EMBEDDINGS = "embeddings"
    DOCUMENTS = "documents"
    SETTINGS = "settings"
    PROVIDER_CHANGED = "provider_changed"
    MODEL_CHANGED = "model_changed"
    CHUNK_SETTINGS_CHANGED = "chunk_settings_changed"
    VECTOR_DIMENSIONS_MISMATCH = "vector_dimensions_mismatch"
    FORCE_SYNC = "force_sync"
    CACHE_INVALIDATION = "cache_invalidation"


@dataclass
class SyncStats:
    """Statistics from a sync operation."""
    started_at: datetime
    completed_at: Optional[datetime] = None
    items_synced: int = 0
    errors: int = 0
    duration_ms: int = 0
    reason: Optional[str] = None


@dataclass
class MemoryManagerConfig:
    """Configuration for memory manager."""
    # Debounce settings
    debounce_ms: int = 1500  # 1.5 seconds (OpenClaw default)
    max_debounce_ms: int = 10000  # Max 10 seconds before forced sync

    # Sync settings
    auto_sync: bool = True
    sync_on_shutdown: bool = True

    # Cache settings
    max_cache_entries: int = 10000
    cache_prune_percent: float = 0.2  # Prune 20% when full

    # Reindex settings
    use_atomic_reindex: bool = True
    validate_before_swap: bool = True


# =============================================================================
# Reindex Triggers
# =============================================================================

REINDEX_TRIGGERS = {
    DirtyReason.PROVIDER_CHANGED,
    DirtyReason.MODEL_CHANGED,
    DirtyReason.CHUNK_SETTINGS_CHANGED,
    DirtyReason.VECTOR_DIMENSIONS_MISMATCH,
    DirtyReason.FORCE_SYNC,
}


# =============================================================================
# Memory Manager
# =============================================================================

class MemoryManager:
    """
    Centralized memory management with dirty tracking and debounced sync.

    Features:
    - Dirty tracking with configurable debounce
    - Atomic reindexing (safe temp file swap)
    - Automatic sync triggers
    - Cache pruning
    - Sync statistics

    Example:
        manager = MemoryManager()

        # Register sync handlers
        manager.register_sync_handler("embeddings", embedding_sync_fn)
        manager.register_sync_handler("documents", document_sync_fn)

        # Mark dirty (triggers debounced sync)
        await manager.mark_dirty(DirtyReason.EMBEDDINGS)

        # Check status
        print(f"Dirty: {manager.is_dirty}, Pending: {manager.pending_reasons}")
    """

    def __init__(self, config: Optional[MemoryManagerConfig] = None):
        self.config = config or MemoryManagerConfig()

        # Dirty state
        self._dirty = False
        self._dirty_reasons: Set[DirtyReason] = set()
        self._first_dirty_at: Optional[datetime] = None

        # Debounce
        self._debounce_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Sync handlers
        self._sync_handlers: Dict[str, Callable] = {}

        # Statistics
        self._sync_history: List[SyncStats] = []
        self._total_syncs = 0
        self._total_errors = 0

        # Shutdown flag
        self._shutting_down = False

        logger.info(
            "Memory manager initialized",
            debounce_ms=self.config.debounce_ms,
            auto_sync=self.config.auto_sync,
        )

    @property
    def is_dirty(self) -> bool:
        """Check if any data is marked dirty."""
        return self._dirty

    @property
    def pending_reasons(self) -> Set[DirtyReason]:
        """Get reasons for pending sync."""
        return self._dirty_reasons.copy()

    @property
    def needs_reindex(self) -> bool:
        """Check if a full reindex is needed."""
        return bool(self._dirty_reasons & REINDEX_TRIGGERS)

    def register_sync_handler(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> None:
        """
        Register a sync handler.

        Args:
            name: Handler name (e.g., "embeddings", "documents")
            handler: Async function to call during sync
        """
        self._sync_handlers[name] = handler
        logger.debug("Registered sync handler", handler=name)

    async def mark_dirty(
        self,
        reason: DirtyReason = DirtyReason.DOCUMENTS,
        immediate: bool = False,
    ) -> None:
        """
        Mark data as dirty, triggering debounced sync.

        Args:
            reason: Why the data is dirty
            immediate: If True, sync immediately without debounce
        """
        async with self._lock:
            self._dirty = True
            self._dirty_reasons.add(reason)

            if self._first_dirty_at is None:
                self._first_dirty_at = datetime.utcnow()

            logger.debug(
                "Marked dirty",
                reason=reason.value,
                total_reasons=len(self._dirty_reasons),
            )

            # Check if max debounce time exceeded
            if self._first_dirty_at:
                elapsed = (datetime.utcnow() - self._first_dirty_at).total_seconds() * 1000
                if elapsed >= self.config.max_debounce_ms:
                    immediate = True
                    logger.info("Max debounce time exceeded, forcing sync")

            if immediate:
                await self._sync_internal()
            elif self.config.auto_sync:
                # Cancel existing debounce task
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()
                    try:
                        await self._debounce_task
                    except asyncio.CancelledError:
                        pass

                # Start new debounce task
                self._debounce_task = asyncio.create_task(
                    self._debounced_sync()
                )

    async def _debounced_sync(self) -> None:
        """Wait for debounce period, then sync."""
        try:
            await asyncio.sleep(self.config.debounce_ms / 1000)
            await self._sync_internal()
        except asyncio.CancelledError:
            # Debounce was reset
            pass

    async def _sync_internal(self) -> SyncStats:
        """Execute sync with all registered handlers."""
        stats = SyncStats(
            started_at=datetime.utcnow(),
            reason=", ".join(r.value for r in self._dirty_reasons),
        )

        try:
            logger.info(
                "Starting sync",
                reasons=list(r.value for r in self._dirty_reasons),
                handlers=list(self._sync_handlers.keys()),
            )

            # Execute handlers
            for name, handler in self._sync_handlers.items():
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(self._dirty_reasons)
                    else:
                        handler(self._dirty_reasons)
                    stats.items_synced += 1
                except Exception as e:
                    logger.error(
                        "Sync handler failed",
                        handler=name,
                        error=str(e),
                    )
                    stats.errors += 1

            # Clear dirty state
            self._dirty = False
            self._dirty_reasons.clear()
            self._first_dirty_at = None

            stats.completed_at = datetime.utcnow()
            stats.duration_ms = int(
                (stats.completed_at - stats.started_at).total_seconds() * 1000
            )

            self._sync_history.append(stats)
            self._total_syncs += 1
            self._total_errors += stats.errors

            # Keep only last 100 sync records
            if len(self._sync_history) > 100:
                self._sync_history = self._sync_history[-100:]

            logger.info(
                "Sync completed",
                duration_ms=stats.duration_ms,
                items_synced=stats.items_synced,
                errors=stats.errors,
            )

        except Exception as e:
            logger.error("Sync failed", error=str(e))
            stats.errors += 1

        return stats

    async def sync_now(self) -> SyncStats:
        """Force immediate sync, bypassing debounce."""
        async with self._lock:
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()
                try:
                    await self._debounce_task
                except asyncio.CancelledError:
                    pass

            return await self._sync_internal()

    async def atomic_reindex(
        self,
        source_path: str,
        build_func: Callable[[str], Any],
        validate_func: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """
        Perform atomic reindexing with safe temp file swap.

        Args:
            source_path: Path to the file/directory to reindex
            build_func: Function to build new index at temp path
            validate_func: Optional function to validate new index

        Returns:
            True if reindex succeeded, False otherwise
        """
        temp_path = f"{source_path}.tmp"

        try:
            logger.info("Starting atomic reindex", source=source_path)

            # Build to temp location
            if asyncio.iscoroutinefunction(build_func):
                await build_func(temp_path)
            else:
                build_func(temp_path)

            # Validate if function provided
            if validate_func and self.config.validate_before_swap:
                if asyncio.iscoroutinefunction(validate_func):
                    valid = await validate_func(temp_path)
                else:
                    valid = validate_func(temp_path)

                if not valid:
                    logger.error("Reindex validation failed", temp=temp_path)
                    if os.path.exists(temp_path):
                        if os.path.isdir(temp_path):
                            shutil.rmtree(temp_path)
                        else:
                            os.remove(temp_path)
                    return False

            # Atomic swap
            if os.path.exists(source_path):
                backup_path = f"{source_path}.backup"
                if os.path.exists(backup_path):
                    if os.path.isdir(backup_path):
                        shutil.rmtree(backup_path)
                    else:
                        os.remove(backup_path)
                os.rename(source_path, backup_path)

            os.rename(temp_path, source_path)

            # Clean up backup
            backup_path = f"{source_path}.backup"
            if os.path.exists(backup_path):
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)

            logger.info("Atomic reindex completed", source=source_path)
            return True

        except Exception as e:
            logger.error(
                "Atomic reindex failed",
                source=source_path,
                error=str(e),
            )

            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    if os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                    else:
                        os.remove(temp_path)
                except Exception:
                    pass

            return False

    async def prune_cache(
        self,
        cache: Dict[str, Any],
        get_access_time: Callable[[str], datetime],
    ) -> int:
        """
        Prune LRU entries from a cache.

        Args:
            cache: Dict cache to prune
            get_access_time: Function to get last access time for a key

        Returns:
            Number of entries pruned
        """
        if len(cache) <= self.config.max_cache_entries:
            return 0

        prune_count = int(self.config.max_cache_entries * self.config.cache_prune_percent)

        # Sort by access time (oldest first)
        sorted_keys = sorted(
            cache.keys(),
            key=lambda k: get_access_time(k),
        )

        # Remove oldest entries
        for key in sorted_keys[:prune_count]:
            del cache[key]

        logger.info("Cache pruned", removed=prune_count, remaining=len(cache))
        return prune_count

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return {
            "is_dirty": self._dirty,
            "pending_reasons": [r.value for r in self._dirty_reasons],
            "needs_reindex": self.needs_reindex,
            "total_syncs": self._total_syncs,
            "total_errors": self._total_errors,
            "handlers_registered": list(self._sync_handlers.keys()),
            "recent_syncs": [
                {
                    "started_at": s.started_at.isoformat(),
                    "duration_ms": s.duration_ms,
                    "items_synced": s.items_synced,
                    "errors": s.errors,
                    "reason": s.reason,
                }
                for s in self._sync_history[-10:]
            ],
        }

    async def shutdown(self) -> None:
        """Shutdown memory manager, syncing if needed."""
        self._shutting_down = True

        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass

        if self.config.sync_on_shutdown and self._dirty:
            logger.info("Syncing before shutdown")
            await self._sync_internal()

        logger.info("Memory manager shutdown complete")


# =============================================================================
# Singleton Instance
# =============================================================================

_memory_manager: Optional[MemoryManager] = None
_manager_lock = threading.Lock()


def get_memory_manager(
    config: Optional[MemoryManagerConfig] = None,
) -> MemoryManager:
    """Get or create the memory manager singleton."""
    global _memory_manager

    if _memory_manager is None:
        with _manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager(config)

    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the memory manager singleton (for testing)."""
    global _memory_manager
    _memory_manager = None
