"""
AIDocumentIndexer - File Watcher Database Service
=================================================

Database-backed file watcher service that persists watched directories
and events across server restarts.
"""

import asyncio
import hashlib
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import structlog
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)

from backend.db.database import get_async_session
from backend.db.models import (
    WatchedDirectory as WatchedDirectoryModel,
    FileWatcherEvent as FileWatcherEventModel,
    FileWatcherConfig as FileWatcherConfigModel,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc", ".odt", ".rtf",
    # Presentations
    ".pptx", ".ppt", ".odp", ".key",
    # Spreadsheets
    ".xlsx", ".xls", ".csv", ".ods",
    # Images
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
    # Text
    ".txt", ".md", ".rst", ".html", ".xml", ".json",
    # Email
    ".eml", ".msg",
    # Archives
    ".zip",
}

# Files/patterns to ignore
IGNORE_PATTERNS = {
    ".DS_Store",
    "Thumbs.db",
    ".git",
    "__pycache__",
    "node_modules",
    ".env",
    "*.tmp",
    "*.temp",
    "~$*",  # Office temp files
}


# =============================================================================
# Enums
# =============================================================================

class WatchEventType(str, Enum):
    """Types of file system events."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


class WatcherStatus(str, Enum):
    """Status of the file watcher."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class EventStatus(str, Enum):
    """Status of a file event."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Data Classes (for in-memory use)
# =============================================================================

@dataclass
class WatchedDirectory:
    """Configuration for a watched directory."""
    id: str
    path: str
    recursive: bool = True
    auto_process: bool = True
    access_tier: int = 1
    collection: Optional[str] = None
    folder_id: Optional[str] = None
    enabled: bool = True
    organization_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FileEvent:
    """A file system event."""
    id: str
    event_type: WatchEventType
    file_path: str
    file_name: str
    file_extension: str
    file_size: int
    file_hash: Optional[str]
    watch_dir_id: str
    timestamp: datetime
    status: EventStatus = EventStatus.PENDING
    processing_error: Optional[str] = None
    retry_count: int = 0


@dataclass
class WatcherStats:
    """Statistics for the file watcher."""
    status: WatcherStatus
    directories_watched: int
    events_queued: int
    events_processed: int
    events_failed: int
    last_event_time: Optional[datetime]
    uptime_seconds: float


# =============================================================================
# Database Helper Functions
# =============================================================================

async def load_watched_directories_from_db(
    db: AsyncSession,
    organization_id: Optional[str] = None,
) -> List[WatchedDirectory]:
    """Load all watched directories from database."""
    query = select(WatchedDirectoryModel).where(WatchedDirectoryModel.enabled == True)

    # PHASE 12 FIX: Include items from user's org AND items without org (legacy/shared)
    if organization_id:
        from sqlalchemy import or_
        import uuid
        org_uuid = uuid.UUID(organization_id) if isinstance(organization_id, str) else organization_id
        query = query.where(
            or_(
                WatchedDirectoryModel.organization_id == org_uuid,
                WatchedDirectoryModel.organization_id.is_(None),
            )
        )

    result = await db.execute(query)
    db_dirs = result.scalars().all()

    directories = []
    for db_dir in db_dirs:
        directories.append(WatchedDirectory(
            id=str(db_dir.id),
            path=db_dir.path,
            recursive=db_dir.recursive,
            auto_process=db_dir.auto_process,
            access_tier=db_dir.access_tier_id if db_dir.access_tier_id else 1,
            collection=db_dir.collection,
            folder_id=str(db_dir.folder_id) if db_dir.folder_id else None,
            enabled=db_dir.enabled,
            organization_id=str(db_dir.organization_id) if db_dir.organization_id else None,
            created_at=db_dir.created_at,
        ))

    return directories


async def save_watch_directory_to_db(
    db: AsyncSession,
    watch_dir: WatchedDirectory,
    created_by_id: Optional[str] = None,
) -> WatchedDirectoryModel:
    """Save a watched directory to database."""
    import uuid

    db_dir = WatchedDirectoryModel(
        id=uuid.UUID(watch_dir.id),
        path=watch_dir.path,
        recursive=watch_dir.recursive,
        auto_process=watch_dir.auto_process,
        access_tier_id=None,  # Will need to look up by level
        collection=watch_dir.collection,
        folder_id=uuid.UUID(watch_dir.folder_id) if watch_dir.folder_id else None,
        enabled=watch_dir.enabled,
        organization_id=uuid.UUID(watch_dir.organization_id) if watch_dir.organization_id else None,
        created_by_id=uuid.UUID(created_by_id) if created_by_id else None,
    )

    db.add(db_dir)
    await db.commit()
    await db.refresh(db_dir)

    return db_dir


async def update_watch_directory_in_db(
    db: AsyncSession,
    watch_dir_id: str,
    **updates,
) -> bool:
    """Update a watched directory in database."""
    import uuid

    stmt = (
        update(WatchedDirectoryModel)
        .where(WatchedDirectoryModel.id == uuid.UUID(watch_dir_id))
        .values(**updates)
    )

    result = await db.execute(stmt)
    await db.commit()

    return result.rowcount > 0


async def delete_watch_directory_from_db(
    db: AsyncSession,
    watch_dir_id: str,
) -> bool:
    """Delete a watched directory from database."""
    import uuid

    stmt = delete(WatchedDirectoryModel).where(
        WatchedDirectoryModel.id == uuid.UUID(watch_dir_id)
    )

    result = await db.execute(stmt)
    await db.commit()

    return result.rowcount > 0


async def save_file_event_to_db(
    db: AsyncSession,
    event: FileEvent,
) -> FileWatcherEventModel:
    """Save a file event to database."""
    import uuid

    db_event = FileWatcherEventModel(
        id=uuid.UUID(event.id),
        watch_dir_id=uuid.UUID(event.watch_dir_id),
        event_type=event.event_type.value,
        file_path=event.file_path,
        file_name=event.file_name,
        file_extension=event.file_extension,
        file_size=event.file_size,
        file_hash=event.file_hash,
        status=event.status.value,
        processing_error=event.processing_error,
        retry_count=event.retry_count,
        detected_at=event.timestamp,
    )

    db.add(db_event)
    await db.commit()

    return db_event


async def update_event_status_in_db(
    db: AsyncSession,
    event_id: str,
    status: EventStatus,
    error: Optional[str] = None,
    document_id: Optional[str] = None,
) -> bool:
    """Update event status in database."""
    import uuid

    updates = {
        "status": status.value,
        "processing_error": error,
    }

    if status == EventStatus.COMPLETED:
        updates["processed_at"] = datetime.now(timezone.utc)

    if document_id:
        updates["document_id"] = uuid.UUID(document_id)

    stmt = (
        update(FileWatcherEventModel)
        .where(FileWatcherEventModel.id == uuid.UUID(event_id))
        .values(**updates)
    )

    result = await db.execute(stmt)
    await db.commit()

    return result.rowcount > 0


async def get_pending_events_from_db(
    db: AsyncSession,
    limit: int = 100,
) -> List[FileEvent]:
    """Get pending events from database."""
    query = (
        select(FileWatcherEventModel)
        .where(FileWatcherEventModel.status == EventStatus.PENDING.value)
        .order_by(FileWatcherEventModel.detected_at)
        .limit(limit)
    )

    result = await db.execute(query)
    db_events = result.scalars().all()

    events = []
    for db_event in db_events:
        events.append(FileEvent(
            id=str(db_event.id),
            event_type=WatchEventType(db_event.event_type),
            file_path=db_event.file_path,
            file_name=db_event.file_name,
            file_extension=db_event.file_extension,
            file_size=db_event.file_size,
            file_hash=db_event.file_hash,
            watch_dir_id=str(db_event.watch_dir_id),
            timestamp=db_event.detected_at,
            status=EventStatus(db_event.status),
            processing_error=db_event.processing_error,
            retry_count=db_event.retry_count,
        ))

    return events


async def get_watcher_config_from_db(
    db: AsyncSession,
    organization_id: Optional[str] = None,
) -> Optional[dict]:
    """Get watcher config from database."""
    import uuid

    query = select(FileWatcherConfigModel)

    if organization_id:
        query = query.where(FileWatcherConfigModel.organization_id == uuid.UUID(organization_id))
    else:
        query = query.where(FileWatcherConfigModel.organization_id.is_(None))

    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if config:
        return {
            "enabled": config.enabled,
            "auto_start": config.auto_start,
            "batch_size": config.batch_size,
            "poll_interval_seconds": config.poll_interval_seconds,
            "max_retries": config.max_retries,
        }

    return None


async def save_watcher_config_to_db(
    db: AsyncSession,
    config: dict,
    organization_id: Optional[str] = None,
) -> None:
    """Save watcher config to database."""
    import uuid

    org_id = uuid.UUID(organization_id) if organization_id else None

    # Try to update existing
    query = select(FileWatcherConfigModel)
    if org_id:
        query = query.where(FileWatcherConfigModel.organization_id == org_id)
    else:
        query = query.where(FileWatcherConfigModel.organization_id.is_(None))

    result = await db.execute(query)
    existing = result.scalar_one_or_none()

    if existing:
        existing.enabled = config.get("enabled", existing.enabled)
        existing.auto_start = config.get("auto_start", existing.auto_start)
        existing.batch_size = config.get("batch_size", existing.batch_size)
        existing.poll_interval_seconds = config.get("poll_interval_seconds", existing.poll_interval_seconds)
        existing.max_retries = config.get("max_retries", existing.max_retries)
    else:
        db_config = FileWatcherConfigModel(
            id=uuid.uuid4(),
            enabled=config.get("enabled", False),
            auto_start=config.get("auto_start", False),
            batch_size=config.get("batch_size", 10),
            poll_interval_seconds=config.get("poll_interval_seconds", 5),
            max_retries=config.get("max_retries", 3),
            organization_id=org_id,
        )
        db.add(db_config)

    await db.commit()


# =============================================================================
# Event Handler (same as original)
# =============================================================================

class DocumentEventHandler(FileSystemEventHandler):
    """Handles file system events and queues documents for processing."""

    def __init__(
        self,
        watcher_service: "FileWatcherServiceDB",
        watch_dir: WatchedDirectory,
    ):
        super().__init__()
        self.watcher_service = watcher_service
        self.watch_dir = watch_dir
        self._debounce_timers: Dict[str, threading.Timer] = {}
        self._debounce_delay = 1.0

    def _should_process(self, path: str) -> bool:
        """Check if the file should be processed."""
        file_path = Path(path)

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False

        name = file_path.name
        for pattern in IGNORE_PATTERNS:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return False
            elif pattern in name:
                return False

        if file_path.is_dir():
            return False

        return True

    def _debounced_process(self, path: str, event_type: WatchEventType):
        """Process event with debouncing."""
        if path in self._debounce_timers:
            self._debounce_timers[path].cancel()

        timer = threading.Timer(
            self._debounce_delay,
            self._queue_event,
            args=[path, event_type],
        )
        self._debounce_timers[path] = timer
        timer.start()

    def _queue_event(self, path: str, event_type: WatchEventType):
        """Queue the event for processing."""
        try:
            self.watcher_service._queue_event(
                path=path,
                event_type=event_type,
                watch_dir=self.watch_dir,
            )
        except Exception as e:
            logger.error("Failed to queue event", path=path, error=str(e))
        finally:
            if path in self._debounce_timers:
                del self._debounce_timers[path]

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.debug("File created", path=event.src_path)
            self._debounced_process(event.src_path, WatchEventType.CREATED)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.debug("File modified", path=event.src_path)
            self._debounced_process(event.src_path, WatchEventType.MODIFIED)

    def on_deleted(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.debug("File deleted", path=event.src_path)
            self._queue_event(event.src_path, WatchEventType.DELETED)

    def on_moved(self, event):
        if event.is_directory:
            return
        src_path = Path(event.src_path)
        if src_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._queue_event(event.src_path, WatchEventType.DELETED)
        if self._should_process(event.dest_path):
            self._debounced_process(event.dest_path, WatchEventType.CREATED)


# =============================================================================
# Database-Backed File Watcher Service
# =============================================================================

class FileWatcherServiceDB:
    """
    Database-backed file watcher service.

    Persists watched directories and events to the database so they survive
    server restarts and work independently of user sessions.
    """

    def __init__(self):
        self._observer: Optional[Observer] = None
        self._watch_dirs: Dict[str, WatchedDirectory] = {}
        self._event_handlers: Dict[str, DocumentEventHandler] = {}
        self._processed_hashes: Set[str] = set()
        self._status = WatcherStatus.STOPPED
        self._start_time: Optional[datetime] = None
        self._events_processed = 0
        self._events_failed = 0
        self._last_event_time: Optional[datetime] = None
        self._callbacks: List[Callable[[FileEvent], None]] = []
        self._lock = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _compute_file_hash(self, path: str) -> Optional[str]:
        """Compute SHA-256 hash of a file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning("Failed to compute hash", path=path, error=str(e))
            return None

    def _queue_event(
        self,
        path: str,
        event_type: WatchEventType,
        watch_dir: WatchedDirectory,
    ):
        """Queue a file event for processing (saves to database)."""
        file_path = Path(path)

        file_size = 0
        file_hash = None

        if event_type != WatchEventType.DELETED:
            try:
                file_size = file_path.stat().st_size
                file_hash = self._compute_file_hash(path)
            except Exception as e:
                logger.warning("Failed to get file info", path=path, error=str(e))

        # Check for duplicates
        if file_hash and file_hash in self._processed_hashes:
            logger.debug("Skipping duplicate file", path=path, hash=file_hash[:8])
            return

        event = FileEvent(
            id=str(uuid4()),
            event_type=event_type,
            file_path=path,
            file_name=file_path.name,
            file_extension=file_path.suffix.lower(),
            file_size=file_size,
            file_hash=file_hash,
            watch_dir_id=watch_dir.id,
            timestamp=datetime.now(timezone.utc),
        )

        with self._lock:
            self._last_event_time = event.timestamp

        # Save to database asynchronously
        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._save_event_async(event),
                self._event_loop,
            )

        logger.info(
            "Queued file event",
            event_type=event_type.value,
            file=file_path.name,
            watch_dir=watch_dir.path,
        )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Callback failed", error=str(e))

    async def _save_event_async(self, event: FileEvent):
        """Save event to database asynchronously."""
        try:
            async for db in get_async_session():
                await save_file_event_to_db(db, event)
                break
        except Exception as e:
            logger.error("Failed to save event to database", error=str(e))

    async def load_from_database(self):
        """Load watched directories from database on startup."""
        try:
            async for db in get_async_session():
                directories = await load_watched_directories_from_db(db)

                for dir_config in directories:
                    self._watch_dirs[dir_config.id] = dir_config
                    logger.info(
                        "Loaded watch directory from database",
                        path=dir_config.path,
                        enabled=dir_config.enabled,
                    )

                break

        except Exception as e:
            logger.error("Failed to load directories from database", error=str(e))

    async def add_watch_directory(
        self,
        path: str,
        recursive: bool = True,
        auto_process: bool = True,
        access_tier: int = 1,
        collection: Optional[str] = None,
        folder_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        created_by_id: Optional[str] = None,
    ) -> WatchedDirectory:
        """Add a directory to watch (persists to database)."""
        dir_path = Path(path)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        watch_dir = WatchedDirectory(
            id=str(uuid4()),
            path=str(dir_path.resolve()),
            recursive=recursive,
            auto_process=auto_process,
            access_tier=access_tier,
            collection=collection,
            folder_id=folder_id,
            organization_id=organization_id,
        )

        # Save to database
        async for db in get_async_session():
            await save_watch_directory_to_db(db, watch_dir, created_by_id)
            break

        self._watch_dirs[watch_dir.id] = watch_dir

        if self._observer and self._status == WatcherStatus.RUNNING:
            self._start_watching_directory(watch_dir)

        logger.info(
            "Added watch directory",
            watch_dir_id=watch_dir.id,
            path=watch_dir.path,
            recursive=recursive,
        )

        return watch_dir

    async def remove_watch_directory(self, watch_dir_id: str) -> bool:
        """Remove a watched directory (deletes from database)."""
        if watch_dir_id not in self._watch_dirs:
            return False

        # Delete from database
        async for db in get_async_session():
            await delete_watch_directory_from_db(db, watch_dir_id)
            break

        if watch_dir_id in self._event_handlers:
            del self._event_handlers[watch_dir_id]

        del self._watch_dirs[watch_dir_id]

        logger.info("Removed watch directory", watch_dir_id=watch_dir_id)
        return True

    async def toggle_watch_directory(self, watch_dir_id: str) -> bool:
        """Toggle a watch directory enabled/disabled."""
        if watch_dir_id not in self._watch_dirs:
            return False

        watch_dir = self._watch_dirs[watch_dir_id]
        new_enabled = not watch_dir.enabled

        # Update in database
        async for db in get_async_session():
            await update_watch_directory_in_db(db, watch_dir_id, enabled=new_enabled)
            break

        watch_dir.enabled = new_enabled

        return new_enabled

    def _start_watching_directory(self, watch_dir: WatchedDirectory):
        """Start watching a specific directory."""
        if not self._observer:
            return

        if not Path(watch_dir.path).exists():
            logger.warning("Directory does not exist, skipping", path=watch_dir.path)
            return

        handler = DocumentEventHandler(self, watch_dir)
        self._event_handlers[watch_dir.id] = handler

        self._observer.schedule(
            handler,
            watch_dir.path,
            recursive=watch_dir.recursive,
        )

        logger.debug("Started watching directory", path=watch_dir.path)

    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start the file watcher service."""
        if self._status == WatcherStatus.RUNNING:
            logger.warning("Watcher already running")
            return

        self._event_loop = event_loop or asyncio.get_event_loop()

        logger.info("Starting file watcher service")

        self._observer = Observer()

        for watch_dir in self._watch_dirs.values():
            if watch_dir.enabled:
                self._start_watching_directory(watch_dir)

        self._observer.start()
        self._status = WatcherStatus.RUNNING
        self._start_time = datetime.now(timezone.utc)

        logger.info(
            "File watcher started",
            directories=len(self._watch_dirs),
        )

    def stop(self):
        """Stop the file watcher service."""
        if self._status != WatcherStatus.RUNNING:
            logger.warning("Watcher not running")
            return

        logger.info("Stopping file watcher service")

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        self._event_handlers.clear()
        self._status = WatcherStatus.STOPPED

        logger.info("File watcher stopped")

    def pause(self):
        """Pause the file watcher."""
        if self._status == WatcherStatus.RUNNING:
            self._status = WatcherStatus.PAUSED
            logger.info("File watcher paused")

    def resume(self):
        """Resume the file watcher."""
        if self._status == WatcherStatus.PAUSED:
            self._status = WatcherStatus.RUNNING
            logger.info("File watcher resumed")

    def get_stats(self) -> WatcherStats:
        """Get watcher statistics."""
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return WatcherStats(
            status=self._status,
            directories_watched=len([d for d in self._watch_dirs.values() if d.enabled]),
            events_queued=0,  # Would need to query DB for accurate count
            events_processed=self._events_processed,
            events_failed=self._events_failed,
            last_event_time=self._last_event_time,
            uptime_seconds=uptime,
        )

    def list_watch_directories(self) -> List[WatchedDirectory]:
        """List all watched directories."""
        return list(self._watch_dirs.values())

    def get_watch_directory(self, watch_dir_id: str) -> Optional[WatchedDirectory]:
        """Get a specific watched directory."""
        return self._watch_dirs.get(watch_dir_id)

    def scan_directory(
        self,
        watch_dir_id: str,
        process_existing: bool = True,
    ) -> List[FileEvent]:
        """Scan a directory for existing files and queue them."""
        watch_dir = self._watch_dirs.get(watch_dir_id)
        if not watch_dir:
            raise ValueError(f"Watch directory not found: {watch_dir_id}")

        events = []
        dir_path = Path(watch_dir.path)

        if watch_dir.recursive:
            files = list(dir_path.rglob("*"))
        else:
            files = list(dir_path.glob("*"))

        for file_path in files:
            if file_path.is_file():
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                skip = False
                for pattern in IGNORE_PATTERNS:
                    if pattern.startswith("*"):
                        if file_path.name.endswith(pattern[1:]):
                            skip = True
                            break
                    elif pattern in file_path.name:
                        skip = True
                        break

                if skip:
                    continue

                file_hash = self._compute_file_hash(str(file_path))

                if file_hash and file_hash in self._processed_hashes:
                    continue

                event = FileEvent(
                    id=str(uuid4()),
                    event_type=WatchEventType.CREATED,
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_extension=file_path.suffix.lower(),
                    file_size=file_path.stat().st_size,
                    file_hash=file_hash,
                    watch_dir_id=watch_dir.id,
                    timestamp=datetime.now(timezone.utc),
                )

                events.append(event)

                if process_existing:
                    # Save to database asynchronously
                    if self._event_loop:
                        asyncio.run_coroutine_threadsafe(
                            self._save_event_async(event),
                            self._event_loop,
                        )

        logger.info(
            "Scanned directory",
            watch_dir=watch_dir.path,
            files_found=len(events),
        )

        return events

    def register_callback(self, callback: Callable[[FileEvent], None]):
        """Register a callback for file events."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[FileEvent], None]):
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)


# =============================================================================
# Module-level singleton
# =============================================================================

_watcher_service_db: Optional[FileWatcherServiceDB] = None


def get_watcher_service_db() -> FileWatcherServiceDB:
    """Get the database-backed file watcher service singleton."""
    global _watcher_service_db
    if _watcher_service_db is None:
        _watcher_service_db = FileWatcherServiceDB()
    return _watcher_service_db


async def initialize_watcher_from_db():
    """Initialize watcher service from database on startup."""
    watcher = get_watcher_service_db()
    await watcher.load_from_database()

    # Check if auto-start is enabled
    async for db in get_async_session():
        config = await get_watcher_config_from_db(db)
        if config and config.get("auto_start"):
            watcher.start()
            logger.info("File watcher auto-started from config")
        break


async def process_watcher_queue_db(
    pipeline_func: Callable,
    batch_size: int = 10,
    interval: float = 5.0,
):
    """Background task to process the watcher queue from database."""
    watcher = get_watcher_service_db()

    while True:
        try:
            async for db in get_async_session():
                events = await get_pending_events_from_db(db, limit=batch_size)

                for event in events:
                    if event.event_type == WatchEventType.DELETED:
                        logger.info("File deleted", path=event.file_path)
                        await update_event_status_in_db(db, event.id, EventStatus.COMPLETED)
                        watcher._events_processed += 1
                        continue

                    try:
                        await update_event_status_in_db(db, event.id, EventStatus.PROCESSING)

                        watch_dir = watcher.get_watch_directory(event.watch_dir_id)
                        metadata = {
                            "access_tier": watch_dir.access_tier if watch_dir else 1,
                            "collection": watch_dir.collection if watch_dir else None,
                            "folder_id": watch_dir.folder_id if watch_dir else None,
                            "source": "file_watcher",
                        }

                        result = await pipeline_func(event.file_path, metadata)
                        document_id = result.get("document_id") if isinstance(result, dict) else None

                        await update_event_status_in_db(
                            db, event.id, EventStatus.COMPLETED, document_id=document_id
                        )
                        watcher._events_processed += 1

                        if event.file_hash:
                            watcher._processed_hashes.add(event.file_hash)

                    except Exception as e:
                        logger.error(
                            "Failed to process file",
                            path=event.file_path,
                            error=str(e),
                        )
                        await update_event_status_in_db(
                            db, event.id, EventStatus.FAILED, error=str(e)
                        )
                        watcher._events_failed += 1

                break

        except Exception as e:
            logger.error("Queue processing error", error=str(e))

        await asyncio.sleep(interval)
