"""
AIDocumentIndexer - File Watcher Service
=========================================

Monitors configured directories for new/modified/deleted files
and queues them for processing using Watchdog.
"""

import asyncio
import hashlib
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import structlog
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
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
# Enums and Types
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


# =============================================================================
# Data Classes
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
    created_at: datetime = field(default_factory=datetime.utcnow)


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
    processed: bool = False
    processing_error: Optional[str] = None


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
# Event Handler
# =============================================================================

class DocumentEventHandler(FileSystemEventHandler):
    """
    Handles file system events and queues documents for processing.
    """

    def __init__(
        self,
        watcher_service: "FileWatcherService",
        watch_dir: WatchedDirectory,
    ):
        super().__init__()
        self.watcher_service = watcher_service
        self.watch_dir = watch_dir
        self._debounce_timers: Dict[str, threading.Timer] = {}
        self._debounce_delay = 1.0  # seconds

    def _should_process(self, path: str) -> bool:
        """Check if the file should be processed."""
        file_path = Path(path)

        # Check extension
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False

        # Check ignore patterns
        name = file_path.name
        for pattern in IGNORE_PATTERNS:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return False
            elif pattern in name:
                return False

        # Check if it's a directory
        if file_path.is_dir():
            return False

        return True

    def _debounced_process(self, path: str, event_type: WatchEventType):
        """Process event with debouncing to handle rapid successive events."""
        # Cancel existing timer for this path
        if path in self._debounce_timers:
            self._debounce_timers[path].cancel()

        # Create new timer
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
            # Clean up timer
            if path in self._debounce_timers:
                del self._debounce_timers[path]

    def on_created(self, event):
        """Handle file creation."""
        if event.is_directory:
            return

        if self._should_process(event.src_path):
            logger.debug("File created", path=event.src_path)
            self._debounced_process(event.src_path, WatchEventType.CREATED)

    def on_modified(self, event):
        """Handle file modification."""
        if event.is_directory:
            return

        if self._should_process(event.src_path):
            logger.debug("File modified", path=event.src_path)
            self._debounced_process(event.src_path, WatchEventType.MODIFIED)

    def on_deleted(self, event):
        """Handle file deletion."""
        if event.is_directory:
            return

        # Check extension for deleted files (can't check SUPPORTED_EXTENSIONS after delete)
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.debug("File deleted", path=event.src_path)
            self._queue_event(event.src_path, WatchEventType.DELETED)

    def on_moved(self, event):
        """Handle file move/rename."""
        if event.is_directory:
            return

        # Handle as delete + create
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)

        if src_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._queue_event(event.src_path, WatchEventType.DELETED)

        if self._should_process(event.dest_path):
            self._debounced_process(event.dest_path, WatchEventType.CREATED)


# =============================================================================
# File Watcher Service
# =============================================================================

class FileWatcherService:
    """
    Service for watching directories and queueing files for processing.

    Uses Watchdog to monitor file system events and maintains a queue
    of files to be processed by the document pipeline.
    """

    def __init__(self):
        self._observer: Optional[Observer] = None
        self._watch_dirs: Dict[str, WatchedDirectory] = {}
        self._event_handlers: Dict[str, DocumentEventHandler] = {}
        self._event_queue: List[FileEvent] = []
        self._processed_hashes: Set[str] = set()
        self._status = WatcherStatus.STOPPED
        self._start_time: Optional[datetime] = None
        self._events_processed = 0
        self._events_failed = 0
        self._last_event_time: Optional[datetime] = None
        self._callbacks: List[Callable[[FileEvent], None]] = []
        self._lock = threading.Lock()

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
        """Queue a file event for processing."""
        file_path = Path(path)

        # Get file info
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
            timestamp=datetime.utcnow(),
        )

        with self._lock:
            self._event_queue.append(event)
            self._last_event_time = event.timestamp

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

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add_watch_directory(
        self,
        path: str,
        recursive: bool = True,
        auto_process: bool = True,
        access_tier: int = 1,
        collection: Optional[str] = None,
        folder_id: Optional[str] = None,
    ) -> WatchedDirectory:
        """Add a directory to watch."""
        # Validate path
        dir_path = Path(path)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Create watch directory config
        watch_dir = WatchedDirectory(
            id=str(uuid4()),
            path=str(dir_path.resolve()),
            recursive=recursive,
            auto_process=auto_process,
            access_tier=access_tier,
            collection=collection,
            folder_id=folder_id,
        )

        self._watch_dirs[watch_dir.id] = watch_dir

        # If observer is running, start watching immediately
        if self._observer and self._status == WatcherStatus.RUNNING:
            self._start_watching_directory(watch_dir)

        logger.info(
            "Added watch directory",
            watch_dir_id=watch_dir.id,
            path=watch_dir.path,
            recursive=recursive,
        )

        return watch_dir

    def remove_watch_directory(self, watch_dir_id: str) -> bool:
        """Remove a watched directory."""
        if watch_dir_id not in self._watch_dirs:
            return False

        watch_dir = self._watch_dirs[watch_dir_id]

        # Stop watching if observer is running
        if watch_dir_id in self._event_handlers:
            # Note: Watchdog doesn't provide direct unschedule by handler
            # We'd need to stop and restart the observer
            del self._event_handlers[watch_dir_id]

        del self._watch_dirs[watch_dir_id]

        logger.info("Removed watch directory", watch_dir_id=watch_dir_id)
        return True

    def _start_watching_directory(self, watch_dir: WatchedDirectory):
        """Start watching a specific directory."""
        if not self._observer:
            return

        handler = DocumentEventHandler(self, watch_dir)
        self._event_handlers[watch_dir.id] = handler

        self._observer.schedule(
            handler,
            watch_dir.path,
            recursive=watch_dir.recursive,
        )

        logger.debug("Started watching directory", path=watch_dir.path)

    def start(self):
        """Start the file watcher service."""
        if self._status == WatcherStatus.RUNNING:
            logger.warning("Watcher already running")
            return

        logger.info("Starting file watcher service")

        self._observer = Observer()

        # Schedule all enabled watch directories
        for watch_dir in self._watch_dirs.values():
            if watch_dir.enabled:
                self._start_watching_directory(watch_dir)

        self._observer.start()
        self._status = WatcherStatus.RUNNING
        self._start_time = datetime.utcnow()

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
        """Pause the file watcher (keeps observer running but ignores events)."""
        if self._status == WatcherStatus.RUNNING:
            self._status = WatcherStatus.PAUSED
            logger.info("File watcher paused")

    def resume(self):
        """Resume the file watcher after pausing."""
        if self._status == WatcherStatus.PAUSED:
            self._status = WatcherStatus.RUNNING
            logger.info("File watcher resumed")

    def get_queued_events(self, limit: int = 100) -> List[FileEvent]:
        """Get queued events for processing."""
        with self._lock:
            events = [e for e in self._event_queue if not e.processed][:limit]
        return events

    def mark_event_processed(
        self,
        event_id: str,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Mark an event as processed."""
        with self._lock:
            for event in self._event_queue:
                if event.id == event_id:
                    event.processed = True
                    if not success:
                        event.processing_error = error
                        self._events_failed += 1
                    else:
                        self._events_processed += 1
                        # Track hash for deduplication
                        if event.file_hash:
                            self._processed_hashes.add(event.file_hash)
                    break

    def clear_processed_events(self):
        """Clear processed events from the queue."""
        with self._lock:
            self._event_queue = [e for e in self._event_queue if not e.processed]

    def register_callback(self, callback: Callable[[FileEvent], None]):
        """Register a callback to be called when events are queued."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[FileEvent], None]):
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_stats(self) -> WatcherStats:
        """Get watcher statistics."""
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return WatcherStats(
            status=self._status,
            directories_watched=len([d for d in self._watch_dirs.values() if d.enabled]),
            events_queued=len([e for e in self._event_queue if not e.processed]),
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
        """
        Scan a directory for existing files and queue them.

        Useful for initial ingestion of existing files.
        """
        watch_dir = self._watch_dirs.get(watch_dir_id)
        if not watch_dir:
            raise ValueError(f"Watch directory not found: {watch_dir_id}")

        events = []
        dir_path = Path(watch_dir.path)

        # Collect files
        if watch_dir.recursive:
            files = list(dir_path.rglob("*"))
        else:
            files = list(dir_path.glob("*"))

        for file_path in files:
            if file_path.is_file():
                # Check extension
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                # Check ignore patterns
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

                # Create event
                file_hash = self._compute_file_hash(str(file_path))

                # Skip if already processed
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
                    timestamp=datetime.utcnow(),
                )

                events.append(event)

                if process_existing:
                    with self._lock:
                        self._event_queue.append(event)

        logger.info(
            "Scanned directory",
            watch_dir=watch_dir.path,
            files_found=len(events),
        )

        return events


# =============================================================================
# Module-level singleton and helpers
# =============================================================================

_watcher_service: Optional[FileWatcherService] = None


def get_watcher_service() -> FileWatcherService:
    """Get the file watcher service singleton."""
    global _watcher_service
    if _watcher_service is None:
        _watcher_service = FileWatcherService()
    return _watcher_service


async def process_watcher_queue(
    pipeline_func: Callable[[str, dict], Any],
    batch_size: int = 10,
    interval: float = 5.0,
):
    """
    Background task to process the watcher queue.

    Args:
        pipeline_func: Function to call for processing files
        batch_size: Number of files to process per batch
        interval: Seconds between processing batches
    """
    watcher = get_watcher_service()

    while True:
        try:
            events = watcher.get_queued_events(limit=batch_size)

            for event in events:
                if event.event_type == WatchEventType.DELETED:
                    # Handle deletion (mark document as deleted in DB)
                    logger.info("File deleted", path=event.file_path)
                    watcher.mark_event_processed(event.id, success=True)
                    continue

                try:
                    # Get watch directory config
                    watch_dir = watcher.get_watch_directory(event.watch_dir_id)
                    metadata = {
                        "access_tier": watch_dir.access_tier if watch_dir else 1,
                        "collection": watch_dir.collection if watch_dir else None,
                        "source": "file_watcher",
                    }

                    # Process file
                    await pipeline_func(event.file_path, metadata)
                    watcher.mark_event_processed(event.id, success=True)

                except Exception as e:
                    logger.error(
                        "Failed to process file",
                        path=event.file_path,
                        error=str(e),
                    )
                    watcher.mark_event_processed(
                        event.id,
                        success=False,
                        error=str(e),
                    )

            # Clean up processed events periodically
            watcher.clear_processed_events()

        except Exception as e:
            logger.error("Queue processing error", error=str(e))

        await asyncio.sleep(interval)
