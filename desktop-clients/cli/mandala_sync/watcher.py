"""
File watching service for Mandala Sync CLI.
"""

import hashlib
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
)

from .api import APIClient, APIError
from .config import get_watched_directories


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

# Patterns to ignore
IGNORE_PATTERNS = {
    ".DS_Store",
    "Thumbs.db",
    ".git",
    "__pycache__",
    "node_modules",
    ".env",
    "*.tmp",
    "*.temp",
    "~$*",
}


class UploadStatus(str, Enum):
    """Status of a file upload."""
    PENDING = "pending"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UploadItem:
    """An item in the upload queue."""
    id: str
    file_path: str
    file_name: str
    file_size: int
    file_hash: Optional[str]
    collection: Optional[str]
    access_tier: int
    folder_id: Optional[str]
    status: UploadStatus = UploadStatus.PENDING
    error: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class FileEventHandler(FileSystemEventHandler):
    """Handles file system events and queues files for upload."""

    def __init__(
        self,
        watcher: "LocalWatcher",
        dir_config: Dict[str, Any],
    ):
        super().__init__()
        self.watcher = watcher
        self.dir_config = dir_config
        self._debounce_timers: Dict[str, threading.Timer] = {}
        self._debounce_delay = 2.0  # Wait 2 seconds for file to be fully written

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

        # Must be a file
        if not file_path.is_file():
            return False

        return True

    def _debounced_queue(self, path: str):
        """Queue file with debouncing to handle partial writes."""
        if path in self._debounce_timers:
            self._debounce_timers[path].cancel()

        timer = threading.Timer(
            self._debounce_delay,
            self._queue_file,
            args=[path],
        )
        self._debounce_timers[path] = timer
        timer.start()

    def _queue_file(self, path: str):
        """Add file to upload queue."""
        try:
            self.watcher.queue_file(path, self.dir_config)
        except Exception as e:
            print(f"Error queuing file {path}: {e}")
        finally:
            if path in self._debounce_timers:
                del self._debounce_timers[path]

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._debounced_queue(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._debounced_queue(event.src_path)


class LocalWatcher:
    """
    Local file watcher that monitors directories and uploads new files.
    """

    def __init__(self, api_client: Optional[APIClient] = None):
        self._observer: Optional[Observer] = None
        self._upload_queue: queue.Queue[UploadItem] = queue.Queue()
        self._processed_hashes: Set[str] = set()
        self._api_client = api_client
        self._running = False
        self._upload_thread: Optional[threading.Thread] = None
        self._handlers: List[FileEventHandler] = []
        self._stats = {
            "files_queued": 0,
            "files_uploaded": 0,
            "files_failed": 0,
        }

    def _compute_file_hash(self, path: str) -> Optional[str]:
        """Compute SHA-256 hash of a file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return None

    def queue_file(self, path: str, dir_config: Dict[str, Any]) -> None:
        """Add a file to the upload queue."""
        file_path = Path(path)

        if not file_path.exists():
            return

        file_hash = self._compute_file_hash(path)

        # Skip duplicates
        if file_hash and file_hash in self._processed_hashes:
            return

        item = UploadItem(
            id=f"{int(time.time() * 1000)}-{file_path.name}",
            file_path=path,
            file_name=file_path.name,
            file_size=file_path.stat().st_size,
            file_hash=file_hash,
            collection=dir_config.get("collection"),
            access_tier=dir_config.get("access_tier", 1),
            folder_id=dir_config.get("folder_id"),
        )

        self._upload_queue.put(item)
        self._stats["files_queued"] += 1

        print(f"Queued: {file_path.name}")

    def _upload_worker(self) -> None:
        """Background worker that processes the upload queue."""
        while self._running:
            try:
                item = self._upload_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not self._api_client:
                print(f"No API client, skipping: {item.file_name}")
                continue

            item.status = UploadStatus.UPLOADING

            try:
                print(f"Uploading: {item.file_name}")

                result = self._api_client.upload_file(
                    file_path=item.file_path,
                    collection=item.collection,
                    access_tier=item.access_tier,
                    folder_id=item.folder_id,
                )

                item.status = UploadStatus.COMPLETED
                self._stats["files_uploaded"] += 1

                if item.file_hash:
                    self._processed_hashes.add(item.file_hash)

                print(f"Uploaded: {item.file_name}")

            except APIError as e:
                item.status = UploadStatus.FAILED
                item.error = str(e)
                item.retry_count += 1
                self._stats["files_failed"] += 1

                print(f"Failed to upload {item.file_name}: {e}")

                # Re-queue for retry if under limit
                if item.retry_count < 3:
                    self._upload_queue.put(item)

            except Exception as e:
                item.status = UploadStatus.FAILED
                item.error = str(e)
                self._stats["files_failed"] += 1
                print(f"Error uploading {item.file_name}: {e}")

    def start(self) -> None:
        """Start the file watcher and upload worker."""
        if self._running:
            return

        self._running = True

        # Start upload worker
        self._upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self._upload_thread.start()

        # Start file observer
        self._observer = Observer()

        for dir_config in get_watched_directories():
            if not dir_config.get("enabled", True):
                continue

            path = dir_config["path"]
            if not Path(path).exists():
                print(f"Warning: Directory not found: {path}")
                continue

            handler = FileEventHandler(self, dir_config)
            self._handlers.append(handler)

            self._observer.schedule(
                handler,
                path,
                recursive=dir_config.get("recursive", True),
            )

            print(f"Watching: {path}")

        self._observer.start()
        print("File watcher started")

    def stop(self) -> None:
        """Stop the file watcher and upload worker."""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._upload_thread:
            self._upload_thread.join(timeout=5.0)
            self._upload_thread = None

        self._handlers.clear()
        print("File watcher stopped")

    def scan_directory(self, path: str, dir_config: Dict[str, Any]) -> int:
        """Scan a directory for existing files and queue them."""
        dir_path = Path(path)
        count = 0

        if dir_config.get("recursive", True):
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

                self.queue_file(str(file_path), dir_config)
                count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        return {
            "running": self._running,
            "queue_size": self._upload_queue.qsize(),
            **self._stats,
        }

    @property
    def is_running(self) -> bool:
        return self._running
