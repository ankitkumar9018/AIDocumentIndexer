"""
AIDocumentIndexer - Bulk Progress Tracking Service
===================================================

Provides distributed progress tracking for bulk uploads and batch operations.
Uses Redis for real-time progress updates across multiple workers.

Features:
- Track progress of individual files in a bulk upload
- Real-time WebSocket updates
- Aggregated statistics (total, completed, failed)
- Per-stage progress (extraction, chunking, embedding, KG)
- Persistent state for recovery after worker crashes
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4
from enum import Enum

import structlog

# Use orjson for faster JSON serialization (2-3x faster)
try:
    import orjson
    def _json_dumps(obj):
        return or_json_dumps(obj).decode()
    def _json_loads(s):
        return or_json_loads(s)
except ImportError:
    import json
    def _json_dumps(obj):
        return _json_dumps(obj)
    def _json_loads(s):
        return _json_loads(s)

logger = structlog.get_logger(__name__)


class ProcessingStage(str, Enum):
    """Processing stages for document pipeline."""
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    KG_EXTRACTION = "kg_extraction"
    COMPLETED = "completed"
    FAILED = "failed"


class FileStatus(str, Enum):
    """Status of individual files in a bulk upload."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Stage weights for progress calculation (total = 100)
STAGE_WEIGHTS = {
    ProcessingStage.QUEUED: 0,
    ProcessingStage.EXTRACTING: 20,
    ProcessingStage.CHUNKING: 40,
    ProcessingStage.EMBEDDING: 70,
    ProcessingStage.STORING: 85,
    ProcessingStage.KG_EXTRACTION: 95,  # Optional, runs in background
    ProcessingStage.COMPLETED: 100,
    ProcessingStage.FAILED: 100,
}


class BulkProgressTracker:
    """
    Tracks progress of bulk uploads using Redis for distributed state.

    Usage:
        tracker = BulkProgressTracker(redis_client)

        # Start tracking a bulk upload
        batch_id = await tracker.create_batch(user_id, file_count=100)

        # Update individual file progress
        await tracker.update_file_status(batch_id, "doc1.pdf", FileStatus.PROCESSING)
        await tracker.update_file_stage(batch_id, "doc1.pdf", ProcessingStage.EXTRACTING)
        await tracker.update_file_stage(batch_id, "doc1.pdf", ProcessingStage.CHUNKING)
        await tracker.update_file_status(batch_id, "doc1.pdf", FileStatus.COMPLETED)

        # Get overall progress
        progress = await tracker.get_batch_progress(batch_id)
        # Returns: {
        #     "batch_id": "...",
        #     "total_files": 100,
        #     "completed": 45,
        #     "failed": 2,
        #     "processing": 5,
        #     "pending": 48,
        #     "overall_progress": 47.3,
        #     "eta_seconds": 1234,
        #     "files": {...}
        # }
    """

    # Redis key prefixes
    BATCH_KEY_PREFIX = "bulk_progress:batch:"
    FILE_KEY_PREFIX = "bulk_progress:file:"
    STATS_KEY_PREFIX = "bulk_progress:stats:"

    # TTL for progress data (7 days)
    TTL_SECONDS = 60 * 60 * 24 * 7

    def __init__(self, redis_client=None):
        """Initialize with optional Redis client."""
        self._redis = redis_client
        self._local_cache = {}  # Fallback when Redis unavailable

    async def _get_redis(self):
        """Get Redis client, creating if needed."""
        if self._redis is None:
            from backend.services.redis_client import get_redis_client
            self._redis = await get_redis_client()
        return self._redis

    async def _is_redis_available(self) -> bool:
        """Check if Redis is available."""
        try:
            redis = await self._get_redis()
            if redis is None:
                return False
            await redis.ping()
            return True
        except Exception:
            return False

    # =========================================================================
    # Batch Management
    # =========================================================================

    async def create_batch(
        self,
        user_id: str,
        file_count: int,
        collection: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new batch for tracking bulk upload progress.

        Args:
            user_id: ID of the user uploading
            file_count: Total number of files in the batch
            collection: Optional collection name
            metadata: Additional metadata

        Returns:
            batch_id: Unique identifier for the batch
        """
        batch_id = str(uuid4())

        batch_data = {
            "batch_id": batch_id,
            "user_id": user_id,
            "file_count": file_count,
            "collection": collection,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "processing",
            "stats": {
                "total": file_count,
                "pending": file_count,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "skipped": 0,
            },
            "files": {},
            "start_time": time.time(),
        }

        if await self._is_redis_available():
            redis = await self._get_redis()
            key = f"{self.BATCH_KEY_PREFIX}{batch_id}"
            await redis.set(key, _json_dumps(batch_data), ex=self.TTL_SECONDS)
        else:
            self._local_cache[batch_id] = batch_data

        logger.info(
            "Bulk upload batch created",
            batch_id=batch_id,
            user_id=user_id,
            file_count=file_count,
        )

        return batch_id

    async def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch data by ID."""
        if await self._is_redis_available():
            redis = await self._get_redis()
            key = f"{self.BATCH_KEY_PREFIX}{batch_id}"
            data = await redis.get(key)
            if data:
                return _json_loads(data)
        else:
            return self._local_cache.get(batch_id)
        return None

    async def _save_batch(self, batch_id: str, batch_data: Dict[str, Any]):
        """Save batch data to Redis or local cache."""
        batch_data["updated_at"] = datetime.utcnow().isoformat()

        if await self._is_redis_available():
            redis = await self._get_redis()
            key = f"{self.BATCH_KEY_PREFIX}{batch_id}"
            await redis.set(key, _json_dumps(batch_data), ex=self.TTL_SECONDS)
        else:
            self._local_cache[batch_id] = batch_data

    # =========================================================================
    # File Progress Tracking
    # =========================================================================

    async def add_file(
        self,
        batch_id: str,
        file_id: str,
        filename: str,
        file_size: Optional[int] = None,
    ):
        """Add a file to the batch for tracking."""
        batch = await self.get_batch(batch_id)
        if not batch:
            logger.warning("Batch not found", batch_id=batch_id)
            return

        batch["files"][file_id] = {
            "file_id": file_id,
            "filename": filename,
            "file_size": file_size,
            "status": FileStatus.PENDING,
            "stage": ProcessingStage.QUEUED,
            "progress": 0,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "document_id": None,
            "chunk_count": 0,
        }

        await self._save_batch(batch_id, batch)

    async def update_file_status(
        self,
        batch_id: str,
        file_id: str,
        status: FileStatus,
        error: Optional[str] = None,
        document_id: Optional[str] = None,
        chunk_count: int = 0,
    ):
        """Update the status of a file in the batch."""
        batch = await self.get_batch(batch_id)
        if not batch:
            return

        file_data = batch["files"].get(file_id)
        if not file_data:
            # Auto-create file entry if not exists
            file_data = {
                "file_id": file_id,
                "filename": file_id,
                "status": FileStatus.PENDING,
                "stage": ProcessingStage.QUEUED,
                "progress": 0,
            }
            batch["files"][file_id] = file_data

        old_status = file_data.get("status", FileStatus.PENDING)
        file_data["status"] = status

        # Update timestamps
        if status == FileStatus.PROCESSING and not file_data.get("started_at"):
            file_data["started_at"] = datetime.utcnow().isoformat()
        elif status in [FileStatus.COMPLETED, FileStatus.FAILED, FileStatus.SKIPPED]:
            file_data["completed_at"] = datetime.utcnow().isoformat()
            file_data["progress"] = 100

        # Set additional data
        if error:
            file_data["error"] = error
        if document_id:
            file_data["document_id"] = document_id
        if chunk_count:
            file_data["chunk_count"] = chunk_count

        # Update batch stats
        stats = batch["stats"]

        # Decrement old status count
        if old_status == FileStatus.PENDING:
            stats["pending"] = max(0, stats["pending"] - 1)
        elif old_status == FileStatus.PROCESSING:
            stats["processing"] = max(0, stats["processing"] - 1)

        # Increment new status count
        if status == FileStatus.PENDING:
            stats["pending"] += 1
        elif status == FileStatus.PROCESSING:
            stats["processing"] += 1
        elif status == FileStatus.COMPLETED:
            stats["completed"] += 1
        elif status == FileStatus.FAILED:
            stats["failed"] += 1
        elif status == FileStatus.SKIPPED:
            stats["skipped"] += 1

        # Check if batch is complete
        total_done = stats["completed"] + stats["failed"] + stats["skipped"]
        if total_done >= stats["total"]:
            batch["status"] = "completed"
            batch["completed_at"] = datetime.utcnow().isoformat()

        await self._save_batch(batch_id, batch)

    async def update_file_stage(
        self,
        batch_id: str,
        file_id: str,
        stage: ProcessingStage,
    ):
        """Update the processing stage of a file."""
        batch = await self.get_batch(batch_id)
        if not batch:
            return

        file_data = batch["files"].get(file_id)
        if not file_data:
            return

        file_data["stage"] = stage
        file_data["progress"] = STAGE_WEIGHTS.get(stage, 0)

        # Auto-update status based on stage
        if stage == ProcessingStage.COMPLETED:
            file_data["status"] = FileStatus.COMPLETED
        elif stage == ProcessingStage.FAILED:
            file_data["status"] = FileStatus.FAILED
        elif stage not in [ProcessingStage.QUEUED]:
            file_data["status"] = FileStatus.PROCESSING

        await self._save_batch(batch_id, batch)

    # =========================================================================
    # Progress Queries
    # =========================================================================

    async def get_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive progress information for a batch.

        Returns detailed stats including:
        - Overall progress percentage
        - Per-status counts
        - ETA calculation
        - Per-file status (paginated)
        """
        batch = await self.get_batch(batch_id)
        if not batch:
            return None

        stats = batch["stats"]
        files = batch["files"]

        # Calculate overall progress
        total_progress = 0
        for file_data in files.values():
            total_progress += file_data.get("progress", 0)

        overall_progress = (
            (total_progress / (stats["total"] * 100)) * 100
            if stats["total"] > 0
            else 0
        )

        # Calculate ETA
        elapsed = time.time() - batch.get("start_time", time.time())
        completed_count = stats["completed"] + stats["failed"] + stats["skipped"]

        if completed_count > 0 and completed_count < stats["total"]:
            avg_time_per_file = elapsed / completed_count
            remaining = stats["total"] - completed_count
            eta_seconds = int(avg_time_per_file * remaining)
        else:
            eta_seconds = None

        # Get processing rate
        if elapsed > 0:
            files_per_minute = (completed_count / elapsed) * 60
        else:
            files_per_minute = 0

        return {
            "batch_id": batch_id,
            "status": batch["status"],
            "user_id": batch["user_id"],
            "collection": batch.get("collection"),
            "created_at": batch["created_at"],
            "updated_at": batch["updated_at"],
            "stats": stats,
            "overall_progress": round(overall_progress, 1),
            "eta_seconds": eta_seconds,
            "files_per_minute": round(files_per_minute, 1),
            "elapsed_seconds": int(elapsed),
        }

    async def get_batch_files(
        self,
        batch_id: str,
        status_filter: Optional[FileStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get paginated list of files in a batch.

        Args:
            batch_id: Batch ID
            status_filter: Optional filter by status
            limit: Max files to return
            offset: Pagination offset

        Returns:
            Dict with files list and pagination info
        """
        batch = await self.get_batch(batch_id)
        if not batch:
            return {"files": [], "total": 0, "limit": limit, "offset": offset}

        files = list(batch["files"].values())

        # Apply filter
        if status_filter:
            files = [f for f in files if f.get("status") == status_filter]

        total = len(files)

        # Apply pagination
        files = files[offset : offset + limit]

        return {
            "files": files,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
        }

    async def get_failed_files(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get list of failed files with their errors."""
        result = await self.get_batch_files(batch_id, status_filter=FileStatus.FAILED)
        return result["files"]

    async def get_user_batches(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get recent batches for a user.

        Note: This requires scanning Redis keys, which can be slow for large datasets.
        Consider implementing a separate index for production use.
        """
        batches = []

        if await self._is_redis_available():
            redis = await self._get_redis()
            pattern = f"{self.BATCH_KEY_PREFIX}*"
            cursor = 0

            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)

                for key in keys:
                    data = await redis.get(key)
                    if data:
                        batch = _json_loads(data)
                        if batch.get("user_id") == user_id:
                            if status is None or batch.get("status") == status:
                                batches.append({
                                    "batch_id": batch["batch_id"],
                                    "status": batch["status"],
                                    "created_at": batch["created_at"],
                                    "stats": batch["stats"],
                                })

                if cursor == 0:
                    break

        else:
            # Local cache fallback
            for batch in self._local_cache.values():
                if batch.get("user_id") == user_id:
                    if status is None or batch.get("status") == status:
                        batches.append({
                            "batch_id": batch["batch_id"],
                            "status": batch["status"],
                            "created_at": batch["created_at"],
                            "stats": batch["stats"],
                        })

        # Sort by created_at descending and limit
        batches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return batches[:limit]

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def delete_batch(self, batch_id: str):
        """Delete a batch and all its data."""
        if await self._is_redis_available():
            redis = await self._get_redis()
            key = f"{self.BATCH_KEY_PREFIX}{batch_id}"
            await redis.delete(key)
        else:
            self._local_cache.pop(batch_id, None)

        logger.info("Batch deleted", batch_id=batch_id)

    async def cleanup_old_batches(self, days: int = 7):
        """Clean up batches older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0

        if await self._is_redis_available():
            redis = await self._get_redis()
            pattern = f"{self.BATCH_KEY_PREFIX}*"
            cursor = 0

            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)

                for key in keys:
                    data = await redis.get(key)
                    if data:
                        batch = _json_loads(data)
                        created_at = datetime.fromisoformat(batch.get("created_at", ""))
                        if created_at < cutoff:
                            await redis.delete(key)
                            deleted_count += 1

                if cursor == 0:
                    break
        else:
            # Local cache cleanup
            to_delete = []
            for batch_id, batch in self._local_cache.items():
                created_at = datetime.fromisoformat(batch.get("created_at", ""))
                if created_at < cutoff:
                    to_delete.append(batch_id)

            for batch_id in to_delete:
                del self._local_cache[batch_id]
                deleted_count += 1

        logger.info("Cleaned up old batches", deleted_count=deleted_count, days=days)
        return deleted_count


# =============================================================================
# Global Instance
# =============================================================================

_progress_tracker: Optional[BulkProgressTracker] = None


async def get_progress_tracker() -> BulkProgressTracker:
    """Get or create the global progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = BulkProgressTracker()
    return _progress_tracker
