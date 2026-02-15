"""
AIDocumentIndexer - File Watcher API Routes
============================================

API endpoints for managing the file watcher service.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.api.middleware.auth import get_current_user, require_admin

logger = structlog.get_logger(__name__)
from backend.db.database import get_async_session
from backend.services.watcher_db import (
    get_watcher_service_db,
    FileWatcherServiceDB,
    WatchedDirectory,
    WatcherStatus,
    WatchEventType,
    FileEvent,
    WatcherStats,
)

router = APIRouter(tags=["File Watcher"])


# =============================================================================
# Request/Response Models
# =============================================================================

class AddWatchDirectoryRequest(BaseModel):
    """Request to add a directory to watch."""
    path: str = Field(..., description="Absolute path to the directory to watch")
    recursive: bool = Field(default=True, description="Watch subdirectories recursively")
    auto_process: bool = Field(default=True, description="Automatically process new files")
    access_tier: int = Field(default=1, ge=1, le=5, description="Access tier for uploaded documents (1-5)")
    collection: Optional[str] = Field(default=None, description="Collection to assign documents to")
    folder_id: Optional[str] = Field(default=None, description="Target folder ID for uploaded documents")


class WatchDirectoryResponse(BaseModel):
    """Response for a watched directory."""
    id: str
    path: str
    recursive: bool
    auto_process: bool
    access_tier: int
    collection: Optional[str]
    folder_id: Optional[str]
    enabled: bool
    created_at: datetime


class FileEventResponse(BaseModel):
    """Response for a file event."""
    id: str
    event_type: str
    file_path: str
    file_name: str
    file_extension: str
    file_size: int
    file_hash: Optional[str]
    watch_dir_id: str
    timestamp: datetime
    processed: bool
    processing_error: Optional[str]


class WatcherStatsResponse(BaseModel):
    """Response for watcher statistics."""
    status: str
    directories_watched: int
    events_queued: int
    events_processed: int
    events_failed: int
    last_event_time: Optional[datetime]
    uptime_seconds: float


class ScanDirectoryRequest(BaseModel):
    """Request to scan a directory for existing files."""
    process_existing: bool = Field(default=True, description="Queue existing files for processing")


class ScanDirectoryResponse(BaseModel):
    """Response for directory scan."""
    files_found: int
    files_queued: int
    files: List[str]


# =============================================================================
# Helper Functions
# =============================================================================

def watch_dir_to_response(watch_dir: WatchedDirectory) -> WatchDirectoryResponse:
    """Convert WatchedDirectory to response model."""
    return WatchDirectoryResponse(
        id=watch_dir.id,
        path=watch_dir.path,
        recursive=watch_dir.recursive,
        auto_process=watch_dir.auto_process,
        access_tier=watch_dir.access_tier,
        collection=watch_dir.collection,
        folder_id=watch_dir.folder_id,
        enabled=watch_dir.enabled,
        created_at=watch_dir.created_at,
    )


def file_event_to_response(event: FileEvent) -> FileEventResponse:
    """Convert FileEvent to response model."""
    return FileEventResponse(
        id=event.id,
        event_type=event.event_type.value,
        file_path=event.file_path,
        file_name=event.file_name,
        file_extension=event.file_extension,
        file_size=event.file_size,
        file_hash=event.file_hash,
        watch_dir_id=event.watch_dir_id,
        timestamp=event.timestamp,
        processed=event.processed,
        processing_error=event.processing_error,
    )


def stats_to_response(stats: WatcherStats) -> WatcherStatsResponse:
    """Convert WatcherStats to response model."""
    return WatcherStatsResponse(
        status=stats.status.value,
        directories_watched=stats.directories_watched,
        events_queued=stats.events_queued,
        events_processed=stats.events_processed,
        events_failed=stats.events_failed,
        last_event_time=stats.last_event_time,
        uptime_seconds=stats.uptime_seconds,
    )


# =============================================================================
# Routes
# =============================================================================

@router.get("/status", response_model=WatcherStatsResponse)
async def get_watcher_status(
    current_user: dict = Depends(require_admin),
):
    """
    Get the current status and statistics of the file watcher service.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()
    stats = watcher.get_stats()
    return stats_to_response(stats)


@router.post("/start")
async def start_watcher(
    current_user: dict = Depends(require_admin),
):
    """
    Start the file watcher service.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()

    if watcher._status == WatcherStatus.RUNNING:
        return {"message": "File watcher is already running", "status": "running"}

    try:
        watcher.start()
        return {"message": "File watcher started successfully", "status": "running"}
    except Exception as e:
        logger.error("Failed to start file watcher", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start file watcher",
        )


@router.post("/stop")
async def stop_watcher(
    current_user: dict = Depends(require_admin),
):
    """
    Stop the file watcher service.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()

    if watcher._status != WatcherStatus.RUNNING:
        return {"message": "File watcher is not running", "status": watcher._status.value}

    try:
        watcher.stop()
        return {"message": "File watcher stopped successfully", "status": "stopped"}
    except Exception as e:
        logger.error("Failed to stop file watcher", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop file watcher",
        )


@router.post("/pause")
async def pause_watcher(
    current_user: dict = Depends(require_admin),
):
    """
    Pause the file watcher service (keeps observer running but ignores events).

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()

    if watcher._status != WatcherStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File watcher must be running to pause",
        )

    watcher.pause()
    return {"message": "File watcher paused", "status": "paused"}


@router.post("/resume")
async def resume_watcher(
    current_user: dict = Depends(require_admin),
):
    """
    Resume the file watcher service after pausing.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()

    if watcher._status != WatcherStatus.PAUSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File watcher must be paused to resume",
        )

    watcher.resume()
    return {"message": "File watcher resumed", "status": "running"}


@router.get("/directories", response_model=List[WatchDirectoryResponse])
async def list_watch_directories(
    current_user: dict = Depends(require_admin),
):
    """
    List all watched directories.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()
    directories = watcher.list_watch_directories()
    return [watch_dir_to_response(d) for d in directories]


@router.post("/directories", response_model=WatchDirectoryResponse)
async def add_watch_directory(
    request: AddWatchDirectoryRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Add a new directory to watch.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()

    try:
        watch_dir = await watcher.add_watch_directory(
            path=request.path,
            recursive=request.recursive,
            auto_process=request.auto_process,
            access_tier=request.access_tier,
            collection=request.collection,
            folder_id=request.folder_id,
            organization_id=current_user.get("organization_id"),
            created_by_id=current_user.get("id"),
        )
        return watch_dir_to_response(watch_dir)
    except ValueError as e:
        logger.warning("Invalid watch directory request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid watch directory configuration",
        )
    except Exception as e:
        logger.error("Failed to add watch directory", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add watch directory",
        )


@router.get("/directories/{watch_dir_id}", response_model=WatchDirectoryResponse)
async def get_watch_directory(
    watch_dir_id: str,
    current_user: dict = Depends(require_admin),
):
    """
    Get a specific watched directory.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()
    watch_dir = watcher.get_watch_directory(watch_dir_id)

    if not watch_dir:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Watch directory not found: {watch_dir_id}",
        )

    return watch_dir_to_response(watch_dir)


@router.delete("/directories/{watch_dir_id}")
async def remove_watch_directory(
    watch_dir_id: str,
    current_user: dict = Depends(require_admin),
):
    """
    Remove a watched directory.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()

    if not await watcher.remove_watch_directory(watch_dir_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Watch directory not found: {watch_dir_id}",
        )

    return {"message": "Watch directory removed", "id": watch_dir_id}


@router.post("/directories/{watch_dir_id}/scan", response_model=ScanDirectoryResponse)
async def scan_directory(
    watch_dir_id: str,
    request: ScanDirectoryRequest = ScanDirectoryRequest(),
    current_user: dict = Depends(require_admin),
):
    """
    Scan a watched directory for existing files.

    Useful for initial ingestion of existing files in a directory.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()

    try:
        events = watcher.scan_directory(
            watch_dir_id=watch_dir_id,
            process_existing=request.process_existing,
        )

        return ScanDirectoryResponse(
            files_found=len(events),
            files_queued=len(events) if request.process_existing else 0,
            files=[e.file_name for e in events[:100]],  # Limit to first 100 for response
        )
    except ValueError as e:
        logger.warning("Watch directory not found", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watch directory not found",
        )
    except Exception as e:
        logger.error("Failed to scan directory", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to scan directory",
        )


@router.post("/directories/{watch_dir_id}/toggle")
async def toggle_watch_directory(
    watch_dir_id: str,
    current_user: dict = Depends(require_admin),
):
    """
    Toggle a watched directory on/off.

    Requires admin privileges.
    """
    watcher = get_watcher_service_db()
    watch_dir = watcher.get_watch_directory(watch_dir_id)

    if not watch_dir:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Watch directory not found: {watch_dir_id}",
        )

    new_enabled = await watcher.toggle_watch_directory(watch_dir_id)

    return {
        "message": f"Watch directory {'enabled' if new_enabled else 'disabled'}",
        "id": watch_dir_id,
        "enabled": new_enabled,
    }


@router.get("/events", response_model=List[FileEventResponse])
async def get_queued_events(
    limit: int = 100,
    include_processed: bool = False,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get queued file events.

    Requires admin privileges.
    """
    from backend.services.watcher_db import get_pending_events_from_db, FileWatcherEventModel, EventStatus
    from sqlalchemy import select

    try:
        if include_processed:
            # Get all events
            query = (
                select(FileWatcherEventModel)
                .order_by(FileWatcherEventModel.detected_at.desc())
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
                    processed=db_event.status in [EventStatus.COMPLETED.value, EventStatus.FAILED.value],
                    processing_error=db_event.processing_error,
                ))
        else:
            events = await get_pending_events_from_db(db, limit=limit)

        return [file_event_to_response(e) for e in events]
    except Exception as e:
        logger.error("Failed to get events", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get events",
        )


@router.post("/events/clear")
async def clear_processed_events(
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Clear processed events from the queue (deletes completed/failed events).

    Requires admin privileges.
    """
    from backend.services.watcher_db import EventStatus
    from sqlalchemy import delete

    try:
        # Count before delete
        from sqlalchemy import func, select
        from backend.db.models import FileWatcherEvent as FileWatcherEventModel

        count_query = select(func.count()).select_from(FileWatcherEventModel).where(
            FileWatcherEventModel.status.in_([EventStatus.COMPLETED.value, EventStatus.FAILED.value])
        )
        result = await db.execute(count_query)
        cleared_count = result.scalar() or 0

        # Delete completed/failed events
        stmt = delete(FileWatcherEventModel).where(
            FileWatcherEventModel.status.in_([EventStatus.COMPLETED.value, EventStatus.FAILED.value])
        )
        await db.execute(stmt)
        await db.commit()

        # Count remaining
        remaining_query = select(func.count()).select_from(FileWatcherEventModel)
        result = await db.execute(remaining_query)
        remaining_count = result.scalar() or 0

        return {
            "message": "Processed events cleared",
            "events_cleared": cleared_count,
            "events_remaining": remaining_count,
        }
    except Exception as e:
        logger.error("Failed to clear events", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear events",
        )


@router.post("/events/{event_id}/retry")
async def retry_failed_event(
    event_id: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Retry a failed event by marking it as pending.

    Requires admin privileges.
    """
    from backend.services.watcher_db import EventStatus, update_event_status_in_db
    from backend.db.models import FileWatcherEvent as FileWatcherEventModel
    from sqlalchemy import select
    import uuid

    try:
        # Get the event
        query = select(FileWatcherEventModel).where(FileWatcherEventModel.id == uuid.UUID(event_id))
        result = await db.execute(query)
        db_event = result.scalar_one_or_none()

        if not db_event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event not found: {event_id}",
            )

        if db_event.status not in [EventStatus.COMPLETED.value, EventStatus.FAILED.value]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Event is not processed yet",
            )

        # Reset to pending
        await update_event_status_in_db(db, event_id, EventStatus.PENDING)

        return {
            "message": "Event marked for retry",
            "id": event_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry event", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry event",
        )
