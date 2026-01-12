"""
AIDocumentIndexer - Upload API Routes
=====================================

Endpoints for file upload and processing management.
Now with persistent storage using the UploadJob model.
"""

import os
import asyncio
import hashlib
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from backend.services.pipeline import (
    DocumentPipeline,
    PipelineConfig,
    ProcessingStatus as PipelineStatus,
    ProcessingResult,
    get_pipeline,
)
from backend.db.models import ProcessingMode, UploadJob, UploadStatus
from backend.api.websocket import (
    notify_processing_started,
    notify_processing_progress,
    notify_processing_complete,
    notify_processing_error,
)
from backend.services.auto_tagger import AutoTaggerService
from backend.db.database import get_async_session
from backend.db.models import Document, Chunk, AccessTier
from backend.api.middleware.auth import get_user_context_optional
from backend.services.permissions import UserContext
from sqlalchemy import select, and_, update, delete
from sqlalchemy.orm import selectinload

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Startup Functions
# =============================================================================

async def reset_stuck_upload_jobs() -> int:
    """
    Reset stuck upload jobs on startup.

    This function:
    1. Marks upload jobs as COMPLETED if their document is already completed
    2. Resets stuck processing jobs (not updated in 5+ minutes) to QUEUED

    Returns:
        Number of jobs reset/fixed
    """
    from datetime import datetime, timedelta

    fixed_count = 0

    async for session in get_async_session():
        # 1. Mark jobs as COMPLETED if document already completed
        result = await session.execute(
            select(UploadJob).where(
                UploadJob.status != UploadStatus.COMPLETED,
                UploadJob.file_hash.in_(
                    select(Document.file_hash).where(Document.processing_status == "COMPLETED")
                )
            )
        )
        already_completed = result.scalars().all()

        for job in already_completed:
            job.status = UploadStatus.COMPLETED
            job.progress = 100
            job.current_step = "Completed"
            fixed_count += 1
            logger.info("Marked upload job as completed (document exists)",
                       job_id=str(job.id), filename=job.filename)

        # 2. Reset stuck jobs (processing for more than 5 minutes)
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        result = await session.execute(
            select(UploadJob).where(
                UploadJob.status.in_([
                    UploadStatus.VALIDATING,
                    UploadStatus.EXTRACTING,
                    UploadStatus.CHUNKING,
                    UploadStatus.EMBEDDING,
                    UploadStatus.INDEXING,
                ]),
                UploadJob.updated_at < cutoff_time
            )
        )
        stuck_jobs = result.scalars().all()

        for job in stuck_jobs:
            # Check if document already completed for this job
            doc_result = await session.execute(
                select(Document).where(Document.file_hash == job.file_hash,
                                       Document.processing_status == "COMPLETED")
            )
            existing_doc = doc_result.scalar_one_or_none()

            if existing_doc:
                job.status = UploadStatus.COMPLETED
                job.progress = 100
                job.current_step = "Completed"
                logger.info("Marked stuck job as completed (document exists)",
                           job_id=str(job.id), filename=job.filename)
            else:
                job.status = UploadStatus.QUEUED
                job.progress = 0
                job.current_step = "Queued"
                job.error_message = None
                logger.info("Reset stuck job to queued",
                           job_id=str(job.id), filename=job.filename)
            fixed_count += 1

        await session.commit()

    return fixed_count


# Track last sync time to avoid running on every request
_last_sync_time: Optional[datetime] = None


async def sync_upload_jobs_with_documents(force: bool = False) -> int:
    """
    Sync upload jobs with completed documents.

    This is a lightweight sync that only updates jobs where the document
    is already completed but the job status wasn't updated.

    Only runs every 30 seconds to avoid unnecessary database load,
    unless force=True.

    Returns:
        Number of jobs synced
    """
    global _last_sync_time

    # Skip if we synced recently (within 30 seconds) unless forced
    if not force and _last_sync_time:
        if (datetime.now() - _last_sync_time).total_seconds() < 30:
            return 0

    _last_sync_time = datetime.now()
    synced_count = 0

    async for session in get_async_session():
        # Find jobs that are not COMPLETED but their document IS completed
        result = await session.execute(
            select(UploadJob).where(
                UploadJob.status != UploadStatus.COMPLETED,
                UploadJob.status != UploadStatus.FAILED,
                UploadJob.status != UploadStatus.CANCELLED,
                UploadJob.file_hash.in_(
                    select(Document.file_hash).where(Document.processing_status == "COMPLETED")
                )
            )
        )
        jobs_to_sync = result.scalars().all()

        for job in jobs_to_sync:
            job.status = UploadStatus.COMPLETED
            job.progress = 100
            job.current_step = "Completed"
            synced_count += 1
            logger.info("Synced upload job to completed",
                       job_id=str(job.id), filename=job.filename)

        if synced_count > 0:
            await session.commit()

    return synced_count


# In-memory cache for fast access (backed by database for persistence)
# This provides both speed (in-memory) and persistence (database)
_processing_status_cache: Dict[str, Dict] = {}

# Upload directory - use persistent storage, not /tmp which gets cleared on reboot
# Default to ./data/uploads relative to project root
_default_upload_dir = Path(__file__).resolve().parents[3] / "data" / "uploads"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(_default_upload_dir)))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Constants
# =============================================================================

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {
    # Documents
    "pdf", "doc", "docx", "txt", "md", "rtf", "odt",
    # Presentations
    "ppt", "pptx", "odp", "key",
    # Spreadsheets
    "xls", "xlsx", "csv", "ods",
    # Images
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "svg",
    # Archives
    "zip", "rar", "7z", "tar", "gz",
    # Audio/Video
    "mp3", "wav", "mp4", "mov", "avi", "mkv",
    # Other
    "json", "xml", "html", "htm",
}


# =============================================================================
# Pydantic Models
# =============================================================================

class UploadResponse(BaseModel):
    """Single file upload response."""
    file_id: UUID
    filename: str
    file_size: int
    file_hash: str
    status: str
    message: str


class BatchUploadResponse(BaseModel):
    """Batch upload response."""
    total_files: int
    successful: int
    failed: int
    files: List[UploadResponse]


class ProcessingStatus(BaseModel):
    """Document processing status."""
    file_id: UUID
    filename: str
    status: str
    progress: int
    current_step: str
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ProcessingQueueResponse(BaseModel):
    """Processing queue status."""
    queue_length: int
    active_tasks: int
    items: List[ProcessingStatus]


class UploadOptions(BaseModel):
    """Upload processing options."""
    collection: Optional[str] = None
    folder_id: Optional[str] = None  # Target folder ID for the uploaded document
    access_tier: int = Field(default=1, ge=1, le=100)
    enable_ocr: bool = True
    enable_image_analysis: bool = True
    smart_chunking: bool = True
    detect_duplicates: bool = True
    auto_generate_tags: bool = False  # Use LLM to auto-generate tags if no collection set
    chunking_strategy: str = Field(
        default="semantic",
        description="Chunking strategy: 'simple', 'semantic', or 'hierarchical'"
    )
    enable_contextual_headers: bool = Field(
        default=True,
        description="Prepend document context (title, section) to each chunk for better retrieval"
    )
    # Multi-tenant fields (populated from user context)
    organization_id: Optional[str] = None
    uploaded_by_id: Optional[str] = None
    is_private: bool = False


# =============================================================================
# Helper Functions
# =============================================================================

def get_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file."""
    # Check extension
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"File type .{ext} is not supported"

    # Check content type
    if file.content_type:
        # Basic MIME type validation
        pass

    return True, ""


# =============================================================================
# Database Helper Functions for UploadJob persistence
# =============================================================================

async def create_upload_job(
    file_id: UUID,
    filename: str,
    file_path: str,
    file_hash: str,
    file_size: int,
    collection: Optional[str] = None,
    access_tier: int = 1,
    enable_ocr: bool = True,
    enable_image_analysis: bool = True,
    auto_generate_tags: bool = False,
) -> UploadJob:
    """Create a new upload job in the database."""
    async for session in get_async_session():
        job = UploadJob(
            id=file_id,
            filename=filename,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            status=UploadStatus.QUEUED,
            progress=0,
            current_step="Queued",
            collection=collection,
            access_tier=access_tier,
            enable_ocr=enable_ocr,
            enable_image_analysis=enable_image_analysis,
            auto_generate_tags=auto_generate_tags,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        # Also cache in memory
        _processing_status_cache[str(file_id)] = {
            "file_id": str(file_id),
            "filename": filename,
            "file_path": file_path,
            "file_hash": file_hash,
            "file_size": file_size,
            "status": UploadStatus.QUEUED.value,
            "progress": 0,
            "current_step": "Queued",
            "collection": collection,
            "access_tier": access_tier,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "error": None,
        }

        return job


async def get_upload_job(file_id: str) -> Optional[Dict]:
    """Get upload job from cache or database."""
    # Check cache first
    if file_id in _processing_status_cache:
        return _processing_status_cache[file_id]

    # Fall back to database
    async for session in get_async_session():
        result = await session.execute(
            select(UploadJob).where(UploadJob.id == file_id)
        )
        job = result.scalar_one_or_none()

        if job:
            # Cache the result
            status_dict = {
                "file_id": str(job.id),
                "filename": job.filename,
                "file_path": job.file_path,
                "file_hash": job.file_hash,
                "file_size": job.file_size,
                "status": job.status.value if job.status else "unknown",
                "progress": job.progress or 0,
                "current_step": job.current_step or "Unknown",
                "collection": job.collection,
                "access_tier": job.access_tier,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "error": job.error_message,
                "chunk_count": job.chunk_count,
                "word_count": job.word_count,
                "document_id": str(job.document_id) if job.document_id else None,
            }
            _processing_status_cache[file_id] = status_dict
            return status_dict

        return None


async def update_upload_job_status(
    file_id: str,
    status: UploadStatus,
    progress: int,
    current_step: str,
    error_message: Optional[str] = None,
    chunk_count: Optional[int] = None,
    word_count: Optional[int] = None,
    document_id: Optional[str] = None,
) -> None:
    """Update upload job status in database and cache."""
    # Convert string file_id to UUID for database comparison
    try:
        file_uuid = UUID(file_id)
    except (ValueError, TypeError):
        logger.error("Invalid file_id format for status update", file_id=file_id)
        return

    async for session in get_async_session():
        # Build update dict
        update_dict = {
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "updated_at": datetime.now(),
        }
        if error_message is not None:
            update_dict["error_message"] = error_message
        if chunk_count is not None:
            update_dict["chunk_count"] = chunk_count
        if word_count is not None:
            update_dict["word_count"] = word_count
        if document_id is not None:
            update_dict["document_id"] = document_id

        await session.execute(
            update(UploadJob)
            .where(UploadJob.id == file_uuid)
            .values(**update_dict)
        )
        await session.commit()

        # Update cache
        if file_id in _processing_status_cache:
            _processing_status_cache[file_id]["status"] = status.value
            _processing_status_cache[file_id]["progress"] = progress
            _processing_status_cache[file_id]["current_step"] = current_step
            _processing_status_cache[file_id]["updated_at"] = datetime.now()
            if error_message is not None:
                _processing_status_cache[file_id]["error"] = error_message
            if chunk_count is not None:
                _processing_status_cache[file_id]["chunk_count"] = chunk_count
            if word_count is not None:
                _processing_status_cache[file_id]["word_count"] = word_count
            if document_id is not None:
                _processing_status_cache[file_id]["document_id"] = document_id


async def check_duplicate_upload(file_hash: str) -> Optional[Dict]:
    """Check if a file with the same hash already exists."""
    async for session in get_async_session():
        result = await session.execute(
            select(UploadJob)
            .where(
                and_(
                    UploadJob.file_hash == file_hash,
                    UploadJob.status == UploadStatus.COMPLETED
                )
            )
            .order_by(UploadJob.created_at.desc())
            .limit(1)
        )
        job = result.scalar_one_or_none()

        if job:
            return {
                "file_id": str(job.id),
                "filename": job.filename,
                "document_id": str(job.document_id) if job.document_id else None,
            }
        return None


async def get_all_upload_jobs(limit: int = 100) -> List[Dict]:
    """Get all recent upload jobs."""
    async for session in get_async_session():
        result = await session.execute(
            select(UploadJob)
            .order_by(UploadJob.created_at.desc())
            .limit(limit)
        )
        jobs = result.scalars().all()

        return [
            {
                "file_id": str(job.id),
                "filename": job.filename,
                "status": job.status.value if job.status else "unknown",
                "progress": job.progress or 0,
                "current_step": job.current_step or "Unknown",
                "error": job.error_message,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            }
            for job in jobs
        ]


# Map PipelineStatus to UploadStatus
def pipeline_to_upload_status(pipeline_status: PipelineStatus) -> UploadStatus:
    """Convert PipelineStatus to UploadStatus."""
    mapping = {
        PipelineStatus.PENDING: UploadStatus.QUEUED,
        PipelineStatus.VALIDATING: UploadStatus.VALIDATING,
        PipelineStatus.EXTRACTING: UploadStatus.EXTRACTING,
        PipelineStatus.CHUNKING: UploadStatus.CHUNKING,
        PipelineStatus.EMBEDDING: UploadStatus.EMBEDDING,
        PipelineStatus.INDEXING: UploadStatus.INDEXING,
        PipelineStatus.COMPLETED: UploadStatus.COMPLETED,
        PipelineStatus.FAILED: UploadStatus.FAILED,
    }
    return mapping.get(pipeline_status, UploadStatus.QUEUED)


async def update_processing_status(file_id: str, status: PipelineStatus):
    """Update processing status in database, cache, and notify via WebSocket."""
    # Map status to progress percentage
    progress_map = {
        PipelineStatus.PENDING: 0,
        PipelineStatus.VALIDATING: 10,
        PipelineStatus.EXTRACTING: 30,
        PipelineStatus.CHUNKING: 50,
        PipelineStatus.EMBEDDING: 70,
        PipelineStatus.INDEXING: 90,
        PipelineStatus.COMPLETED: 100,
        PipelineStatus.FAILED: 0,
    }
    progress = progress_map.get(status, 0)
    current_step = status.value.replace("_", " ").title()

    # Convert to UploadStatus
    upload_status = pipeline_to_upload_status(status)

    # Update database and cache
    await update_upload_job_status(
        file_id=file_id,
        status=upload_status,
        progress=progress,
        current_step=current_step,
    )

    # Send WebSocket notification
    await notify_processing_progress(
        file_id=file_id,
        status=status.value,
        progress=progress,
        current_step=current_step,
    )


def run_async_task(coro):
    """
    Run an async coroutine in the current event loop.

    FastAPI's BackgroundTasks can run async functions, but we need to
    ensure proper exception handling and logging.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is running, create a task
            asyncio.create_task(coro)
        else:
            # If no loop is running, run until complete
            loop.run_until_complete(coro)
    except Exception as e:
        logger.error("Failed to run async task", error=str(e))


async def _auto_tag_document(document_id: str, filename: str):
    """
    Auto-generate tags for a document using LLM.

    Gets document chunks from database, calls AutoTaggerService,
    and updates the document with generated tags.
    """
    try:
        # Convert string to UUID for database queries
        from uuid import UUID as PyUUID
        try:
            doc_uuid = PyUUID(document_id)
        except (ValueError, TypeError):
            logger.error("Invalid document ID for auto-tagging", document_id=document_id)
            return

        async for session in get_async_session():
            # Get document
            doc_query = select(Document).where(Document.id == doc_uuid)
            result = await session.execute(doc_query)
            document = result.scalar_one_or_none()

            if not document:
                logger.warning("Document not found for auto-tagging", document_id=document_id)
                return

            # Get first few chunks for content sample
            chunks_query = (
                select(Chunk)
                .where(Chunk.document_id == doc_uuid)
                .order_by(Chunk.chunk_index)
                .limit(3)
            )
            chunks_result = await session.execute(chunks_query)
            chunks = chunks_result.scalars().all()

            if not chunks:
                logger.warning("No chunks found for auto-tagging", document_id=document_id)
                return

            content_sample = "\n".join([c.content for c in chunks if c.content])

            # Get existing collections for context
            collections_query = (
                select(Document.tags)
                .where(Document.tags.isnot(None))
                .distinct()
            )
            collections_result = await session.execute(collections_query)
            existing_tags = collections_result.scalars().all()

            # Flatten and deduplicate tags
            existing_collections = list(set(
                tag for tags in existing_tags if tags for tag in tags
            ))

            # Generate tags using LLM
            auto_tagger = AutoTaggerService()
            tags = await auto_tagger.generate_tags(
                document_name=filename,
                content_sample=content_sample,
                existing_collections=existing_collections,
                max_tags=5
            )

            if tags:
                # Merge auto-generated tags with existing user tags (preserve user tags)
                existing_tags = document.tags or []
                # Use dict.fromkeys to preserve order and remove duplicates (existing tags first)
                merged_tags = list(dict.fromkeys(existing_tags + tags))
                document.tags = merged_tags
                await session.commit()

                logger.info(
                    "Auto-generated tags for document",
                    document_id=document_id,
                    filename=filename,
                    tags=tags
                )
            else:
                logger.warning(
                    "No tags generated for document",
                    document_id=document_id,
                    filename=filename
                )

    except Exception as e:
        logger.error(
            "Failed to auto-tag document",
            document_id=document_id,
            filename=filename,
            error=str(e),
            exc_info=True
        )


async def process_document_background(
    file_id: UUID,
    file_path: str,
    options: UploadOptions,
):
    """
    Background task for document processing.

    Uses the DocumentPipeline for processing with Ray support.
    Sends real-time updates via WebSocket.
    """
    file_id_str = str(file_id)

    # Get filename from cache or database
    job_info = await get_upload_job(file_id_str)
    filename = job_info.get("filename", "unknown") if job_info else "unknown"

    logger.info(
        "Starting background processing",
        file_id=file_id_str,
        file_path=file_path,
    )

    try:
        # Notify that processing has started
        await notify_processing_started(file_id_str, filename)
    except Exception as e:
        logger.warning("Failed to send WebSocket notification", error=str(e))

    # Determine processing mode
    if options.enable_ocr and options.enable_image_analysis:
        processing_mode = ProcessingMode.FULL
    elif options.enable_ocr:
        processing_mode = ProcessingMode.OCR_ENABLED
    else:
        processing_mode = ProcessingMode.BASIC

    # Create async status callback
    async def status_callback(doc_id: str, status: PipelineStatus):
        await update_processing_status(file_id_str, status)

    # Create async progress callback for real-time progress updates
    async def progress_callback(doc_id: str, current: int, total: int):
        progress_percent = int((current / total) * 100) if total > 0 else 0
        step_names = {
            1: "Extracting text",
            2: "Chunking document",
            3: "Generating embeddings",
            4: "Storing in database",
            5: "Finalizing",
        }
        current_step = step_names.get(current, f"Step {current}/{total}")
        await notify_processing_progress(
            file_id=file_id_str,
            status="processing",
            progress=progress_percent,
            current_step=current_step,
        )

    # Create pipeline config with status and progress callbacks
    config = PipelineConfig(
        processing_mode=processing_mode,
        use_ray=True,
        check_duplicates=options.detect_duplicates,
        on_status_change=status_callback,
        on_progress=progress_callback,
    )

    pipeline = DocumentPipeline(config=config)

    try:
        result = await pipeline.process_document(
            file_path=file_path,
            document_id=file_id_str,
            metadata={"original_filename": filename},
            access_tier=options.access_tier,
            collection=options.collection,
            folder_id=options.folder_id,
            # Multi-tenant parameters for organization isolation
            organization_id=options.organization_id,
            uploaded_by_id=options.uploaded_by_id,
            is_private=options.is_private,
        )

        # Update final status in database
        if result.error_message:
            await update_upload_job_status(
                file_id=file_id_str,
                status=UploadStatus.FAILED,
                progress=0,
                current_step="Failed",
                error_message=result.error_message,
            )
            await notify_processing_error(file_id_str, result.error_message)
        else:
            await update_upload_job_status(
                file_id=file_id_str,
                status=UploadStatus.COMPLETED,
                progress=100,
                current_step="Completed",
                chunk_count=result.chunk_count,
                word_count=result.word_count,
                document_id=result.document_id,
            )

            # Auto-generate tags if enabled
            if options.auto_generate_tags:
                await _auto_tag_document(
                    document_id=result.document_id or file_id_str,
                    filename=filename,
                )

            # Notify successful completion via WebSocket
            await notify_processing_complete(
                file_id=file_id_str,
                document_id=result.document_id or file_id_str,
                chunk_count=result.chunk_count,
                word_count=result.word_count,
            )

        logger.info(
            "Background processing complete",
            file_id=file_id_str,
            status=result.status.value,
            chunks=result.chunk_count,
        )

    except Exception as e:
        logger.error(
            "Background processing failed",
            file_id=file_id_str,
            error=str(e),
        )
        # Update database with failure status
        await update_upload_job_status(
            file_id=file_id_str,
            status=UploadStatus.FAILED,
            progress=0,
            current_step="Failed",
            error_message=str(e),
        )

        # Notify error via WebSocket
        await notify_processing_error(file_id_str, str(e))


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/single", response_model=UploadResponse)
async def upload_single_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
    folder_id: Optional[str] = Form(None, description="Target folder ID for the document"),
    access_tier: int = Form(default=1, ge=1, le=100),
    enable_ocr: bool = Form(True),
    enable_image_analysis: bool = Form(True),
    smart_chunking: bool = Form(True),
    detect_duplicates: bool = Form(True),
    auto_generate_tags: bool = Form(False),
    is_private: bool = Form(False, description="Make document private (only visible to uploader)"),
    user: Optional["UserContext"] = Depends(get_user_context_optional),
):
    """
    Upload a single file for processing.

    The file will be validated, stored, and queued for processing.
    Processing happens asynchronously using Ray.
    """
    logger.info(
        "Uploading single file",
        filename=file.filename,
        content_type=file.content_type,
    )

    # Validate file
    is_valid, error_message = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    # Calculate hash
    file_hash = get_file_hash(content)

    # Check for duplicates if enabled (now using database)
    if detect_duplicates:
        existing = await check_duplicate_upload(file_hash)
        if existing:
            return UploadResponse(
                file_id=UUID(existing["file_id"]),
                filename=file.filename or "unknown",
                file_size=len(content),
                file_hash=file_hash,
                status="duplicate",
                message=f"File already exists with ID {existing['file_id']}",
            )

    # Generate file ID
    file_id = uuid4()

    # Save file to storage
    file_ext = Path(file.filename or "file").suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"

    with open(file_path, "wb") as f:
        f.write(content)

    logger.info("File saved", file_id=str(file_id), path=str(file_path))

    # Create processing options with multi-tenant fields from user context
    options = UploadOptions(
        collection=collection,
        folder_id=folder_id,
        access_tier=access_tier,
        enable_ocr=enable_ocr,
        enable_image_analysis=enable_image_analysis,
        smart_chunking=smart_chunking,
        detect_duplicates=detect_duplicates,
        auto_generate_tags=auto_generate_tags,
        # Multi-tenant fields from authenticated user
        organization_id=user.organization_id if user else None,
        uploaded_by_id=user.user_id if user else None,
        is_private=is_private,
    )

    # Create upload job in database (persistent storage)
    await create_upload_job(
        file_id=file_id,
        filename=file.filename or "unknown",
        file_path=str(file_path),
        file_hash=file_hash,
        file_size=len(content),
        collection=collection,
        access_tier=access_tier,
        enable_ocr=enable_ocr,
        enable_image_analysis=enable_image_analysis,
        auto_generate_tags=auto_generate_tags,
    )

    # Queue for background processing
    background_tasks.add_task(
        process_document_background,
        file_id=file_id,
        file_path=str(file_path),
        options=options,
    )

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or "unknown",
        file_size=len(content),
        file_hash=file_hash,
        status="queued",
        message="File uploaded successfully. Processing will begin shortly.",
    )


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection: Optional[str] = Form(None),
    folder_id: Optional[str] = Form(None, description="Target folder ID for the documents"),
    access_tier: int = Form(default=1, ge=1, le=100),
    enable_ocr: bool = Form(True),
    is_private: bool = Form(False, description="Make documents private (only visible to uploader)"),
    user: Optional[UserContext] = Depends(get_user_context_optional),
):
    """
    Upload multiple files at once.

    All files are validated and queued for parallel processing with Ray.
    """
    logger.info("Uploading batch", file_count=len(files))

    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 files per batch upload",
        )

    results = []
    successful = 0
    failed = 0

    for file in files:
        try:
            # Validate
            is_valid, error_message = validate_file(file)
            if not is_valid:
                results.append(
                    UploadResponse(
                        file_id=uuid4(),
                        filename=file.filename or "unknown",
                        file_size=0,
                        file_hash="",
                        status="failed",
                        message=error_message,
                    )
                )
                failed += 1
                continue

            # Read and process
            content = await file.read()

            if len(content) > MAX_FILE_SIZE:
                results.append(
                    UploadResponse(
                        file_id=uuid4(),
                        filename=file.filename or "unknown",
                        file_size=len(content),
                        file_hash="",
                        status="failed",
                        message="File too large",
                    )
                )
                failed += 1
                continue

            file_id = uuid4()
            file_hash = get_file_hash(content)

            # Save file to storage
            file_ext = Path(file.filename or "file").suffix
            file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
            with open(file_path, "wb") as f:
                f.write(content)

            # Create upload job in database (persistent storage)
            await create_upload_job(
                file_id=file_id,
                filename=file.filename or "unknown",
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=len(content),
                collection=collection,
                access_tier=access_tier,
                enable_ocr=enable_ocr,
            )

            # Queue for background processing with multi-tenant fields
            options = UploadOptions(
                collection=collection,
                folder_id=folder_id,
                access_tier=access_tier,
                enable_ocr=enable_ocr,
                # Multi-tenant fields from authenticated user
                organization_id=user.organization_id if user else None,
                uploaded_by_id=user.user_id if user else None,
                is_private=is_private,
            )
            background_tasks.add_task(
                process_document_background,
                file_id=file_id,
                file_path=str(file_path),
                options=options,
            )

            results.append(
                UploadResponse(
                    file_id=file_id,
                    filename=file.filename or "unknown",
                    file_size=len(content),
                    file_hash=file_hash,
                    status="queued",
                    message="File uploaded successfully",
                )
            )
            successful += 1

        except Exception as e:
            logger.error("Error uploading file", filename=file.filename, error=str(e))
            results.append(
                UploadResponse(
                    file_id=uuid4(),
                    filename=file.filename or "unknown",
                    file_size=0,
                    file_hash="",
                    status="failed",
                    message=str(e),
                )
            )
            failed += 1

    return BatchUploadResponse(
        total_files=len(files),
        successful=successful,
        failed=failed,
        files=results,
    )


@router.get("/status/{file_id}", response_model=ProcessingStatus)
async def get_processing_status_endpoint(
    file_id: UUID,
    # user = Depends(get_current_user),
):
    """
    Get the processing status of an uploaded file.
    Now retrieves from database (with cache).
    """
    logger.info("Getting processing status", file_id=str(file_id))

    status = await get_upload_job(str(file_id))
    if not status:
        raise HTTPException(status_code=404, detail="File not found")

    return ProcessingStatus(
        file_id=file_id,
        filename=status.get("filename", "unknown"),
        status=status.get("status", "unknown"),
        progress=status.get("progress", 0),
        current_step=status.get("current_step", "Unknown"),
        error=status.get("error"),
        created_at=status.get("created_at", datetime.now()),
        updated_at=status.get("updated_at", datetime.now()),
    )


@router.get("/queue", response_model=ProcessingQueueResponse)
async def get_processing_queue(
    # user = Depends(get_current_user),
):
    """
    Get the current processing queue status.

    Now retrieves from database (persists across server restarts).
    Also performs automatic sync to update stuck jobs.
    """
    # Auto-sync stuck jobs with completed documents (runs on every queue fetch)
    try:
        await sync_upload_jobs_with_documents()
    except Exception as e:
        logger.warning("Failed to sync upload jobs", error=str(e))

    # Get all items from database
    all_jobs = await get_all_upload_jobs(limit=100)

    items = []
    active_count = 0

    for job in all_jobs:
        status_value = job.get("status", "")
        items.append(ProcessingStatus(
            file_id=UUID(job["file_id"]),
            filename=job.get("filename", "unknown"),
            status=status_value,
            progress=job.get("progress", 0),
            current_step=job.get("current_step", "Unknown"),
            error=job.get("error"),
            created_at=job.get("created_at", datetime.now()),
            updated_at=job.get("updated_at", datetime.now()),
        ))

        # Active = not queued, pending, completed, failed, or cancelled
        if status_value not in ["queued", "pending", "completed", "failed", "cancelled", "duplicate"]:
            active_count += 1

    return ProcessingQueueResponse(
        queue_length=len(items),
        active_tasks=active_count,
        items=items,
    )


@router.delete("/cancel/{file_id}")
async def cancel_processing(
    file_id: UUID,
    # user = Depends(get_current_user),
):
    """
    Cancel processing of a queued file.

    Only works for files that haven't started processing yet.
    """
    logger.info("Cancelling processing", file_id=str(file_id))

    # Check if file exists and is in a cancellable state (from database)
    status = await get_upload_job(str(file_id))
    if not status:
        raise HTTPException(status_code=404, detail="File not found")

    if status.get("status") not in ["queued", "pending"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel file in '{status.get('status')}' state. Only queued files can be cancelled."
        )

    # Mark as cancelled in database
    await update_upload_job_status(
        file_id=str(file_id),
        status=UploadStatus.CANCELLED,
        progress=0,
        current_step="Cancelled",
    )
    logger.info("Processing cancelled", file_id=str(file_id))

    return {"message": "Processing cancelled", "file_id": str(file_id)}


@router.post("/retry/{file_id}")
async def retry_processing(
    file_id: UUID,
    background_tasks: BackgroundTasks,
    # user = Depends(get_current_user),
):
    """
    Retry processing a failed file.
    """
    logger.info("Retrying processing", file_id=str(file_id))

    # Check if file exists and is in a retriable state (from database)
    status = await get_upload_job(str(file_id))
    if not status:
        raise HTTPException(status_code=404, detail="File not found")

    if status.get("status") not in ["failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry file in '{status.get('status')}' state. Only failed or cancelled files can be retried."
        )

    # Check if file still exists on disk
    file_path = status.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=400,
            detail="Original file no longer exists. Please re-upload."
        )

    # Reset status and re-queue in database
    await update_upload_job_status(
        file_id=str(file_id),
        status=UploadStatus.QUEUED,
        progress=0,
        current_step="Queued",
        error_message=None,  # Clear previous error
    )

    # Re-queue for background processing
    options = UploadOptions(
        collection=status.get("collection"),
        access_tier=status.get("access_tier", 1),
    )
    background_tasks.add_task(
        process_document_background,
        file_id=file_id,
        file_path=file_path,
        options=options,
    )

    logger.info("File re-queued for processing", file_id=str(file_id))
    return {"message": "File queued for reprocessing", "file_id": str(file_id)}


@router.post("/retry-queued")
async def retry_all_queued(
    background_tasks: BackgroundTasks,
):
    """
    Retry processing all files stuck in QUEUED status.

    This is useful after a server restart when background tasks were lost.
    """
    logger.info("Retrying all queued uploads")

    queued_count = 0
    started_count = 0
    skipped_count = 0
    errors = []

    async for session in get_async_session():
        # Get all queued upload jobs
        result = await session.execute(
            select(UploadJob).where(UploadJob.status == UploadStatus.QUEUED)
        )
        queued_jobs = result.scalars().all()
        queued_count = len(queued_jobs)

        for job in queued_jobs:
            file_path = job.file_path

            # Check if file still exists
            if not file_path or not Path(file_path).exists():
                logger.warning("File not found for queued job", file_id=str(job.id), file_path=file_path)
                skipped_count += 1
                errors.append(f"{job.filename}: File not found")
                continue

            # Queue for processing
            options = UploadOptions(
                collection=job.collection,
                folder_id=None,  # folder_id not stored in UploadJob
                access_tier=job.access_tier or 1,
                enable_ocr=job.enable_ocr if job.enable_ocr is not None else True,
                enable_image_analysis=job.enable_image_analysis if job.enable_image_analysis is not None else True,
                auto_generate_tags=job.auto_generate_tags if job.auto_generate_tags is not None else False,
            )

            background_tasks.add_task(
                process_document_background,
                file_id=job.id,
                file_path=file_path,
                options=options,
            )
            started_count += 1
            logger.info("Re-queued file for processing", file_id=str(job.id), filename=job.filename)

    logger.info("Retry queued complete", queued=queued_count, started=started_count, skipped=skipped_count)

    return {
        "message": f"Started processing {started_count} of {queued_count} queued files",
        "queued_count": queued_count,
        "started_count": started_count,
        "skipped_count": skipped_count,
        "errors": errors if errors else None,
    }


@router.get("/supported-types")
async def get_supported_types():
    """
    Get list of supported file types.
    """
    return {
        "supported_extensions": sorted(list(ALLOWED_EXTENSIONS)),
        "max_file_size": MAX_FILE_SIZE,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "categories": {
            "documents": ["pdf", "doc", "docx", "txt", "md", "rtf", "odt"],
            "presentations": ["ppt", "pptx", "odp", "key"],
            "spreadsheets": ["xls", "xlsx", "csv", "ods"],
            "images": ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "svg"],
            "archives": ["zip", "rar", "7z", "tar", "gz"],
            "media": ["mp3", "wav", "mp4", "mov", "avi", "mkv"],
            "other": ["json", "xml", "html", "htm"],
        },
    }


@router.delete("/queue/completed")
async def clear_completed_uploads():
    """
    Clear all completed upload jobs from the queue.

    This removes only completed jobs, leaving failed, processing, and queued items intact.
    Also clears the in-memory cache for completed items.
    """
    logger.info("Clearing completed upload jobs")

    deleted_count = 0
    async for session in get_async_session():
        # Delete completed upload jobs
        result = await session.execute(
            delete(UploadJob).where(UploadJob.status == UploadStatus.COMPLETED)
        )
        deleted_count = result.rowcount
        await session.commit()

    # Clear completed items from in-memory cache
    completed_cache_keys = [
        key for key, value in _processing_status_cache.items()
        if value.get("status") == "completed"
    ]
    for key in completed_cache_keys:
        del _processing_status_cache[key]

    logger.info("Cleared completed upload jobs", deleted_count=deleted_count)
    return {"deleted_count": deleted_count, "message": f"Cleared {deleted_count} completed upload jobs"}


# =============================================================================
# Folder Upload Endpoint
# =============================================================================

class FolderUploadResponse(BaseModel):
    """Response for folder upload."""
    folder_id: str
    folder_name: str
    files_queued: int
    file_ids: List[str]
    message: str


@router.post("/folder", response_model=FolderUploadResponse)
async def upload_folder(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP file containing folder structure"),
    folder_id: Optional[str] = Form(None, description="Target folder ID (upload into existing folder)"),
    new_folder_name: Optional[str] = Form(None, description="Name for new root folder (creates new folder)"),
    access_tier: int = Form(1, ge=1, le=100, description="Access tier for uploaded documents"),
    collection: Optional[str] = Form(None, description="Collection/tag for all documents"),
    enable_ocr: bool = Form(True, description="Enable OCR for scanned documents"),
    enable_image_analysis: bool = Form(True, description="Enable image analysis"),
    user_id: Optional[str] = Form(None, description="User ID for ownership tracking"),
    access_tier_id: Optional[str] = Form(None, description="Access tier ID for new folders"),
):
    """
    Upload a folder as a ZIP file.

    The ZIP file structure is preserved as a folder hierarchy:
    - If folder_id is provided: extracts into existing folder
    - If new_folder_name is provided: creates new root folder first
    - If neither: creates folder named after ZIP file

    Each file in the ZIP is queued for processing.
    Returns the root folder ID and list of file IDs.
    """
    import zipfile
    from backend.services.folder_service import get_folder_service

    # Validate ZIP file
    if not file.filename or not file.filename.lower().endswith('.zip'):
        raise HTTPException(
            status_code=400,
            detail="File must be a ZIP archive"
        )

    # Read ZIP content
    try:
        content = await file.read()
        file_size = len(content)

        if file_size > MAX_FILE_SIZE * 10:  # Allow 10x max size for folders
            raise HTTPException(
                status_code=400,
                detail=f"ZIP file too large. Maximum size: {MAX_FILE_SIZE * 10 // (1024*1024)}MB"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Save ZIP to temp file
    temp_zip_path = UPLOAD_DIR / f"temp_{uuid4()}.zip"
    with open(temp_zip_path, 'wb') as f:
        f.write(content)

    try:
        # Open and validate ZIP
        with zipfile.ZipFile(temp_zip_path, 'r') as zf:
            # Check for valid structure
            file_list = [f for f in zf.namelist() if not f.endswith('/')]

            if not file_list:
                raise HTTPException(status_code=400, detail="ZIP file is empty")

            # Filter to supported file types
            processable_files = []
            for filepath in file_list:
                filename = Path(filepath).name
                if filename.startswith('.') or filename.startswith('__'):
                    continue  # Skip hidden files
                ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
                if ext in ALLOWED_EXTENSIONS:
                    processable_files.append(filepath)

            if not processable_files:
                raise HTTPException(
                    status_code=400,
                    detail="No supported files found in ZIP"
                )

            # Determine root folder
            folder_service = get_folder_service()

            if folder_id:
                # Use existing folder
                target_folder = await folder_service.get_folder_by_id(folder_id)
                if not target_folder:
                    raise HTTPException(status_code=404, detail="Target folder not found")
                root_folder_id = folder_id
                root_folder_name = target_folder.name
            else:
                # Create new folder
                folder_name = new_folder_name or Path(file.filename).stem
                new_folder = await folder_service.create_folder(
                    name=folder_name,
                    parent_folder_id=None,
                    access_tier_id=access_tier_id,
                    created_by_id=user_id,
                    inherit_permissions=True,
                )
                root_folder_id = str(new_folder.id)
                root_folder_name = folder_name

            # Create extraction directory
            extract_dir = UPLOAD_DIR / f"folder_{root_folder_id}"
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Extract and process files
            file_ids = []
            folder_cache = {"/": root_folder_id}  # Cache folder paths to IDs

            for filepath in processable_files:
                # Get relative directory path
                parts = Path(filepath).parts
                filename = parts[-1]
                dir_parts = parts[:-1]

                # Create intermediate folders if needed
                current_path = "/"
                parent_folder_id = root_folder_id

                for part in dir_parts:
                    current_path = f"{current_path}{part}/"
                    if current_path not in folder_cache:
                        # Create subfolder
                        subfolder = await folder_service.create_folder(
                            name=part,
                            parent_folder_id=parent_folder_id,
                            access_tier_id=access_tier_id,
                            created_by_id=user_id,
                            inherit_permissions=True,
                        )
                        folder_cache[current_path] = str(subfolder.id)
                    parent_folder_id = folder_cache[current_path]

                # Extract file
                extracted_path = extract_dir / filepath
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(filepath) as src, open(extracted_path, 'wb') as dst:
                    dst.write(src.read())

                # Create upload job
                file_content = extracted_path.read_bytes()
                file_hash = get_file_hash(file_content)
                file_id = uuid4()

                await create_upload_job(
                    file_id=file_id,
                    filename=filename,
                    file_path=str(extracted_path),
                    file_hash=file_hash,
                    file_size=len(file_content),
                    collection=collection,
                    access_tier=access_tier,
                    enable_ocr=enable_ocr,
                    enable_image_analysis=enable_image_analysis,
                )

                file_ids.append(str(file_id))

                # Queue for processing
                options = UploadOptions(
                    collection=collection,
                    access_tier=access_tier,
                    enable_ocr=enable_ocr,
                    enable_image_analysis=enable_image_analysis,
                )
                background_tasks.add_task(
                    process_document_background,
                    file_id=file_id,
                    file_path=str(extracted_path),
                    options=options,
                )

                # Update document with folder_id after processing
                # This is done via a callback in process_document_background

            logger.info(
                "Folder upload completed",
                folder_id=root_folder_id,
                files_queued=len(file_ids),
            )

            return FolderUploadResponse(
                folder_id=root_folder_id,
                folder_name=root_folder_name,
                files_queued=len(file_ids),
                file_ids=file_ids,
                message=f"Uploaded {len(file_ids)} files to folder '{root_folder_name}'"
            )

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Folder upload failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Folder upload failed: {str(e)}")
    finally:
        # Clean up temp ZIP
        if temp_zip_path.exists():
            temp_zip_path.unlink()
