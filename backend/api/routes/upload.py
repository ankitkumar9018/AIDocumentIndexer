"""
AIDocumentIndexer - Upload API Routes
=====================================

Endpoints for file upload and processing management.
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
from backend.db.models import ProcessingMode
from backend.api.websocket import (
    notify_processing_started,
    notify_processing_progress,
    notify_processing_complete,
    notify_processing_error,
)
from backend.services.auto_tagger import AutoTaggerService
from backend.db.database import get_async_session
from backend.db.models import Document, Chunk, AccessTier
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

logger = structlog.get_logger(__name__)

router = APIRouter()

# In-memory storage for processing status (would be Redis/DB in production)
_processing_status: Dict[str, Dict] = {}

# Upload directory
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/aidocindexer/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Constants
# =============================================================================

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
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
    access_tier: int = Field(default=1, ge=1, le=100)
    enable_ocr: bool = True
    enable_image_analysis: bool = True
    smart_chunking: bool = True
    detect_duplicates: bool = True
    auto_generate_tags: bool = False  # Use LLM to auto-generate tags if no collection set


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


async def update_processing_status(file_id: str, status: PipelineStatus):
    """Update processing status in memory store and notify via WebSocket."""
    if file_id in _processing_status:
        _processing_status[file_id]["status"] = status.value
        _processing_status[file_id]["updated_at"] = datetime.now()

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

        _processing_status[file_id]["progress"] = progress
        _processing_status[file_id]["current_step"] = current_step

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
        async for session in get_async_session():
            # Get document
            doc_query = select(Document).where(Document.id == document_id)
            result = await session.execute(doc_query)
            document = result.scalar_one_or_none()

            if not document:
                logger.warning("Document not found for auto-tagging", document_id=document_id)
                return

            # Get first few chunks for content sample
            chunks_query = (
                select(Chunk)
                .where(Chunk.document_id == document_id)
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
                # Update document with generated tags
                document.tags = tags
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
    filename = _processing_status.get(file_id_str, {}).get("filename", "unknown")

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
        processing_mode = ProcessingMode.SMART
    elif options.enable_ocr:
        processing_mode = ProcessingMode.FULL
    else:
        processing_mode = ProcessingMode.TEXT_ONLY

    # Create async status callback
    async def status_callback(doc_id: str, status: PipelineStatus):
        await update_processing_status(file_id_str, status)

    # Create pipeline config with status callback
    config = PipelineConfig(
        processing_mode=processing_mode,
        use_ray=True,
        check_duplicates=options.detect_duplicates,
        on_status_change=status_callback,
    )

    pipeline = DocumentPipeline(config=config)

    try:
        result = await pipeline.process_document(
            file_path=file_path,
            document_id=file_id_str,
            metadata={"original_filename": filename},
            access_tier=options.access_tier,
            collection=options.collection,
        )

        # Update final status
        _processing_status[file_id_str]["status"] = result.status.value
        _processing_status[file_id_str]["progress"] = 100 if result.status == PipelineStatus.COMPLETED else 0
        _processing_status[file_id_str]["chunk_count"] = result.chunk_count
        _processing_status[file_id_str]["word_count"] = result.word_count
        _processing_status[file_id_str]["updated_at"] = datetime.now()

        if result.error_message:
            _processing_status[file_id_str]["error"] = result.error_message
            await notify_processing_error(file_id_str, result.error_message)
        else:
            # Auto-generate tags if enabled and no collection was set
            if options.auto_generate_tags and not options.collection:
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
        _processing_status[file_id_str]["status"] = "failed"
        _processing_status[file_id_str]["error"] = str(e)
        _processing_status[file_id_str]["updated_at"] = datetime.now()

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
    access_tier: int = Form(default=1, ge=1, le=100),
    enable_ocr: bool = Form(True),
    enable_image_analysis: bool = Form(True),
    smart_chunking: bool = Form(True),
    detect_duplicates: bool = Form(True),
    auto_generate_tags: bool = Form(False),
    # user = Depends(get_current_user),
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

    # Check for duplicates if enabled
    if detect_duplicates:
        # Check in processing status store (would check DB in production)
        for fid, status in _processing_status.items():
            if status.get("file_hash") == file_hash and status.get("status") == "completed":
                return UploadResponse(
                    file_id=UUID(fid),
                    filename=file.filename or "unknown",
                    file_size=len(content),
                    file_hash=file_hash,
                    status="duplicate",
                    message=f"File already exists with ID {fid}",
                )

    # Generate file ID
    file_id = uuid4()

    # Save file to storage
    file_ext = Path(file.filename or "file").suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"

    with open(file_path, "wb") as f:
        f.write(content)

    logger.info("File saved", file_id=str(file_id), path=str(file_path))

    # Create processing options
    options = UploadOptions(
        collection=collection,
        access_tier=access_tier,
        enable_ocr=enable_ocr,
        enable_image_analysis=enable_image_analysis,
        smart_chunking=smart_chunking,
        detect_duplicates=detect_duplicates,
        auto_generate_tags=auto_generate_tags,
    )

    # Initialize processing status
    _processing_status[str(file_id)] = {
        "file_id": str(file_id),
        "filename": file.filename or "unknown",
        "file_path": str(file_path),
        "file_hash": file_hash,
        "file_size": len(content),
        "status": "queued",
        "progress": 0,
        "current_step": "Queued",
        "collection": collection,
        "access_tier": access_tier,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "error": None,
    }

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
    access_tier: int = Form(default=1, ge=1, le=100),
    enable_ocr: bool = Form(True),
    # user = Depends(get_current_user),
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

            # Initialize processing status
            _processing_status[str(file_id)] = {
                "file_id": str(file_id),
                "filename": file.filename or "unknown",
                "file_path": str(file_path),
                "file_hash": file_hash,
                "file_size": len(content),
                "status": "queued",
                "progress": 0,
                "current_step": "Queued",
                "collection": collection,
                "access_tier": access_tier,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "error": None,
            }

            # Queue for background processing
            options = UploadOptions(
                collection=collection,
                access_tier=access_tier,
                enable_ocr=enable_ocr,
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
async def get_processing_status(
    file_id: UUID,
    # user = Depends(get_current_user),
):
    """
    Get the processing status of an uploaded file.
    """
    logger.info("Getting processing status", file_id=str(file_id))

    status = _processing_status.get(str(file_id))
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

    Shows all files pending or currently being processed.
    """
    logger.info("Getting processing queue")

    # Get all items from status store (including completed and failed)
    items = []
    active_count = 0

    for fid, status in _processing_status.items():
        status_value = status.get("status", "")
        items.append(ProcessingStatus(
            file_id=UUID(fid),
            filename=status.get("filename", "unknown"),
            status=status_value,
            progress=status.get("progress", 0),
            current_step=status.get("current_step", "Unknown"),
            error=status.get("error"),
            created_at=status.get("created_at", datetime.now()),
            updated_at=status.get("updated_at", datetime.now()),
        ))

        if status_value not in ["queued", "pending", "completed", "failed"]:
            active_count += 1

    # Sort by created_at
    items.sort(key=lambda x: x.created_at, reverse=True)

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

    # Check if file exists and is in a cancellable state
    status = _processing_status.get(str(file_id))
    if not status:
        raise HTTPException(status_code=404, detail="File not found")

    if status.get("status") not in ["queued", "pending"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel file in '{status.get('status')}' state. Only queued files can be cancelled."
        )

    # Mark as cancelled
    _processing_status[str(file_id)]["status"] = "cancelled"
    _processing_status[str(file_id)]["updated_at"] = datetime.now()
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

    # Check if file exists and is in a retriable state
    status = _processing_status.get(str(file_id))
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

    # Reset status and re-queue
    _processing_status[str(file_id)]["status"] = "queued"
    _processing_status[str(file_id)]["progress"] = 0
    _processing_status[str(file_id)]["current_step"] = "Queued"
    _processing_status[str(file_id)]["error"] = None
    _processing_status[str(file_id)]["updated_at"] = datetime.now()

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
