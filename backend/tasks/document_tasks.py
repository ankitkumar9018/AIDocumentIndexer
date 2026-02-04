"""
AIDocumentIndexer - Document Processing Tasks
==============================================

Celery tasks for asynchronous document processing.
These tasks handle document ingestion, OCR, chunking, and embedding
in the background, allowing the API to return immediately.
"""

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional
from uuid import UUID

from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


def run_async(coro):
    """Helper to run async code in Celery tasks.

    Handles both cases where there is or isn't a running event loop.
    Uses asyncio.run() in most cases (recommended by Python docs).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run() which handles loop lifecycle
        return asyncio.run(coro)
    else:
        # Loop is already running - use thread executor to avoid blocking
        # This happens when called from within async context
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


@shared_task(bind=True, name="backend.tasks.document_tasks.process_document_task")
def process_document_task(
    self,
    file_path: str,
    original_filename: str,
    user_id: str,
    collection: Optional[str] = None,
    access_tier: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single document asynchronously.

    Args:
        file_path: Path to the uploaded file
        original_filename: Original name of the file
        user_id: ID of the user who uploaded
        collection: Optional collection/tag
        access_tier: Access tier level
        metadata: Additional metadata

    Returns:
        Dict with document_id and processing result
    """
    logger.info(f"Starting document processing: {original_filename}")

    # Update task state to show progress
    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 10,
            "message": "Initializing document processing...",
            "filename": original_filename,
        }
    )

    try:
        # Import here to avoid circular imports
        from backend.db.database import async_session_context
        from backend.services.pipeline import DocumentPipeline

        async def _process():
            pipeline = DocumentPipeline()

            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": 20,
                    "message": "Processing document...",
                    "filename": original_filename,
                }
            )

            # Process the document - pipeline expects file_path, not file_content
            result = await pipeline.process_document(
                file_path=file_path,
                metadata={
                    "original_filename": original_filename,
                    **(metadata or {}),
                },
                collection=collection,
                access_tier=access_tier,
                uploaded_by_id=user_id,
            )

            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": 90,
                    "message": "Finalizing...",
                    "filename": original_filename,
                }
            )

            return result

        result = run_async(_process())

        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Auto-generate tags if enabled in metadata
        document_id = str(result.document_id) if hasattr(result, "document_id") and result.document_id else None
        if metadata and metadata.get("auto_generate_tags") and document_id:
            logger.info(f"Auto-generating tags for document: {original_filename}")
            try:
                from backend.services.auto_tagger import AutoTaggerService

                async def _auto_tag():
                    from backend.db.database import async_session_context
                    from backend.db.models import Document, Chunk
                    from sqlalchemy import select
                    from uuid import UUID as PyUUID

                    doc_uuid = PyUUID(document_id)
                    async with async_session_context() as session:
                        # Get document
                        doc_query = select(Document).where(Document.id == doc_uuid)
                        doc_result = await session.execute(doc_query)
                        document = doc_result.scalar_one_or_none()

                        if not document:
                            logger.warning(f"Document not found for auto-tagging: {document_id}")
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
                            logger.warning(f"No chunks found for auto-tagging: {document_id}")
                            return

                        content_sample = "\n".join([c.content for c in chunks if c.content])

                        # Generate tags using LLM
                        auto_tagger = AutoTaggerService()
                        tags = await auto_tagger.generate_tags(
                            document_name=original_filename,
                            content_sample=content_sample,
                            existing_collections=None,
                            max_tags=5
                        )

                        if tags:
                            existing_tags = document.tags or []
                            merged_tags = list(dict.fromkeys(existing_tags + tags))
                            document.tags = merged_tags
                            await session.commit()
                            logger.info(f"Auto-generated tags for {original_filename}: {tags}")

                run_async(_auto_tag())
            except Exception as e:
                logger.error(f"Auto-tagging failed for {original_filename}: {str(e)}")
                # Don't fail the task if auto-tagging fails

        # Auto-enhance document if enabled (per-upload override or global setting)
        if document_id:
            try:
                async def _auto_enhance():
                    # Check per-upload override first, then fall back to global setting
                    per_upload_enhance = metadata.get("auto_enhance", None) if metadata else None
                    if per_upload_enhance is not None:
                        should_enhance = per_upload_enhance
                    else:
                        from backend.services.settings import get_settings_service
                        settings = get_settings_service()
                        should_enhance = await settings.get_setting("upload.auto_enhance")
                    if should_enhance:
                        logger.info(f"Auto-enhancing document: {original_filename}")
                        from backend.services.document_enhancer import get_document_enhancer
                        enhancer = get_document_enhancer()
                        await enhancer.enhance_document(document_id)
                        logger.info(f"Auto-enhancement complete: {original_filename}")

                run_async(_auto_enhance())
            except Exception as e:
                logger.error(f"Auto-enhancement failed for {original_filename}: {str(e)}")
                # Don't fail the upload - enhancement is optional

        logger.info(f"Document processed successfully: {original_filename}")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": original_filename,
            "chunks": result.chunk_count if hasattr(result, "chunk_count") else 0,
        }

    except Exception as e:
        logger.error(f"Document processing failed: {original_filename} - {str(e)}")
        # Clean up temp file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


@shared_task(bind=True, name="backend.tasks.document_tasks.process_batch_task")
def process_batch_task(
    self,
    file_infos: List[Dict[str, Any]],
    user_id: str,
    collection: Optional[str] = None,
    access_tier: int = 1,
) -> Dict[str, Any]:
    """
    Process multiple documents as a batch.

    Args:
        file_infos: List of dicts with file_path, original_filename
        user_id: ID of the user who uploaded
        collection: Optional collection/tag for all files
        access_tier: Access tier level for all files

    Returns:
        Dict with batch results
    """
    total = len(file_infos)
    logger.info(f"Starting batch processing: {total} documents")

    results = {
        "total": total,
        "successful": 0,
        "failed": 0,
        "documents": [],
        "errors": [],
    }

    for idx, file_info in enumerate(file_infos):
        progress = int((idx / total) * 100)

        self.update_state(
            state="PROGRESS",
            meta={
                "progress": progress,
                "current": idx + 1,
                "total": total,
                "message": f"Processing {file_info['original_filename']}...",
            }
        )

        try:
            # Process each document
            doc_result = process_document_task.apply(
                args=[
                    file_info["file_path"],
                    file_info["original_filename"],
                    user_id,
                    collection,
                    access_tier,
                    file_info.get("metadata"),
                ]
            ).get(timeout=600)  # 10 minute timeout per document

            results["successful"] += 1
            results["documents"].append(doc_result)

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "filename": file_info["original_filename"],
                "error": str(e),
            })
            logger.error(f"Batch item failed: {file_info['original_filename']} - {str(e)}")

    logger.info(
        f"Batch processing complete: {results['successful']}/{total} successful"
    )
    return results


@shared_task(bind=True, name="backend.tasks.document_tasks.reprocess_document_task")
def reprocess_document_task(
    self,
    document_id: str,
    user_id: str,
    force_ocr: bool = False,
) -> Dict[str, Any]:
    """
    Reprocess an existing document.

    Args:
        document_id: UUID of the document to reprocess
        user_id: ID of the user requesting reprocessing
        force_ocr: Force OCR even if text was extracted

    Returns:
        Dict with reprocessing result
    """
    logger.info(f"Starting document reprocessing: {document_id}")

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 10,
            "message": "Loading document...",
            "document_id": document_id,
        }
    )

    try:
        from backend.db.database import async_session_context
        from backend.db.models import Document
        from backend.services.pipeline import DocumentPipeline
        from sqlalchemy import select

        async def _reprocess():
            async with async_session_context() as session:
                # Get document
                result = await session.execute(
                    select(Document).where(Document.id == document_id)
                )
                document = result.scalar_one_or_none()

                if not document:
                    raise ValueError(f"Document not found: {document_id}")

                self.update_state(
                    state="PROGRESS",
                    meta={
                        "progress": 30,
                        "message": "Reprocessing document...",
                        "document_id": document_id,
                    }
                )

                pipeline = DocumentPipeline()
                result = await pipeline.reprocess_document(
                    document_id=document_id,
                    session=session,
                    force_ocr=force_ocr,
                )

                return result

        result = run_async(_reprocess())

        logger.info(f"Document reprocessed successfully: {document_id}")
        return {
            "status": "success",
            "document_id": document_id,
            "chunks": result.chunk_count if hasattr(result, "chunk_count") else 0,
        }

    except Exception as e:
        logger.error(f"Document reprocessing failed: {document_id} - {str(e)}")
        raise


@shared_task(bind=True, name="backend.tasks.document_tasks.ocr_task")
def ocr_task(
    self,
    file_path: str,
    page_numbers: Optional[List[int]] = None,
    language: str = "eng",
) -> Dict[str, Any]:
    """
    Run OCR on a document or specific pages.

    Args:
        file_path: Path to the file
        page_numbers: Optional list of page numbers to OCR
        language: OCR language

    Returns:
        Dict with OCR results
    """
    logger.info(f"Starting OCR task: {file_path}")

    try:
        from backend.processors.universal import UniversalProcessor

        async def _ocr():
            processor = UniversalProcessor()

            # Read file
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Extract text with OCR
            result = await processor.extract_text(
                file_content=file_content,
                filename=os.path.basename(file_path),
                force_ocr=True,
                ocr_language=language,
            )

            return result

        result = run_async(_ocr())

        return {
            "status": "success",
            "text_length": len(result) if result else 0,
            "pages_processed": len(page_numbers) if page_numbers else "all",
        }

    except Exception as e:
        logger.error(f"OCR task failed: {file_path} - {str(e)}")
        raise


@shared_task(bind=True, name="backend.tasks.document_tasks.embedding_task")
def embedding_task(
    self,
    texts: List[str],
    chunk_ids: List[str],
    model: Optional[str] = None,
    use_ray: bool = True,
) -> Dict[str, Any]:
    """
    Generate embeddings for a batch of text chunks.

    Uses Ray for parallel embedding when available (2-4x speedup for large batches).

    Args:
        texts: List of text chunks to embed
        chunk_ids: Corresponding chunk IDs
        model: Optional embedding model to use
        use_ray: Whether to use Ray for parallel processing (default True)

    Returns:
        Dict with embedding results
    """
    batch_size = len(texts)
    # Ray provides 2-4x speedup for batches >= 50 texts
    should_use_ray = use_ray and batch_size >= 50
    logger.info(f"Starting embedding task: {batch_size} chunks, use_ray={should_use_ray}")

    try:
        from backend.services.embeddings import get_embedding_service

        async def _embed():
            service = get_embedding_service(use_ray=should_use_ray)
            embeddings = await service.embed_texts(texts, model=model)
            return embeddings

        embeddings = run_async(_embed())

        return {
            "status": "success",
            "count": len(embeddings),
            "chunk_ids": chunk_ids,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "used_ray": should_use_ray,
        }

    except Exception as e:
        logger.error(f"Embedding task failed: {str(e)}")
        raise


# =============================================================================
# Parallel Bulk Upload Tasks
# =============================================================================

@shared_task(bind=True, name="backend.tasks.document_tasks.process_bulk_upload_task")
def process_bulk_upload_task(
    self,
    batch_id: str,
    file_infos: List[Dict[str, Any]],
    user_id: str,
    collection: Optional[str] = None,
    access_tier: int = 1,
    max_concurrent: int = 4,
) -> Dict[str, Any]:
    """
    Process multiple documents in parallel using a worker pool.

    This task spawns individual document processing tasks and tracks
    their progress through the bulk progress tracker.

    Args:
        batch_id: Batch ID for progress tracking
        file_infos: List of dicts with file_path, original_filename
        user_id: ID of the user who uploaded
        collection: Optional collection/tag for all files
        access_tier: Access tier level for all files
        max_concurrent: Maximum concurrent processing tasks

    Returns:
        Dict with batch results
    """
    from celery import group
    import time

    total = len(file_infos)
    logger.info(f"Starting parallel bulk upload: {total} documents, batch_id={batch_id}")

    # Initialize progress tracking
    async def _init_progress():
        from backend.services.bulk_progress import get_progress_tracker, FileStatus
        tracker = await get_progress_tracker()

        # Add all files to tracking
        for file_info in file_infos:
            await tracker.add_file(
                batch_id=batch_id,
                file_id=file_info.get("file_id", file_info["original_filename"]),
                filename=file_info["original_filename"],
                file_size=file_info.get("file_size"),
            )

    run_async(_init_progress())

    # Create task signatures for all documents
    task_signatures = []
    for file_info in file_infos:
        sig = process_document_with_progress.s(
            file_path=file_info["file_path"],
            original_filename=file_info["original_filename"],
            user_id=user_id,
            collection=collection,
            access_tier=access_tier,
            metadata=file_info.get("metadata"),
            batch_id=batch_id,
            file_id=file_info.get("file_id", file_info["original_filename"]),
        )
        task_signatures.append(sig)

    # Execute in parallel batches
    results = {
        "batch_id": batch_id,
        "total": total,
        "successful": 0,
        "failed": 0,
        "documents": [],
        "errors": [],
    }

    # Process in chunks of max_concurrent
    for i in range(0, len(task_signatures), max_concurrent):
        chunk = task_signatures[i:i + max_concurrent]

        # Execute chunk in parallel
        job = group(chunk)
        group_result = job.apply_async()

        # Wait for this chunk to complete
        try:
            chunk_results = group_result.get(timeout=600 * max_concurrent)

            for result in chunk_results:
                if result.get("status") == "success":
                    results["successful"] += 1
                    results["documents"].append(result)
                else:
                    results["failed"] += 1
                    results["errors"].append(result)

        except Exception as e:
            logger.error(f"Chunk processing failed: {str(e)}")
            # Mark remaining as failed
            for _ in chunk:
                results["failed"] += 1

        # Update overall progress
        progress = int(((i + len(chunk)) / total) * 100)
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": progress,
                "completed": results["successful"] + results["failed"],
                "total": total,
                "batch_id": batch_id,
            }
        )

    logger.info(
        f"Bulk upload complete: {results['successful']}/{total} successful, batch_id={batch_id}"
    )
    return results


@shared_task(bind=True, name="backend.tasks.document_tasks.process_document_with_progress")
def process_document_with_progress(
    self,
    file_path: str,
    original_filename: str,
    user_id: str,
    collection: Optional[str] = None,
    access_tier: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    batch_id: Optional[str] = None,
    file_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single document with progress tracking for bulk uploads.

    Updates the bulk progress tracker at each processing stage.
    Also updates the UploadJob status in the database for single file uploads.
    """
    logger.info(f"Processing document: {original_filename} (batch={batch_id}, file_id={file_id})")

    file_id = file_id or original_filename

    async def _update_upload_job(status_str: str, progress: int, step: str, error: str = None, chunk_count: int = None, document_id: str = None):
        """Helper to update UploadJob status in database."""
        if not file_id:
            return
        try:
            from backend.api.routes.upload import update_upload_job_status
            from backend.db.models import UploadStatus

            # Map string to UploadStatus enum
            status_map = {
                "extracting": UploadStatus.EXTRACTING,
                "chunking": UploadStatus.CHUNKING,
                "embedding": UploadStatus.EMBEDDING,
                "indexing": UploadStatus.INDEXING,
                "completed": UploadStatus.COMPLETED,
                "failed": UploadStatus.FAILED,
            }
            status = status_map.get(status_str, UploadStatus.EXTRACTING)

            await update_upload_job_status(
                file_id=file_id,
                status=status,
                progress=progress,
                current_step=step,
                error_message=error,
                chunk_count=chunk_count,
                document_id=document_id,
            )
        except Exception as e:
            logger.warning(f"Failed to update upload job status: {e}")

    async def _update_progress(stage, error=None, document_id=None, chunk_count=0):
        """Helper to update progress tracker."""
        if not batch_id:
            return

        from backend.services.bulk_progress import (
            get_progress_tracker,
            ProcessingStage,
            FileStatus,
        )

        tracker = await get_progress_tracker()

        if error:
            await tracker.update_file_status(
                batch_id, file_id, FileStatus.FAILED, error=error
            )
        elif stage == ProcessingStage.COMPLETED:
            await tracker.update_file_status(
                batch_id, file_id, FileStatus.COMPLETED,
                document_id=document_id, chunk_count=chunk_count
            )
        else:
            await tracker.update_file_stage(batch_id, file_id, stage)

    try:
        from backend.db.database import async_session_context
        from backend.services.pipeline import DocumentPipeline
        from backend.services.bulk_progress import ProcessingStage

        async def _process():
            pipeline = DocumentPipeline()

            # Stage: Extracting
            await _update_progress(ProcessingStage.EXTRACTING)
            await _update_upload_job("extracting", 20, "Extracting text")

            # Stage: Chunking
            await _update_progress(ProcessingStage.CHUNKING)
            await _update_upload_job("chunking", 40, "Chunking document")

            # Stage: Embedding (happens inside pipeline)
            await _update_progress(ProcessingStage.EMBEDDING)
            await _update_upload_job("embedding", 60, "Generating embeddings")

            # Process the document - pipeline expects file_path, not file_content
            result = await pipeline.process_document(
                file_path=file_path,
                document_id=file_id,
                metadata={
                    "original_filename": original_filename,
                    **(metadata or {}),
                },
                collection=collection,
                access_tier=access_tier,
                uploaded_by_id=user_id,
            )

            # Stage: Storing
            await _update_progress(ProcessingStage.STORING)
            await _update_upload_job("indexing", 80, "Indexing document")

            return result

        result = run_async(_process())

        # Stage: Completed
        document_id = str(result.document_id) if hasattr(result, "document_id") else None
        chunk_count = result.chunk_count if hasattr(result, "chunk_count") else 0

        run_async(_update_progress(
            ProcessingStage.COMPLETED,
            document_id=document_id,
            chunk_count=chunk_count,
        ))

        # Update UploadJob as completed
        run_async(_update_upload_job(
            "completed", 100, "Completed",
            chunk_count=chunk_count, document_id=document_id
        ))

        # Auto-generate tags if enabled in metadata
        if metadata and metadata.get("auto_generate_tags") and document_id:
            logger.info(f"Auto-generating tags for document: {original_filename}")
            try:
                async def _auto_tag():
                    from backend.services.auto_tagger import AutoTaggerService
                    from backend.db.database import async_session_context
                    from backend.db.models import Document, Chunk
                    from sqlalchemy import select
                    from uuid import UUID as PyUUID

                    doc_uuid = PyUUID(document_id)
                    async with async_session_context() as session:
                        # Get document
                        doc_query = select(Document).where(Document.id == doc_uuid)
                        doc_result = await session.execute(doc_query)
                        document = doc_result.scalar_one_or_none()

                        if not document:
                            logger.warning(f"Document not found for auto-tagging: {document_id}")
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
                            logger.warning(f"No chunks found for auto-tagging: {document_id}")
                            return

                        content_sample = "\n".join([c.content for c in chunks if c.content])

                        # Generate tags using LLM
                        auto_tagger = AutoTaggerService()
                        tags = await auto_tagger.generate_tags(
                            document_name=original_filename,
                            content_sample=content_sample,
                            existing_collections=None,
                            max_tags=5
                        )

                        if tags:
                            existing_tags = document.tags or []
                            merged_tags = list(dict.fromkeys(existing_tags + tags))
                            document.tags = merged_tags
                            await session.commit()
                            logger.info(f"Auto-generated tags for {original_filename}: {tags}")

                run_async(_auto_tag())
            except Exception as e:
                logger.error(f"Auto-tagging failed for {original_filename}: {str(e)}")
                # Don't fail the task if auto-tagging fails

        # Auto-enhance document if enabled (per-upload override or global setting)
        if document_id:
            try:
                async def _auto_enhance_with_progress():
                    # Check per-upload override first, then fall back to global setting
                    per_upload_enhance = metadata.get("auto_enhance", None) if metadata else None
                    if per_upload_enhance is not None:
                        should_enhance = per_upload_enhance
                    else:
                        from backend.services.settings import get_settings_service
                        settings = get_settings_service()
                        should_enhance = await settings.get_setting("upload.auto_enhance")
                    if should_enhance:
                        logger.info(f"Auto-enhancing document: {original_filename}")
                        from backend.services.document_enhancer import get_document_enhancer
                        enhancer = get_document_enhancer()
                        await enhancer.enhance_document(document_id)
                        logger.info(f"Auto-enhancement complete: {original_filename}")

                run_async(_auto_enhance_with_progress())
            except Exception as e:
                logger.error(f"Auto-enhancement failed for {original_filename}: {str(e)}")
                # Don't fail the upload - enhancement is optional

        logger.info(f"Document processed: {original_filename}, chunks={chunk_count}")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": original_filename,
            "chunks": chunk_count,
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Document processing failed: {original_filename} - {error_msg}")

        # Update progress with error
        run_async(_update_progress(ProcessingStage.FAILED, error=error_msg))

        # Update UploadJob as failed
        run_async(_update_upload_job("failed", 0, "Failed", error=error_msg))

        return {
            "status": "failed",
            "filename": original_filename,
            "error": error_msg,
        }


@shared_task(bind=True, name="backend.tasks.document_tasks.extract_kg_task")
def extract_kg_task(
    self,
    document_id: str,
    user_id: str,
    batch_id: Optional[str] = None,
    file_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract knowledge graph from a document (background task).

    This runs in the background queue after document processing is complete.
    """
    logger.info(f"Starting KG extraction: document_id={document_id}")

    try:
        from backend.services.knowledge_graph import get_kg_service

        async def _extract_kg():
            kg_service = await get_kg_service()
            result = await kg_service.extract_from_document(document_id)
            return result

        result = run_async(_extract_kg())

        # Update progress if part of batch
        if batch_id and file_id:
            async def _update():
                from backend.services.bulk_progress import (
                    get_progress_tracker,
                    ProcessingStage,
                )
                tracker = await get_progress_tracker()
                await tracker.update_file_stage(
                    batch_id, file_id, ProcessingStage.KG_EXTRACTION
                )

            run_async(_update())

        return {
            "status": "success",
            "document_id": document_id,
            "entities": result.get("entity_count", 0) if result else 0,
            "relationships": result.get("relationship_count", 0) if result else 0,
        }

    except Exception as e:
        logger.error(f"KG extraction failed: document_id={document_id} - {str(e)}")
        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(e),
        }


@shared_task(bind=True, name="backend.tasks.document_tasks.run_kg_extraction_job")
def run_kg_extraction_job(
    self,
    job_id: str,
    user_id: str,
    organization_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    use_ray: bool = True,
) -> Dict[str, Any]:
    """
    Run a bulk KG extraction job in Celery with Ray for parallel processing.

    Architecture:
    - Celery: Handles job dispatch (so API returns immediately)
    - Ray: Handles parallel document processing (fast, distributed)

    This runs in Celery worker, NOT in the backend event loop,
    so it won't block the API.
    """
    import uuid
    from datetime import datetime, timezone

    logger.info(f"Starting bulk KG extraction job: job_id={job_id}, use_ray={use_ray}")

    try:
        from backend.services.kg_extraction_job import KGExtractionJobRunner
        from backend.db.database import get_async_session_factory

        async def _run_job():
            session_factory = get_async_session_factory()
            async with session_factory() as db_session:
                runner = KGExtractionJobRunner(
                    job_id=uuid.UUID(job_id),
                    db_session=db_session,
                    organization_id=uuid.UUID(organization_id) if organization_id else None,
                    provider_id=uuid.UUID(provider_id) if provider_id else None,
                )

                # Check if Ray is available and enabled
                try:
                    import ray
                    ray_available = True
                except ImportError:
                    ray_available = False

                if use_ray and ray_available:
                    # Use Ray for distributed parallel processing (fastest)
                    # run_with_ray() will auto-fallback to run_parallel()
                    # if Ray can't initialize (e.g. in Celery workers)
                    logger.info(f"Using Ray for KG extraction: job_id={job_id}")
                    await runner.run_with_ray()
                else:
                    # Fallback to asyncio parallel processing
                    logger.info(f"Using asyncio parallel for KG extraction: job_id={job_id}")
                    await runner.run_parallel()

        result = run_async(_run_job())

        logger.info(f"Bulk KG extraction job completed: job_id={job_id}")
        return {
            "status": "success",
            "job_id": job_id,
        }

    except Exception as e:
        logger.error(f"Bulk KG extraction job failed: job_id={job_id} - {str(e)}")

        # Update job status to failed
        try:
            from backend.db.database import get_async_session_factory
            from backend.db.models import KGExtractionJob
            from sqlalchemy import select
            import uuid

            async def _mark_failed():
                session_factory = get_async_session_factory()
                async with session_factory() as db_session:
                    result = await db_session.execute(
                        select(KGExtractionJob).where(KGExtractionJob.id == uuid.UUID(job_id))
                    )
                    job = result.scalar_one_or_none()
                    if job and job.status in ("queued", "running"):
                        job.status = "failed"
                        job.completed_at = datetime.now(timezone.utc)
                        error_log = job.error_log or []
                        error_log.append({
                            "error": f"Celery task failed: {str(e)}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        job.error_log = error_log
                        await db_session.commit()

            run_async(_mark_failed())
        except Exception as cleanup_error:
            logger.error(f"Failed to update job status after failure: {cleanup_error}")

        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        }


@shared_task(bind=True, name="backend.tasks.document_tasks.run_reindex_all_job")
def run_reindex_all_job(
    self,
    job_id: str,
    processing_mode: str = "linear",
    parallel_count: int = 2,
    batch_size: int = 5,
    delay_seconds: int = 15,
    force_reembed: bool = True,
) -> Dict[str, Any]:
    """
    Run a batch reindex job in Celery worker.

    Offloads the CPU/memory-intensive embedding work from the main backend
    process so the API stays responsive during reindexing.
    """
    logger.info(
        f"Starting reindex job in Celery: job_id={job_id}, "
        f"mode={processing_mode}, parallel={parallel_count}"
    )

    try:
        async def _run_reindex():
            from backend.api.routes.embeddings import _run_batch_reindex
            await _run_batch_reindex(
                job_id=job_id,
                processing_mode=processing_mode,
                parallel_count=parallel_count,
                batch_size=batch_size,
                delay_seconds=delay_seconds,
                force_reembed=force_reembed,
            )

        run_async(_run_reindex())

        logger.info(f"Reindex job completed: job_id={job_id}")
        return {
            "status": "success",
            "job_id": job_id,
        }

    except Exception as e:
        logger.error(f"Reindex job failed: job_id={job_id} - {str(e)}")

        # Update job status to failed via Redis tracker
        try:
            from backend.api.routes.embeddings import _job_tracker
            from datetime import datetime
            _job_tracker.update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=str(e),
            )
        except Exception as cleanup_error:
            logger.error(f"Failed to update job status: {cleanup_error}")

        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        }
