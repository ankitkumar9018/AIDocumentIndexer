"""
AIDocumentIndexer - Knowledge Graph Extraction Job Service
============================================================

Background job management for knowledge graph entity extraction.
Handles starting, tracking, pausing, and cancelling extraction jobs.
"""

import asyncio
import os
import uuid
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

import structlog
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Document, Chunk, KGExtractionJob, ProcessingStatus
from backend.services.knowledge_graph import KnowledgeGraphService
from backend.db.database import get_async_session_factory
from backend.api.websocket import (
    notify_kg_extraction_started,
    notify_kg_extraction_progress,
    notify_kg_extraction_complete,
)
from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Phase 60: Ray support for distributed KG extraction
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    pass


# Global registry of running jobs (for cancellation)
_running_jobs: Dict[str, "KGExtractionJobRunner"] = {}
_running_jobs_lock = asyncio.Lock()

# Pause configuration
MAX_PAUSE_DURATION_SECONDS = 3600  # 1 hour max pause
PAUSE_CHECK_INTERVAL = 5  # Check every 5 seconds instead of 1


class KGExtractionJobRunner:
    """
    Runner for a single knowledge graph extraction job.

    Processes documents in the background, updates progress,
    and supports cancellation.
    """

    def __init__(
        self,
        job_id: uuid.UUID,
        db_session: AsyncSession,
        organization_id: Optional[uuid.UUID] = None,
        provider_id: Optional[str] = None,
    ):
        self.job_id = job_id
        self.db = db_session
        self.organization_id = organization_id
        self.provider_id = provider_id
        self._cancelled = False
        self._paused = False
        # Per-document chunk progress: {doc_id: {"done": N, "total": M}}
        self._chunk_progress: Dict[str, Dict[str, int]] = {}

    def get_chunk_progress(self) -> Dict[str, Dict[str, int]]:
        """Get current chunk-level progress for all documents being processed."""
        return dict(self._chunk_progress)

    def _make_chunk_progress_callback(self, doc_id: str):
        """Create a progress callback that updates _chunk_progress for a specific document."""
        def callback(chunks_done: int, total_chunks: int):
            self._chunk_progress[doc_id] = {"done": chunks_done, "total": total_chunks}
        return callback

    async def run(self):
        """Execute the extraction job."""
        job = await self._get_job()
        if not job:
            logger.error("Job not found", job_id=str(self.job_id))
            return

        # Register for cancellation (thread-safe)
        async with _running_jobs_lock:
            _running_jobs[str(self.job_id)] = self

        try:
            # Update status to running
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            job.last_activity_at = datetime.now(timezone.utc)  # Initial heartbeat
            await self.db.commit()

            # Get documents to process
            documents = await self._get_documents_for_job(job)

            job.total_documents = len(documents)
            # Persist document IDs for detail endpoint
            job.document_ids = [str(doc.id) for doc in documents]
            await self.db.commit()

            logger.info(
                "Starting KG extraction job",
                job_id=str(self.job_id),
                total_documents=len(documents),
            )

            # Process each document
            processing_times: List[float] = []

            for i, doc in enumerate(documents):
                # Refresh job from DB to check for external pause/cancel requests
                # This handles cases where pause/cancel was set via DB when runner wasn't in registry
                await self.db.refresh(job)
                if job.status == "paused":
                    self._paused = True
                elif job.status == "cancelled":
                    self._cancelled = True

                # Check for cancellation
                if self._cancelled:
                    job.status = "cancelled"
                    job.completed_at = datetime.now(timezone.utc)
                    await self.db.commit()
                    logger.info("Job cancelled", job_id=str(self.job_id))
                    return

                # Check for pause (with max duration to prevent infinite loop)
                pause_start = time.time()
                pause_committed = False
                while self._paused and not self._cancelled:
                    if not pause_committed:
                        job.status = "paused"
                        await self.db.commit()
                        pause_committed = True

                    # Check for max pause duration
                    if time.time() - pause_start > MAX_PAUSE_DURATION_SECONDS:
                        logger.warning("Job paused too long, auto-resuming", job_id=str(self.job_id))
                        self._paused = False
                        break

                    await asyncio.sleep(PAUSE_CHECK_INTERVAL)

                    # Check DB for external resume request
                    await self.db.refresh(job)
                    if job.status == "running":
                        self._paused = False
                    elif job.status == "cancelled":
                        self._cancelled = True

                if self._cancelled:
                    continue

                job.status = "running"

                # Update current document and heartbeat
                job.current_document_id = doc.id
                job.current_document_name = doc.filename
                job.last_activity_at = datetime.now(timezone.utc)  # Heartbeat
                await self.db.commit()

                # Send WebSocket notification for document start
                await notify_kg_extraction_started(
                    job_id=str(self.job_id),
                    document_id=str(doc.id),
                    document_name=doc.filename or str(doc.id),
                )

                # Process the document
                start_time = time.time()
                try:
                    # Update document status
                    doc.kg_extraction_status = "processing"
                    await self.db.commit()

                    # Create KG service and process
                    # If a specific provider is requested, create an LLM service for it
                    llm_service = None
                    if self.provider_id:
                        from backend.services.llm import LLMConfigManager, LLMFactory
                        config = await LLMConfigManager.get_config_for_provider_id(self.provider_id)
                        if config:
                            llm_service = LLMFactory.get_chat_model(
                                provider=config.provider_type,
                                model=config.model,
                                temperature=config.temperature,
                            )
                    kg_service = KnowledgeGraphService(self.db, llm_service=llm_service)

                    # Clean up any stale entity data from previous extraction runs
                    await kg_service.delete_document_data(doc.id)

                    # Create chunk progress callback for live tracking
                    progress_cb = self._make_chunk_progress_callback(str(doc.id))
                    stats = await kg_service.process_document_for_graph(
                        doc.id, progress_callback=progress_cb
                    )

                    # Clear chunk progress after document completes
                    self._chunk_progress.pop(str(doc.id), None)

                    # Update document with results
                    doc.kg_extraction_status = "completed"
                    doc.kg_extracted_at = datetime.now(timezone.utc)
                    doc.kg_entity_count = stats.get("entities", 0)
                    doc.kg_relation_count = stats.get("relations", 0)

                    # Update job counters
                    job.processed_documents += 1
                    job.total_entities += stats.get("entities", 0)
                    job.total_relations += stats.get("relations", 0)

                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)

                    # Update average processing time for ETA and heartbeat
                    job.avg_doc_processing_time = sum(processing_times) / len(processing_times)
                    job.last_activity_at = datetime.now(timezone.utc)  # Heartbeat on completion

                    await self.db.commit()

                    logger.info(
                        "Processed document for KG",
                        job_id=str(self.job_id),
                        document_id=str(doc.id),
                        entities=stats.get("entities", 0),
                        relations=stats.get("relations", 0),
                        processing_time=processing_time,
                        progress=f"{job.processed_documents}/{job.total_documents}",
                    )

                    # Send WebSocket progress notification
                    progress_pct = (job.processed_documents / job.total_documents * 100) if job.total_documents > 0 else 0
                    await notify_kg_extraction_progress(
                        job_id=str(self.job_id),
                        entities_found=job.total_entities,
                        relations_found=job.total_relations,
                        progress=progress_pct,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process document for KG",
                        job_id=str(self.job_id),
                        document_id=str(doc.id),
                        error=str(e),
                        exc_info=True,
                    )

                    # Update document status
                    doc.kg_extraction_status = "failed"
                    job.failed_documents += 1
                    # NOTE: Do NOT increment processed_documents on failure
                    # processed_documents should only count successful extractions

                    # Log error
                    error_log = job.error_log or []
                    error_log.append({
                        "doc_id": str(doc.id),
                        "doc_name": doc.filename or str(doc.id),
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    job.error_log = error_log

                    await self.db.commit()

                    # Send WebSocket notification for failed doc
                    total_done = job.processed_documents + job.failed_documents
                    progress_pct = (total_done / job.total_documents * 100) if job.total_documents > 0 else 0
                    await notify_kg_extraction_progress(
                        job_id=str(self.job_id),
                        entities_found=job.total_entities,
                        relations_found=job.total_relations,
                        progress=progress_pct,
                    )

            # Mark job as completed
            job.status = "completed" if job.failed_documents == 0 else "completed_with_errors"
            job.completed_at = datetime.now(timezone.utc)
            job.current_document_id = None
            job.current_document_name = None
            await self.db.commit()

            logger.info(
                "KG extraction job completed",
                job_id=str(self.job_id),
                processed=job.processed_documents,
                failed=job.failed_documents,
                total_entities=job.total_entities,
                total_relations=job.total_relations,
            )

            # Send WebSocket completion notification
            await notify_kg_extraction_complete(
                job_id=str(self.job_id),
                total_entities=job.total_entities,
                total_relations=job.total_relations,
            )

        except Exception as e:
            logger.error(
                "KG extraction job failed",
                job_id=str(self.job_id),
                error=str(e),
            )
            job.status = "failed"
            job.completed_at = datetime.now(timezone.utc)
            await self.db.commit()

        finally:
            # Unregister from running jobs (thread-safe)
            async with _running_jobs_lock:
                _running_jobs.pop(str(self.job_id), None)

    async def cancel(self):
        """Cancel the running job."""
        self._cancelled = True
        logger.info("Cancellation requested", job_id=str(self.job_id))

    async def pause(self):
        """Pause the running job."""
        self._paused = True
        logger.info("Pause requested", job_id=str(self.job_id))

    async def resume(self):
        """Resume a paused job."""
        self._paused = False
        logger.info("Resume requested", job_id=str(self.job_id))

    async def _get_job(self) -> Optional[KGExtractionJob]:
        """Get the job from database."""
        result = await self.db.execute(
            select(KGExtractionJob).where(KGExtractionJob.id == self.job_id)
        )
        return result.scalar_one_or_none()

    async def _get_documents_for_job(self, job: KGExtractionJob) -> List[Document]:
        """Get documents to process for this job."""
        query = select(Document).where(
            Document.processing_status == ProcessingStatus.COMPLETED,
        )

        # Apply organization filter
        if self.organization_id:
            query = query.where(
                or_(
                    Document.organization_id == self.organization_id,
                    Document.organization_id.is_(None),
                )
            )

        # Filter to specific documents if provided
        if job.document_ids:
            doc_uuids = [uuid.UUID(d) for d in job.document_ids if d]
            if doc_uuids:
                query = query.where(Document.id.in_(doc_uuids))

        # Only new = skip already successfully extracted (include failed for retry)
        if job.only_new_documents:
            query = query.where(
                or_(
                    Document.kg_extraction_status == "pending",
                    Document.kg_extraction_status == "failed",  # Include failed docs for retry
                    Document.kg_extraction_status.is_(None),
                )
            )

        # Order by creation date (oldest first)
        query = query.order_by(Document.created_at.asc())

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def _should_skip_document(self, doc: Document) -> Tuple[bool, str]:
        """
        Pre-filter documents to skip those unlikely to benefit from KG extraction.

        Returns:
            Tuple of (should_skip, reason)
        """
        if not settings.KG_PRE_FILTER_ENABLED:
            return False, ""

        # Skip very small documents (< 500 chars) - unlikely to have meaningful entities
        content_length = getattr(doc, 'content_length', None) or getattr(doc, 'word_count', None)
        if content_length and content_length < 500:
            return True, "document_too_small"

        # Skip documents with no chunks
        chunk_count = await self.db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
        )
        chunk_count = chunk_count.scalar() or 0
        if chunk_count == 0:
            return True, "no_chunks"

        # Skip certain document types that are typically lists/tables without prose
        simple_types = {"csv", "tsv", "xlsx", "xls"}
        if doc.file_type and doc.file_type.lower() in simple_types:
            return True, "tabular_data"

        return False, ""

    async def _process_single_document(
        self,
        doc: Document,
        job: KGExtractionJob,
        semaphore: asyncio.Semaphore,
        stats_lock: asyncio.Lock,
        processing_times: List[float],
    ) -> Dict[str, Any]:
        """
        Process a single document with semaphore-controlled concurrency.

        Each document gets its own DB session to avoid transaction conflicts
        when processing documents concurrently.

        Returns:
            Dict with status, entities, relations, error info
        """
        doc_id = doc.id
        doc_name = doc.filename or str(doc.id)

        async with semaphore:
            # Check for cancellation before processing
            if self._cancelled:
                return {"status": "cancelled", "doc_id": str(doc_id)}

            # Check for pause
            while self._paused and not self._cancelled:
                await asyncio.sleep(PAUSE_CHECK_INTERVAL)

            if self._cancelled:
                return {"status": "cancelled", "doc_id": str(doc_id)}

            start_time = time.time()
            result = {
                "doc_id": str(doc_id),
                "doc_name": doc_name,
                "status": "pending",
                "entities": 0,
                "relations": 0,
                "error": None,
            }

            # Use a dedicated session for each document to avoid
            # transaction conflicts with concurrent processing
            session_factory = get_async_session_factory()
            async with session_factory() as doc_db:
                try:
                    # Re-fetch document in this session
                    doc_result = await doc_db.execute(
                        select(Document).where(Document.id == doc_id)
                    )
                    local_doc = doc_result.scalar_one_or_none()
                    if not local_doc:
                        result["status"] = "failed"
                        result["error"] = "Document not found"
                        return result

                    # Check pre-filter
                    should_skip, skip_reason = await self._should_skip_document_with_session(
                        local_doc, doc_db
                    )
                    if should_skip:
                        result["status"] = "skipped"
                        result["skip_reason"] = skip_reason
                        logger.debug(
                            "Skipping document for KG extraction",
                            doc_id=str(doc_id),
                            reason=skip_reason,
                        )
                        return result

                    # Update document status to processing
                    local_doc.kg_extraction_status = "processing"
                    await doc_db.commit()

                    # Send WebSocket notification
                    await notify_kg_extraction_started(
                        job_id=str(self.job_id),
                        document_id=str(doc_id),
                        document_name=doc_name,
                    )

                    # Create KG service and process
                    llm_service = None
                    if self.provider_id:
                        from backend.services.llm import LLMConfigManager, LLMFactory
                        config = await LLMConfigManager.get_config_for_provider_id(self.provider_id)
                        if config:
                            llm_service = LLMFactory.get_chat_model(
                                provider=config.provider_type,
                                model=config.model,
                                temperature=config.temperature,
                            )

                    kg_service = KnowledgeGraphService(doc_db, llm_service=llm_service)

                    # Clean up any stale entity data from previous extraction runs
                    await kg_service.delete_document_data(doc_id)

                    # Create chunk progress callback for live tracking
                    progress_cb = self._make_chunk_progress_callback(str(doc_id))
                    stats = await kg_service.process_document_for_graph(
                        doc_id, progress_callback=progress_cb
                    )

                    # Clear chunk progress after document completes
                    self._chunk_progress.pop(str(doc_id), None)

                    # Update document with results
                    local_doc.kg_extraction_status = "completed"
                    local_doc.kg_extracted_at = datetime.now(timezone.utc)
                    local_doc.kg_entity_count = stats.get("entities", 0)
                    local_doc.kg_relation_count = stats.get("relations", 0)
                    await doc_db.commit()

                    processing_time = time.time() - start_time

                    # Thread-safe update of processing times
                    async with stats_lock:
                        processing_times.append(processing_time)

                    result["status"] = "completed"
                    result["entities"] = stats.get("entities", 0)
                    result["relations"] = stats.get("relations", 0)
                    result["processing_time"] = processing_time

                    logger.info(
                        "Processed document for KG (parallel)",
                        job_id=str(self.job_id),
                        document_id=str(doc_id),
                        entities=stats.get("entities", 0),
                        relations=stats.get("relations", 0),
                        processing_time=processing_time,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process document for KG (parallel)",
                        job_id=str(self.job_id),
                        document_id=str(doc_id),
                        error=str(e),
                        exc_info=True,
                    )

                    try:
                        local_doc.kg_extraction_status = "failed"
                        await doc_db.commit()
                    except Exception:
                        pass  # Session may already be in error state

                    result["status"] = "failed"
                    result["error"] = str(e)

            return result

    async def _should_skip_document_with_session(
        self, doc: Document, db: AsyncSession
    ) -> Tuple[bool, str]:
        """Pre-filter using a specific DB session (for parallel processing)."""
        if not settings.KG_PRE_FILTER_ENABLED:
            return False, ""

        content_length = getattr(doc, 'content_length', None) or getattr(doc, 'word_count', None)
        if content_length and content_length < 500:
            return True, "document_too_small"

        chunk_count = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
        )
        chunk_count = chunk_count.scalar() or 0
        if chunk_count == 0:
            return True, "no_chunks"

        simple_types = {"csv", "tsv", "xlsx", "xls"}
        if doc.file_type and doc.file_type.lower() in simple_types:
            return True, "tabular_data"

        return False, ""

    async def _get_extraction_settings(self) -> Tuple[int, int]:
        """Read KG extraction settings from DB (concurrency & ray timeout)."""
        try:
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            concurrency = await settings_svc.get_setting("kg.extraction_concurrency")
            ray_timeout = await settings_svc.get_setting("kg.ray_task_timeout")
            return (
                int(concurrency) if concurrency else settings.KG_EXTRACTION_CONCURRENCY,
                int(ray_timeout) if ray_timeout else 600,
            )
        except Exception:
            return settings.KG_EXTRACTION_CONCURRENCY, 600

    async def run_parallel(self):
        """
        Execute the extraction job with parallel document processing.

        Uses semaphore-controlled concurrency to process multiple documents
        simultaneously while respecting the kg.extraction_concurrency setting.
        """
        job = await self._get_job()
        if not job:
            logger.error("Job not found", job_id=str(self.job_id))
            return

        # Read settings from DB (configurable via Admin UI)
        concurrency, _ = await self._get_extraction_settings()

        # Register for cancellation
        async with _running_jobs_lock:
            _running_jobs[str(self.job_id)] = self

        try:
            # Update status to running
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            job.last_activity_at = datetime.now(timezone.utc)
            await self.db.commit()

            # Get documents to process
            documents = await self._get_documents_for_job(job)

            job.total_documents = len(documents)
            # Persist document IDs for detail endpoint
            job.document_ids = [str(doc.id) for doc in documents]
            await self.db.commit()

            logger.info(
                "Starting parallel KG extraction job",
                job_id=str(self.job_id),
                total_documents=len(documents),
                concurrency=concurrency,
            )

            if not documents:
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                await self.db.commit()
                await notify_kg_extraction_complete(
                    job_id=str(self.job_id),
                    total_entities=0,
                    total_relations=0,
                )
                return

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrency)
            stats_lock = asyncio.Lock()
            processing_times: List[float] = []

            # Process documents in parallel batches
            batch_size = concurrency * 2  # 2x concurrency for better utilization
            total_entities = 0
            total_relations = 0
            processed_count = 0
            failed_count = 0
            skipped_count = 0

            for batch_start in range(0, len(documents), batch_size):
                if self._cancelled:
                    break

                # Wait for pause if needed
                while self._paused and not self._cancelled:
                    job.status = "paused"
                    await self.db.commit()
                    await asyncio.sleep(PAUSE_CHECK_INTERVAL)
                    await self.db.refresh(job)
                    if job.status == "running":
                        self._paused = False
                    elif job.status == "cancelled":
                        self._cancelled = True

                if self._cancelled:
                    break

                job.status = "running"
                batch = documents[batch_start:batch_start + batch_size]

                # Process batch concurrently
                tasks = [
                    self._process_single_document(doc, job, semaphore, stats_lock, processing_times)
                    for doc in batch
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Aggregate results
                for result in results:
                    if isinstance(result, Exception):
                        failed_count += 1
                        logger.error("Unexpected error in parallel processing", error=str(result))
                        continue

                    if result["status"] == "completed":
                        processed_count += 1
                        total_entities += result.get("entities", 0)
                        total_relations += result.get("relations", 0)
                    elif result["status"] == "failed":
                        failed_count += 1
                        # Add to error log
                        error_log = job.error_log or []
                        error_log.append({
                            "doc_id": result["doc_id"],
                            "doc_name": result["doc_name"],
                            "error": result.get("error", "Unknown error"),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        job.error_log = error_log
                    elif result["status"] == "skipped":
                        skipped_count += 1
                    elif result["status"] == "cancelled":
                        break

                # Update job progress
                job.processed_documents = processed_count
                job.failed_documents = failed_count
                job.total_entities = total_entities
                job.total_relations = total_relations
                job.last_activity_at = datetime.now(timezone.utc)

                if processing_times:
                    job.avg_doc_processing_time = sum(processing_times) / len(processing_times)

                await self.db.commit()

                # Send WebSocket progress notification
                total_done = processed_count + failed_count + skipped_count
                progress_pct = (total_done / job.total_documents * 100) if job.total_documents > 0 else 0
                await notify_kg_extraction_progress(
                    job_id=str(self.job_id),
                    entities_found=total_entities,
                    relations_found=total_relations,
                    progress=progress_pct,
                )

                logger.info(
                    "Batch processed",
                    job_id=str(self.job_id),
                    batch_start=batch_start,
                    batch_size=len(batch),
                    processed=processed_count,
                    failed=failed_count,
                    skipped=skipped_count,
                    progress_pct=progress_pct,
                )

            # Mark job as completed
            if self._cancelled:
                job.status = "cancelled"
            elif failed_count > 0:
                job.status = "completed_with_errors"
            else:
                job.status = "completed"

            job.completed_at = datetime.now(timezone.utc)
            job.current_document_id = None
            job.current_document_name = None
            await self.db.commit()

            logger.info(
                "Parallel KG extraction job completed",
                job_id=str(self.job_id),
                processed=processed_count,
                failed=failed_count,
                skipped=skipped_count,
                total_entities=total_entities,
                total_relations=total_relations,
            )

            await notify_kg_extraction_complete(
                job_id=str(self.job_id),
                total_entities=total_entities,
                total_relations=total_relations,
            )

        except Exception as e:
            logger.error(
                "Parallel KG extraction job failed",
                job_id=str(self.job_id),
                error=str(e),
                exc_info=True,
            )
            job.status = "failed"
            job.completed_at = datetime.now(timezone.utc)
            await self.db.commit()

        finally:
            async with _running_jobs_lock:
                _running_jobs.pop(str(self.job_id), None)

    async def run_with_ray(self):
        """
        Execute the extraction job using Ray distributed processing.

        Phase 60: Uses Ray actors for true distributed parallelism across
        multiple workers/machines for large-scale KG extraction.
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, falling back to parallel mode")
            return await self.run_parallel()

        # Read settings from DB (configurable via Admin UI)
        concurrency, ray_task_timeout = await self._get_extraction_settings()

        # Initialize Ray - handle Celery workers where stdout/stderr are
        # replaced with LoggingProxy objects lacking fileno() (needed by
        # Ray's faulthandler.enable()). Fix: temporarily swap in the
        # original Python file descriptors during ray.init().
        try:
            if not ray.is_initialized():
                import sys
                orig_stdout = sys.stdout
                orig_stderr = sys.stderr
                try:
                    # Use Python's original file objects (before Celery replaced them)
                    if not hasattr(sys.stdout, 'fileno'):
                        sys.stdout = sys.__stdout__ or open(os.devnull, 'w')
                    if not hasattr(sys.stderr, 'fileno'):
                        sys.stderr = sys.__stderr__ or open(os.devnull, 'w')
                    # Pass the real database URL so Ray workers don't
                    # use a path inside the packaged runtime_env.
                    # In LOCAL_MODE, DatabaseConfig computes the SQLite URL
                    # dynamically, so we must get it from there (not settings).
                    from backend.db.database import DatabaseConfig
                    db_config = DatabaseConfig()
                    db_url = db_config.database_url
                    ray.init(
                        ignore_reinit_error=True,
                        log_to_driver=False,
                        num_cpus=concurrency,
                        runtime_env={
                            "env_vars": {
                                "DATABASE_URL": db_url,
                                "LOCAL_MODE": os.getenv("LOCAL_MODE", "true"),
                                "DEV_MODE": os.getenv("DEV_MODE", "true"),
                            },
                            "excludes": [
                                ".git/",
                                "node_modules/",
                                "logs/",
                                "data/",
                                "backend/data/",
                                "__pycache__/",
                                "*.pyc",
                                "frontend/",
                            ]
                        },
                    )
                finally:
                    sys.stdout = orig_stdout
                    sys.stderr = orig_stderr
                logger.info("Ray initialized successfully for KG extraction")
        except Exception as ray_err:
            logger.warning(
                "Ray initialization failed, falling back to parallel mode",
                error=str(ray_err),
            )
            return await self.run_parallel()

        job = await self._get_job()
        if not job:
            logger.error("Job not found", job_id=str(self.job_id))
            return

        # Register for cancellation
        async with _running_jobs_lock:
            _running_jobs[str(self.job_id)] = self

        try:

            # Update status to running
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            job.last_activity_at = datetime.now(timezone.utc)
            await self.db.commit()

            # Get documents to process
            documents = await self._get_documents_for_job(job)
            job.total_documents = len(documents)
            # Persist document IDs for detail endpoint
            job.document_ids = [str(doc.id) for doc in documents]
            await self.db.commit()

            logger.info(
                "Starting Ray-distributed KG extraction job",
                job_id=str(self.job_id),
                total_documents=len(documents),
                concurrency=concurrency,
            )

            if not documents:
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                await self.db.commit()
                await notify_kg_extraction_complete(
                    job_id=str(self.job_id),
                    total_entities=0,
                    total_relations=0,
                )
                return

            # Define Ray remote function for document processing
            @ray.remote
            def process_document_ray(doc_id: str, provider_id: Optional[str] = None) -> Dict[str, Any]:
                """Process a single document for KG extraction using Ray."""
                import asyncio

                async def _process():
                    from backend.db.database import get_async_session_factory
                    from backend.services.knowledge_graph import KnowledgeGraphService

                    session_factory = get_async_session_factory()
                    async with session_factory() as session:
                        try:
                            # Get the document
                            from backend.db.models import Document
                            result = await session.execute(
                                select(Document).where(Document.id == uuid.UUID(doc_id))
                            )
                            doc = result.scalar_one_or_none()
                            if not doc:
                                return {"success": False, "error": "Document not found", "doc_id": doc_id}

                            # Create KG service
                            llm_service = None
                            if provider_id:
                                from backend.services.llm import LLMConfigManager, LLMFactory
                                config = await LLMConfigManager.get_config_for_provider_id(provider_id)
                                if config:
                                    llm_service = LLMFactory.get_chat_model(
                                        provider=config.provider_type,
                                        model=config.model,
                                        temperature=config.temperature,
                                    )

                            kg_service = KnowledgeGraphService(session, llm_service=llm_service)

                            # Clean up any stale entity data from previous extraction runs
                            await kg_service.delete_document_data(uuid.UUID(doc_id))

                            stats = await kg_service.process_document_for_graph(uuid.UUID(doc_id))

                            return {
                                "success": True,
                                "doc_id": doc_id,
                                "entities": stats.get("entities", 0),
                                "relations": stats.get("relations", 0),
                            }
                        except Exception as e:
                            return {"success": False, "error": str(e), "doc_id": doc_id}

                return asyncio.run(_process())

            # Process documents in batches using Ray
            # Batch size equals concurrency to limit peak memory usage
            batch_size = concurrency
            total_entities = 0
            total_relations = 0
            processed_count = 0
            failed_count = 0

            for batch_start in range(0, len(documents), batch_size):
                if self._cancelled:
                    break

                # Handle pause
                while self._paused and not self._cancelled:
                    job.status = "paused"
                    await self.db.commit()
                    await asyncio.sleep(PAUSE_CHECK_INTERVAL)
                    await self.db.refresh(job)
                    if job.status == "running":
                        self._paused = False
                    elif job.status == "cancelled":
                        self._cancelled = True

                if self._cancelled:
                    break

                batch = documents[batch_start:batch_start + batch_size]

                # Submit batch to Ray
                futures = [
                    process_document_ray.remote(str(doc.id), self.provider_id)
                    for doc in batch
                ]
                # Map futures to documents for result tracking
                future_to_doc = dict(zip(futures, batch))

                # Use ray.wait() to process results as each document completes,
                # giving real-time progress updates instead of waiting for entire batch
                # Adaptive timeout: base setting + 5s per chunk for the largest doc in batch
                max_chunks = max((doc.chunk_count or 0) for doc in batch)
                ray_timeout = float(max(ray_task_timeout, 600 + max_chunks * 5))
                remaining = list(futures)

                while remaining:
                    done, remaining = ray.wait(
                        remaining, num_returns=1, timeout=ray_timeout
                    )

                    if not done:
                        # Timeout: cancel remaining tasks
                        logger.warning(
                            "Ray KG extraction timed out, cancelling remaining tasks",
                            timeout=ray_timeout,
                            remaining_tasks=len(remaining),
                        )
                        for ref in remaining:
                            doc = future_to_doc[ref]
                            doc.kg_extraction_status = "failed"
                            failed_count += 1
                            # Log the timeout error
                            error_log = job.error_log or []
                            error_log.append({
                                "doc_id": str(doc.id),
                                "doc_name": doc.filename or str(doc.id),
                                "error": f"Ray task timed out after {ray_timeout}s",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            job.error_log = error_log
                            try:
                                ray.cancel(ref, force=True)
                            except Exception:
                                pass
                        remaining = []
                        break

                    # Process completed task
                    for ref in done:
                        doc = future_to_doc[ref]
                        try:
                            result = ray.get(ref)
                            if result.get("success"):
                                doc.kg_extraction_status = "completed"
                                doc.kg_extracted_at = datetime.now(timezone.utc)
                                doc.kg_entity_count = result.get("entities", 0)
                                doc.kg_relation_count = result.get("relations", 0)
                                total_entities += result.get("entities", 0)
                                total_relations += result.get("relations", 0)
                                processed_count += 1
                            else:
                                doc.kg_extraction_status = "failed"
                                failed_count += 1
                                # Log the error
                                error_log = job.error_log or []
                                error_log.append({
                                    "doc_id": result.get("doc_id", str(doc.id)),
                                    "doc_name": doc.filename or str(doc.id),
                                    "error": result.get("error", "Unknown error"),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                                job.error_log = error_log
                                logger.warning(
                                    "Ray KG extraction failed for document",
                                    doc_id=result.get("doc_id"),
                                    error=result.get("error"),
                                )
                        except Exception as e:
                            doc.kg_extraction_status = "failed"
                            failed_count += 1
                            # Log the error
                            error_log = job.error_log or []
                            error_log.append({
                                "doc_id": str(doc.id),
                                "doc_name": doc.filename or str(doc.id),
                                "error": str(e),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            job.error_log = error_log
                            logger.warning(
                                "Ray task error",
                                doc_id=str(doc.id),
                                error=str(e),
                            )

                    # Update job progress after each document completes
                    job.processed_documents = processed_count
                    job.failed_documents = failed_count
                    job.total_entities = total_entities
                    job.total_relations = total_relations
                    job.last_activity_at = datetime.now(timezone.utc)
                    await self.db.commit()

                    # Send progress notification
                    total_done = processed_count + failed_count
                    progress_pct = (total_done / len(documents) * 100) if documents else 0
                    await notify_kg_extraction_progress(
                        job_id=str(self.job_id),
                        entities_found=total_entities,
                        relations_found=total_relations,
                        progress=progress_pct,
                    )

                # Final batch commit for any timeout failures
                job.processed_documents = processed_count
                job.failed_documents = failed_count
                job.last_activity_at = datetime.now(timezone.utc)
                await self.db.commit()

            # Mark job as completed
            if self._cancelled:
                job.status = "cancelled"
            elif failed_count > 0:
                job.status = "completed_with_errors"
            else:
                job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            await self.db.commit()

            await notify_kg_extraction_complete(
                job_id=str(self.job_id),
                total_entities=total_entities,
                total_relations=total_relations,
            )

            logger.info(
                "Ray KG extraction job completed",
                job_id=str(self.job_id),
                total_documents=len(documents),
                processed=processed_count,
                failed=failed_count,
                total_entities=total_entities,
                total_relations=total_relations,
            )

        except Exception as e:
            logger.error(
                "Ray KG extraction job failed",
                job_id=str(self.job_id),
                error=str(e),
                exc_info=True,
            )
            job.status = "failed"
            job.completed_at = datetime.now(timezone.utc)
            await self.db.commit()

        finally:
            async with _running_jobs_lock:
                _running_jobs.pop(str(self.job_id), None)

            # Shutdown Ray to free memory after job completes
            try:
                if RAY_AVAILABLE and ray.is_initialized():
                    ray.shutdown()
                    logger.info("Ray shut down after KG extraction job completed")
            except Exception as shutdown_err:
                logger.warning(
                    "Failed to shut down Ray",
                    error=str(shutdown_err),
                )


class KGExtractionJobService:
    """
    Service for managing KG extraction jobs.

    Provides high-level operations for creating, monitoring,
    and controlling extraction jobs.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_job(
        self,
        user_id: uuid.UUID,
        organization_id: Optional[uuid.UUID] = None,
        only_new_documents: bool = True,
        document_ids: Optional[List[str]] = None,
        provider_id: Optional[str] = None,
    ) -> KGExtractionJob:
        """
        Create a new extraction job.

        Args:
            user_id: User starting the job
            organization_id: Organization scope
            only_new_documents: Only process documents not yet extracted
            document_ids: Optional list of specific document IDs
            provider_id: Optional LLM provider ID to use for extraction

        Returns:
            The created job
        """
        # Check for existing running job for this org
        existing = await self.get_running_job(organization_id)
        if existing:
            raise ValueError("An extraction job is already running for this organization")

        job = KGExtractionJob(
            user_id=user_id,
            organization_id=organization_id,
            only_new_documents=only_new_documents,
            document_ids=document_ids,
            provider_id=provider_id,
            status="queued",
        )

        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)

        logger.info(
            "Created KG extraction job",
            job_id=str(job.id),
            user_id=str(user_id),
            organization_id=str(organization_id) if organization_id else None,
            only_new=only_new_documents,
            provider_id=provider_id,
        )

        return job

    async def start_job(
        self,
        job_id: uuid.UUID,
        background_task_runner=None,
        use_parallel: bool = True,
    ) -> KGExtractionJob:
        """
        Start running an extraction job.

        Args:
            job_id: Job to start
            background_task_runner: Optional background task runner (e.g., FastAPI BackgroundTasks)
            use_parallel: Use parallel processing (default True, uses KG_EXTRACTION_CONCURRENCY)

        Returns:
            The updated job
        """
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status not in ("queued", "paused"):
            raise ValueError(f"Job cannot be started from status: {job.status}")

        # Store job info for background task
        org_id = job.organization_id
        provider_id = job.provider_id  # Get provider_id from job

        async def run_extraction():
            """Run extraction with its own database session and exception handling."""
            session_factory = get_async_session_factory()
            try:
                async with session_factory() as bg_session:
                    runner = KGExtractionJobRunner(
                        job_id=job_id,
                        db_session=bg_session,
                        organization_id=org_id,
                        provider_id=provider_id,
                    )
                    # Phase 60: Choose processing method based on settings and availability
                    # Priority: Ray (if enabled) > Parallel (asyncio) > Sequential
                    use_ray = (
                        RAY_AVAILABLE
                        and getattr(settings, 'USE_RAY_FOR_KG', True)
                        and use_parallel
                    )

                    if use_ray:
                        logger.info("Using Ray distributed processing for KG extraction", job_id=str(job_id))
                        await runner.run_with_ray()
                    elif use_parallel:
                        await runner.run_parallel()
                    else:
                        await runner.run()
            except Exception as e:
                logger.error(
                    "Background extraction task crashed",
                    job_id=str(job_id),
                    error=str(e),
                    exc_info=True,
                )
                # Update job status to failed in database
                try:
                    async with session_factory() as cleanup_session:
                        result = await cleanup_session.execute(
                            select(KGExtractionJob).where(KGExtractionJob.id == job_id)
                        )
                        failed_job = result.scalar_one_or_none()
                        if failed_job and failed_job.status in ("queued", "running", "paused"):
                            failed_job.status = "failed"
                            failed_job.completed_at = datetime.now(timezone.utc)
                            error_log = failed_job.error_log or []
                            error_log.append({
                                "error": f"Background task crashed: {str(e)}",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            failed_job.error_log = error_log
                            await cleanup_session.commit()
                except Exception as cleanup_error:
                    logger.error(
                        "Failed to update job status after crash",
                        job_id=str(job_id),
                        error=str(cleanup_error),
                    )
                # Clean up running jobs registry
                async with _running_jobs_lock:
                    _running_jobs.pop(str(job_id), None)

        # Run in background
        if background_task_runner:
            background_task_runner.add_task(run_extraction)
        else:
            # Run as asyncio task
            asyncio.create_task(run_extraction())

        return job

    async def get_job(self, job_id: uuid.UUID) -> Optional[KGExtractionJob]:
        """Get a job by ID with fresh data from database."""
        # CRITICAL: Expire cached objects to ensure fresh read
        # This fixes session isolation issue where background task updates
        # aren't visible to the API session due to SQLAlchemy caching
        self.db.expire_all()

        result = await self.db.execute(
            select(KGExtractionJob).where(KGExtractionJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def get_running_job(
        self,
        organization_id: Optional[uuid.UUID] = None,
    ) -> Optional[KGExtractionJob]:
        """Get the currently running job for an organization with fresh data."""
        # CRITICAL: Expire cached objects to ensure fresh read
        self.db.expire_all()

        query = select(KGExtractionJob).where(
            KGExtractionJob.status.in_(["queued", "running", "paused"])
        )

        if organization_id:
            query = query.where(
                or_(
                    KGExtractionJob.organization_id == organization_id,
                    KGExtractionJob.organization_id.is_(None),
                )
            )

        result = await self.db.execute(query.order_by(KGExtractionJob.created_at.desc()).limit(1))
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        user_id: Optional[uuid.UUID] = None,
        organization_id: Optional[uuid.UUID] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[KGExtractionJob]:
        """List extraction jobs with optional filters."""
        query = select(KGExtractionJob)

        if user_id:
            query = query.where(KGExtractionJob.user_id == user_id)

        if organization_id:
            query = query.where(
                or_(
                    KGExtractionJob.organization_id == organization_id,
                    KGExtractionJob.organization_id.is_(None),
                )
            )

        if status:
            query = query.where(KGExtractionJob.status == status)

        query = query.order_by(KGExtractionJob.created_at.desc()).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def cancel_job(self, job_id: uuid.UUID) -> bool:
        """
        Cancel a running job.

        Returns True if cancellation was requested, False if job not running.
        """
        async with _running_jobs_lock:
            runner = _running_jobs.get(str(job_id))
        if runner:
            await runner.cancel()
            return True

        # Job not in memory, update database directly
        job = await self.get_job(job_id)
        if job and job.status in ("queued", "running", "paused"):
            job.status = "cancelled"
            job.completed_at = datetime.now(timezone.utc)
            await self.db.commit()
            return True

        return False

    async def pause_job(self, job_id: uuid.UUID) -> bool:
        """Pause a running job."""
        async with _running_jobs_lock:
            runner = _running_jobs.get(str(job_id))
        if runner:
            await runner.pause()
            return True

        # Job not in memory registry - check if it's running/queued in DB
        # This can happen in race conditions or if server restarted
        job = await self.get_job(job_id)
        if job and job.status in ("running", "queued"):
            # Update status directly - the runner will pick it up if it exists
            job.status = "paused"
            await self.db.commit()
            logger.warning(
                "Paused job via DB (runner not in registry)",
                job_id=str(job_id),
                original_status=job.status,
            )
            return True

        return False

    async def resume_job(self, job_id: uuid.UUID) -> bool:
        """Resume a paused job."""
        async with _running_jobs_lock:
            runner = _running_jobs.get(str(job_id))
        if runner:
            await runner.resume()
            return True

        # If job not in memory but status is paused, restart it
        job = await self.get_job(job_id)
        if job and job.status == "paused":
            await self.start_job(job_id)
            return True

        return False

    async def get_progress(self, job_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get detailed progress for a job."""
        job = await self.get_job(job_id)
        if not job:
            return None

        # Calculate progress as (processed + failed) / total
        # Since processed now only counts successes, we need to add them
        total_done = job.processed_documents + job.failed_documents
        progress_percent = (total_done / job.total_documents * 100) if job.total_documents > 0 else 0.0

        return {
            "job_id": str(job.id),
            "status": job.status,
            "progress_percent": progress_percent,
            "processed_documents": job.processed_documents,  # Successfully completed only
            "total_documents": job.total_documents,
            "failed_documents": job.failed_documents,
            "total_entities": job.total_entities,
            "total_relations": job.total_relations,
            "current_document": job.current_document_name,
            "current_document_id": str(job.current_document_id) if job.current_document_id else None,
            "estimated_remaining_seconds": job.get_estimated_remaining_seconds(),
            "avg_doc_processing_time": job.avg_doc_processing_time,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "last_activity_at": job.last_activity_at.isoformat() if job.last_activity_at else None,
            "can_cancel": job.status in ("queued", "running", "paused"),
            "can_pause": job.status == "running",
            "can_resume": job.status == "paused",
            "error_count": len(job.error_log or []),
            "only_new_documents": job.only_new_documents,
            # Include last 5 errors for UI display
            "errors": (job.error_log or [])[-5:],
        }

    async def get_documents_pending_extraction(
        self,
        organization_id: Optional[uuid.UUID] = None,
    ) -> int:
        """Get count of documents pending extraction."""
        query = select(func.count(Document.id)).where(
            Document.processing_status == ProcessingStatus.COMPLETED,
            or_(
                Document.kg_extraction_status == "pending",
                Document.kg_extraction_status.is_(None),
            )
        )

        if organization_id:
            query = query.where(
                or_(
                    Document.organization_id == organization_id,
                    Document.organization_id.is_(None),
                )
            )

        result = await self.db.execute(query)
        return result.scalar() or 0

    async def get_job_documents_detail(self, job_id: uuid.UUID) -> Optional[List[Dict[str, Any]]]:
        """
        Get per-document detail for a job, including status, chunk progress, and entity/relation counts.

        Returns None if the job doesn't exist.
        For running jobs, includes live chunk progress from the in-memory runner.
        """
        job = await self.get_job(job_id)
        if not job:
            return None

        # Get document IDs from the persisted list
        doc_id_strs = job.document_ids
        if not doc_id_strs:
            return []

        # Convert to UUIDs
        try:
            doc_ids = [uuid.UUID(d) for d in doc_id_strs]
        except (ValueError, AttributeError):
            return []

        # Query documents with their current state
        result = await self.db.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        documents = result.scalars().all()

        # Get chunk counts for each document
        chunk_counts_result = await self.db.execute(
            select(
                Chunk.document_id,
                func.count(Chunk.id).label("chunk_count"),
            )
            .where(Chunk.document_id.in_(doc_ids))
            .group_by(Chunk.document_id)
        )
        chunk_counts = {str(row.document_id): row.chunk_count for row in chunk_counts_result}

        # Get live chunk progress from running job (if any)
        chunk_progress = {}
        runner = _running_jobs.get(str(job_id))
        if runner:
            chunk_progress = runner.get_chunk_progress()

        # Build response
        details = []
        for doc in documents:
            doc_id_str = str(doc.id)

            # Map kg_extraction_status to display status
            status = doc.kg_extraction_status or "pending"

            # Get live chunk progress for processing documents
            live_progress = chunk_progress.get(doc_id_str)
            chunks_processed = None
            if live_progress:
                chunks_processed = live_progress.get("done", 0)

            details.append({
                "document_id": doc_id_str,
                "filename": doc.original_filename or doc.filename or doc_id_str,
                "status": status,
                "chunk_count": chunk_counts.get(doc_id_str, doc.chunk_count or 0),
                "chunks_processed": chunks_processed,
                "kg_entity_count": doc.kg_entity_count or 0,
                "kg_relation_count": doc.kg_relation_count or 0,
            })

        # Sort: processing first, then pending, then completed, then failed
        status_order = {"processing": 0, "pending": 1, "completed": 2, "failed": 3}
        details.sort(key=lambda d: status_order.get(d["status"], 4))

        return details
