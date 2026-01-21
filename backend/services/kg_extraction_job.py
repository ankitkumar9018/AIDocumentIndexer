"""
AIDocumentIndexer - Knowledge Graph Extraction Job Service
============================================================

Background job management for knowledge graph entity extraction.
Handles starting, tracking, pausing, and cancelling extraction jobs.
"""

import asyncio
import uuid
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

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

logger = structlog.get_logger(__name__)


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
                    stats = await kg_service.process_document_for_graph(doc.id)

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
    ) -> KGExtractionJob:
        """
        Start running an extraction job.

        Args:
            job_id: Job to start
            background_task_runner: Optional background task runner (e.g., FastAPI BackgroundTasks)

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
