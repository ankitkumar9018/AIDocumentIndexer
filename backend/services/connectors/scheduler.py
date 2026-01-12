"""
AIDocumentIndexer - Connector Sync Scheduler
=============================================

Manages scheduled synchronization for external connectors.

Features:
- Cron-based scheduling
- Background sync with Ray or asyncio fallback
- Incremental sync with change detection
- Rate limiting and backoff
- Health monitoring
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

import structlog

# Optional dependency for cron scheduling
try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    croniter = None
    HAS_CRONITER = False

from backend.services.base import BaseService, ServiceException
from backend.services.connectors.base import BaseConnector, Resource, Change
from backend.services.connectors.registry import ConnectorRegistry, ConnectorType

logger = structlog.get_logger(__name__)


class SyncStatus(str, Enum):
    """Sync job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SyncJob:
    """Represents a sync job."""
    id: str
    connector_instance_id: str
    connector_type: ConnectorType
    status: SyncStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    resources_synced: int = 0
    resources_failed: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleEntry:
    """Represents a scheduled sync."""
    connector_instance_id: str
    cron_expression: str
    organization_id: str
    enabled: bool = True
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    consecutive_failures: int = 0
    max_failures: int = 5


class ConnectorSyncScheduler(BaseService):
    """
    Manages scheduled synchronization for connectors.

    Supports both Ray-based and asyncio-based execution for flexibility.
    """

    def __init__(
        self,
        session=None,
        organization_id=None,
        user_id=None,
        use_ray: bool = True,
    ):
        super().__init__(session, organization_id, user_id)
        self._schedules: Dict[str, ScheduleEntry] = {}
        self._running_jobs: Dict[str, SyncJob] = {}
        self._job_history: List[SyncJob] = []
        self._max_history = 100
        self._use_ray = use_ray and self._is_ray_available()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Rate limiting
        self._rate_limiter: Dict[str, datetime] = {}
        self._min_sync_interval = timedelta(minutes=5)

    def _is_ray_available(self) -> bool:
        """Check if Ray is available and initialized."""
        try:
            import ray
            return ray.is_initialized()
        except ImportError:
            return False
        except Exception:
            return False

    async def start(self):
        """Start the scheduler background task."""
        if self._scheduler_task is not None:
            self.log_warning("Scheduler already running")
            return

        self._shutdown_event.clear()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.log_info("Connector sync scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._shutdown_event.set()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        self.log_info("Connector sync scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop - checks for due syncs every minute."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.utcnow()

                # Check each schedule
                for schedule in list(self._schedules.values()):
                    if not schedule.enabled:
                        continue

                    # Skip if in backoff due to failures
                    if schedule.consecutive_failures >= schedule.max_failures:
                        continue

                    if schedule.next_run_at and schedule.next_run_at <= now:
                        # Time to sync
                        await self._trigger_sync(schedule)

                # Sleep until next check
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60.0,  # Check every minute
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error("Scheduler loop error", error=e)
                await asyncio.sleep(60)

    async def _trigger_sync(self, schedule: ScheduleEntry):
        """Trigger a sync for a schedule."""
        # Rate limiting check
        if schedule.connector_instance_id in self._rate_limiter:
            last_sync = self._rate_limiter[schedule.connector_instance_id]
            if datetime.utcnow() - last_sync < self._min_sync_interval:
                self.log_debug(
                    "Skipping sync due to rate limit",
                    connector_instance_id=schedule.connector_instance_id,
                )
                return

        # Update rate limiter
        self._rate_limiter[schedule.connector_instance_id] = datetime.utcnow()

        # Create sync job
        job = SyncJob(
            id=str(uuid.uuid4()),
            connector_instance_id=schedule.connector_instance_id,
            connector_type=ConnectorType.GOOGLE_DRIVE,  # Will be loaded from DB
            status=SyncStatus.PENDING,
            created_at=datetime.utcnow(),
        )

        self._running_jobs[job.id] = job

        try:
            # Execute sync (Ray or asyncio)
            if self._use_ray:
                await self._execute_sync_ray(job, schedule)
            else:
                await self._execute_sync_async(job, schedule)

            # Update schedule on success
            schedule.last_run_at = datetime.utcnow()
            schedule.next_run_at = self._calculate_next_run(schedule.cron_expression)
            schedule.consecutive_failures = 0

        except Exception as e:
            self.log_error("Sync failed", error=e, job_id=job.id)
            job.status = SyncStatus.FAILED
            job.error_message = str(e)
            schedule.consecutive_failures += 1

            # Calculate next run with exponential backoff
            backoff_minutes = min(60 * 2 ** schedule.consecutive_failures, 1440)  # Max 24h
            schedule.next_run_at = datetime.utcnow() + timedelta(minutes=backoff_minutes)

        finally:
            # Move to history
            del self._running_jobs[job.id]
            self._job_history.append(job)

            # Trim history
            if len(self._job_history) > self._max_history:
                self._job_history = self._job_history[-self._max_history:]

    async def _execute_sync_ray(self, job: SyncJob, schedule: ScheduleEntry):
        """Execute sync using Ray for distributed processing."""
        import ray
        from backend.ray_workers.config import init_ray

        # Ensure Ray is initialized
        if not ray.is_initialized():
            if not init_ray():
                # Fallback to async
                self.log_warning("Ray not available, falling back to async")
                return await self._execute_sync_async(job, schedule)

        @ray.remote
        def sync_connector_task(
            connector_instance_id: str,
            organization_id: str,
            sync_token: Optional[str],
        ) -> Dict[str, Any]:
            """Ray task for connector sync."""
            import asyncio

            async def _sync():
                from backend.db.database import async_session_context
                from backend.services.connectors.sync_service import ConnectorSyncService

                async with async_session_context() as session:
                    sync_service = ConnectorSyncService(
                        session=session,
                        organization_id=organization_id,
                    )
                    return await sync_service.execute_sync(
                        connector_instance_id=connector_instance_id,
                        sync_token=sync_token,
                    )

            return asyncio.run(_sync())

        # Submit Ray task
        job.status = SyncStatus.RUNNING
        job.started_at = datetime.utcnow()

        future = sync_connector_task.remote(
            connector_instance_id=schedule.connector_instance_id,
            organization_id=schedule.organization_id,
            sync_token=None,  # Will be loaded from DB
        )

        # Wait for result with timeout
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ray.get(future, timeout=3600),  # 1 hour timeout
            )

            job.status = SyncStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.resources_synced = result.get("resources_synced", 0)
            job.resources_failed = result.get("resources_failed", 0)
            job.metadata = result

        except Exception as e:
            job.status = SyncStatus.FAILED
            job.error_message = str(e)
            raise

    async def _execute_sync_async(self, job: SyncJob, schedule: ScheduleEntry):
        """Execute sync using asyncio (fallback when Ray unavailable)."""
        from backend.db.database import async_session_context

        job.status = SyncStatus.RUNNING
        job.started_at = datetime.utcnow()

        try:
            async with async_session_context() as session:
                # Import here to avoid circular imports
                from backend.services.connectors.sync_service import ConnectorSyncService

                sync_service = ConnectorSyncService(
                    session=session,
                    organization_id=schedule.organization_id,
                )

                result = await sync_service.execute_sync(
                    connector_instance_id=schedule.connector_instance_id,
                    sync_token=None,
                )

                job.status = SyncStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.resources_synced = result.get("resources_synced", 0)
                job.resources_failed = result.get("resources_failed", 0)
                job.metadata = result

        except Exception as e:
            job.status = SyncStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            raise

    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """Calculate the next run time from a cron expression."""
        if not HAS_CRONITER:
            raise ServiceException("croniter package is required for cron scheduling. Install with: pip install croniter")
        cron = croniter(cron_expression, datetime.utcnow())
        return cron.get_next(datetime)

    # ==========================================================================
    # Schedule Management
    # ==========================================================================

    def add_schedule(
        self,
        connector_instance_id: str,
        cron_expression: str,
        organization_id: str,
        enabled: bool = True,
    ) -> ScheduleEntry:
        """Add or update a sync schedule."""
        # Validate cron expression
        if not HAS_CRONITER:
            raise ServiceException("croniter package is required for cron scheduling. Install with: pip install croniter")
        try:
            croniter(cron_expression)
        except Exception as e:
            raise ServiceException(f"Invalid cron expression: {e}")

        schedule = ScheduleEntry(
            connector_instance_id=connector_instance_id,
            cron_expression=cron_expression,
            organization_id=organization_id,
            enabled=enabled,
            next_run_at=self._calculate_next_run(cron_expression) if enabled else None,
        )

        self._schedules[connector_instance_id] = schedule

        self.log_info(
            "Schedule added",
            connector_instance_id=connector_instance_id,
            cron=cron_expression,
            next_run=schedule.next_run_at.isoformat() if schedule.next_run_at else None,
        )

        return schedule

    def remove_schedule(self, connector_instance_id: str):
        """Remove a sync schedule."""
        if connector_instance_id in self._schedules:
            del self._schedules[connector_instance_id]
            self.log_info("Schedule removed", connector_instance_id=connector_instance_id)

    def enable_schedule(self, connector_instance_id: str):
        """Enable a sync schedule."""
        if connector_instance_id in self._schedules:
            schedule = self._schedules[connector_instance_id]
            schedule.enabled = True
            schedule.next_run_at = self._calculate_next_run(schedule.cron_expression)
            schedule.consecutive_failures = 0

    def disable_schedule(self, connector_instance_id: str):
        """Disable a sync schedule."""
        if connector_instance_id in self._schedules:
            schedule = self._schedules[connector_instance_id]
            schedule.enabled = False
            schedule.next_run_at = None

    def get_schedule(self, connector_instance_id: str) -> Optional[ScheduleEntry]:
        """Get a schedule by connector instance ID."""
        return self._schedules.get(connector_instance_id)

    def list_schedules(self) -> List[ScheduleEntry]:
        """List all schedules."""
        return list(self._schedules.values())

    # ==========================================================================
    # Manual Sync
    # ==========================================================================

    async def trigger_manual_sync(
        self,
        connector_instance_id: str,
        organization_id: str,
    ) -> SyncJob:
        """Trigger a manual sync for a connector."""
        # Create a temporary schedule for manual sync
        temp_schedule = ScheduleEntry(
            connector_instance_id=connector_instance_id,
            cron_expression="* * * * *",  # Unused for manual
            organization_id=organization_id,
        )

        job = SyncJob(
            id=str(uuid.uuid4()),
            connector_instance_id=connector_instance_id,
            connector_type=ConnectorType.GOOGLE_DRIVE,
            status=SyncStatus.PENDING,
            created_at=datetime.utcnow(),
        )

        self._running_jobs[job.id] = job

        try:
            if self._use_ray:
                await self._execute_sync_ray(job, temp_schedule)
            else:
                await self._execute_sync_async(job, temp_schedule)
        except Exception as e:
            job.status = SyncStatus.FAILED
            job.error_message = str(e)
        finally:
            del self._running_jobs[job.id]
            self._job_history.append(job)

        return job

    # ==========================================================================
    # Job Status
    # ==========================================================================

    def get_running_jobs(self) -> List[SyncJob]:
        """Get currently running sync jobs."""
        return list(self._running_jobs.values())

    def get_job_history(
        self,
        connector_instance_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[SyncJob]:
        """Get sync job history."""
        jobs = self._job_history

        if connector_instance_id:
            jobs = [j for j in jobs if j.connector_instance_id == connector_instance_id]

        return jobs[-limit:]

    def get_job(self, job_id: str) -> Optional[SyncJob]:
        """Get a specific job by ID."""
        # Check running first
        if job_id in self._running_jobs:
            return self._running_jobs[job_id]

        # Check history
        for job in self._job_history:
            if job.id == job_id:
                return job

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "schedules_count": len(self._schedules),
            "schedules_enabled": sum(1 for s in self._schedules.values() if s.enabled),
            "running_jobs": len(self._running_jobs),
            "history_count": len(self._job_history),
            "use_ray": self._use_ray,
            "ray_available": self._is_ray_available(),
        }


class ConnectorSyncService(BaseService):
    """
    Service for executing connector synchronization.

    Handles the actual sync logic including:
    - Incremental change detection
    - Resource download and processing
    - Document creation/update
    """

    async def execute_sync(
        self,
        connector_instance_id: str,
        sync_token: Optional[str] = None,
        full_sync: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a sync for a connector instance.

        Args:
            connector_instance_id: The connector instance to sync
            sync_token: Token for incremental sync (from previous sync)
            full_sync: Force full sync instead of incremental

        Returns:
            Sync result with statistics
        """
        from backend.db.database import async_session_context
        from backend.db.models import ConnectorInstance, SyncedResource, Document
        from sqlalchemy import select

        session = await self.get_session()

        # Load connector instance
        result = await session.execute(
            select(ConnectorInstance).where(ConnectorInstance.id == connector_instance_id)
        )
        instance = result.scalar_one_or_none()

        if not instance:
            raise ServiceException(f"Connector instance not found: {connector_instance_id}")

        # Get connector class
        connector_class = ConnectorRegistry.get(ConnectorType(instance.connector_type))
        if not connector_class:
            raise ServiceException(f"Unknown connector type: {instance.connector_type}")

        # Create connector instance
        from backend.services.connectors.base import ConnectorConfig

        connector = connector_class(
            config=ConnectorConfig(
                credentials=instance.credentials or {},
                settings=instance.sync_config or {},
            ),
            session=session,
            organization_id=self._organization_id,
        )

        # Authenticate
        if not await connector.authenticate():
            raise ServiceException("Connector authentication failed")

        stats = {
            "resources_synced": 0,
            "resources_failed": 0,
            "resources_deleted": 0,
            "new_documents": 0,
            "updated_documents": 0,
        }

        try:
            if full_sync or not sync_token:
                # Full sync - list all resources
                await self._full_sync(connector, instance, session, stats)
            else:
                # Incremental sync - get changes since last sync
                await self._incremental_sync(connector, instance, sync_token, session, stats)

            # Update instance
            instance.last_sync_at = datetime.utcnow()
            instance.status = "active"
            instance.error_message = None
            await session.commit()

        except Exception as e:
            instance.status = "error"
            instance.error_message = str(e)
            await session.commit()
            raise

        return stats

    async def _full_sync(
        self,
        connector: BaseConnector,
        instance,
        session,
        stats: Dict[str, int],
    ):
        """Perform a full sync of all resources."""
        from backend.db.models import SyncedResource

        folders_to_process = [None]  # Start from root

        while folders_to_process:
            folder_id = folders_to_process.pop(0)
            page_token = None

            while True:
                resources, next_token = await connector.list_resources(
                    folder_id=folder_id,
                    page_token=page_token,
                )

                for resource in resources:
                    try:
                        if resource.resource_type.value == "folder":
                            # Add folder to processing queue
                            folders_to_process.append(resource.id)
                        else:
                            # Process file
                            await self._sync_resource(connector, resource, instance, session)
                            stats["resources_synced"] += 1
                    except Exception as e:
                        self.log_error("Failed to sync resource", error=e, resource_id=resource.id)
                        stats["resources_failed"] += 1

                if not next_token:
                    break
                page_token = next_token

    async def _incremental_sync(
        self,
        connector: BaseConnector,
        instance,
        sync_token: str,
        session,
        stats: Dict[str, int],
    ):
        """Perform an incremental sync based on changes."""
        changes, new_token = await connector.get_changes(since_token=sync_token)

        for change in changes:
            try:
                if change.change_type == "deleted":
                    await self._handle_deletion(change.resource_id, instance, session)
                    stats["resources_deleted"] += 1
                else:
                    if change.resource:
                        await self._sync_resource(connector, change.resource, instance, session)
                        stats["resources_synced"] += 1
            except Exception as e:
                self.log_error("Failed to process change", error=e, resource_id=change.resource_id)
                stats["resources_failed"] += 1

        # Store new sync token
        if instance.sync_config is None:
            instance.sync_config = {}
        instance.sync_config["sync_token"] = new_token

    async def _sync_resource(
        self,
        connector: BaseConnector,
        resource: Resource,
        instance,
        session,
    ):
        """Sync a single resource to the database."""
        from backend.db.models import SyncedResource, Document
        from sqlalchemy import select

        # Check if resource already exists
        result = await session.execute(
            select(SyncedResource).where(
                SyncedResource.connector_instance_id == instance.id,
                SyncedResource.external_id == resource.id,
            )
        )
        synced_resource = result.scalar_one_or_none()

        # Skip if not modified
        if synced_resource and resource.modified_at:
            if synced_resource.external_modified_at >= resource.modified_at:
                return

        # Download content
        content = await connector.download_resource(resource.id)
        if not content:
            self.log_warning("No content for resource", resource_id=resource.id)
            return

        # Create or update document
        if synced_resource and synced_resource.document_id:
            # Update existing document
            doc_result = await session.execute(
                select(Document).where(Document.id == synced_resource.document_id)
            )
            document = doc_result.scalar_one_or_none()
            if document:
                # Update document content (would trigger reprocessing)
                document.title = resource.name
                document.updated_at = datetime.utcnow()
        else:
            # Create new document
            document = Document(
                id=uuid.uuid4(),
                organization_id=self._organization_id,
                title=resource.name,
                filename=resource.name,
                file_type=self._get_file_extension(resource.mime_type, resource.name),
                file_size=len(content),
                source_url=resource.web_url,
                source_type="connector",
                processing_status="pending",
            )
            session.add(document)
            await session.flush()

            # Save content to storage
            await self._save_content(document.id, content, resource.name)

        # Update synced resource record
        if synced_resource:
            synced_resource.external_modified_at = resource.modified_at
            synced_resource.synced_at = datetime.utcnow()
            synced_resource.metadata = resource.metadata
        else:
            synced_resource = SyncedResource(
                id=uuid.uuid4(),
                connector_instance_id=instance.id,
                external_id=resource.id,
                document_id=document.id,
                external_modified_at=resource.modified_at,
                synced_at=datetime.utcnow(),
                metadata=resource.metadata,
            )
            session.add(synced_resource)

    async def _handle_deletion(
        self,
        resource_id: str,
        instance,
        session,
    ):
        """Handle a deleted resource."""
        from backend.db.models import SyncedResource, Document
        from sqlalchemy import select

        result = await session.execute(
            select(SyncedResource).where(
                SyncedResource.connector_instance_id == instance.id,
                SyncedResource.external_id == resource_id,
            )
        )
        synced_resource = result.scalar_one_or_none()

        if synced_resource:
            # Optionally delete the document or just unlink
            if synced_resource.document_id:
                doc_result = await session.execute(
                    select(Document).where(Document.id == synced_resource.document_id)
                )
                document = doc_result.scalar_one_or_none()
                if document:
                    # Soft delete - mark as archived
                    document.is_archived = True
                    document.archived_at = datetime.utcnow()

            # Remove synced resource record
            await session.delete(synced_resource)

    def _get_file_extension(self, mime_type: Optional[str], filename: str) -> str:
        """Get file extension from mime type or filename."""
        if filename and "." in filename:
            return filename.rsplit(".", 1)[-1].lower()

        mime_to_ext = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "text/plain": "txt",
            "text/markdown": "md",
        }

        return mime_to_ext.get(mime_type, "bin")

    async def _save_content(self, document_id: uuid.UUID, content: bytes, filename: str):
        """Save file content to storage."""
        from pathlib import Path
        from backend.core.config import settings

        storage_path = Path(getattr(settings, "DOCUMENT_STORAGE_PATH", "./storage/documents"))
        storage_path.mkdir(parents=True, exist_ok=True)

        file_path = storage_path / str(document_id) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(content)


# Singleton scheduler instance
_scheduler: Optional[ConnectorSyncScheduler] = None


def get_scheduler() -> ConnectorSyncScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ConnectorSyncScheduler()
    return _scheduler


async def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()


async def stop_scheduler():
    """Stop the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop()
