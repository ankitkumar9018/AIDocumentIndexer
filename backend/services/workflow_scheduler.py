"""
AIDocumentIndexer - Workflow Scheduler Service
===============================================

Scheduled workflow execution using Celery Beat.

Provides:
- Cron-based scheduling for recurring workflow executions
- Timezone-aware scheduling
- Integration with Celery Beat for periodic task registration
- Automatic schedule synchronization from database

Usage:
    from backend.services.workflow_scheduler import get_workflow_scheduler

    scheduler = get_workflow_scheduler()
    await scheduler.sync_schedules()  # Load schedules from DB
    await scheduler.schedule_workflow(workflow_id, "0 9 * * *", "UTC")
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

# Try to import Celery components
try:
    from celery import Celery
    from celery.schedules import crontab
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False
    Celery = None  # type: ignore
    crontab = None  # type: ignore

logger = structlog.get_logger(__name__)


# =============================================================================
# Cron Parsing Helpers
# =============================================================================

def _parse_cron_expression(expression: str) -> Optional[Any]:
    """
    Parse a standard cron expression into a Celery crontab object.

    Supports the standard five-field cron format:
        minute hour day_of_month month_of_year day_of_week

    Examples:
        "0 9 * * *"     -> 9:00 AM daily
        "0 9 * * 1-5"   -> 9:00 AM weekdays
        "*/15 * * * *"  -> every 15 minutes
        "0 0 1 * *"     -> midnight on 1st of each month

    Args:
        expression: A cron expression string with five space-separated fields.

    Returns:
        A celery.schedules.crontab instance, or None if Celery is unavailable.

    Raises:
        ValueError: If the expression does not contain exactly five fields.
    """
    if not HAS_CELERY:
        return None

    parts = expression.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"Invalid cron expression '{expression}': expected 5 fields "
            f"(minute hour day_of_month month_of_year day_of_week), got {len(parts)}"
        )

    minute, hour, day_of_month, month_of_year, day_of_week = parts
    return crontab(
        minute=minute,
        hour=hour,
        day_of_month=day_of_month,
        month_of_year=month_of_year,
        day_of_week=day_of_week,
    )


def validate_cron_expression(expression: str) -> bool:
    """
    Validate a cron expression without creating a crontab object.

    Args:
        expression: Cron expression to validate.

    Returns:
        True if valid, False otherwise.
    """
    try:
        parts = expression.strip().split()
        if len(parts) != 5:
            return False

        # Basic validation of each field
        for i, part in enumerate(parts):
            if not part:
                return False
            # Allow *, numbers, ranges (1-5), steps (*/15), lists (1,2,3)
            valid_chars = set("0123456789*,/-")
            if not all(c in valid_chars for c in part):
                return False

        return True
    except Exception:
        return False


# =============================================================================
# Workflow Scheduler Service
# =============================================================================

class WorkflowSchedulerService:
    """
    Service for managing scheduled workflow executions.

    Integrates with Celery Beat for periodic task dispatch and
    synchronizes schedules from the database.
    """

    def __init__(self) -> None:
        """Initialize the scheduler with optional Celery app."""
        self._celery_app: Optional[Any] = None
        self._scheduled_workflows: Dict[str, Dict[str, Any]] = {}

        if HAS_CELERY:
            try:
                from backend.core.celery_app import celery_app
                self._celery_app = celery_app
                logger.info("WorkflowSchedulerService initialized with Celery Beat support")
            except ImportError:
                logger.warning(
                    "Celery app not found. Scheduled workflows will not be dispatched."
                )
        else:
            logger.warning(
                "Celery not available. Scheduled workflows will not be dispatched."
            )

    # -------------------------------------------------------------------------
    # Schedule Management
    # -------------------------------------------------------------------------

    async def sync_schedules(self) -> int:
        """
        Synchronize schedules from database to Celery Beat.

        Loads all active scheduled workflows and registers them with Celery.

        Returns:
            Number of schedules synchronized.
        """
        from backend.db.database import get_async_session_context
        from backend.db.models import Workflow, WorkflowTriggerType

        count = 0

        try:
            async with get_async_session_context() as session:
                from sqlalchemy import select

                # Get all active scheduled workflows
                result = await session.execute(
                    select(Workflow).where(
                        Workflow.trigger_type == WorkflowTriggerType.SCHEDULED.value,
                        Workflow.is_active == True,
                        Workflow.is_draft == False,
                    )
                )
                workflows = result.scalars().all()

                for workflow in workflows:
                    trigger_config = workflow.trigger_config or {}
                    cron_expr = trigger_config.get("cron")
                    timezone = trigger_config.get("timezone", "UTC")

                    if cron_expr and validate_cron_expression(cron_expr):
                        self._register_workflow_schedule(
                            workflow_id=str(workflow.id),
                            cron_expression=cron_expr,
                            timezone=timezone,
                            workflow_name=workflow.name,
                        )
                        count += 1

                logger.info(
                    "Synchronized workflow schedules",
                    count=count,
                    total_workflows=len(workflows),
                )

        except Exception as e:
            logger.error("Failed to sync workflow schedules", error=str(e))

        return count

    async def schedule_workflow(
        self,
        workflow_id: str,
        cron_expression: str,
        timezone: str = "UTC",
    ) -> Dict[str, Any]:
        """
        Schedule a workflow for periodic execution.

        Args:
            workflow_id: UUID of the workflow to schedule.
            cron_expression: Cron expression (e.g., "0 9 * * *").
            timezone: Timezone for the schedule (default: UTC).

        Returns:
            Dict with schedule details.

        Raises:
            ValueError: If cron expression is invalid.
        """
        from backend.db.database import get_async_session_context
        from backend.db.models import Workflow, WorkflowTriggerType

        # Validate cron expression
        if not validate_cron_expression(cron_expression):
            raise ValueError(f"Invalid cron expression: {cron_expression}")

        async with get_async_session_context() as session:
            from sqlalchemy import select

            # Get workflow
            result = await session.execute(
                select(Workflow).where(Workflow.id == UUID(workflow_id))
            )
            workflow = result.scalar_one_or_none()

            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Update workflow trigger configuration
            workflow.trigger_type = WorkflowTriggerType.SCHEDULED.value
            workflow.trigger_config = {
                "cron": cron_expression,
                "timezone": timezone,
            }

            await session.commit()

            # Register with Celery Beat
            self._register_workflow_schedule(
                workflow_id=workflow_id,
                cron_expression=cron_expression,
                timezone=timezone,
                workflow_name=workflow.name,
            )

            logger.info(
                "Workflow scheduled",
                workflow_id=workflow_id,
                cron=cron_expression,
                timezone=timezone,
            )

            return {
                "workflow_id": workflow_id,
                "cron": cron_expression,
                "timezone": timezone,
                "status": "scheduled",
            }

    async def unschedule_workflow(self, workflow_id: str) -> bool:
        """
        Remove a workflow from the schedule.

        Args:
            workflow_id: UUID of the workflow to unschedule.

        Returns:
            True if unscheduled, False if not found.
        """
        self._unregister_workflow_schedule(workflow_id)

        if workflow_id in self._scheduled_workflows:
            del self._scheduled_workflows[workflow_id]
            logger.info("Workflow unscheduled", workflow_id=workflow_id)
            return True

        return False

    def get_schedule(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current schedule for a workflow.

        Args:
            workflow_id: UUID of the workflow.

        Returns:
            Schedule details if found, None otherwise.
        """
        return self._scheduled_workflows.get(workflow_id)

    def list_schedules(self) -> List[Dict[str, Any]]:
        """
        List all registered workflow schedules.

        Returns:
            List of schedule details.
        """
        return list(self._scheduled_workflows.values())

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute_scheduled_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a scheduled workflow.

        Called by Celery Beat when the schedule triggers.

        Args:
            workflow_id: UUID of the workflow to execute.

        Returns:
            Execution result with execution_id and status.
        """
        from backend.db.database import get_async_session_context
        from backend.db.models import Workflow, WorkflowTriggerType
        from backend.services.workflow_engine import get_execution_engine

        logger.info("Executing scheduled workflow", workflow_id=workflow_id)

        try:
            async with get_async_session_context() as session:
                from sqlalchemy import select

                # Get workflow
                result = await session.execute(
                    select(Workflow).where(Workflow.id == UUID(workflow_id))
                )
                workflow = result.scalar_one_or_none()

                if not workflow:
                    raise ValueError(f"Workflow not found: {workflow_id}")

                if not workflow.is_active:
                    logger.warning(
                        "Skipping inactive workflow",
                        workflow_id=workflow_id,
                    )
                    return {
                        "workflow_id": workflow_id,
                        "status": "skipped",
                        "reason": "workflow_inactive",
                    }

                # Get execution engine
                engine = get_execution_engine(
                    session=session,
                    organization_id=workflow.organization_id,
                )

                # Execute the workflow
                execution = await engine.execute(
                    workflow_id=workflow.id,
                    trigger_type=WorkflowTriggerType.SCHEDULED.value,
                    trigger_data={
                        "scheduled_at": datetime.utcnow().isoformat(),
                        "cron": (workflow.trigger_config or {}).get("cron"),
                    },
                    input_data={},
                    triggered_by_id=workflow.created_by_id,
                )

                logger.info(
                    "Scheduled workflow execution started",
                    workflow_id=workflow_id,
                    execution_id=str(execution.id),
                )

                return {
                    "workflow_id": workflow_id,
                    "execution_id": str(execution.id),
                    "status": execution.status,
                }

        except Exception as e:
            logger.error(
                "Scheduled workflow execution failed",
                workflow_id=workflow_id,
                error=str(e),
            )
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _register_workflow_schedule(
        self,
        workflow_id: str,
        cron_expression: str,
        timezone: str,
        workflow_name: str,
    ) -> None:
        """
        Register a workflow schedule with Celery Beat.

        Args:
            workflow_id: UUID of the workflow.
            cron_expression: Cron expression.
            timezone: Timezone for the schedule.
            workflow_name: Name of the workflow (for logging).
        """
        # Store in local cache
        self._scheduled_workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "cron": cron_expression,
            "timezone": timezone,
            "registered_at": datetime.utcnow().isoformat(),
        }

        if not self._celery_app or not HAS_CELERY:
            logger.debug(
                "Celery not available, schedule stored locally only",
                workflow_id=workflow_id,
            )
            return

        try:
            cron_schedule = _parse_cron_expression(cron_expression)
            if cron_schedule is None:
                return

            task_name = f"workflow_scheduler_{workflow_id}"

            # Add/update the periodic task in Celery Beat's schedule
            self._celery_app.conf.beat_schedule[task_name] = {
                "task": "backend.tasks.workflow_tasks.execute_scheduled_workflow",
                "schedule": cron_schedule,
                "args": (workflow_id,),
                "options": {"queue": "workflows"},
            }

            logger.info(
                "Registered Celery Beat task for workflow",
                task_name=task_name,
                workflow_id=workflow_id,
                cron=cron_expression,
            )

        except Exception as e:
            logger.error(
                "Failed to register Celery Beat task",
                workflow_id=workflow_id,
                error=str(e),
            )

    def _unregister_workflow_schedule(self, workflow_id: str) -> None:
        """
        Remove a workflow schedule from Celery Beat.

        Args:
            workflow_id: UUID of the workflow.
        """
        if not self._celery_app or not HAS_CELERY:
            return

        task_name = f"workflow_scheduler_{workflow_id}"

        try:
            beat_schedule = getattr(self._celery_app.conf, "beat_schedule", {})
            if task_name in beat_schedule:
                del beat_schedule[task_name]
                logger.info("Unregistered Celery Beat task", task_name=task_name)
        except Exception as e:
            logger.error(
                "Failed to unregister Celery Beat task",
                workflow_id=workflow_id,
                error=str(e),
            )


# =============================================================================
# Celery Tasks
# =============================================================================

def execute_scheduled_workflow_task(workflow_id: str) -> Dict[str, Any]:
    """
    Celery task for executing scheduled workflows.

    This function is called by Celery Beat and runs the async
    execute_scheduled_workflow method in a sync context.

    Args:
        workflow_id: UUID of the workflow to execute.

    Returns:
        Execution result dict.
    """
    scheduler = get_workflow_scheduler()
    return asyncio.run(scheduler.execute_scheduled_workflow(workflow_id))


# Register the task with Celery if available
if HAS_CELERY:
    try:
        from backend.core.celery_app import celery_app

        @celery_app.task(
            name="backend.tasks.workflow_tasks.execute_scheduled_workflow",
            bind=True,
            max_retries=3,
            default_retry_delay=60,
        )
        def celery_execute_scheduled_workflow(self, workflow_id: str) -> Dict[str, Any]:
            """Celery task wrapper for scheduled workflow execution."""
            try:
                return execute_scheduled_workflow_task(workflow_id)
            except Exception as e:
                logger.error(
                    "Celery task failed",
                    workflow_id=workflow_id,
                    error=str(e),
                )
                raise self.retry(exc=e)

    except ImportError:
        pass


# =============================================================================
# Module-level Singleton
# =============================================================================

_scheduler: Optional[WorkflowSchedulerService] = None


def get_workflow_scheduler() -> WorkflowSchedulerService:
    """
    Get or create the singleton WorkflowSchedulerService instance.

    Returns:
        The WorkflowSchedulerService singleton.
    """
    global _scheduler

    if _scheduler is None:
        _scheduler = WorkflowSchedulerService()

    return _scheduler


# =============================================================================
# API Helper Functions
# =============================================================================

async def get_next_run_time(cron_expression: str, timezone: str = "UTC") -> Optional[datetime]:
    """
    Calculate the next run time for a cron expression.

    Args:
        cron_expression: Cron expression.
        timezone: Timezone for calculation.

    Returns:
        Next run datetime or None if invalid.
    """
    try:
        from croniter import croniter
        from datetime import datetime
        import pytz

        tz = pytz.timezone(timezone)
        now = datetime.now(tz)

        cron = croniter(cron_expression, now)
        next_run = cron.get_next(datetime)

        return next_run

    except ImportError:
        # croniter not installed, return None
        logger.debug("croniter not installed, cannot calculate next run time")
        return None
    except Exception as e:
        logger.error("Failed to calculate next run time", error=str(e))
        return None


async def list_upcoming_executions(
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    List upcoming scheduled workflow executions.

    Args:
        limit: Maximum number of upcoming executions to return.

    Returns:
        List of upcoming execution details with times.
    """
    scheduler = get_workflow_scheduler()
    schedules = scheduler.list_schedules()

    upcoming = []
    for schedule in schedules:
        next_run = await get_next_run_time(
            schedule["cron"],
            schedule.get("timezone", "UTC"),
        )
        if next_run:
            upcoming.append({
                "workflow_id": schedule["workflow_id"],
                "workflow_name": schedule["workflow_name"],
                "cron": schedule["cron"],
                "timezone": schedule.get("timezone", "UTC"),
                "next_run": next_run.isoformat(),
            })

    # Sort by next run time
    upcoming.sort(key=lambda x: x["next_run"])

    return upcoming[:limit]
