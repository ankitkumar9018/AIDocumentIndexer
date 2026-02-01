"""
AIDocumentIndexer - Crawl Scheduler Service
=============================================

Scheduled/recurring web crawls using Celery Beat.

Provides:
- Cron-based scheduling for recurring crawls
- Incremental re-indexing via content hash change detection
- CRUD operations for managing scheduled crawls
- Integration with Celery Beat for periodic task registration

Usage:
    from backend.services.crawl_scheduler import get_scheduler_service

    scheduler = get_scheduler_service()
    schedule = scheduler.create_schedule(
        url="https://example.com",
        schedule="0 */6 * * *",
        config={"max_pages": 10, "max_depth": 2},
        user_id="user-123",
    )
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

# Try to import Celery components
try:
    from celery import Celery
    from celery.schedules import crontab
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False
    Celery = None  # type: ignore[assignment,misc]
    crontab = None  # type: ignore[assignment,misc]
    logger.warning("celery not installed. Scheduled crawls will not be dispatched.")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScheduledCrawl:
    """A scheduled/recurring web crawl configuration."""

    id: str
    url: str
    schedule: str  # Cron expression, e.g. "0 */6 * * *"
    crawl_config: Dict[str, Any]  # max_pages, max_depth, storage_mode, etc.
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_content_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scheduled crawl to a dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "schedule": self.schedule,
            "crawl_config": self.crawl_config,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_content_hash": self.last_content_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
        }


# =============================================================================
# Cron Parsing Helpers
# =============================================================================

def _parse_cron_expression(expression: str) -> Optional[Any]:
    """
    Parse a standard cron expression into a Celery crontab object.

    Supports the standard five-field cron format:
        minute hour day_of_month month_of_year day_of_week

    Examples:
        "0 */6 * * *"   -> every 6 hours at minute 0
        "30 2 * * 1"    -> 2:30 AM every Monday
        "*/15 * * * *"  -> every 15 minutes

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


# =============================================================================
# Crawl Scheduler Service
# =============================================================================

class CrawlSchedulerService:
    """
    Service for managing scheduled/recurring web crawls.

    Uses an in-memory store (dict) for schedule persistence and
    integrates with Celery Beat for periodic task dispatch.
    """

    def __init__(self) -> None:
        """Initialize the scheduler with an in-memory store and optional Celery app."""
        self._schedules: Dict[str, ScheduledCrawl] = {}
        self._celery_app: Optional[Any] = None

        if HAS_CELERY:
            try:
                from backend.core.celery_app import celery_app
                self._celery_app = celery_app
                logger.info("CrawlSchedulerService initialized with Celery Beat support")
            except ImportError:
                logger.warning(
                    "Celery app not found at backend.core.celery_app. "
                    "Scheduled crawls will be stored but not dispatched."
                )
        else:
            logger.warning(
                "Celery not available. Scheduled crawls will be stored but not dispatched."
            )

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def create_schedule(
        self,
        url: str,
        schedule: str,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> ScheduledCrawl:
        """
        Create a new scheduled crawl.

        Args:
            url: The target URL to crawl on the schedule.
            schedule: Cron expression (e.g. "0 */6 * * *").
            config: Crawl configuration dict (max_pages, max_depth, storage_mode, etc.).
            user_id: ID of the user creating the schedule.

        Returns:
            The newly created ScheduledCrawl instance.

        Raises:
            ValueError: If the cron expression is invalid.
        """
        # Validate the cron expression early
        _parse_cron_expression(schedule)

        crawl_config = config or {
            "max_pages": 50,
            "max_depth": 3,
            "storage_mode": "permanent",
        }

        scheduled_crawl = ScheduledCrawl(
            id=str(uuid4()),
            url=url,
            schedule=schedule,
            crawl_config=crawl_config,
            enabled=True,
            created_by=user_id,
        )

        self._schedules[scheduled_crawl.id] = scheduled_crawl

        # Register with Celery Beat
        self._register_celery_task(scheduled_crawl)

        logger.info(
            "Created scheduled crawl",
            schedule_id=scheduled_crawl.id,
            url=url,
            schedule=schedule,
            user_id=user_id,
        )

        return scheduled_crawl

    def update_schedule(
        self,
        schedule_id: str,
        updates: Dict[str, Any],
    ) -> ScheduledCrawl:
        """
        Update an existing scheduled crawl.

        Supported update fields: url, schedule, crawl_config, enabled.

        Args:
            schedule_id: ID of the schedule to update.
            updates: Dictionary of fields to update.

        Returns:
            The updated ScheduledCrawl instance.

        Raises:
            ValueError: If the schedule_id is not found or the cron expression is invalid.
        """
        if schedule_id not in self._schedules:
            raise ValueError(f"Scheduled crawl not found: {schedule_id}")

        scheduled_crawl = self._schedules[schedule_id]

        # Validate new cron expression if provided
        if "schedule" in updates:
            _parse_cron_expression(updates["schedule"])

        # Apply updates
        allowed_fields = {"url", "schedule", "crawl_config", "enabled"}
        for key, value in updates.items():
            if key in allowed_fields and hasattr(scheduled_crawl, key):
                setattr(scheduled_crawl, key, value)

        scheduled_crawl.updated_at = datetime.utcnow()

        # Re-register with Celery Beat if schedule or enabled changed
        if "schedule" in updates or "enabled" in updates:
            if scheduled_crawl.enabled:
                self._register_celery_task(scheduled_crawl)
            else:
                self._unregister_celery_task(schedule_id)

        logger.info(
            "Updated scheduled crawl",
            schedule_id=schedule_id,
            updates=list(updates.keys()),
        )

        return scheduled_crawl

    def delete_schedule(self, schedule_id: str) -> bool:
        """
        Delete a scheduled crawl.

        Args:
            schedule_id: ID of the schedule to delete.

        Returns:
            True if the schedule was deleted, False if not found.
        """
        if schedule_id not in self._schedules:
            return False

        # Unregister from Celery Beat
        self._unregister_celery_task(schedule_id)

        del self._schedules[schedule_id]

        logger.info("Deleted scheduled crawl", schedule_id=schedule_id)
        return True

    def get_schedule(self, schedule_id: str) -> Optional[ScheduledCrawl]:
        """
        Get a specific scheduled crawl by ID.

        Args:
            schedule_id: ID of the schedule.

        Returns:
            The ScheduledCrawl if found, otherwise None.
        """
        return self._schedules.get(schedule_id)

    def list_schedules(self, user_id: Optional[str] = None) -> List[ScheduledCrawl]:
        """
        List all scheduled crawls, optionally filtered by user.

        Args:
            user_id: If provided, only return schedules created by this user.

        Returns:
            List of ScheduledCrawl instances.
        """
        schedules = list(self._schedules.values())
        if user_id is not None:
            schedules = [s for s in schedules if s.created_by == user_id]
        return schedules

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute_scheduled_crawl(self, schedule_id: str) -> Dict[str, Any]:
        """
        Execute a scheduled crawl immediately.

        Performs the crawl using WebScraperService, computes a content hash
        of the results, and triggers re-embedding only if the content has
        changed since the last run.

        Args:
            schedule_id: ID of the schedule to execute.

        Returns:
            Dict with execution results including pages_crawled, content_changed,
            and content_hash.

        Raises:
            ValueError: If the schedule_id is not found.
        """
        scheduled_crawl = self._schedules.get(schedule_id)
        if not scheduled_crawl:
            raise ValueError(f"Scheduled crawl not found: {schedule_id}")

        logger.info(
            "Executing scheduled crawl",
            schedule_id=schedule_id,
            url=scheduled_crawl.url,
        )

        try:
            from services.web_crawler import WebScraperService
        except ImportError:
            # Fallback: use the EnhancedWebCrawler from web_crawler module
            from backend.services.web_crawler import get_web_crawler

        from backend.services.web_crawler import get_web_crawler

        crawler = get_web_crawler()

        # Extract crawl parameters from config
        config = scheduled_crawl.crawl_config
        max_pages = config.get("max_pages", 50)

        # Perform the crawl
        results = await crawler.crawl_site(
            start_url=scheduled_crawl.url,
            max_pages=max_pages,
            same_domain_only=config.get("same_domain_only", True),
        )

        # Compute content hash for change detection
        content_hash = self._compute_content_hash(results)
        content_changed = content_hash != scheduled_crawl.last_content_hash

        # Update schedule metadata
        now = datetime.utcnow()
        scheduled_crawl.last_run = now
        scheduled_crawl.last_content_hash = content_hash
        scheduled_crawl.updated_at = now

        pages_crawled = len(results)
        pages_successful = sum(1 for r in results if r.success)
        re_indexed = False

        # If content changed, trigger re-embedding
        if content_changed and pages_successful > 0:
            storage_mode = config.get("storage_mode", "permanent")
            if storage_mode == "permanent":
                try:
                    from backend.services.scraper import ScrapedPage, get_scraper_service

                    scraper_service = get_scraper_service()
                    scraped_pages = []
                    for r in results:
                        if r.success:
                            scraped_pages.append(ScrapedPage(
                                url=r.url,
                                title=r.title,
                                content=r.markdown or r.content,
                                word_count=r.word_count,
                            ))

                    if scraped_pages:
                        await scraper_service.index_pages_content(
                            pages=scraped_pages,
                            source_id=f"scheduled_{schedule_id}",
                        )
                        re_indexed = True
                        logger.info(
                            "Re-indexed content for scheduled crawl",
                            schedule_id=schedule_id,
                            pages_indexed=len(scraped_pages),
                        )
                except Exception as e:
                    logger.error(
                        "Failed to re-index content for scheduled crawl",
                        schedule_id=schedule_id,
                        error=str(e),
                    )

        result = {
            "schedule_id": schedule_id,
            "url": scheduled_crawl.url,
            "pages_crawled": pages_crawled,
            "pages_successful": pages_successful,
            "content_hash": content_hash,
            "content_changed": content_changed,
            "re_indexed": re_indexed,
            "executed_at": now.isoformat(),
        }

        logger.info(
            "Scheduled crawl execution completed",
            schedule_id=schedule_id,
            pages_crawled=pages_crawled,
            content_changed=content_changed,
            re_indexed=re_indexed,
        )

        return result

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _compute_content_hash(self, results: list) -> str:
        """
        Compute a SHA-256 hash of the crawled content for change detection.

        Concatenates the markdown/content of all successful crawl results
        and produces a deterministic hash. This allows comparing successive
        crawls to detect whether re-indexing is necessary.

        Args:
            results: List of CrawlResult objects from a crawl operation.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        hasher = hashlib.sha256()
        for r in results:
            if hasattr(r, "success") and r.success:
                content = getattr(r, "markdown", "") or getattr(r, "content", "")
                hasher.update(content.encode("utf-8"))
        return hasher.hexdigest()

    def _register_celery_task(self, scheduled_crawl: ScheduledCrawl) -> None:
        """
        Register or update a periodic task with Celery Beat.

        Creates a Celery Beat schedule entry that will trigger
        execute_scheduled_crawl at the configured cron interval.

        Args:
            scheduled_crawl: The ScheduledCrawl to register.
        """
        if not self._celery_app or not HAS_CELERY:
            logger.debug(
                "Celery not available, skipping task registration",
                schedule_id=scheduled_crawl.id,
            )
            return

        if not scheduled_crawl.enabled:
            logger.debug(
                "Schedule is disabled, skipping registration",
                schedule_id=scheduled_crawl.id,
            )
            return

        try:
            cron_schedule = _parse_cron_expression(scheduled_crawl.schedule)
            if cron_schedule is None:
                return

            task_name = f"crawl_scheduler_{scheduled_crawl.id}"

            # Add/update the periodic task in Celery Beat's schedule
            self._celery_app.conf.beat_schedule[task_name] = {
                "task": "backend.services.crawl_scheduler.execute_scheduled_crawl_task",
                "schedule": cron_schedule,
                "args": (scheduled_crawl.id,),
                "options": {"queue": "crawl"},
            }

            logger.info(
                "Registered Celery Beat task",
                task_name=task_name,
                schedule=scheduled_crawl.schedule,
                url=scheduled_crawl.url,
            )
        except Exception as e:
            logger.error(
                "Failed to register Celery Beat task",
                schedule_id=scheduled_crawl.id,
                error=str(e),
            )

    def _unregister_celery_task(self, schedule_id: str) -> None:
        """
        Remove a periodic task from Celery Beat.

        Args:
            schedule_id: ID of the schedule whose task should be removed.
        """
        if not self._celery_app or not HAS_CELERY:
            return

        task_name = f"crawl_scheduler_{schedule_id}"

        try:
            beat_schedule = getattr(self._celery_app.conf, "beat_schedule", {})
            if task_name in beat_schedule:
                del beat_schedule[task_name]
                logger.info("Unregistered Celery Beat task", task_name=task_name)
        except Exception as e:
            logger.error(
                "Failed to unregister Celery Beat task",
                schedule_id=schedule_id,
                error=str(e),
            )


# =============================================================================
# Module-level Singleton
# =============================================================================

_scheduler_service: Optional[CrawlSchedulerService] = None


def get_scheduler_service() -> CrawlSchedulerService:
    """
    Get or create the singleton CrawlSchedulerService instance.

    Returns:
        The CrawlSchedulerService singleton.
    """
    global _scheduler_service

    if _scheduler_service is None:
        _scheduler_service = CrawlSchedulerService()

    return _scheduler_service
