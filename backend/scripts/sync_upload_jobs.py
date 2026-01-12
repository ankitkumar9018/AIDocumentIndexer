#!/usr/bin/env python3
"""
Sync UploadJob Statuses with Document Statuses
===============================================

This script synchronizes the UploadJob table with the actual Document
processing status. It fixes stale UploadJob records that show incorrect
statuses (e.g., "processing" when the document is actually "completed").

Usage:
    python -m backend.scripts.sync_upload_jobs

    # Dry run (no changes)
    python -m backend.scripts.sync_upload_jobs --dry-run

    # Also cleanup orphaned upload jobs (no matching document)
    python -m backend.scripts.sync_upload_jobs --cleanup-orphans
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, Set

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import structlog
from sqlalchemy import select, update, delete

from backend.db.database import async_session_context
from backend.db.models import Document, UploadJob, ProcessingStatus
from backend.api.routes.upload import UploadStatus

logger = structlog.get_logger(__name__)


# Map Document ProcessingStatus to UploadJob UploadStatus
STATUS_MAP = {
    ProcessingStatus.PENDING: UploadStatus.QUEUED,
    ProcessingStatus.PROCESSING: UploadStatus.EXTRACTING,
    ProcessingStatus.COMPLETED: UploadStatus.COMPLETED,
    ProcessingStatus.FAILED: UploadStatus.FAILED,
}


async def sync_upload_jobs(dry_run: bool = False, cleanup_orphans: bool = False) -> Dict[str, int]:
    """
    Sync UploadJob statuses with actual Document statuses.

    Returns:
        Dict with counts: {synced, already_correct, orphaned, cleaned}
    """
    stats = {
        "total_upload_jobs": 0,
        "total_documents": 0,
        "synced": 0,
        "already_correct": 0,
        "orphaned": 0,
        "cleaned": 0,
    }

    async with async_session_context() as db:
        # Get all documents with their statuses
        doc_result = await db.execute(
            select(Document.id, Document.filename, Document.processing_status)
        )
        documents = {str(row[0]): (row[1], row[2]) for row in doc_result.all()}
        stats["total_documents"] = len(documents)

        # Get all upload jobs
        job_result = await db.execute(select(UploadJob))
        jobs = job_result.scalars().all()
        stats["total_upload_jobs"] = len(jobs)

        # Track document IDs that have upload jobs
        docs_with_jobs: Set[str] = set()

        for job in jobs:
            job_id = str(job.id)

            # Try to find matching document by ID or filename
            doc_info = documents.get(job_id)

            # If not found by ID, try by filename
            if not doc_info:
                for doc_id, (filename, status) in documents.items():
                    if filename == job.filename:
                        doc_info = (filename, status)
                        docs_with_jobs.add(doc_id)
                        break
            else:
                docs_with_jobs.add(job_id)

            if doc_info:
                filename, doc_status = doc_info
                expected_upload_status = STATUS_MAP.get(doc_status, UploadStatus.COMPLETED)

                if job.status != expected_upload_status:
                    print(f"  Sync: {job.filename[:40]} - {job.status.value} -> {expected_upload_status.value}")

                    if not dry_run:
                        job.status = expected_upload_status
                        job.progress = 100 if expected_upload_status == UploadStatus.COMPLETED else 0
                        job.current_step = "Completed" if expected_upload_status == UploadStatus.COMPLETED else "Unknown"

                    stats["synced"] += 1
                else:
                    stats["already_correct"] += 1
            else:
                # Orphaned upload job - no matching document
                stats["orphaned"] += 1

                if cleanup_orphans:
                    print(f"  Orphan: {job.filename[:40]} - removing")
                    if not dry_run:
                        await db.delete(job)
                    stats["cleaned"] += 1

        if not dry_run:
            await db.commit()

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Sync UploadJob statuses with Document statuses"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make changes, just report what would be done",
    )
    parser.add_argument(
        "--cleanup-orphans",
        action="store_true",
        help="Remove upload jobs that don't have matching documents",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("UploadJob Status Sync Script")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    stats = await sync_upload_jobs(
        dry_run=args.dry_run,
        cleanup_orphans=args.cleanup_orphans,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Total Upload Jobs:    {stats['total_upload_jobs']}")
    print(f"  Total Documents:      {stats['total_documents']}")
    print(f"  Already Correct:      {stats['already_correct']}")
    print(f"  Synced:               {stats['synced']}")
    print(f"  Orphaned:             {stats['orphaned']}")
    if args.cleanup_orphans:
        print(f"  Cleaned Up:           {stats['cleaned']}")

    if args.dry_run:
        print("\n*** This was a dry run. Run without --dry-run to apply changes. ***")


if __name__ == "__main__":
    asyncio.run(main())
