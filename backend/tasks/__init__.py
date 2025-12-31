"""
AIDocumentIndexer - Celery Tasks Package
=========================================

Contains async task definitions for background processing.
"""

from backend.tasks.document_tasks import (
    process_document_task,
    process_batch_task,
    reprocess_document_task,
)

__all__ = [
    "process_document_task",
    "process_batch_task",
    "reprocess_document_task",
]
