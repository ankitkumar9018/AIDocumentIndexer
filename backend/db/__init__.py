"""
AIDocumentIndexer - Database Module
====================================

Database configuration, models, and utilities with multi-database support.
"""

from backend.db.models import (
    Base,
    AccessTier,
    User,
    Document,
    Chunk,
    ScrapedContent,
    ChatSession,
    ChatMessage,
    AuditLog,
    ProcessingQueue,
    ProcessingStatus,
    ProcessingMode,
    StorageMode,
    MessageRole,
)

__all__ = [
    "Base",
    "AccessTier",
    "User",
    "Document",
    "Chunk",
    "ScrapedContent",
    "ChatSession",
    "ChatMessage",
    "AuditLog",
    "ProcessingQueue",
    "ProcessingStatus",
    "ProcessingMode",
    "StorageMode",
    "MessageRole",
]
