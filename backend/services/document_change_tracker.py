"""
AIDocumentIndexer - Document Change Tracker
============================================

Tracks document modifications and triggers re-indexing of:
- Chunks
- Embeddings
- Knowledge Graph entities/relations

Features:
- Content hash-based change detection
- Smart incremental updates (only changed chunks)
- Automatic KG entity update
- File modification time tracking
- Background processing with queue

Supported editing:
- Direct file modifications (via file watcher)
- In-app document editing
- External editor changes
- Collaborative editing hooks
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Document, Chunk, Entity, EntityMention
from backend.db.database import async_session_context
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class ChangeType(str, Enum):
    """Types of document changes."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class DocumentChange:
    """Represents a change to a document."""
    document_id: UUID
    change_type: ChangeType
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    changed_chunks: List[int] = field(default_factory=list)  # Chunk indices
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkDiff:
    """Represents differences between old and new chunks."""
    added: List[int]        # New chunk indices
    removed: List[int]      # Removed chunk indices
    modified: List[int]     # Modified chunk indices
    unchanged: List[int]    # Unchanged chunk indices


@dataclass
class ReindexResult:
    """Result of a re-indexing operation."""
    document_id: UUID
    chunks_added: int
    chunks_removed: int
    chunks_modified: int
    entities_updated: int
    relations_updated: int
    processing_time_ms: float
    success: bool
    error: Optional[str] = None


class DocumentChangeTracker:
    """
    Tracks document changes and coordinates re-indexing.

    Uses content hashing to detect changes and smart diffing
    to minimize re-processing work.
    """

    def __init__(self):
        self._change_queue: asyncio.Queue[DocumentChange] = asyncio.Queue()
        self._processing = False
        self._callbacks: List[Callable[[ReindexResult], None]] = []
        self._content_hashes: Dict[str, str] = {}  # doc_id -> content_hash

    def register_callback(self, callback: Callable[[ReindexResult], None]) -> None:
        """Register a callback for reindex completion."""
        self._callbacks.append(callback)

    async def start_processing(self) -> None:
        """Start background processing of changes."""
        if self._processing:
            return
        self._processing = True
        asyncio.create_task(self._process_queue())
        logger.info("Document change tracker started")

    async def stop_processing(self) -> None:
        """Stop background processing."""
        self._processing = False
        logger.info("Document change tracker stopped")

    async def track_change(
        self,
        document_id: UUID,
        change_type: ChangeType,
        new_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track a document change.

        Args:
            document_id: ID of the changed document
            change_type: Type of change
            new_content: New document content (for hash computation)
            metadata: Additional metadata about the change
        """
        old_hash = self._content_hashes.get(str(document_id))
        new_hash = None

        if new_content:
            new_hash = self._compute_hash(new_content)
            self._content_hashes[str(document_id)] = new_hash

        # Skip if content hasn't changed
        if old_hash and new_hash and old_hash == new_hash:
            logger.debug("Document content unchanged, skipping", document_id=str(document_id))
            return

        change = DocumentChange(
            document_id=document_id,
            change_type=change_type,
            old_hash=old_hash,
            new_hash=new_hash,
            metadata=metadata or {},
        )

        await self._change_queue.put(change)
        logger.info(
            "Document change tracked",
            document_id=str(document_id),
            change_type=change_type.value,
        )

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def _process_queue(self) -> None:
        """Process changes from the queue."""
        while self._processing:
            try:
                # Get change with timeout to allow checking _processing flag
                try:
                    change = await asyncio.wait_for(
                        self._change_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the change
                result = await self._process_change(change)

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error("Callback error", error=str(e))

            except Exception as e:
                logger.error("Queue processing error", error=str(e))
                await asyncio.sleep(1)

    async def _process_change(self, change: DocumentChange) -> ReindexResult:
        """Process a single document change."""
        import time
        start_time = time.time()

        try:
            if change.change_type == ChangeType.DELETED:
                return await self._handle_deletion(change, start_time)
            elif change.change_type in [ChangeType.CREATED, ChangeType.MODIFIED]:
                return await self._handle_modification(change, start_time)
            else:
                # RENAMED - just update metadata, no reindex needed
                return ReindexResult(
                    document_id=change.document_id,
                    chunks_added=0,
                    chunks_removed=0,
                    chunks_modified=0,
                    entities_updated=0,
                    relations_updated=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    success=True,
                )
        except Exception as e:
            logger.error(
                "Change processing failed",
                document_id=str(change.document_id),
                error=str(e),
            )
            return ReindexResult(
                document_id=change.document_id,
                chunks_added=0,
                chunks_removed=0,
                chunks_modified=0,
                entities_updated=0,
                relations_updated=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )

    async def _handle_deletion(
        self,
        change: DocumentChange,
        start_time: float,
    ) -> ReindexResult:
        """Handle document deletion - remove chunks and entities."""
        import time

        async with async_session_context() as db:
            # Get chunk count before deletion
            result = await db.execute(
                select(Chunk).where(Chunk.document_id == change.document_id)
            )
            chunks = result.scalars().all()
            chunk_count = len(chunks)

            # Delete entity mentions for this document's chunks
            chunk_ids = [c.id for c in chunks]
            if chunk_ids:
                await db.execute(
                    EntityMention.__table__.delete().where(
                        EntityMention.chunk_id.in_(chunk_ids)
                    )
                )

            # Delete chunks (cascades to embeddings via DB constraint)
            await db.execute(
                Chunk.__table__.delete().where(Chunk.document_id == change.document_id)
            )

            await db.commit()

        # Remove from hash cache
        self._content_hashes.pop(str(change.document_id), None)

        return ReindexResult(
            document_id=change.document_id,
            chunks_added=0,
            chunks_removed=chunk_count,
            chunks_modified=0,
            entities_updated=0,
            relations_updated=0,
            processing_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    async def _handle_modification(
        self,
        change: DocumentChange,
        start_time: float,
    ) -> ReindexResult:
        """Handle document modification - smart incremental update."""
        import time
        from backend.services.chunking import chunk_document
        from backend.services.embeddings import get_embedding_service
        from backend.services.knowledge_graph import get_knowledge_graph_service

        async with async_session_context() as db:
            # Get document
            doc_result = await db.execute(
                select(Document).where(Document.id == change.document_id)
            )
            document = doc_result.scalar_one_or_none()
            if not document:
                raise ValueError(f"Document not found: {change.document_id}")

            # Get old chunks
            old_chunks_result = await db.execute(
                select(Chunk)
                .where(Chunk.document_id == change.document_id)
                .order_by(Chunk.chunk_index)
            )
            old_chunks = old_chunks_result.scalars().all()
            old_chunk_hashes = {c.chunk_index: c.content_hash for c in old_chunks}

            # Get new content and chunk it
            # Note: In production, content would come from storage
            new_content = change.metadata.get("content", "")
            new_chunks_data = await chunk_document(new_content, document.file_type)

            # Compute diff
            diff = self._compute_chunk_diff(old_chunk_hashes, new_chunks_data)

            # Remove deleted chunks
            if diff.removed:
                chunk_ids_to_remove = [
                    c.id for c in old_chunks if c.chunk_index in diff.removed
                ]
                if chunk_ids_to_remove:
                    # Delete entity mentions
                    await db.execute(
                        EntityMention.__table__.delete().where(
                            EntityMention.chunk_id.in_(chunk_ids_to_remove)
                        )
                    )
                    # Delete chunks
                    await db.execute(
                        Chunk.__table__.delete().where(
                            Chunk.id.in_(chunk_ids_to_remove)
                        )
                    )

            # Add/update chunks
            embedding_service = get_embedding_service()
            chunks_to_embed = []

            for idx in diff.added + diff.modified:
                chunk_content = new_chunks_data[idx]["content"]
                chunk_hash = self._compute_hash(chunk_content)

                if idx in diff.added:
                    # Create new chunk
                    new_chunk = Chunk(
                        document_id=change.document_id,
                        chunk_index=idx,
                        content=chunk_content,
                        content_hash=chunk_hash,
                    )
                    db.add(new_chunk)
                    chunks_to_embed.append((new_chunk, chunk_content))
                else:
                    # Update existing chunk
                    old_chunk = next(c for c in old_chunks if c.chunk_index == idx)
                    old_chunk.content = chunk_content
                    old_chunk.content_hash = chunk_hash
                    chunks_to_embed.append((old_chunk, chunk_content))

            await db.flush()

            # Generate embeddings for new/modified chunks
            if chunks_to_embed:
                texts = [content for _, content in chunks_to_embed]
                embeddings = embedding_service.embed_texts(texts)

                for (chunk, _), embedding in zip(chunks_to_embed, embeddings):
                    chunk.embedding = embedding

            # Update document hash
            await db.execute(
                update(Document)
                .where(Document.id == change.document_id)
                .values(
                    content_hash=change.new_hash,
                    updated_at=datetime.utcnow(),
                )
            )

            await db.commit()

            # Update knowledge graph for modified chunks
            kg_service = await get_knowledge_graph_service(db)
            entities_updated = 0

            try:
                # Re-extract entities from modified chunks
                modified_chunk_ids = [
                    c.id for c in old_chunks if c.chunk_index in diff.modified
                ]
                for chunk_id in modified_chunk_ids:
                    await kg_service.reextract_entities_for_chunk(chunk_id)
                    entities_updated += 1
            except Exception as e:
                logger.warning("KG update failed", error=str(e))

        return ReindexResult(
            document_id=change.document_id,
            chunks_added=len(diff.added),
            chunks_removed=len(diff.removed),
            chunks_modified=len(diff.modified),
            entities_updated=entities_updated,
            relations_updated=0,
            processing_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    def _compute_chunk_diff(
        self,
        old_hashes: Dict[int, str],
        new_chunks: List[Dict[str, Any]],
    ) -> ChunkDiff:
        """
        Compute the difference between old and new chunks.

        Uses content hashing for efficient comparison.
        """
        new_hashes = {
            i: self._compute_hash(c["content"])
            for i, c in enumerate(new_chunks)
        }

        old_indices = set(old_hashes.keys())
        new_indices = set(new_hashes.keys())

        # Find added, removed, and potentially modified
        added = list(new_indices - old_indices)
        removed = list(old_indices - new_indices)
        common = old_indices & new_indices

        # Check which common chunks are actually modified
        modified = []
        unchanged = []
        for idx in common:
            if old_hashes[idx] != new_hashes[idx]:
                modified.append(idx)
            else:
                unchanged.append(idx)

        return ChunkDiff(
            added=sorted(added),
            removed=sorted(removed),
            modified=sorted(modified),
            unchanged=sorted(unchanged),
        )

    async def get_pending_changes(self) -> int:
        """Get number of pending changes in queue."""
        return self._change_queue.qsize()


# =============================================================================
# Document Edit API Mixin
# =============================================================================

class DocumentEditMixin:
    """
    Mixin for document editing with change tracking.

    Add this to document service classes to enable edit tracking.
    """

    _change_tracker: Optional[DocumentChangeTracker] = None

    @classmethod
    def get_change_tracker(cls) -> DocumentChangeTracker:
        """Get or create the change tracker."""
        if cls._change_tracker is None:
            cls._change_tracker = DocumentChangeTracker()
        return cls._change_tracker

    async def update_document_content(
        self,
        document_id: UUID,
        new_content: str,
        user_id: str,
    ) -> None:
        """
        Update document content and trigger re-indexing.

        Args:
            document_id: Document ID
            new_content: New document content
            user_id: User making the change
        """
        tracker = self.get_change_tracker()

        await tracker.track_change(
            document_id=document_id,
            change_type=ChangeType.MODIFIED,
            new_content=new_content,
            metadata={
                "content": new_content,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


# =============================================================================
# Singleton Access
# =============================================================================

_change_tracker: Optional[DocumentChangeTracker] = None


def get_change_tracker() -> DocumentChangeTracker:
    """Get or create the document change tracker."""
    global _change_tracker
    if _change_tracker is None:
        _change_tracker = DocumentChangeTracker()
    return _change_tracker


async def start_change_tracker() -> None:
    """Start the change tracker background processing."""
    tracker = get_change_tracker()
    await tracker.start_processing()


async def stop_change_tracker() -> None:
    """Stop the change tracker."""
    if _change_tracker:
        await _change_tracker.stop_processing()
