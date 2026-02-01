"""
AIDocumentIndexer - Real-Time Indexing Service
==============================================

Implements real-time document updates and incremental indexing.

Features:
- Incremental indexing (update only changed chunks)
- Document freshness tracking
- Stale content detection
- Change detection using content hashes
- Webhook notifications for updates
- Batch processing for efficiency
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
import uuid

import structlog
from sqlalchemy import select, func, and_, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Document, Chunk, ProcessingStatus

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Environment-based configuration
FRESHNESS_THRESHOLD_DAYS = int(os.getenv("FRESHNESS_THRESHOLD_DAYS", "30"))
STALE_THRESHOLD_DAYS = int(os.getenv("STALE_THRESHOLD_DAYS", "90"))
BATCH_SIZE = int(os.getenv("INCREMENTAL_BATCH_SIZE", "50"))


# =============================================================================
# Data Classes
# =============================================================================

class FreshnessLevel(str, Enum):
    """Content freshness levels."""
    FRESH = "fresh"          # Recently updated
    CURRENT = "current"      # Within acceptable age
    AGING = "aging"          # Getting old, may need review
    STALE = "stale"          # Potentially outdated
    UNKNOWN = "unknown"      # No timestamp available


@dataclass
class ContentChange:
    """Detected change in document content."""
    document_id: uuid.UUID
    change_type: str  # "added", "modified", "deleted"
    chunk_ids: List[uuid.UUID] = field(default_factory=list)
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class IndexingTask:
    """A task for incremental indexing."""
    document_id: uuid.UUID
    task_type: str  # "reindex", "update_chunks", "delete"
    priority: int = 0
    chunks_to_update: List[uuid.UUID] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FreshnessInfo:
    """Information about content freshness."""
    document_id: uuid.UUID
    filename: str
    last_modified: datetime
    freshness_level: FreshnessLevel
    days_since_update: int
    recommendation: Optional[str] = None


@dataclass
class IndexingStats:
    """Statistics for indexing operations."""
    documents_checked: int = 0
    documents_updated: int = 0
    chunks_added: int = 0
    chunks_modified: int = 0
    chunks_deleted: int = 0
    processing_time_ms: float = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Real-Time Indexing Service
# =============================================================================

class RealTimeIndexerService:
    """
    Service for real-time and incremental document indexing.

    Provides:
    - Change detection between document versions
    - Incremental updates (only changed chunks)
    - Freshness tracking and stale content detection
    - Batch processing for efficiency
    """

    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service=None,
        pipeline_service=None,
    ):
        self.db = db_session
        self.embeddings = embedding_service
        self.pipeline = pipeline_service

        self._pending_tasks: List[IndexingTask] = []
        self._task_lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Content Hashing
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute hash of content for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    @staticmethod
    def compute_chunk_hash(content: str, metadata: Dict[str, Any] = None) -> str:
        """Compute hash for a chunk including metadata."""
        hash_input = content
        if metadata:
            # Include relevant metadata in hash
            for key in sorted(metadata.keys()):
                if key in ('page_number', 'section_title'):
                    hash_input += f"|{key}:{metadata[key]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Change Detection
    # -------------------------------------------------------------------------

    async def detect_document_changes(
        self,
        document_id: uuid.UUID,
        new_content: str,
        new_chunks: List[Dict[str, Any]],
    ) -> List[ContentChange]:
        """
        Detect changes between current and new document content.

        Args:
            document_id: Document to check
            new_content: New document content
            new_chunks: New chunks with content and metadata

        Returns:
            List of detected changes
        """
        changes = []

        # Get existing document
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            # New document
            changes.append(ContentChange(
                document_id=document_id,
                change_type="added",
                new_hash=self.compute_content_hash(new_content),
            ))
            return changes

        # Get existing chunks
        result = await self.db.execute(
            select(Chunk).where(Chunk.document_id == document_id)
        )
        existing_chunks = {c.content_hash: c for c in result.scalars().all()}

        # Compare chunks
        new_chunk_hashes: Set[str] = set()

        for i, chunk_data in enumerate(new_chunks):
            content = chunk_data.get("content", "")
            metadata = chunk_data.get("metadata", {})
            chunk_hash = self.compute_chunk_hash(content, metadata)
            new_chunk_hashes.add(chunk_hash)

            if chunk_hash not in existing_chunks:
                # New or modified chunk
                changes.append(ContentChange(
                    document_id=document_id,
                    change_type="modified",
                    new_hash=chunk_hash,
                ))

        # Find deleted chunks
        for old_hash, chunk in existing_chunks.items():
            if old_hash not in new_chunk_hashes:
                changes.append(ContentChange(
                    document_id=document_id,
                    change_type="deleted",
                    chunk_ids=[chunk.id],
                    old_hash=old_hash,
                ))

        return changes

    async def has_document_changed(
        self,
        document_id: uuid.UUID,
        new_file_hash: str,
    ) -> bool:
        """
        Check if document file has changed based on hash.

        Args:
            document_id: Document to check
            new_file_hash: Hash of new file content

        Returns:
            True if document has changed
        """
        result = await self.db.execute(
            select(Document.file_hash).where(Document.id == document_id)
        )
        old_hash = result.scalar_one_or_none()

        return old_hash != new_file_hash

    # -------------------------------------------------------------------------
    # Incremental Indexing
    # -------------------------------------------------------------------------

    async def incremental_update(
        self,
        document_id: uuid.UUID,
        new_chunks: List[Dict[str, Any]],
        access_tier_id: uuid.UUID,
    ) -> IndexingStats:
        """
        Perform incremental update of document chunks.

        Only updates chunks that have actually changed.

        Args:
            document_id: Document to update
            new_chunks: New chunks with content and metadata
            access_tier_id: Access tier for new chunks

        Returns:
            IndexingStats with operation counts
        """
        import time
        start_time = time.time()

        stats = IndexingStats()

        try:
            # Get existing chunks indexed by hash
            result = await self.db.execute(
                select(Chunk).where(Chunk.document_id == document_id)
            )
            existing_chunks = {c.content_hash: c for c in result.scalars().all()}
            existing_hashes = set(existing_chunks.keys())

            # Track what we've seen
            seen_hashes: Set[str] = set()
            chunks_to_add: List[Chunk] = []

            for i, chunk_data in enumerate(new_chunks):
                content = chunk_data.get("content", "")
                metadata = chunk_data.get("metadata", {})
                chunk_hash = self.compute_chunk_hash(content, metadata)

                seen_hashes.add(chunk_hash)

                if chunk_hash in existing_chunks:
                    # Chunk unchanged - skip
                    continue

                # Generate embedding if service available
                embedding = None
                if self.embeddings:
                    try:
                        embedding = await self.embeddings.embed_query(content)
                    except Exception as e:
                        logger.warning("Embedding generation failed", error=str(e))

                # Create new chunk
                new_chunk = Chunk(
                    content=content,
                    content_hash=chunk_hash,
                    embedding=embedding,
                    chunk_index=i,
                    page_number=metadata.get("page_number"),
                    section_title=metadata.get("section_title"),
                    token_count=len(content.split()),
                    char_count=len(content),
                    document_id=document_id,
                    access_tier_id=access_tier_id,
                )
                chunks_to_add.append(new_chunk)
                stats.chunks_added += 1

            # Delete removed chunks
            hashes_to_delete = existing_hashes - seen_hashes
            for hash_to_delete in hashes_to_delete:
                chunk = existing_chunks[hash_to_delete]
                await self.db.delete(chunk)
                stats.chunks_deleted += 1

            # Add new chunks in batch
            if chunks_to_add:
                self.db.add_all(chunks_to_add)

            await self.db.commit()

            stats.documents_updated = 1
            stats.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Incremental update complete",
                document_id=str(document_id),
                added=stats.chunks_added,
                deleted=stats.chunks_deleted,
                time_ms=stats.processing_time_ms,
            )

        except Exception as e:
            logger.error("Incremental update failed", error=str(e))
            stats.errors.append(str(e))
            await self.db.rollback()

        return stats

    async def handle_content_diff_update(
        self,
        document_id: uuid.UUID,
        old_content: str,
        new_content: str,
        access_tier_id: uuid.UUID,
    ) -> IndexingStats:
        """
        Update chunks incrementally based on content diff.

        Uses difflib to identify changed sections and only re-chunks/re-embeds
        the affected portions. This is more efficient than full re-indexing
        for documents with small edits.

        Args:
            document_id: Document being updated
            old_content: Previous document content
            new_content: New document content
            access_tier_id: Access tier for new chunks

        Returns:
            IndexingStats with update details
        """
        import time
        from difflib import SequenceMatcher

        start_time = time.time()
        stats = IndexingStats()

        try:
            # Get existing chunks
            result = await self.db.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
            )
            existing_chunks = list(result.scalars().all())

            if not existing_chunks:
                # No existing chunks - do full indexing
                logger.info("No existing chunks for diff update, using full indexing")
                # The full indexing would be handled by the caller
                return stats

            # Use SequenceMatcher to find differences
            matcher = SequenceMatcher(None, old_content, new_content)
            opcodes = matcher.get_opcodes()

            # Track affected character ranges
            affected_ranges: List[Tuple[int, int]] = []  # (start, end) in new content

            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'replace':
                    # Content replaced - need to re-process this section
                    affected_ranges.append((j1, j2))
                elif tag == 'insert':
                    # New content inserted - need to process insertion
                    affected_ranges.append((j1, j2))
                elif tag == 'delete':
                    # Content deleted - chunks in this range should be removed
                    # Mark the position where deletion occurred
                    affected_ranges.append((j1, j1))
                # 'equal' sections don't need processing

            if not affected_ranges:
                logger.info("No differences found in content diff update")
                stats.processing_time_ms = (time.time() - start_time) * 1000
                return stats

            # Build position map for existing chunks (approximate character positions)
            chunk_positions = self._estimate_chunk_positions(existing_chunks, old_content)

            # Find chunks affected by the changes
            affected_chunk_indices = set()
            for start, end in affected_ranges:
                for idx, (chunk_start, chunk_end) in enumerate(chunk_positions):
                    # Check if ranges overlap
                    if self._ranges_overlap(start, end, chunk_start, chunk_end):
                        affected_chunk_indices.add(idx)

            logger.info(
                "Diff analysis complete",
                document_id=str(document_id),
                change_regions=len(affected_ranges),
                affected_chunks=len(affected_chunk_indices),
                total_chunks=len(existing_chunks),
            )

            # Delete affected chunks
            for idx in sorted(affected_chunk_indices, reverse=True):
                if idx < len(existing_chunks):
                    chunk = existing_chunks[idx]
                    await self.db.delete(chunk)
                    stats.chunks_deleted += 1

            # Re-chunk the affected sections of new content
            # Merge overlapping/adjacent affected ranges
            merged_ranges = self._merge_ranges(sorted(affected_ranges))

            for start, end in merged_ranges:
                if start >= end:
                    continue

                # Extract affected section with some context
                context_chars = 200  # Characters of context on each side
                section_start = max(0, start - context_chars)
                section_end = min(len(new_content), end + context_chars)
                section_content = new_content[section_start:section_end]

                # Create new chunk for this section
                chunk_hash = self.compute_chunk_hash(section_content, {})

                # Generate embedding if service available
                embedding = None
                if self.embeddings:
                    try:
                        embedding = await self.embeddings.embed_query(section_content)
                    except Exception as e:
                        logger.warning("Embedding generation failed for diff chunk", error=str(e))

                # Create new chunk
                new_chunk = Chunk(
                    content=section_content,
                    content_hash=chunk_hash,
                    embedding=embedding,
                    chunk_index=len(existing_chunks) - len(affected_chunk_indices) + stats.chunks_added,
                    document_id=document_id,
                    access_tier_id=access_tier_id,
                    token_count=len(section_content.split()),
                    char_count=len(section_content),
                )
                self.db.add(new_chunk)
                stats.chunks_added += 1

            await self.db.commit()
            stats.documents_updated = 1
            stats.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Diff-based chunk update complete",
                document_id=str(document_id),
                added=stats.chunks_added,
                deleted=stats.chunks_deleted,
                time_ms=stats.processing_time_ms,
            )

        except Exception as e:
            logger.error("Diff-based update failed", error=str(e))
            stats.errors.append(str(e))
            await self.db.rollback()

        return stats

    def _estimate_chunk_positions(
        self,
        chunks: List[Chunk],
        content: str,
    ) -> List[Tuple[int, int]]:
        """
        Estimate character positions for each chunk in content.

        Since chunks may overlap or have gaps, this finds approximate
        positions by searching for chunk content in the document.
        """
        positions = []
        search_start = 0

        for chunk in chunks:
            chunk_text = chunk.content[:200]  # Use first 200 chars for matching
            pos = content.find(chunk_text, search_start)

            if pos >= 0:
                chunk_start = pos
                chunk_end = pos + len(chunk.content)
                search_start = chunk_end  # Continue search after this chunk
            else:
                # Chunk not found - estimate based on index
                avg_chunk_size = len(content) // max(len(chunks), 1)
                chunk_start = chunk.chunk_index * avg_chunk_size
                chunk_end = chunk_start + len(chunk.content)

            positions.append((chunk_start, chunk_end))

        return positions

    def _ranges_overlap(
        self,
        start1: int, end1: int,
        start2: int, end2: int,
    ) -> bool:
        """Check if two ranges overlap."""
        return start1 < end2 and start2 < end1

    def _merge_ranges(
        self,
        ranges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent ranges."""
        if not ranges:
            return []

        merged = [ranges[0]]

        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]

            if start <= last_end + 100:  # Merge if within 100 chars
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    # -------------------------------------------------------------------------
    # Freshness Tracking
    # -------------------------------------------------------------------------

    def get_freshness_level(
        self,
        last_modified: Optional[datetime],
        fresh_days: int = FRESHNESS_THRESHOLD_DAYS,
        stale_days: int = STALE_THRESHOLD_DAYS,
    ) -> Tuple[FreshnessLevel, int]:
        """
        Determine freshness level based on last modification date.

        Args:
            last_modified: Last modification timestamp
            fresh_days: Days threshold for "fresh"
            stale_days: Days threshold for "stale"

        Returns:
            Tuple of (FreshnessLevel, days_since_update)
        """
        if not last_modified:
            return FreshnessLevel.UNKNOWN, -1

        now = datetime.now(last_modified.tzinfo) if last_modified.tzinfo else datetime.now()
        days_old = (now - last_modified).days

        if days_old <= 7:
            return FreshnessLevel.FRESH, days_old
        elif days_old <= fresh_days:
            return FreshnessLevel.CURRENT, days_old
        elif days_old <= stale_days:
            return FreshnessLevel.AGING, days_old
        else:
            return FreshnessLevel.STALE, days_old

    async def get_document_freshness(
        self,
        document_id: uuid.UUID,
    ) -> Optional[FreshnessInfo]:
        """
        Get freshness information for a document.

        Args:
            document_id: Document to check

        Returns:
            FreshnessInfo or None if document not found
        """
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            return None

        last_modified = document.updated_at or document.created_at
        level, days = self.get_freshness_level(last_modified)

        # Generate recommendation
        recommendation = None
        if level == FreshnessLevel.AGING:
            recommendation = "Consider reviewing this document for accuracy."
        elif level == FreshnessLevel.STALE:
            recommendation = "This document may contain outdated information. Recommend updating."

        return FreshnessInfo(
            document_id=document_id,
            filename=document.filename,
            last_modified=last_modified,
            freshness_level=level,
            days_since_update=days,
            recommendation=recommendation,
        )

    async def get_stale_documents(
        self,
        threshold_days: int = STALE_THRESHOLD_DAYS,
        limit: int = 100,
    ) -> List[FreshnessInfo]:
        """
        Get documents that are considered stale.

        Args:
            threshold_days: Days after which content is stale
            limit: Maximum documents to return

        Returns:
            List of FreshnessInfo for stale documents
        """
        threshold_date = datetime.now() - timedelta(days=threshold_days)

        result = await self.db.execute(
            select(Document)
            .where(
                Document.processing_status == ProcessingStatus.COMPLETED,
                or_(
                    Document.updated_at < threshold_date,
                    and_(
                        Document.updated_at.is_(None),
                        Document.created_at < threshold_date,
                    )
                )
            )
            .order_by(Document.updated_at.asc().nullsfirst())
            .limit(limit)
        )
        documents = result.scalars().all()

        freshness_list = []
        for doc in documents:
            last_modified = doc.updated_at or doc.created_at
            level, days = self.get_freshness_level(last_modified)

            freshness_list.append(FreshnessInfo(
                document_id=doc.id,
                filename=doc.filename,
                last_modified=last_modified,
                freshness_level=level,
                days_since_update=days,
                recommendation="This document may contain outdated information.",
            ))

        return freshness_list

    async def get_freshness_summary(self) -> Dict[str, Any]:
        """
        Get overall freshness statistics for the document archive.

        Returns:
            Summary statistics
        """
        now = datetime.now()
        fresh_threshold = now - timedelta(days=7)
        current_threshold = now - timedelta(days=FRESHNESS_THRESHOLD_DAYS)
        aging_threshold = now - timedelta(days=STALE_THRESHOLD_DAYS)

        # Count by freshness level
        fresh_count = await self.db.scalar(
            select(func.count(Document.id)).where(
                Document.processing_status == ProcessingStatus.COMPLETED,
                Document.updated_at >= fresh_threshold,
            )
        )

        current_count = await self.db.scalar(
            select(func.count(Document.id)).where(
                Document.processing_status == ProcessingStatus.COMPLETED,
                Document.updated_at >= current_threshold,
                Document.updated_at < fresh_threshold,
            )
        )

        aging_count = await self.db.scalar(
            select(func.count(Document.id)).where(
                Document.processing_status == ProcessingStatus.COMPLETED,
                Document.updated_at >= aging_threshold,
                Document.updated_at < current_threshold,
            )
        )

        stale_count = await self.db.scalar(
            select(func.count(Document.id)).where(
                Document.processing_status == ProcessingStatus.COMPLETED,
                Document.updated_at < aging_threshold,
            )
        )

        total = (fresh_count or 0) + (current_count or 0) + (aging_count or 0) + (stale_count or 0)

        return {
            "total_documents": total,
            "fresh": fresh_count or 0,
            "current": current_count or 0,
            "aging": aging_count or 0,
            "stale": stale_count or 0,
            "freshness_thresholds": {
                "fresh_days": 7,
                "current_days": FRESHNESS_THRESHOLD_DAYS,
                "stale_days": STALE_THRESHOLD_DAYS,
            }
        }

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    async def queue_reindex_task(
        self,
        document_id: uuid.UUID,
        priority: int = 0,
    ):
        """
        Queue a document for reindexing.

        Args:
            document_id: Document to reindex
            priority: Task priority (higher = more urgent)
        """
        async with self._task_lock:
            task = IndexingTask(
                document_id=document_id,
                task_type="reindex",
                priority=priority,
            )
            self._pending_tasks.append(task)
            # Sort by priority (descending)
            self._pending_tasks.sort(key=lambda t: -t.priority)

    async def process_pending_tasks(
        self,
        batch_size: int = BATCH_SIZE,
    ) -> IndexingStats:
        """
        Process pending indexing tasks in batch.

        Args:
            batch_size: Maximum tasks to process

        Returns:
            Combined IndexingStats
        """
        stats = IndexingStats()

        async with self._task_lock:
            tasks_to_process = self._pending_tasks[:batch_size]
            self._pending_tasks = self._pending_tasks[batch_size:]

        for task in tasks_to_process:
            try:
                if task.task_type == "reindex":
                    # Full reindex: delete existing chunks and regenerate
                    reindex_stats = await self._perform_reindex(task.document_id)
                    stats.documents_checked += 1
                    stats.documents_updated += reindex_stats.documents_updated
                    stats.chunks_added += reindex_stats.chunks_added
                    stats.chunks_deleted += reindex_stats.chunks_deleted

                elif task.task_type == "update_chunks":
                    # Partial update: only regenerate specified chunks
                    update_stats = await self._perform_chunk_update(
                        task.document_id,
                        task.chunks_to_update,
                    )
                    stats.chunks_modified += update_stats.chunks_modified
                    stats.documents_updated += 1

                elif task.task_type == "delete":
                    # Delete document and chunks
                    result = await self.db.execute(
                        select(Document).where(Document.id == task.document_id)
                    )
                    doc = result.scalar_one_or_none()
                    if doc:
                        await self.db.delete(doc)
                        stats.documents_updated += 1

            except Exception as e:
                stats.errors.append(f"Task {task.document_id}: {str(e)}")

        if tasks_to_process:
            await self.db.commit()

        return stats

    async def get_pending_task_count(self) -> int:
        """Get number of pending indexing tasks."""
        async with self._task_lock:
            return len(self._pending_tasks)

    # -------------------------------------------------------------------------
    # Reindex and Chunk Update Implementation
    # -------------------------------------------------------------------------

    async def _perform_reindex(
        self,
        document_id: uuid.UUID,
    ) -> IndexingStats:
        """
        Perform full reindex of a document.

        1. Fetch the document
        2. Delete existing chunks
        3. Re-chunk and re-embed the content
        4. Update document status

        Args:
            document_id: Document to reindex

        Returns:
            IndexingStats with operation counts
        """
        stats = IndexingStats()

        # Get the document
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()

        if not doc:
            logger.warning("Document not found for reindex", document_id=str(document_id))
            stats.errors.append(f"Document {document_id} not found")
            return stats

        try:
            # Count existing chunks
            chunk_count_result = await self.db.execute(
                select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
            )
            old_chunk_count = chunk_count_result.scalar() or 0

            # Delete existing chunks
            await self.db.execute(
                Chunk.__table__.delete().where(Chunk.document_id == document_id)
            )
            stats.chunks_deleted = old_chunk_count

            # Re-chunk the document content
            from backend.services.chunker import TextChunker

            chunker = TextChunker()
            content = doc.content or ""

            if content:
                chunks_data = chunker.chunk_text(
                    text=content,
                    metadata={
                        "document_id": str(document_id),
                        "filename": doc.filename,
                        "reindexed_at": datetime.now().isoformat(),
                    }
                )

                # Create new chunks
                for i, chunk_data in enumerate(chunks_data):
                    chunk = Chunk(
                        document_id=document_id,
                        content=chunk_data.get("text", ""),
                        chunk_index=i,
                        token_count=len(chunk_data.get("text", "").split()),
                        metadata=chunk_data.get("metadata", {}),
                    )
                    self.db.add(chunk)
                    stats.chunks_added += 1

            # Update document status
            doc.processing_status = ProcessingStatus.COMPLETED
            doc.updated_at = datetime.now()
            stats.documents_updated = 1

            logger.info(
                "Document reindexed",
                document_id=str(document_id),
                chunks_deleted=stats.chunks_deleted,
                chunks_added=stats.chunks_added,
            )

        except Exception as e:
            logger.error("Reindex failed", document_id=str(document_id), error=str(e))
            stats.errors.append(f"Reindex failed: {str(e)}")
            doc.processing_status = ProcessingStatus.FAILED

        return stats

    async def _perform_chunk_update(
        self,
        document_id: uuid.UUID,
        chunk_ids: List[uuid.UUID],
    ) -> IndexingStats:
        """
        Update specific chunks of a document.

        This is more efficient than full reindex when only
        some chunks need regeneration (e.g., embedding update).

        Args:
            document_id: Parent document
            chunk_ids: Specific chunks to update

        Returns:
            IndexingStats with operation counts
        """
        stats = IndexingStats()

        if not chunk_ids:
            return stats

        # Get the chunks to update
        result = await self.db.execute(
            select(Chunk).where(
                and_(
                    Chunk.document_id == document_id,
                    Chunk.id.in_(chunk_ids),
                )
            )
        )
        chunks = result.scalars().all()

        for chunk in chunks:
            try:
                # Regenerate embedding for this chunk
                from backend.services.embeddings import get_embedding_service

                embedding_service = get_embedding_service()
                new_embedding = await embedding_service.embed_text(chunk.content)

                # Update the chunk's embedding
                chunk.embedding = new_embedding
                chunk.metadata = {
                    **chunk.metadata,
                    "embedding_updated_at": datetime.now().isoformat(),
                }

                stats.chunks_modified += 1

            except Exception as e:
                logger.error(
                    "Chunk update failed",
                    chunk_id=str(chunk.id),
                    error=str(e),
                )
                stats.errors.append(f"Chunk {chunk.id}: {str(e)}")

        if stats.chunks_modified > 0:
            logger.info(
                "Chunks updated",
                document_id=str(document_id),
                chunks_modified=stats.chunks_modified,
            )

        return stats

    # -------------------------------------------------------------------------
    # Document Touch (Update Timestamp)
    # -------------------------------------------------------------------------

    async def touch_document(
        self,
        document_id: uuid.UUID,
    ) -> bool:
        """
        Update document's last modified timestamp.

        Use this when document is accessed/verified as current.

        Args:
            document_id: Document to touch

        Returns:
            True if document was updated
        """
        result = await self.db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(updated_at=datetime.now())
        )
        await self.db.commit()
        return result.rowcount > 0


# =============================================================================
# Factory Function
# =============================================================================

async def get_realtime_indexer_service(
    db_session: AsyncSession,
    embedding_service=None,
) -> RealTimeIndexerService:
    """Create configured real-time indexer service."""
    return RealTimeIndexerService(
        db_session=db_session,
        embedding_service=embedding_service,
    )
