"""
AIDocumentIndexer - Agent Knowledge Management
===============================================

Phase 23B: Implements incremental document update pipeline for agents.

Features:
- Version-tracked knowledge base per agent
- Incremental updates without full rebuild
- Document change detection and diff processing
- Async embedding generation queuing
- Soft delete with version tombstones

Based on:
- Incremental RAG patterns (2025 research)
- Version-controlled vector stores
- Efficient index update strategies
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
import uuid

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class ChangeType(str, Enum):
    """Types of document changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class DocumentVersion:
    """Tracks a document's version in an agent's knowledge base."""
    document_id: str
    version: int
    content_hash: str
    chunk_count: int
    embedding_status: str  # "pending", "processing", "complete", "failed"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
            "embedding_status": self.embedding_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "metadata": self.metadata,
        }


@dataclass
class DocumentChange:
    """Represents a change to a document."""
    document_id: str
    change_type: ChangeType
    content: Optional[str] = None
    chunks: List[str] = field(default_factory=list)
    old_version: Optional[int] = None
    new_version: Optional[int] = None
    diff_summary: Optional[str] = None


@dataclass
class DocumentChanges:
    """Collection of document changes."""
    added: List[DocumentChange] = field(default_factory=list)
    modified: List[DocumentChange] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added_count": len(self.added),
            "modified_count": len(self.modified),
            "deleted_count": len(self.deleted),
            "added": [c.document_id for c in self.added],
            "modified": [c.document_id for c in self.modified],
            "deleted": self.deleted,
        }


@dataclass
class KnowledgeBaseStats:
    """Statistics for an agent's knowledge base."""
    agent_id: str
    total_documents: int
    total_chunks: int
    current_version: int
    pending_updates: int
    last_updated: Optional[datetime] = None
    embedding_coverage: float = 0.0  # Percentage with embeddings


# =============================================================================
# Agent Knowledge Service
# =============================================================================

class AgentKnowledgeService:
    """
    Manages incremental knowledge base updates for agents.

    Provides:
    - Version tracking per document
    - Incremental index updates
    - Change detection
    - Async embedding queue
    """

    def __init__(
        self,
        agent_id: str,
        embedding_service=None,
        vector_store=None,
    ):
        self.agent_id = agent_id
        self.embedding_service = embedding_service
        self.vector_store = vector_store

        # In-memory state (should be persisted in production)
        self._versions: Dict[str, DocumentVersion] = {}
        self._current_version = 0
        self._pending_embeddings: List[str] = []
        self._redis = None
        self._initialized = False

    async def initialize(self):
        """Initialize the service and load state."""
        if self._initialized:
            return

        try:
            from backend.services.redis_client import get_redis_client
            self._redis = await get_redis_client()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")

        # Load state from storage
        await self._load_state()
        self._initialized = True

        logger.info(
            "Agent knowledge service initialized",
            agent_id=self.agent_id,
            documents=len(self._versions),
            version=self._current_version,
        )

    async def _load_state(self):
        """Load state from Redis."""
        if not self._redis:
            return

        try:
            key = f"agent:knowledge:{self.agent_id}"
            data = await self._redis.get(key)
            if data:
                parsed = json.loads(data)
                self._current_version = parsed.get("version", 0)
                self._pending_embeddings = parsed.get("pending", [])

                for doc_id, version_data in parsed.get("versions", {}).items():
                    version_data["created_at"] = datetime.fromisoformat(version_data["created_at"])
                    version_data["updated_at"] = datetime.fromisoformat(version_data["updated_at"])
                    if version_data.get("deleted_at"):
                        version_data["deleted_at"] = datetime.fromisoformat(version_data["deleted_at"])
                    self._versions[doc_id] = DocumentVersion(**version_data)
        except Exception as e:
            logger.error(f"Failed to load knowledge state: {e}")

    async def _save_state(self):
        """Save state to Redis."""
        if not self._redis:
            return

        try:
            key = f"agent:knowledge:{self.agent_id}"
            data = {
                "version": self._current_version,
                "pending": self._pending_embeddings,
                "versions": {
                    doc_id: v.to_dict()
                    for doc_id, v in self._versions.items()
                },
            }
            await self._redis.set(key, json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to save knowledge state: {e}")

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_current_version(self) -> int:
        """Get current knowledge base version."""
        return self._current_version

    async def detect_changes(
        self,
        documents: List[Dict[str, Any]],
    ) -> DocumentChanges:
        """
        Detect changes between current state and new documents.

        Args:
            documents: List of documents with id and content

        Returns:
            DocumentChanges with added, modified, deleted
        """
        changes = DocumentChanges()
        seen_ids: Set[str] = set()

        for doc in documents:
            doc_id = doc.get("id") or doc.get("document_id")
            content = doc.get("content", "")
            if not doc_id:
                continue

            seen_ids.add(doc_id)
            content_hash = self._compute_hash(content)

            if doc_id not in self._versions:
                # New document
                changes.added.append(DocumentChange(
                    document_id=doc_id,
                    change_type=ChangeType.ADDED,
                    content=content,
                    new_version=self._current_version + 1,
                ))
            else:
                existing = self._versions[doc_id]
                if existing.deleted_at:
                    # Re-adding deleted document
                    changes.added.append(DocumentChange(
                        document_id=doc_id,
                        change_type=ChangeType.ADDED,
                        content=content,
                        old_version=existing.version,
                        new_version=self._current_version + 1,
                    ))
                elif existing.content_hash != content_hash:
                    # Modified document
                    changes.modified.append(DocumentChange(
                        document_id=doc_id,
                        change_type=ChangeType.MODIFIED,
                        content=content,
                        old_version=existing.version,
                        new_version=self._current_version + 1,
                    ))

        # Find deleted documents
        for doc_id, version in self._versions.items():
            if doc_id not in seen_ids and not version.deleted_at:
                changes.deleted.append(doc_id)

        return changes

    async def apply_changes(
        self,
        changes: DocumentChanges,
        queue_embeddings: bool = True,
    ) -> int:
        """
        Apply document changes to the knowledge base.

        Args:
            changes: Changes to apply
            queue_embeddings: Whether to queue embedding generation

        Returns:
            New version number
        """
        if changes.total_changes == 0:
            return self._current_version

        self._current_version += 1
        new_version = self._current_version

        # Process additions
        for change in changes.added:
            self._versions[change.document_id] = DocumentVersion(
                document_id=change.document_id,
                version=new_version,
                content_hash=self._compute_hash(change.content or ""),
                chunk_count=len(change.chunks) if change.chunks else 0,
                embedding_status="pending" if queue_embeddings else "complete",
            )

            if queue_embeddings:
                self._pending_embeddings.append(change.document_id)

            # Add to vector store
            if self.vector_store and change.content:
                await self._index_document(change.document_id, change.content, change.chunks)

        # Process modifications
        for change in changes.modified:
            if change.document_id in self._versions:
                version = self._versions[change.document_id]
                version.version = new_version
                version.content_hash = self._compute_hash(change.content or "")
                version.chunk_count = len(change.chunks) if change.chunks else 0
                version.embedding_status = "pending" if queue_embeddings else "complete"
                version.updated_at = datetime.utcnow()

                if queue_embeddings:
                    self._pending_embeddings.append(change.document_id)

                # Update in vector store
                if self.vector_store and change.content:
                    await self._update_document(change.document_id, change.content, change.chunks)

        # Process deletions (soft delete)
        for doc_id in changes.deleted:
            if doc_id in self._versions:
                self._versions[doc_id].deleted_at = datetime.utcnow()

                # Remove from vector store
                if self.vector_store:
                    await self._remove_document(doc_id)

        # Save state
        await self._save_state()

        logger.info(
            "Applied knowledge changes",
            agent_id=self.agent_id,
            added=len(changes.added),
            modified=len(changes.modified),
            deleted=len(changes.deleted),
            new_version=new_version,
        )

        return new_version

    async def _index_document(
        self,
        doc_id: str,
        content: str,
        chunks: Optional[List[str]] = None,
    ):
        """Index a document in the vector store."""
        try:
            if not chunks:
                # Simple chunking if not provided
                chunks = self._simple_chunk(content)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"

                # Generate embedding if service available
                embedding = None
                if self.embedding_service:
                    embedding = self.embedding_service.embed_text(chunk)

                # Store in vector store
                if self.vector_store:
                    await self.vector_store.upsert(
                        id=chunk_id,
                        content=chunk,
                        embedding=embedding,
                        metadata={
                            "document_id": doc_id,
                            "agent_id": self.agent_id,
                            "chunk_index": i,
                        },
                    )

        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            if doc_id in self._versions:
                self._versions[doc_id].embedding_status = "failed"

    async def _update_document(
        self,
        doc_id: str,
        content: str,
        chunks: Optional[List[str]] = None,
    ):
        """Update a document in the vector store."""
        # Remove old chunks first
        await self._remove_document(doc_id)
        # Re-index
        await self._index_document(doc_id, content, chunks)

    async def _remove_document(self, doc_id: str):
        """Remove a document from the vector store."""
        try:
            if self.vector_store:
                # Remove all chunks for this document
                await self.vector_store.delete_by_metadata(
                    filter={"document_id": doc_id, "agent_id": self.agent_id}
                )
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")

    def _simple_chunk(self, content: str, chunk_size: int = 500) -> List[str]:
        """Simple text chunking."""
        chunks = []
        words = content.split()

        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def process_pending_embeddings(self, batch_size: int = 10) -> int:
        """
        Process pending embedding generation.

        Args:
            batch_size: Number of documents to process

        Returns:
            Number of documents processed
        """
        if not self._pending_embeddings or not self.embedding_service:
            return 0

        processed = 0
        batch = self._pending_embeddings[:batch_size]

        for doc_id in batch:
            if doc_id in self._versions:
                version = self._versions[doc_id]
                version.embedding_status = "processing"

                try:
                    # Embeddings would be generated here
                    # In practice, this is done during indexing
                    version.embedding_status = "complete"
                    processed += 1
                except Exception as e:
                    logger.error(f"Embedding generation failed for {doc_id}: {e}")
                    version.embedding_status = "failed"

        # Remove processed from pending
        self._pending_embeddings = self._pending_embeddings[batch_size:]
        await self._save_state()

        return processed

    async def get_stats(self) -> KnowledgeBaseStats:
        """Get knowledge base statistics."""
        active_docs = [v for v in self._versions.values() if not v.deleted_at]
        total_chunks = sum(v.chunk_count for v in active_docs)
        complete_embeddings = sum(
            1 for v in active_docs if v.embedding_status == "complete"
        )

        return KnowledgeBaseStats(
            agent_id=self.agent_id,
            total_documents=len(active_docs),
            total_chunks=total_chunks,
            current_version=self._current_version,
            pending_updates=len(self._pending_embeddings),
            last_updated=max(
                (v.updated_at for v in active_docs), default=None
            ),
            embedding_coverage=complete_embeddings / len(active_docs) if active_docs else 0.0,
        )

    async def rollback_to_version(self, target_version: int) -> bool:
        """
        Rollback knowledge base to a previous version.

        Note: This is a simplified implementation. Full rollback would
        require storing historical state.
        """
        if target_version >= self._current_version or target_version < 0:
            return False

        # Remove documents added after target version
        to_remove = [
            doc_id for doc_id, v in self._versions.items()
            if v.version > target_version
        ]

        for doc_id in to_remove:
            del self._versions[doc_id]
            if self.vector_store:
                await self._remove_document(doc_id)

        self._current_version = target_version
        await self._save_state()

        logger.info(
            "Rolled back knowledge base",
            agent_id=self.agent_id,
            target_version=target_version,
            removed_docs=len(to_remove),
        )

        return True


# =============================================================================
# Update Pipeline
# =============================================================================

class IncrementalUpdatePipeline:
    """
    Pipeline for incremental knowledge base updates.

    Handles:
    - Document change detection
    - Chunking and embedding
    - Index updates
    - Version management
    """

    def __init__(
        self,
        knowledge_service: AgentKnowledgeService,
        chunker=None,
        embedding_service=None,
    ):
        self.knowledge = knowledge_service
        self.chunker = chunker
        self.embedding_service = embedding_service

    async def update(
        self,
        documents: List[Dict[str, Any]],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Update knowledge base with new documents.

        Args:
            documents: List of documents to sync
            force_rebuild: Force full rebuild instead of incremental

        Returns:
            Update result with statistics
        """
        await self.knowledge.initialize()

        if force_rebuild:
            # Clear all and re-add
            for doc_id in list(self.knowledge._versions.keys()):
                self.knowledge._versions[doc_id].deleted_at = datetime.utcnow()

        # Detect changes
        changes = await self.knowledge.detect_changes(documents)

        if changes.total_changes == 0:
            return {
                "status": "no_changes",
                "version": await self.knowledge.get_current_version(),
            }

        # Chunk documents if chunker available
        if self.chunker:
            for change in changes.added + changes.modified:
                if change.content:
                    change.chunks = await self._chunk_document(change.content)

        # Apply changes
        new_version = await self.knowledge.apply_changes(changes)

        # Get stats
        stats = await self.knowledge.get_stats()

        return {
            "status": "updated",
            "version": new_version,
            "changes": changes.to_dict(),
            "stats": {
                "total_documents": stats.total_documents,
                "total_chunks": stats.total_chunks,
                "embedding_coverage": stats.embedding_coverage,
            },
        }

    async def _chunk_document(self, content: str) -> List[str]:
        """Chunk a document using the configured chunker."""
        if self.chunker:
            try:
                return await self.chunker.chunk(content)
            except Exception as e:
                logger.warning(f"Chunking failed: {e}")

        # Fallback to simple chunking
        return self.knowledge._simple_chunk(content)


# =============================================================================
# Factory Function
# =============================================================================

async def get_agent_knowledge_service(
    agent_id: str,
    embedding_service=None,
    vector_store=None,
) -> AgentKnowledgeService:
    """
    Get or create agent knowledge service.

    Args:
        agent_id: Agent ID
        embedding_service: Embedding generation service
        vector_store: Vector store for retrieval

    Returns:
        Initialized AgentKnowledgeService
    """
    service = AgentKnowledgeService(
        agent_id=agent_id,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    await service.initialize()
    return service
