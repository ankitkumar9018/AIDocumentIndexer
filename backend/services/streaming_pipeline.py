"""
AIDocumentIndexer - Real-Time Streaming Pipeline
=================================================

Phase 15: Real-time streaming ingestion and processing pipeline.

Key Features:
- Streaming document ingestion (process chunks as they arrive)
- Partial query support (answer from partially processed docs)
- Progressive embeddings (no waiting for full document)
- Event-driven architecture with async streaming
- Priority queue integration for real-time vs batch

Architecture:
- StreamingIngestionService: Handles chunked document uploads
- PartialQueryService: Queries partially indexed documents
- ProgressiveEmbeddingService: Generates embeddings as chunks arrive
- StreamingEventBus: Async event distribution

Based on requirements from the optimization plan:
- 5s partial query availability
- Progressive embedding generation
- Event-driven processing
"""

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class StreamingStatus(str, Enum):
    """Status of streaming ingestion."""
    RECEIVING = "receiving"        # Receiving chunks
    PROCESSING = "processing"      # Processing received chunks
    PARTIALLY_READY = "partial"    # Some chunks ready for query
    COMPLETE = "complete"          # All chunks processed
    FAILED = "failed"             # Ingestion failed
    CANCELLED = "cancelled"       # Ingestion cancelled


class ChunkStatus(str, Enum):
    """Status of individual chunk."""
    RECEIVED = "received"          # Chunk received
    PARSING = "parsing"            # Extracting text
    EMBEDDING = "embedding"        # Generating embedding
    INDEXED = "indexed"            # Stored in vector DB
    FAILED = "failed"             # Processing failed


@dataclass
class StreamingChunk:
    """A chunk of streaming document data."""
    chunk_id: str
    document_id: str
    content: bytes
    chunk_index: int
    is_final: bool = False
    content_type: str = "application/octet-stream"
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ChunkStatus = ChunkStatus.RECEIVED
    text_content: Optional[str] = None
    embedding: Optional[List[float]] = None
    received_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


@dataclass
class StreamingDocument:
    """A document being streamed."""
    document_id: str
    filename: str
    content_type: str
    total_chunks: Optional[int] = None
    received_chunks: int = 0
    processed_chunks: int = 0
    indexed_chunks: int = 0
    status: StreamingStatus = StreamingStatus.RECEIVING
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    collection: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: Dict[int, StreamingChunk] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    first_query_ready_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class StreamingEvent:
    """Event for streaming pipeline."""
    event_type: str
    document_id: str
    chunk_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PartialQueryResult:
    """Result from a partial query."""
    query: str
    document_id: str
    chunks_available: int
    chunks_total: Optional[int]
    is_complete: bool
    results: List[Dict[str, Any]]
    confidence: float  # Lower for partial results
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Streaming Event Bus
# =============================================================================

class StreamingEventBus:
    """
    Async event bus for streaming pipeline events.

    Supports:
    - Multiple subscribers per event type
    - Async event handlers
    - Event filtering by document_id
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._document_subscribers: Dict[str, List[Callable]] = {}

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[StreamingEvent], Any],
        document_id: Optional[str] = None,
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_type: Type of event (or "*" for all)
            handler: Async callback function
            document_id: Optional filter for specific document

        Returns:
            Subscription ID for unsubscribing
        """
        sub_id = f"{event_type}:{uuid.uuid4().hex[:8]}"

        if document_id:
            key = f"{event_type}:{document_id}"
            if key not in self._document_subscribers:
                self._document_subscribers[key] = []
            self._document_subscribers[key].append((sub_id, handler))
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append((sub_id, handler))

        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Unsubscribe from events."""
        event_type = sub_id.split(":")[0]

        # Check global subscribers
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                (sid, h) for sid, h in self._subscribers[event_type]
                if sid != sub_id
            ]
            return True

        # Check document-specific subscribers
        for key, handlers in self._document_subscribers.items():
            self._document_subscribers[key] = [
                (sid, h) for sid, h in handlers if sid != sub_id
            ]

        return False

    async def publish(self, event: StreamingEvent) -> int:
        """
        Publish event to all subscribers.

        Returns number of handlers notified.
        """
        handlers_called = 0

        # Global subscribers for this event type
        if event.event_type in self._subscribers:
            for _, handler in self._subscribers[event.event_type]:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                    handlers_called += 1
                except Exception as e:
                    logger.warning(f"Event handler failed: {e}")

        # Wildcard subscribers
        if "*" in self._subscribers:
            for _, handler in self._subscribers["*"]:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                    handlers_called += 1
                except Exception as e:
                    logger.warning(f"Wildcard handler failed: {e}")

        # Document-specific subscribers
        doc_key = f"{event.event_type}:{event.document_id}"
        if doc_key in self._document_subscribers:
            for _, handler in self._document_subscribers[doc_key]:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                    handlers_called += 1
                except Exception as e:
                    logger.warning(f"Document handler failed: {e}")

        return handlers_called


# Global event bus
_event_bus: Optional[StreamingEventBus] = None


def get_event_bus() -> StreamingEventBus:
    """Get or create the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = StreamingEventBus()
    return _event_bus


# =============================================================================
# Streaming Ingestion Service
# =============================================================================

class StreamingIngestionService:
    """
    Service for streaming document ingestion.

    Features:
    - Accept chunks as they arrive
    - Process chunks in parallel
    - Generate embeddings progressively
    - Index chunks for partial queries
    """

    def __init__(
        self,
        max_concurrent_chunks: int = 10,
        min_chunks_for_query: int = 1,
        embedding_batch_size: int = 10,
    ):
        self.max_concurrent_chunks = max_concurrent_chunks
        self.min_chunks_for_query = min_chunks_for_query
        self.embedding_batch_size = embedding_batch_size

        self._documents: Dict[str, StreamingDocument] = {}
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_chunks)
        self._event_bus = get_event_bus()

        # Pending chunks for batch embedding
        self._pending_embeddings: Dict[str, List[StreamingChunk]] = {}
        self._embedding_lock = asyncio.Lock()

    async def start_streaming(
        self,
        filename: str,
        content_type: str,
        total_chunks: Optional[int] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        collection: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new streaming document upload.

        Returns document_id for subsequent chunk uploads.
        """
        document_id = str(uuid.uuid4())

        doc = StreamingDocument(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            total_chunks=total_chunks,
            organization_id=organization_id,
            user_id=user_id,
            collection=collection,
            metadata=metadata or {},
        )

        self._documents[document_id] = doc
        self._pending_embeddings[document_id] = []

        # Publish event
        await self._event_bus.publish(StreamingEvent(
            event_type="document.started",
            document_id=document_id,
            data={"filename": filename, "total_chunks": total_chunks},
        ))

        logger.info(
            "Started streaming document",
            document_id=document_id,
            filename=filename,
            total_chunks=total_chunks,
        )

        return document_id

    async def receive_chunk(
        self,
        document_id: str,
        chunk_data: bytes,
        chunk_index: int,
        is_final: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StreamingChunk:
        """
        Receive a chunk of document data.

        The chunk will be processed asynchronously.
        """
        if document_id not in self._documents:
            raise ValueError(f"Unknown document: {document_id}")

        doc = self._documents[document_id]

        if doc.status == StreamingStatus.FAILED:
            raise ValueError(f"Document {document_id} is in failed state")

        # Create chunk
        chunk = StreamingChunk(
            chunk_id=f"{document_id}:{chunk_index}",
            document_id=document_id,
            content=chunk_data,
            chunk_index=chunk_index,
            is_final=is_final,
            content_type=doc.content_type,
            metadata=metadata or {},
        )

        doc.chunks[chunk_index] = chunk
        doc.received_chunks += 1

        if is_final and doc.total_chunks is None:
            doc.total_chunks = chunk_index + 1

        # Publish event
        await self._event_bus.publish(StreamingEvent(
            event_type="chunk.received",
            document_id=document_id,
            chunk_id=chunk.chunk_id,
            data={"chunk_index": chunk_index, "is_final": is_final},
        ))

        # Start processing chunk asynchronously
        asyncio.create_task(self._process_chunk(chunk))

        return chunk

    async def _process_chunk(self, chunk: StreamingChunk) -> None:
        """Process a single chunk: parse, embed, index."""
        async with self._processing_semaphore:
            try:
                doc = self._documents.get(chunk.document_id)
                if not doc:
                    return

                # Step 1: Parse content
                chunk.status = ChunkStatus.PARSING
                chunk.text_content = await self._parse_chunk_content(chunk)

                if not chunk.text_content:
                    chunk.status = ChunkStatus.FAILED
                    return

                # Step 2: Add to embedding batch
                chunk.status = ChunkStatus.EMBEDDING
                await self._queue_for_embedding(chunk)

                # Check if we have enough for a batch
                await self._process_embedding_batch(chunk.document_id)

            except Exception as e:
                logger.error(f"Chunk processing failed: {e}", chunk_id=chunk.chunk_id)
                chunk.status = ChunkStatus.FAILED
                await self._event_bus.publish(StreamingEvent(
                    event_type="chunk.failed",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    data={"error": str(e)},
                ))

    async def _parse_chunk_content(self, chunk: StreamingChunk) -> Optional[str]:
        """Parse raw chunk content to text."""
        # For text-based content, decode directly
        if chunk.content_type.startswith("text/"):
            return chunk.content.decode("utf-8", errors="replace")

        # For binary content (PDF, DOCX, etc.), use document parser
        try:
            # Import here to avoid circular imports
            from backend.services.document_parser import DocumentParser

            parser = DocumentParser()
            text = await parser.parse_binary_chunk(
                chunk.content,
                chunk.content_type,
                chunk.metadata,
            )
            return text

        except Exception as e:
            logger.warning(f"Failed to parse chunk: {e}")
            # Fallback: try to decode as text
            try:
                return chunk.content.decode("utf-8", errors="replace")
            except Exception:
                return None

    async def _queue_for_embedding(self, chunk: StreamingChunk) -> None:
        """Add chunk to embedding queue."""
        async with self._embedding_lock:
            if chunk.document_id not in self._pending_embeddings:
                self._pending_embeddings[chunk.document_id] = []
            self._pending_embeddings[chunk.document_id].append(chunk)

    async def _process_embedding_batch(
        self,
        document_id: str,
        force: bool = False,
    ) -> None:
        """Process pending embeddings in batch."""
        async with self._embedding_lock:
            pending = self._pending_embeddings.get(document_id, [])

            if not pending:
                return

            # Process if batch is full or forced
            if len(pending) < self.embedding_batch_size and not force:
                return

            # Take the batch
            batch = pending[:self.embedding_batch_size]
            self._pending_embeddings[document_id] = pending[self.embedding_batch_size:]

        # Generate embeddings outside the lock
        texts = [c.text_content for c in batch if c.text_content]

        if not texts:
            return

        try:
            embeddings = await self._generate_embeddings(texts)

            # Update chunks with embeddings
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
                chunk.status = ChunkStatus.INDEXED
                chunk.processed_at = datetime.utcnow()

                # Index in vector store
                await self._index_chunk(chunk)

            # Update document status
            await self._update_document_status(document_id)

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            for chunk in batch:
                chunk.status = ChunkStatus.FAILED

    async def _generate_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            from backend.services.embeddings import EmbeddingService

            service = EmbeddingService()
            embeddings = await service.embed_texts(texts)
            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def _index_chunk(self, chunk: StreamingChunk) -> None:
        """Index chunk in vector store for retrieval."""
        if not chunk.embedding or not chunk.text_content:
            return

        try:
            # Import here to avoid circular imports
            # Use factory function for proper initialization (thread-safe, properly configured)
            from backend.services.rag import get_rag_service

            rag = get_rag_service()
            doc = self._documents.get(chunk.document_id)

            await rag.index_chunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.text_content,
                embedding=chunk.embedding,
                metadata={
                    "chunk_index": chunk.chunk_index,
                    "filename": doc.filename if doc else None,
                    "collection": doc.collection if doc else None,
                    "streaming": True,
                    **chunk.metadata,
                },
            )

            # Publish event
            await self._event_bus.publish(StreamingEvent(
                event_type="chunk.indexed",
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                data={"chunk_index": chunk.chunk_index},
            ))

        except Exception as e:
            logger.error(f"Chunk indexing failed: {e}")
            raise

    async def _update_document_status(self, document_id: str) -> None:
        """Update document status based on chunk progress."""
        doc = self._documents.get(document_id)
        if not doc:
            return

        # Count indexed chunks
        indexed_count = sum(
            1 for c in doc.chunks.values()
            if c.status == ChunkStatus.INDEXED
        )
        doc.indexed_chunks = indexed_count

        # Check if ready for partial queries
        if indexed_count >= self.min_chunks_for_query:
            if doc.status == StreamingStatus.RECEIVING:
                doc.status = StreamingStatus.PARTIALLY_READY
                doc.first_query_ready_at = datetime.utcnow()

                await self._event_bus.publish(StreamingEvent(
                    event_type="document.partial_ready",
                    document_id=document_id,
                    data={
                        "indexed_chunks": indexed_count,
                        "time_to_ready_ms": (
                            doc.first_query_ready_at - doc.created_at
                        ).total_seconds() * 1000,
                    },
                ))

        # Check if complete
        if doc.total_chunks and indexed_count >= doc.total_chunks:
            doc.status = StreamingStatus.COMPLETE
            doc.completed_at = datetime.utcnow()

            await self._event_bus.publish(StreamingEvent(
                event_type="document.complete",
                document_id=document_id,
                data={
                    "total_chunks": doc.total_chunks,
                    "processing_time_ms": (
                        doc.completed_at - doc.created_at
                    ).total_seconds() * 1000,
                },
            ))

    async def finalize_streaming(self, document_id: str) -> StreamingDocument:
        """
        Finalize streaming document.

        Processes any remaining embeddings and marks document complete.
        """
        if document_id not in self._documents:
            raise ValueError(f"Unknown document: {document_id}")

        doc = self._documents[document_id]

        # Process remaining embeddings
        await self._process_embedding_batch(document_id, force=True)

        # Wait for all chunks to be processed
        max_wait = 30  # seconds
        start = time.time()

        while time.time() - start < max_wait:
            pending_count = sum(
                1 for c in doc.chunks.values()
                if c.status not in (ChunkStatus.INDEXED, ChunkStatus.FAILED)
            )
            if pending_count == 0:
                break
            await asyncio.sleep(0.1)

        # Update final status
        doc.total_chunks = len(doc.chunks)
        await self._update_document_status(document_id)

        logger.info(
            "Streaming document finalized",
            document_id=document_id,
            total_chunks=doc.total_chunks,
            indexed_chunks=doc.indexed_chunks,
            status=doc.status.value,
        )

        return doc

    async def get_document_status(
        self,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get current status of streaming document."""
        doc = self._documents.get(document_id)
        if not doc:
            return None

        return {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "status": doc.status.value,
            "total_chunks": doc.total_chunks,
            "received_chunks": doc.received_chunks,
            "indexed_chunks": doc.indexed_chunks,
            "is_query_ready": doc.status in (
                StreamingStatus.PARTIALLY_READY,
                StreamingStatus.COMPLETE,
            ),
            "created_at": doc.created_at.isoformat(),
            "first_query_ready_at": (
                doc.first_query_ready_at.isoformat()
                if doc.first_query_ready_at else None
            ),
            "completed_at": (
                doc.completed_at.isoformat()
                if doc.completed_at else None
            ),
        }

    async def cancel_streaming(self, document_id: str) -> bool:
        """Cancel an ongoing streaming upload."""
        doc = self._documents.get(document_id)
        if not doc:
            return False

        doc.status = StreamingStatus.CANCELLED

        await self._event_bus.publish(StreamingEvent(
            event_type="document.cancelled",
            document_id=document_id,
        ))

        return True


# =============================================================================
# Partial Query Service
# =============================================================================

class PartialQueryService:
    """
    Service for querying partially indexed documents.

    Features:
    - Query documents before fully indexed
    - Confidence scoring based on completeness
    - Progressive result improvement
    """

    def __init__(self, ingestion_service: StreamingIngestionService):
        self.ingestion_service = ingestion_service
        self._event_bus = get_event_bus()

    async def query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        min_chunks_available: int = 1,
        include_metadata: bool = True,
    ) -> PartialQueryResult:
        """
        Query partially indexed documents.

        Args:
            query: Search query
            document_ids: Filter to specific documents
            top_k: Number of results to return
            min_chunks_available: Minimum chunks required
            include_metadata: Include chunk metadata

        Returns:
            PartialQueryResult with confidence score
        """
        # Filter documents that are ready for query
        queryable_docs = []
        for doc_id, doc in self.ingestion_service._documents.items():
            if document_ids and doc_id not in document_ids:
                continue
            if doc.indexed_chunks >= min_chunks_available:
                queryable_docs.append(doc)

        if not queryable_docs:
            return PartialQueryResult(
                query=query,
                document_id="",
                chunks_available=0,
                chunks_total=None,
                is_complete=False,
                results=[],
                confidence=0.0,
                metadata={"error": "No queryable documents"},
            )

        # Perform search
        try:
            # Use factory function for proper initialization (thread-safe, properly configured)
            from backend.services.rag import get_rag_service

            rag = get_rag_service()
            doc_id_list = [d.document_id for d in queryable_docs]

            results = await rag.search(
                query=query,
                top_k=top_k,
                document_ids=doc_id_list,
                include_metadata=include_metadata,
            )

            # Calculate confidence based on completeness
            total_chunks = sum(d.total_chunks or 0 for d in queryable_docs)
            indexed_chunks = sum(d.indexed_chunks for d in queryable_docs)

            if total_chunks > 0:
                completeness = indexed_chunks / total_chunks
            else:
                completeness = 0.5  # Unknown total

            # Confidence is lower for partial results
            base_confidence = 1.0 if total_chunks and indexed_chunks >= total_chunks else 0.5
            confidence = base_confidence * completeness

            return PartialQueryResult(
                query=query,
                document_id=doc_id_list[0] if len(doc_id_list) == 1 else "",
                chunks_available=indexed_chunks,
                chunks_total=total_chunks if total_chunks > 0 else None,
                is_complete=all(
                    d.status == StreamingStatus.COMPLETE
                    for d in queryable_docs
                ),
                results=results,
                confidence=round(confidence, 3),
                metadata={
                    "document_count": len(queryable_docs),
                    "completeness": round(completeness, 3),
                },
            )

        except Exception as e:
            logger.error(f"Partial query failed: {e}")
            return PartialQueryResult(
                query=query,
                document_id="",
                chunks_available=0,
                chunks_total=None,
                is_complete=False,
                results=[],
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def subscribe_to_improvements(
        self,
        document_id: str,
        callback: Callable[[PartialQueryResult], Any],
    ) -> str:
        """
        Subscribe to query result improvements.

        The callback is called when more chunks become available.
        """
        return self._event_bus.subscribe(
            "chunk.indexed",
            callback,
            document_id=document_id,
        )


# =============================================================================
# Streaming Upload API Helpers
# =============================================================================

async def create_streaming_upload(
    filename: str,
    content_type: str,
    total_size: Optional[int] = None,
    total_chunks: Optional[int] = None,
    **kwargs,
) -> Tuple[str, StreamingIngestionService]:
    """
    Create a new streaming upload session.

    Returns:
        Tuple of (document_id, service)
    """
    service = StreamingIngestionService()

    document_id = await service.start_streaming(
        filename=filename,
        content_type=content_type,
        total_chunks=total_chunks,
        **kwargs,
    )

    return document_id, service


async def process_streaming_chunk(
    service: StreamingIngestionService,
    document_id: str,
    chunk_data: bytes,
    chunk_index: int,
    is_final: bool = False,
) -> Dict[str, Any]:
    """
    Process a streaming chunk.

    Returns status update.
    """
    chunk = await service.receive_chunk(
        document_id=document_id,
        chunk_data=chunk_data,
        chunk_index=chunk_index,
        is_final=is_final,
    )

    status = await service.get_document_status(document_id)

    return {
        "chunk_id": chunk.chunk_id,
        "chunk_status": chunk.status.value,
        "document_status": status,
    }


# =============================================================================
# Streaming Response Generator
# =============================================================================

async def stream_query_results(
    query: str,
    document_id: str,
    service: StreamingIngestionService,
    poll_interval: float = 0.5,
    max_polls: int = 100,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream query results as they become available.

    Yields results with improving confidence as more chunks are indexed.
    """
    partial_service = PartialQueryService(service)
    last_indexed = 0

    for _ in range(max_polls):
        status = await service.get_document_status(document_id)

        if not status:
            yield {"error": "Document not found"}
            return

        current_indexed = status.get("indexed_chunks", 0)

        # Only query if new chunks are available
        if current_indexed > last_indexed:
            result = await partial_service.query(
                query=query,
                document_ids=[document_id],
            )

            yield {
                "type": "result",
                "result": {
                    "results": result.results,
                    "confidence": result.confidence,
                    "chunks_available": result.chunks_available,
                    "is_complete": result.is_complete,
                },
            }

            last_indexed = current_indexed

            if result.is_complete:
                yield {"type": "complete", "message": "All chunks indexed"}
                return

        # Yield progress update
        yield {
            "type": "progress",
            "status": status.get("status"),
            "indexed_chunks": current_indexed,
            "total_chunks": status.get("total_chunks"),
        }

        await asyncio.sleep(poll_interval)

    yield {"type": "timeout", "message": "Max polls reached"}


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "StreamingStatus",
    "ChunkStatus",
    "StreamingChunk",
    "StreamingDocument",
    "StreamingEvent",
    "PartialQueryResult",
    "StreamingEventBus",
    "get_event_bus",
    "StreamingIngestionService",
    "PartialQueryService",
    "create_streaming_upload",
    "process_streaming_chunk",
    "stream_query_results",
]
