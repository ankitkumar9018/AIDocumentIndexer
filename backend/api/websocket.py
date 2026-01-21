"""
AIDocumentIndexer - WebSocket Manager
=====================================

Manages WebSocket connections for real-time updates on document processing,
chat streaming, and system notifications.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Set, Any, Callable
from uuid import UUID
import json

from fastapi import WebSocket, WebSocketDisconnect
import structlog

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time communication.

    Supports:
    - User-specific connections (authenticated users)
    - File-specific subscriptions (processing updates)
    - Broadcast messages (system-wide notifications)
    """

    def __init__(self):
        # Active connections by user_id
        self._user_connections: Dict[str, Set[WebSocket]] = {}

        # Connections subscribed to specific file processing updates
        self._file_subscriptions: Dict[str, Set[WebSocket]] = {}

        # All active connections (for broadcasts)
        self._all_connections: Set[WebSocket] = set()

        # Connection metadata
        self._connection_info: Dict[WebSocket, Dict[str, Any]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        require_auth: bool = False,
    ) -> bool:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            user_id: Optional authenticated user ID
            metadata: Optional connection metadata
            require_auth: If True, reject connections without user_id

        Returns:
            True if connection was accepted, False if rejected
        """
        # SECURITY: Reject unauthenticated connections if auth is required
        if require_auth and not user_id:
            logger.warning(
                "WebSocket connection rejected - authentication required",
                metadata=metadata,
            )
            await websocket.close(code=4001, reason="Authentication required")
            return False

        await websocket.accept()

        # Track whether this is an authenticated connection
        is_authenticated = user_id is not None

        async with self._lock:
            self._all_connections.add(websocket)

            # Store connection info
            self._connection_info[websocket] = {
                "user_id": user_id,
                "connected_at": datetime.now(),
                "subscriptions": set(),
                "is_authenticated": is_authenticated,
                **(metadata or {}),
            }

            # Add to user connections if authenticated
            if user_id:
                if user_id not in self._user_connections:
                    self._user_connections[user_id] = set()
                self._user_connections[user_id].add(websocket)

        logger.info(
            "WebSocket connected",
            user_id=user_id,
            is_authenticated=is_authenticated,
            total_connections=len(self._all_connections),
        )
        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: The disconnected WebSocket
        """
        async with self._lock:
            # Remove from all connections
            self._all_connections.discard(websocket)

            # Get connection info
            info = self._connection_info.pop(websocket, {})
            user_id = info.get("user_id")
            subscriptions = info.get("subscriptions", set())

            # Remove from user connections
            if user_id and user_id in self._user_connections:
                self._user_connections[user_id].discard(websocket)
                if not self._user_connections[user_id]:
                    del self._user_connections[user_id]

            # Remove from file subscriptions
            for file_id in subscriptions:
                if file_id in self._file_subscriptions:
                    self._file_subscriptions[file_id].discard(websocket)
                    if not self._file_subscriptions[file_id]:
                        del self._file_subscriptions[file_id]

        logger.info(
            "WebSocket disconnected",
            user_id=user_id,
            remaining_connections=len(self._all_connections),
        )

    async def subscribe_to_file(self, websocket: WebSocket, file_id: str) -> None:
        """
        Subscribe a connection to file processing updates.

        Args:
            websocket: The WebSocket connection
            file_id: The file ID to subscribe to
        """
        async with self._lock:
            if file_id not in self._file_subscriptions:
                self._file_subscriptions[file_id] = set()
            self._file_subscriptions[file_id].add(websocket)

            # Track subscription in connection info
            if websocket in self._connection_info:
                self._connection_info[websocket]["subscriptions"].add(file_id)

        logger.debug("Subscribed to file updates", file_id=file_id)

    async def unsubscribe_from_file(self, websocket: WebSocket, file_id: str) -> None:
        """
        Unsubscribe a connection from file processing updates.

        Args:
            websocket: The WebSocket connection
            file_id: The file ID to unsubscribe from
        """
        async with self._lock:
            if file_id in self._file_subscriptions:
                self._file_subscriptions[file_id].discard(websocket)
                if not self._file_subscriptions[file_id]:
                    del self._file_subscriptions[file_id]

            # Remove from connection info
            if websocket in self._connection_info:
                self._connection_info[websocket]["subscriptions"].discard(file_id)

        logger.debug("Unsubscribed from file updates", file_id=file_id)

    async def send_personal_message(
        self,
        message: Dict[str, Any],
        websocket: WebSocket,
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            message: The message to send
            websocket: The target WebSocket

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error("Failed to send personal message", error=str(e))
            return False

    async def send_to_user(
        self,
        message: Dict[str, Any],
        user_id: str,
    ) -> int:
        """
        Send a message to all connections of a specific user.

        Args:
            message: The message to send
            user_id: The target user ID

        Returns:
            Number of connections message was sent to
        """
        sent_count = 0

        async with self._lock:
            connections = self._user_connections.get(user_id, set()).copy()

        for websocket in connections:
            if await self.send_personal_message(message, websocket):
                sent_count += 1

        return sent_count

    async def send_file_update(
        self,
        file_id: str,
        status: str,
        progress: int,
        current_step: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Send a processing update to all subscribers of a file.

        Args:
            file_id: The file being processed
            status: Current processing status
            progress: Progress percentage (0-100)
            current_step: Description of current step
            error: Optional error message
            metadata: Optional additional metadata

        Returns:
            Number of connections message was sent to
        """
        message = {
            "type": "file_update",
            "file_id": file_id,
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        sent_count = 0

        async with self._lock:
            connections = self._file_subscriptions.get(file_id, set()).copy()

        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            await self.disconnect(ws)

        logger.debug(
            "Sent file update",
            file_id=file_id,
            status=status,
            recipients=sent_count,
        )

        return sent_count

    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude: Optional[Set[WebSocket]] = None,
        authenticated_only: bool = False,
    ) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: The message to broadcast
            exclude: Optional set of connections to exclude
            authenticated_only: If True, only send to authenticated connections

        Returns:
            Number of connections message was sent to
        """
        exclude = exclude or set()
        sent_count = 0

        async with self._lock:
            connections = self._all_connections - exclude
            # Filter to authenticated only if requested
            if authenticated_only:
                connections = {
                    ws for ws in connections
                    if self._connection_info.get(ws, {}).get("is_authenticated", False)
                }

        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            await self.disconnect(ws)

        logger.debug(
            "Broadcast message",
            recipients=sent_count,
            authenticated_only=authenticated_only,
        )

        return sent_count

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self._all_connections)

    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a specific user."""
        return len(self._user_connections.get(user_id, set()))

    def get_file_subscriber_count(self, file_id: str) -> int:
        """Get number of subscribers for a specific file."""
        return len(self._file_subscriptions.get(file_id, set()))

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self._all_connections),
            "users_connected": len(self._user_connections),
            "files_with_subscribers": len(self._file_subscriptions),
        }


# Global connection manager instance
manager = ConnectionManager()


# =============================================================================
# Message Types
# =============================================================================

class MessageType:
    """WebSocket message types."""
    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"

    # Server -> Client - File Processing
    FILE_UPDATE = "file_update"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"
    SYSTEM_NOTIFICATION = "system_notification"
    PONG = "pong"
    ERROR = "error"

    # Server -> Client - Phase 2 Event Types
    # Web Scraping
    SCRAPE_STARTED = "scrape.started"
    SCRAPE_PROGRESS = "scrape.progress"
    SCRAPE_PAGE_COMPLETE = "scrape.page_complete"
    SCRAPE_COMPLETE = "scrape.complete"
    SCRAPE_ERROR = "scrape.error"

    # Document Generation
    GENERATION_STARTED = "generation.started"
    GENERATION_PROGRESS = "generation.progress"
    GENERATION_SECTION_COMPLETE = "generation.section_complete"
    GENERATION_COMPLETE = "generation.complete"
    GENERATION_ERROR = "generation.error"

    # Knowledge Graph Extraction
    KG_EXTRACTION_STARTED = "kg.extraction.started"
    KG_EXTRACTION_PROGRESS = "kg.extraction.progress"
    KG_ENTITY_EXTRACTED = "kg.entity.extracted"
    KG_EXTRACTION_COMPLETE = "kg.extraction.complete"
    KG_EXTRACTION_ERROR = "kg.extraction.error"

    # Audio Synthesis
    AUDIO_SYNTHESIS_STARTED = "audio.synthesis.started"
    AUDIO_SYNTHESIS_PROGRESS = "audio.synthesis.progress"
    AUDIO_SECTION_COMPLETE = "audio.section_complete"
    AUDIO_SYNTHESIS_COMPLETE = "audio.synthesis.complete"
    AUDIO_SYNTHESIS_ERROR = "audio.synthesis.error"

    # Embedding/Indexing
    EMBEDDING_STARTED = "embedding.started"
    EMBEDDING_PROGRESS = "embedding.progress"
    EMBEDDING_COMPLETE = "embedding.complete"
    EMBEDDING_ERROR = "embedding.error"

    # Chat Streaming
    CHAT_MESSAGE_START = "chat.message.start"
    CHAT_MESSAGE_STREAM = "chat.message.stream"
    CHAT_MESSAGE_COMPLETE = "chat.message.complete"
    CHAT_SOURCE_UPDATE = "chat.source.update"

    # Fact Checking
    FACT_CHECK_STARTED = "factcheck.started"
    FACT_CHECK_PROGRESS = "factcheck.progress"
    FACT_CHECK_COMPLETE = "factcheck.complete"


# =============================================================================
# WebSocket Handlers
# =============================================================================

async def handle_websocket_message(
    websocket: WebSocket,
    message: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Handle incoming WebSocket messages.

    Args:
        websocket: The WebSocket connection
        message: The received message

    Returns:
        Optional response message
    """
    msg_type = message.get("type")

    if msg_type == MessageType.SUBSCRIBE:
        file_id = message.get("file_id")
        if file_id:
            await manager.subscribe_to_file(websocket, file_id)
            return {
                "type": "subscribed",
                "file_id": file_id,
                "message": f"Subscribed to updates for file {file_id}",
            }
        return {
            "type": MessageType.ERROR,
            "message": "file_id required for subscription",
        }

    elif msg_type == MessageType.UNSUBSCRIBE:
        file_id = message.get("file_id")
        if file_id:
            await manager.unsubscribe_from_file(websocket, file_id)
            return {
                "type": "unsubscribed",
                "file_id": file_id,
                "message": f"Unsubscribed from updates for file {file_id}",
            }
        return {
            "type": MessageType.ERROR,
            "message": "file_id required for unsubscription",
        }

    elif msg_type == MessageType.PING:
        return {
            "type": MessageType.PONG,
            "timestamp": datetime.now().isoformat(),
        }

    else:
        return {
            "type": MessageType.ERROR,
            "message": f"Unknown message type: {msg_type}",
        }


# =============================================================================
# Helper Functions for Processing Updates
# =============================================================================

async def notify_processing_started(file_id: str, filename: str) -> None:
    """Notify subscribers that processing has started."""
    await manager.send_file_update(
        file_id=file_id,
        status="processing",
        progress=0,
        current_step="Starting processing",
        metadata={"filename": filename},
    )


async def notify_processing_progress(
    file_id: str,
    status: str,
    progress: int,
    current_step: str,
) -> None:
    """Notify subscribers of processing progress."""
    await manager.send_file_update(
        file_id=file_id,
        status=status,
        progress=progress,
        current_step=current_step,
    )


async def notify_processing_complete(
    file_id: str,
    document_id: str,
    chunk_count: int,
    word_count: int,
) -> None:
    """Notify subscribers that processing is complete."""
    await manager.send_file_update(
        file_id=file_id,
        status="completed",
        progress=100,
        current_step="Processing complete",
        metadata={
            "document_id": document_id,
            "chunk_count": chunk_count,
            "word_count": word_count,
        },
    )


async def notify_processing_error(
    file_id: str,
    error_message: str,
) -> None:
    """Notify subscribers of a processing error."""
    await manager.send_file_update(
        file_id=file_id,
        status="failed",
        progress=0,
        current_step="Processing failed",
        error=error_message,
    )


# =============================================================================
# Phase 2 Progress Notifications
# =============================================================================

async def broadcast_progress(
    event_type: str,
    job_id: str,
    progress: float,
    message: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Broadcast progress update for any operation.

    Args:
        event_type: Type of event (from MessageType)
        job_id: Unique identifier for the job/operation
        progress: Progress percentage (0-100)
        message: Human-readable progress message
        user_id: Optional user ID to send to specific user
        metadata: Optional additional metadata

    Returns:
        Number of recipients
    """
    payload = {
        "type": event_type,
        "job_id": job_id,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        **(metadata or {}),
    }

    if user_id:
        return await manager.send_to_user(payload, user_id)
    else:
        return await manager.broadcast(payload)


# Web Scraping Events
async def notify_scrape_started(
    job_id: str,
    url: str,
    total_pages: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify that web scraping has started."""
    return await broadcast_progress(
        event_type=MessageType.SCRAPE_STARTED,
        job_id=job_id,
        progress=0,
        message=f"Starting to scrape {url}",
        user_id=user_id,
        metadata={"url": url, "total_pages": total_pages},
    )


async def notify_scrape_progress(
    job_id: str,
    current_page: int,
    total_pages: int,
    current_url: str,
    user_id: Optional[str] = None,
) -> int:
    """Notify scraping progress."""
    progress = (current_page / total_pages) * 100 if total_pages > 0 else 0
    return await broadcast_progress(
        event_type=MessageType.SCRAPE_PROGRESS,
        job_id=job_id,
        progress=progress,
        message=f"Scraping page {current_page}/{total_pages}",
        user_id=user_id,
        metadata={"current_url": current_url, "current_page": current_page, "total_pages": total_pages},
    )


async def notify_scrape_complete(
    job_id: str,
    pages_scraped: int,
    documents_created: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify that scraping is complete."""
    return await broadcast_progress(
        event_type=MessageType.SCRAPE_COMPLETE,
        job_id=job_id,
        progress=100,
        message=f"Scraping complete: {pages_scraped} pages, {documents_created} documents",
        user_id=user_id,
        metadata={"pages_scraped": pages_scraped, "documents_created": documents_created},
    )


# Document Generation Events
async def notify_generation_started(
    job_id: str,
    document_type: str,
    total_sections: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify that document generation has started."""
    return await broadcast_progress(
        event_type=MessageType.GENERATION_STARTED,
        job_id=job_id,
        progress=0,
        message=f"Starting {document_type} generation",
        user_id=user_id,
        metadata={"document_type": document_type, "total_sections": total_sections},
    )


async def notify_generation_progress(
    job_id: str,
    current_section: int,
    total_sections: int,
    section_name: str,
    user_id: Optional[str] = None,
) -> int:
    """Notify generation progress."""
    progress = (current_section / total_sections) * 100 if total_sections > 0 else 0
    return await broadcast_progress(
        event_type=MessageType.GENERATION_PROGRESS,
        job_id=job_id,
        progress=progress,
        message=f"Generating section: {section_name}",
        user_id=user_id,
        metadata={"current_section": current_section, "total_sections": total_sections, "section_name": section_name},
    )


async def notify_generation_complete(
    job_id: str,
    output_path: str,
    document_type: str,
    user_id: Optional[str] = None,
) -> int:
    """Notify that generation is complete."""
    return await broadcast_progress(
        event_type=MessageType.GENERATION_COMPLETE,
        job_id=job_id,
        progress=100,
        message=f"{document_type} generation complete",
        user_id=user_id,
        metadata={"output_path": output_path, "document_type": document_type},
    )


# Knowledge Graph Events
async def notify_kg_extraction_started(
    job_id: str,
    document_id: str,
    document_name: str,
    user_id: Optional[str] = None,
) -> int:
    """Notify that KG extraction has started."""
    return await broadcast_progress(
        event_type=MessageType.KG_EXTRACTION_STARTED,
        job_id=job_id,
        progress=0,
        message=f"Extracting entities from {document_name}",
        user_id=user_id,
        metadata={"document_id": document_id, "document_name": document_name},
    )


async def notify_kg_extraction_progress(
    job_id: str,
    entities_found: int,
    relations_found: int,
    progress: float,
    user_id: Optional[str] = None,
) -> int:
    """Notify KG extraction progress."""
    return await broadcast_progress(
        event_type=MessageType.KG_EXTRACTION_PROGRESS,
        job_id=job_id,
        progress=progress,
        message=f"Found {entities_found} entities, {relations_found} relations",
        user_id=user_id,
        metadata={"entities_found": entities_found, "relations_found": relations_found},
    )


async def notify_kg_extraction_complete(
    job_id: str,
    total_entities: int,
    total_relations: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify that KG extraction is complete."""
    return await broadcast_progress(
        event_type=MessageType.KG_EXTRACTION_COMPLETE,
        job_id=job_id,
        progress=100,
        message=f"Extraction complete: {total_entities} entities, {total_relations} relations",
        user_id=user_id,
        metadata={"total_entities": total_entities, "total_relations": total_relations},
    )


# Audio Synthesis Events
async def notify_audio_synthesis_started(
    job_id: str,
    total_segments: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify that audio synthesis has started."""
    return await broadcast_progress(
        event_type=MessageType.AUDIO_SYNTHESIS_STARTED,
        job_id=job_id,
        progress=0,
        message=f"Starting audio synthesis ({total_segments} segments)",
        user_id=user_id,
        metadata={"total_segments": total_segments},
    )


async def notify_audio_synthesis_progress(
    job_id: str,
    current_segment: int,
    total_segments: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify audio synthesis progress."""
    progress = (current_segment / total_segments) * 100 if total_segments > 0 else 0
    return await broadcast_progress(
        event_type=MessageType.AUDIO_SYNTHESIS_PROGRESS,
        job_id=job_id,
        progress=progress,
        message=f"Synthesizing segment {current_segment}/{total_segments}",
        user_id=user_id,
        metadata={"current_segment": current_segment, "total_segments": total_segments},
    )


async def notify_audio_synthesis_complete(
    job_id: str,
    audio_path: str,
    duration_seconds: float,
    user_id: Optional[str] = None,
) -> int:
    """Notify that audio synthesis is complete."""
    return await broadcast_progress(
        event_type=MessageType.AUDIO_SYNTHESIS_COMPLETE,
        job_id=job_id,
        progress=100,
        message=f"Audio synthesis complete ({duration_seconds:.1f}s)",
        user_id=user_id,
        metadata={"audio_path": audio_path, "duration_seconds": duration_seconds},
    )


# Embedding Events
async def notify_embedding_started(
    job_id: str,
    total_chunks: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify that embedding generation has started."""
    return await broadcast_progress(
        event_type=MessageType.EMBEDDING_STARTED,
        job_id=job_id,
        progress=0,
        message=f"Generating embeddings for {total_chunks} chunks",
        user_id=user_id,
        metadata={"total_chunks": total_chunks},
    )


async def notify_embedding_progress(
    job_id: str,
    current_chunk: int,
    total_chunks: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify embedding progress."""
    progress = (current_chunk / total_chunks) * 100 if total_chunks > 0 else 0
    return await broadcast_progress(
        event_type=MessageType.EMBEDDING_PROGRESS,
        job_id=job_id,
        progress=progress,
        message=f"Embedding chunk {current_chunk}/{total_chunks}",
        user_id=user_id,
        metadata={"current_chunk": current_chunk, "total_chunks": total_chunks},
    )


async def notify_embedding_complete(
    job_id: str,
    total_embedded: int,
    user_id: Optional[str] = None,
) -> int:
    """Notify that embedding is complete."""
    return await broadcast_progress(
        event_type=MessageType.EMBEDDING_COMPLETE,
        job_id=job_id,
        progress=100,
        message=f"Embedded {total_embedded} chunks",
        user_id=user_id,
        metadata={"total_embedded": total_embedded},
    )
