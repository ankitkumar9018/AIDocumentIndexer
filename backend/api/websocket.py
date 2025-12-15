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
    ) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            user_id: Optional authenticated user ID
            metadata: Optional connection metadata
        """
        await websocket.accept()

        async with self._lock:
            self._all_connections.add(websocket)

            # Store connection info
            self._connection_info[websocket] = {
                "user_id": user_id,
                "connected_at": datetime.now(),
                "subscriptions": set(),
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
            total_connections=len(self._all_connections),
        )

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
    ) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: The message to broadcast
            exclude: Optional set of connections to exclude

        Returns:
            Number of connections message was sent to
        """
        exclude = exclude or set()
        sent_count = 0

        async with self._lock:
            connections = self._all_connections - exclude

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

        logger.debug("Broadcast message", recipients=sent_count)

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

    # Server -> Client
    FILE_UPDATE = "file_update"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"
    SYSTEM_NOTIFICATION = "system_notification"
    PONG = "pong"
    ERROR = "error"


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
