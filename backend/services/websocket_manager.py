"""
WebSocket Manager for Real-Time Updates
========================================

Provides real-time notifications for:
- Document processing progress
- KG extraction status
- Chat streaming
- System events
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from weakref import WeakSet

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of WebSocket events."""
    # Document events
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSING = "document.processing"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_FAILED = "document.failed"
    DOCUMENT_DELETED = "document.deleted"

    # Chunk events
    CHUNK_CREATED = "chunk.created"
    CHUNK_EMBEDDED = "chunk.embedded"

    # KG events
    KG_EXTRACTION_STARTED = "kg.extraction.started"
    KG_EXTRACTION_PROGRESS = "kg.extraction.progress"
    KG_EXTRACTION_COMPLETED = "kg.extraction.completed"
    KG_EXTRACTION_FAILED = "kg.extraction.failed"
    KG_ENTITY_CREATED = "kg.entity.created"
    KG_RELATIONSHIP_CREATED = "kg.relationship.created"

    # Chat events
    CHAT_STARTED = "chat.started"
    CHAT_CHUNK = "chat.chunk"
    CHAT_SOURCES = "chat.sources"
    CHAT_COMPLETED = "chat.completed"
    CHAT_ERROR = "chat.error"

    # Collection events
    COLLECTION_CREATED = "collection.created"
    COLLECTION_UPDATED = "collection.updated"
    COLLECTION_DELETED = "collection.deleted"

    # System events
    SYSTEM_STATUS = "system.status"
    SYSTEM_ERROR = "system.error"
    CONNECTION_ACK = "connection.ack"
    HEARTBEAT = "heartbeat"


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    event: str
    data: dict
    timestamp: str = ""
    channel: Optional[str] = None
    request_id: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class ConnectionInfo(BaseModel):
    """Information about a WebSocket connection."""
    id: str
    user_id: Optional[str] = None
    subscriptions: set[str] = set()
    connected_at: datetime = None

    class Config:
        arbitrary_types_allowed = True


class WebSocketManager:
    """
    Manages WebSocket connections and message broadcasting.

    Features:
    - Connection tracking
    - Channel-based subscriptions
    - Broadcast to all or specific channels
    - Heartbeat monitoring
    - Automatic reconnection handling
    """

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._connection_info: dict[str, ConnectionInfo] = {}
        self._channel_subscribers: dict[str, set[str]] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._broadcast_task: Optional[asyncio.Task] = None

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)

    @property
    def connections(self) -> list[str]:
        """Get list of connection IDs."""
        return list(self._connections.keys())

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None,
    ) -> ConnectionInfo:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            connection_id: Unique identifier for this connection
            user_id: Optional user identifier

        Returns:
            ConnectionInfo for the new connection
        """
        await websocket.accept()

        info = ConnectionInfo(
            id=connection_id,
            user_id=user_id,
            connected_at=datetime.utcnow(),
        )

        self._connections[connection_id] = websocket
        self._connection_info[connection_id] = info

        logger.info(f"WebSocket connected: {connection_id}")

        # Send acknowledgment
        await self.send_to_connection(
            connection_id,
            WebSocketMessage(
                event=EventType.CONNECTION_ACK,
                data={
                    "connection_id": connection_id,
                    "server_time": datetime.utcnow().isoformat(),
                },
            ),
        )

        return info

    async def disconnect(self, connection_id: str):
        """
        Disconnect and clean up a WebSocket connection.

        Args:
            connection_id: ID of connection to disconnect
        """
        if connection_id in self._connections:
            # Remove from all channels
            info = self._connection_info.get(connection_id)
            if info:
                for channel in info.subscriptions:
                    self._unsubscribe_from_channel(connection_id, channel)

            del self._connections[connection_id]

        if connection_id in self._connection_info:
            del self._connection_info[connection_id]

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def close_connection(self, connection_id: str, code: int = 1000):
        """Close a connection gracefully."""
        if connection_id in self._connections:
            websocket = self._connections[connection_id]
            try:
                await websocket.close(code)
            except Exception:
                pass
            await self.disconnect(connection_id)

    # =========================================================================
    # Channel Subscriptions
    # =========================================================================

    def subscribe(self, connection_id: str, channel: str):
        """
        Subscribe a connection to a channel.

        Args:
            connection_id: Connection to subscribe
            channel: Channel name to subscribe to
        """
        if channel not in self._channel_subscribers:
            self._channel_subscribers[channel] = set()

        self._channel_subscribers[channel].add(connection_id)

        if connection_id in self._connection_info:
            self._connection_info[connection_id].subscriptions.add(channel)

        logger.debug(f"Connection {connection_id} subscribed to {channel}")

    def unsubscribe(self, connection_id: str, channel: str):
        """
        Unsubscribe a connection from a channel.

        Args:
            connection_id: Connection to unsubscribe
            channel: Channel name to unsubscribe from
        """
        self._unsubscribe_from_channel(connection_id, channel)

        if connection_id in self._connection_info:
            self._connection_info[connection_id].subscriptions.discard(channel)

    def _unsubscribe_from_channel(self, connection_id: str, channel: str):
        """Internal method to remove connection from channel."""
        if channel in self._channel_subscribers:
            self._channel_subscribers[channel].discard(connection_id)
            if not self._channel_subscribers[channel]:
                del self._channel_subscribers[channel]

    def get_channel_subscribers(self, channel: str) -> set[str]:
        """Get all connections subscribed to a channel."""
        return self._channel_subscribers.get(channel, set())

    # =========================================================================
    # Message Sending
    # =========================================================================

    async def send_to_connection(
        self,
        connection_id: str,
        message: WebSocketMessage,
    ):
        """
        Send a message to a specific connection.

        Args:
            connection_id: Target connection
            message: Message to send
        """
        if connection_id not in self._connections:
            return

        websocket = self._connections[connection_id]
        try:
            await websocket.send_json(message.model_dump())
        except Exception as e:
            logger.error(f"Failed to send to {connection_id}: {e}")
            await self.disconnect(connection_id)

    async def broadcast(
        self,
        message: WebSocketMessage,
        exclude: Optional[set[str]] = None,
    ):
        """
        Broadcast a message to all connections.

        Args:
            message: Message to broadcast
            exclude: Connection IDs to exclude
        """
        exclude = exclude or set()
        tasks = []

        for conn_id in list(self._connections.keys()):
            if conn_id not in exclude:
                tasks.append(self.send_to_connection(conn_id, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_to_channel(
        self,
        channel: str,
        message: WebSocketMessage,
    ):
        """
        Broadcast a message to all subscribers of a channel.

        Args:
            channel: Target channel
            message: Message to broadcast
        """
        message.channel = channel
        subscribers = self.get_channel_subscribers(channel)
        tasks = [
            self.send_to_connection(conn_id, message)
            for conn_id in subscribers
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to_user(
        self,
        user_id: str,
        message: WebSocketMessage,
    ):
        """
        Send a message to all connections of a specific user.

        Args:
            user_id: Target user
            message: Message to send
        """
        tasks = []
        for conn_id, info in self._connection_info.items():
            if info.user_id == user_id:
                tasks.append(self.send_to_connection(conn_id, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # =========================================================================
    # Event Emission Helpers
    # =========================================================================

    async def emit_document_progress(
        self,
        document_id: str,
        progress: float,
        status: str,
        message: Optional[str] = None,
    ):
        """Emit document processing progress."""
        await self.broadcast_to_channel(
            f"document:{document_id}",
            WebSocketMessage(
                event=EventType.DOCUMENT_PROCESSING,
                data={
                    "document_id": document_id,
                    "progress": progress,
                    "status": status,
                    "message": message,
                },
            ),
        )

    async def emit_kg_progress(
        self,
        job_id: str,
        progress: float,
        entities_count: int,
        relationships_count: int,
    ):
        """Emit KG extraction progress."""
        await self.broadcast_to_channel(
            f"kg:{job_id}",
            WebSocketMessage(
                event=EventType.KG_EXTRACTION_PROGRESS,
                data={
                    "job_id": job_id,
                    "progress": progress,
                    "entities_count": entities_count,
                    "relationships_count": relationships_count,
                },
            ),
        )

    async def emit_chat_chunk(
        self,
        conversation_id: str,
        chunk: str,
        request_id: Optional[str] = None,
    ):
        """Emit chat streaming chunk."""
        await self.broadcast_to_channel(
            f"chat:{conversation_id}",
            WebSocketMessage(
                event=EventType.CHAT_CHUNK,
                data={"chunk": chunk},
                request_id=request_id,
            ),
        )

    async def emit_system_status(
        self,
        component: str,
        status: str,
        details: Optional[dict] = None,
    ):
        """Emit system status update."""
        await self.broadcast(
            WebSocketMessage(
                event=EventType.SYSTEM_STATUS,
                data={
                    "component": component,
                    "status": status,
                    "details": details or {},
                },
            ),
        )

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def start_heartbeat(self, interval: float = 30.0):
        """Start periodic heartbeat to all connections."""
        async def heartbeat_loop():
            while True:
                await asyncio.sleep(interval)
                await self.broadcast(
                    WebSocketMessage(
                        event=EventType.HEARTBEAT,
                        data={"server_time": datetime.utcnow().isoformat()},
                    ),
                )

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def stop_heartbeat(self):
        """Stop heartbeat task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def handle_connection(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None,
    ):
        """
        Handle a WebSocket connection lifecycle.

        Args:
            websocket: The WebSocket connection
            connection_id: Unique connection identifier
            user_id: Optional user identifier
        """
        await self.connect(websocket, connection_id, user_id)

        try:
            while True:
                # Receive and process messages
                data = await websocket.receive_json()

                # Handle subscription messages
                if data.get("action") == "subscribe":
                    channel = data.get("channel")
                    if channel:
                        self.subscribe(connection_id, channel)
                        await self.send_to_connection(
                            connection_id,
                            WebSocketMessage(
                                event="subscription.confirmed",
                                data={"channel": channel},
                            ),
                        )

                elif data.get("action") == "unsubscribe":
                    channel = data.get("channel")
                    if channel:
                        self.unsubscribe(connection_id, channel)

                # Handle ping
                elif data.get("action") == "ping":
                    await self.send_to_connection(
                        connection_id,
                        WebSocketMessage(
                            event="pong",
                            data={"client_time": data.get("timestamp")},
                        ),
                    )

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            await self.disconnect(connection_id)


# Global WebSocket manager instance
_ws_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


async def emit_event(
    event: EventType,
    data: dict,
    channel: Optional[str] = None,
):
    """
    Convenience function to emit an event.

    Args:
        event: Event type
        data: Event data
        channel: Optional channel to broadcast to
    """
    manager = get_websocket_manager()
    message = WebSocketMessage(event=event, data=data)

    if channel:
        await manager.broadcast_to_channel(channel, message)
    else:
        await manager.broadcast(message)
