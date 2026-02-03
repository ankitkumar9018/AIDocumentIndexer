"""
WebSocket API Routes
====================

Real-time WebSocket endpoints for live updates.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from backend.services.websocket_manager import (
    get_websocket_manager,
    WebSocketMessage,
    EventType,
)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    Main WebSocket endpoint for real-time updates.

    Query Parameters:
        token: Optional authentication token

    Message Format (client -> server):
        {
            "action": "subscribe" | "unsubscribe" | "ping",
            "channel": "document:123" | "kg:456" | "chat:789",
            "timestamp": "2024-01-01T00:00:00Z"
        }

    Message Format (server -> client):
        {
            "event": "document.processing" | "chat.chunk" | ...,
            "data": { ... },
            "timestamp": "2024-01-01T00:00:00Z",
            "channel": "document:123"
        }

    Channels:
        - document:{id} - Document processing updates
        - kg:{job_id} - KG extraction progress
        - chat:{conversation_id} - Chat streaming
        - collection:{id} - Collection updates
        - system - System-wide events
    """
    manager = get_websocket_manager()
    connection_id = str(uuid.uuid4())

    # TODO: Validate token and get user_id
    user_id = None

    await manager.handle_connection(websocket, connection_id, user_id)


@router.websocket("/ws/chat/{conversation_id}")
async def chat_websocket(
    websocket: WebSocket,
    conversation_id: str,
    token: Optional[str] = Query(None),
):
    """
    Dedicated WebSocket endpoint for chat streaming.

    Automatically subscribes to chat:{conversation_id} channel.
    """
    manager = get_websocket_manager()
    connection_id = str(uuid.uuid4())

    await manager.connect(websocket, connection_id)
    manager.subscribe(connection_id, f"chat:{conversation_id}")

    try:
        while True:
            data = await websocket.receive_json()

            # Handle ping
            if data.get("action") == "ping":
                await manager.send_to_connection(
                    connection_id,
                    WebSocketMessage(
                        event="pong",
                        data={"client_time": data.get("timestamp")},
                    ),
                )

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(connection_id)


@router.websocket("/ws/documents")
async def documents_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for document processing updates.

    Subscribes to all document events by default.
    Send subscription messages to filter specific documents.
    """
    manager = get_websocket_manager()
    connection_id = str(uuid.uuid4())

    await manager.connect(websocket, connection_id)
    manager.subscribe(connection_id, "documents")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "subscribe" and data.get("document_id"):
                manager.subscribe(
                    connection_id,
                    f"document:{data['document_id']}",
                )

            elif data.get("action") == "unsubscribe" and data.get("document_id"):
                manager.unsubscribe(
                    connection_id,
                    f"document:{data['document_id']}",
                )

            elif data.get("action") == "ping":
                await manager.send_to_connection(
                    connection_id,
                    WebSocketMessage(
                        event="pong",
                        data={},
                    ),
                )

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(connection_id)


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket server status."""
    manager = get_websocket_manager()
    return {
        "connections": manager.connection_count,
        "status": "online",
    }
