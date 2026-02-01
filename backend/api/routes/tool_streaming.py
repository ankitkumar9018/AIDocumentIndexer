"""
AIDocumentIndexer - Tool Streaming API Routes
==============================================

SSE endpoints for streaming tool execution events in real-time.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tool-streaming", tags=["Tool Streaming"])


# =============================================================================
# Request Models
# =============================================================================

class StartExecutionRequest(BaseModel):
    """Request to start a streaming tool execution."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/start")
async def start_tool_execution(
    request: StartExecutionRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Start a streaming tool execution.

    Returns an execution ID that can be used to subscribe to events.
    """
    from backend.services.tool_streaming import get_tool_streaming_service

    service = get_tool_streaming_service()

    execution_id = await service.start_execution(
        tool_name=request.tool_name,
        arguments=request.arguments,
    )

    return {
        "execution_id": execution_id,
        "tool_name": request.tool_name,
        "status": "started",
        "subscribe_url": f"/api/v1/tool-streaming/subscribe/{execution_id}",
    }


@router.get("/subscribe/{execution_id}")
async def subscribe_to_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Subscribe to tool execution events via Server-Sent Events (SSE).

    Returns a stream of events as the tool executes.
    """
    from backend.services.tool_streaming import get_tool_streaming_service

    service = get_tool_streaming_service()

    execution = service.get_execution(execution_id)
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}",
        )

    async def event_generator():
        async for event in service.subscribe(execution_id):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status/{execution_id}")
async def get_execution_status(
    execution_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get current status of a tool execution.
    """
    from backend.services.tool_streaming import get_tool_streaming_service

    service = get_tool_streaming_service()

    execution = service.get_execution(execution_id)
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}",
        )

    return {
        "execution_id": execution.execution_id,
        "tool_name": execution.tool_name,
        "status": execution.status,
        "progress": execution.progress,
        "started_at": execution.started_at.isoformat(),
        "result": execution.result if execution.status == "completed" else None,
        "error": execution.error if execution.status == "failed" else None,
        "events_count": len(execution.events),
    }


@router.post("/progress/{execution_id}")
async def emit_progress(
    execution_id: str,
    progress: float,
    message: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Manually emit a progress update for an execution.

    Useful for long-running tools to report intermediate progress.
    """
    from backend.services.tool_streaming import get_tool_streaming_service

    service = get_tool_streaming_service()

    execution = service.get_execution(execution_id)
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}",
        )

    await service.emit_progress(
        execution_id=execution_id,
        progress=progress,
        message=message,
    )

    return {"status": "ok", "progress": progress}


@router.post("/complete/{execution_id}")
async def complete_execution(
    execution_id: str,
    result: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Mark an execution as complete with a result.
    """
    from backend.services.tool_streaming import get_tool_streaming_service

    service = get_tool_streaming_service()

    execution = service.get_execution(execution_id)
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}",
        )

    await service.complete_execution(
        execution_id=execution_id,
        result=result,
    )

    return {"status": "completed", "execution_id": execution_id}


@router.get("/active")
async def list_active_executions(
    current_user: dict = Depends(get_current_user),
):
    """
    List all currently active (running) tool executions.
    """
    from backend.services.tool_streaming import get_tool_streaming_service

    service = get_tool_streaming_service()

    active = service.get_active_executions()

    return {
        "active_executions": [
            {
                "execution_id": e.execution_id,
                "tool_name": e.tool_name,
                "progress": e.progress,
                "started_at": e.started_at.isoformat(),
            }
            for e in active
        ],
        "count": len(active),
    }
