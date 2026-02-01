"""
AIDocumentIndexer - Tool Use Streaming Service
===============================================

Provides real-time streaming of tool execution events.
Allows UIs to show progress during long-running operations.

Events:
1. tool_start - Tool execution started
2. tool_progress - Progress update (for multi-step tools)
3. tool_output - Partial output (for streaming results)
4. tool_complete - Tool execution finished
5. tool_error - Tool execution failed
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncGenerator, Callable, Awaitable
from enum import Enum
from datetime import datetime
import asyncio
import json
import time
import uuid

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


class ToolEventType(str, Enum):
    """Types of tool streaming events."""
    START = "tool_start"
    PROGRESS = "tool_progress"
    OUTPUT = "tool_output"
    COMPLETE = "tool_complete"
    ERROR = "tool_error"


@dataclass
class ToolEvent:
    """A single tool execution event."""
    event_type: ToolEventType
    tool_name: str
    execution_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[float] = None  # 0.0 to 1.0
    message: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type.value,
            "tool": self.tool_name,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class StreamingToolExecution:
    """Tracks a streaming tool execution."""
    execution_id: str
    tool_name: str
    started_at: datetime
    status: str = "running"
    progress: float = 0.0
    events: List[ToolEvent] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None


class ToolStreamingService:
    """
    Service for streaming tool execution events.

    Provides:
    1. Event emission for tool execution steps
    2. SSE streaming for real-time UI updates
    3. Execution tracking and history
    """

    def __init__(self):
        """Initialize the streaming service."""
        self._executions: Dict[str, StreamingToolExecution] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._max_history = 100

    async def start_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
    ) -> str:
        """
        Start a new streaming tool execution.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments (for logging)

        Returns:
            Unique execution ID
        """
        execution_id = str(uuid.uuid4())

        execution = StreamingToolExecution(
            execution_id=execution_id,
            tool_name=tool_name,
            started_at=datetime.utcnow(),
        )
        self._executions[execution_id] = execution

        # Emit start event
        await self._emit_event(ToolEvent(
            event_type=ToolEventType.START,
            tool_name=tool_name,
            execution_id=execution_id,
            message=f"Starting {tool_name}",
            data={"arguments": arguments or {}},
        ))

        logger.info(
            "Tool execution started",
            execution_id=execution_id,
            tool=tool_name,
        )

        return execution_id

    async def emit_progress(
        self,
        execution_id: str,
        progress: float,
        message: str = None,
        data: Dict[str, Any] = None,
    ):
        """
        Emit a progress update event.

        Args:
            execution_id: Execution ID
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
            data: Optional additional data
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return

        execution.progress = progress

        await self._emit_event(ToolEvent(
            event_type=ToolEventType.PROGRESS,
            tool_name=execution.tool_name,
            execution_id=execution_id,
            progress=progress,
            message=message or f"Progress: {int(progress * 100)}%",
            data=data or {},
        ))

    async def emit_output(
        self,
        execution_id: str,
        output: Any,
        is_partial: bool = True,
    ):
        """
        Emit a tool output event (for streaming results).

        Args:
            execution_id: Execution ID
            output: The output data
            is_partial: Whether this is a partial result
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return

        await self._emit_event(ToolEvent(
            event_type=ToolEventType.OUTPUT,
            tool_name=execution.tool_name,
            execution_id=execution_id,
            data={"output": output, "is_partial": is_partial},
        ))

    async def complete_execution(
        self,
        execution_id: str,
        result: Any,
        message: str = None,
    ):
        """
        Mark execution as complete and emit final event.

        Args:
            execution_id: Execution ID
            result: Final result
            message: Optional completion message
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return

        execution.status = "completed"
        execution.progress = 1.0
        execution.result = result

        await self._emit_event(ToolEvent(
            event_type=ToolEventType.COMPLETE,
            tool_name=execution.tool_name,
            execution_id=execution_id,
            progress=1.0,
            message=message or f"{execution.tool_name} completed",
            data={"result": result},
        ))

        logger.info(
            "Tool execution completed",
            execution_id=execution_id,
            tool=execution.tool_name,
        )

    async def fail_execution(
        self,
        execution_id: str,
        error: str,
    ):
        """
        Mark execution as failed and emit error event.

        Args:
            execution_id: Execution ID
            error: Error message
        """
        execution = self._executions.get(execution_id)
        if not execution:
            return

        execution.status = "failed"
        execution.error = error

        await self._emit_event(ToolEvent(
            event_type=ToolEventType.ERROR,
            tool_name=execution.tool_name,
            execution_id=execution_id,
            error=error,
            message=f"{execution.tool_name} failed: {error}",
        ))

        logger.error(
            "Tool execution failed",
            execution_id=execution_id,
            tool=execution.tool_name,
            error=error,
        )

    async def _emit_event(self, event: ToolEvent):
        """Emit an event to all subscribers."""
        execution = self._executions.get(event.execution_id)
        if execution:
            execution.events.append(event)

        # Notify subscribers
        if event.execution_id in self._subscribers:
            for queue in self._subscribers[event.execution_id]:
                try:
                    await queue.put(event)
                except asyncio.QueueFull:
                    pass

    async def subscribe(
        self,
        execution_id: str,
    ) -> AsyncGenerator[ToolEvent, None]:
        """
        Subscribe to events for an execution.

        Args:
            execution_id: Execution ID to subscribe to

        Yields:
            ToolEvent objects as they occur
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        if execution_id not in self._subscribers:
            self._subscribers[execution_id] = []
        self._subscribers[execution_id].append(queue)

        try:
            # Replay existing events
            execution = self._executions.get(execution_id)
            if execution:
                for event in execution.events:
                    yield event

            # Wait for new events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event

                    # Stop if execution is complete
                    if event.event_type in [ToolEventType.COMPLETE, ToolEventType.ERROR]:
                        break

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ToolEvent(
                        event_type=ToolEventType.PROGRESS,
                        tool_name="keepalive",
                        execution_id=execution_id,
                        message="keepalive",
                    )

        finally:
            # Cleanup subscription
            if execution_id in self._subscribers:
                self._subscribers[execution_id].remove(queue)
                if not self._subscribers[execution_id]:
                    del self._subscribers[execution_id]

    def get_execution(self, execution_id: str) -> Optional[StreamingToolExecution]:
        """Get an execution by ID."""
        return self._executions.get(execution_id)

    def get_active_executions(self) -> List[StreamingToolExecution]:
        """Get all active (running) executions."""
        return [
            e for e in self._executions.values()
            if e.status == "running"
        ]

    async def stream_tool_execution(
        self,
        tool_name: str,
        tool_func: Callable[..., Awaitable[Any]],
        arguments: Dict[str, Any] = None,
        progress_callback: Callable[[float, str], Awaitable[None]] = None,
    ) -> AsyncGenerator[ToolEvent, None]:
        """
        Execute a tool with streaming events.

        Args:
            tool_name: Name of the tool
            tool_func: Async function to execute
            arguments: Arguments to pass to the function
            progress_callback: Optional callback to report progress

        Yields:
            ToolEvent objects during execution
        """
        execution_id = await self.start_execution(tool_name, arguments)

        try:
            # Start subscribing to events
            async def run_tool():
                try:
                    # If tool supports progress callback, use it
                    if progress_callback:
                        result = await tool_func(
                            **(arguments or {}),
                            progress_callback=lambda p, m: self.emit_progress(execution_id, p, m),
                        )
                    else:
                        result = await tool_func(**(arguments or {}))

                    await self.complete_execution(execution_id, result)

                except Exception as e:
                    await self.fail_execution(execution_id, str(e))

            # Run tool in background
            asyncio.create_task(run_tool())

            # Stream events
            async for event in self.subscribe(execution_id):
                yield event

        except Exception as e:
            await self.fail_execution(execution_id, str(e))
            yield ToolEvent(
                event_type=ToolEventType.ERROR,
                tool_name=tool_name,
                execution_id=execution_id,
                error=str(e),
            )


# Singleton instance
_tool_streaming: Optional[ToolStreamingService] = None


def get_tool_streaming_service() -> ToolStreamingService:
    """Get or create the tool streaming service singleton."""
    global _tool_streaming
    if _tool_streaming is None:
        _tool_streaming = ToolStreamingService()
    return _tool_streaming
