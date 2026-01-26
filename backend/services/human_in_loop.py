"""
AIDocumentIndexer - Human-in-the-Loop Interrupt Support
========================================================

Phase 77: Human-in-the-loop (HITL) for agent workflows.

Provides mechanisms for users to:
- Pause running agent workflows
- Inspect intermediate results
- Provide feedback or corrections
- Resume or abort workflows
- Inject new instructions mid-execution

Key Features:
- Async interrupt handling
- Checkpoint-based pause/resume
- User approval gates for critical actions
- Feedback collection for learning
- Timeout handling for unresponsive users

Architecture:
- InterruptManager: Central coordinator for interrupts
- Checkpoint: Saveable workflow state
- ApprovalGate: Wait for user approval
- FeedbackCollector: Gather user input

Usage:
    from backend.services.human_in_loop import (
        InterruptManager,
        get_interrupt_manager,
    )

    manager = get_interrupt_manager()

    # In agent workflow
    if await manager.check_interrupt(session_id):
        checkpoint = await manager.create_checkpoint(session_id, state)
        action = await manager.wait_for_user(session_id, timeout=300)
        if action.type == "abort":
            raise WorkflowAborted()
        elif action.type == "modify":
            state = action.new_state
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================

class InterruptType(str, Enum):
    """Types of interrupts."""
    PAUSE = "pause"           # Pause for user inspection
    APPROVAL = "approval"     # Wait for user approval
    FEEDBACK = "feedback"     # Request user feedback
    ABORT = "abort"           # Abort the workflow
    MODIFY = "modify"         # User wants to modify state
    TIMEOUT = "timeout"       # User didn't respond in time


class WorkflowState(str, Enum):
    """States of a workflow."""
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_FEEDBACK = "waiting_feedback"
    COMPLETED = "completed"
    ABORTED = "aborted"
    TIMED_OUT = "timed_out"


class ApprovalLevel(str, Enum):
    """Levels of approval required."""
    NONE = "none"             # No approval needed
    INFORM = "inform"         # Inform user, don't wait
    OPTIONAL = "optional"     # Wait briefly, continue if no response
    REQUIRED = "required"     # Must have approval to continue
    CRITICAL = "critical"     # Requires explicit confirmation


@dataclass
class Checkpoint:
    """A saveable workflow state."""
    id: str
    session_id: str
    workflow_id: str
    step_name: str
    state: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "step_name": self.step_name,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class InterruptRequest:
    """A request to interrupt a workflow."""
    id: str
    session_id: str
    type: InterruptType
    message: str
    options: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserResponse:
    """User's response to an interrupt."""
    request_id: str
    session_id: str
    action: InterruptType
    selected_option: Optional[str] = None
    feedback: Optional[str] = None
    new_state: Optional[Dict[str, Any]] = None
    responded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ApprovalGate:
    """A gate that requires user approval."""
    id: str
    session_id: str
    action_description: str
    level: ApprovalLevel
    details: Dict[str, Any]
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Interrupt Manager
# =============================================================================

class InterruptManager:
    """
    Central coordinator for human-in-the-loop interrupts.

    Manages pause/resume, approval gates, and user feedback collection.

    Usage:
        manager = InterruptManager()

        # Request interrupt
        await manager.request_interrupt(
            session_id="session-123",
            type=InterruptType.APPROVAL,
            message="About to delete 50 documents. Continue?",
            options=["Yes, delete", "No, cancel"],
        )

        # Check for interrupts (in agent loop)
        if await manager.check_interrupt(session_id):
            response = await manager.wait_for_user(session_id, timeout=60)
    """

    def __init__(self):
        # Active sessions and their states
        self._sessions: Dict[str, WorkflowState] = {}

        # Pending interrupt requests
        self._pending_interrupts: Dict[str, InterruptRequest] = {}

        # User responses
        self._responses: Dict[str, UserResponse] = {}

        # Checkpoints for resume
        self._checkpoints: Dict[str, List[Checkpoint]] = defaultdict(list)

        # Approval gates
        self._approval_gates: Dict[str, ApprovalGate] = {}

        # Event for async waiting
        self._response_events: Dict[str, asyncio.Event] = {}

        # Callbacks for notifications
        self._interrupt_callbacks: List[Callable] = []
        self._response_callbacks: List[Callable] = []

        logger.info("InterruptManager initialized")

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def register_session(self, session_id: str, workflow_id: Optional[str] = None) -> None:
        """Register a new workflow session."""
        self._sessions[session_id] = WorkflowState.RUNNING
        logger.debug("Session registered", session_id=session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister a workflow session."""
        self._sessions.pop(session_id, None)
        self._pending_interrupts.pop(session_id, None)
        self._responses.pop(session_id, None)
        self._checkpoints.pop(session_id, None)
        if session_id in self._response_events:
            self._response_events[session_id].set()
            del self._response_events[session_id]
        logger.debug("Session unregistered", session_id=session_id)

    def get_session_state(self, session_id: str) -> Optional[WorkflowState]:
        """Get current state of a session."""
        return self._sessions.get(session_id)

    # -------------------------------------------------------------------------
    # Interrupt Handling
    # -------------------------------------------------------------------------

    async def request_interrupt(
        self,
        session_id: str,
        type: InterruptType,
        message: str,
        options: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
    ) -> InterruptRequest:
        """
        Request an interrupt for a session.

        Args:
            session_id: The session to interrupt
            type: Type of interrupt
            message: Message to show user
            options: Optional list of choices
            data: Additional data to pass
            timeout_seconds: How long to wait for response

        Returns:
            The created interrupt request
        """
        request = InterruptRequest(
            id=str(uuid.uuid4()),
            session_id=session_id,
            type=type,
            message=message,
            options=options or [],
            data=data or {},
            timeout_seconds=timeout_seconds,
        )

        self._pending_interrupts[session_id] = request
        self._sessions[session_id] = self._state_for_interrupt(type)
        self._response_events[session_id] = asyncio.Event()

        # Notify callbacks
        for callback in self._interrupt_callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.warning("Interrupt callback failed", error=str(e))

        logger.info(
            "Interrupt requested",
            session_id=session_id,
            type=type.value,
            message=message[:100],
        )

        return request

    def _state_for_interrupt(self, type: InterruptType) -> WorkflowState:
        """Map interrupt type to workflow state."""
        mapping = {
            InterruptType.PAUSE: WorkflowState.PAUSED,
            InterruptType.APPROVAL: WorkflowState.WAITING_APPROVAL,
            InterruptType.FEEDBACK: WorkflowState.WAITING_FEEDBACK,
            InterruptType.ABORT: WorkflowState.ABORTED,
        }
        return mapping.get(type, WorkflowState.PAUSED)

    async def check_interrupt(self, session_id: str) -> bool:
        """
        Check if there's a pending interrupt for this session.

        Call this periodically in agent loops to check for user requests.

        Returns:
            True if there's a pending interrupt
        """
        return session_id in self._pending_interrupts

    async def wait_for_user(
        self,
        session_id: str,
        timeout: Optional[int] = None,
    ) -> UserResponse:
        """
        Wait for user response to an interrupt.

        Args:
            session_id: The session waiting
            timeout: Override default timeout (seconds)

        Returns:
            UserResponse with user's action
        """
        request = self._pending_interrupts.get(session_id)
        if not request:
            return UserResponse(
                request_id="",
                session_id=session_id,
                action=InterruptType.ABORT,
                feedback="No pending interrupt",
            )

        timeout = timeout or request.timeout_seconds
        event = self._response_events.get(session_id)

        if not event:
            event = asyncio.Event()
            self._response_events[session_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)

            # Get response
            response = self._responses.get(session_id)
            if response:
                return response

            return UserResponse(
                request_id=request.id,
                session_id=session_id,
                action=InterruptType.TIMEOUT,
            )

        except asyncio.TimeoutError:
            self._sessions[session_id] = WorkflowState.TIMED_OUT
            return UserResponse(
                request_id=request.id,
                session_id=session_id,
                action=InterruptType.TIMEOUT,
            )

    async def submit_response(
        self,
        session_id: str,
        action: InterruptType,
        selected_option: Optional[str] = None,
        feedback: Optional[str] = None,
        new_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Submit user response to an interrupt.

        Args:
            session_id: The session
            action: User's chosen action
            selected_option: If options were provided, which was selected
            feedback: Optional text feedback
            new_state: Optional modified state

        Returns:
            True if response was accepted
        """
        request = self._pending_interrupts.get(session_id)
        if not request:
            logger.warning("No pending interrupt for session", session_id=session_id)
            return False

        response = UserResponse(
            request_id=request.id,
            session_id=session_id,
            action=action,
            selected_option=selected_option,
            feedback=feedback,
            new_state=new_state,
        )

        self._responses[session_id] = response
        del self._pending_interrupts[session_id]

        # Update session state
        if action == InterruptType.ABORT:
            self._sessions[session_id] = WorkflowState.ABORTED
        else:
            self._sessions[session_id] = WorkflowState.RUNNING

        # Signal waiting coroutine
        event = self._response_events.get(session_id)
        if event:
            event.set()

        # Notify callbacks
        for callback in self._response_callbacks:
            try:
                await callback(response)
            except Exception as e:
                logger.warning("Response callback failed", error=str(e))

        logger.info(
            "User response submitted",
            session_id=session_id,
            action=action.value,
        )

        return True

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    async def create_checkpoint(
        self,
        session_id: str,
        workflow_id: str,
        step_name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a checkpoint for later resume.

        Args:
            session_id: The session
            workflow_id: Workflow identifier
            step_name: Current step name
            state: State to save
            metadata: Optional metadata

        Returns:
            The created checkpoint
        """
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            session_id=session_id,
            workflow_id=workflow_id,
            step_name=step_name,
            state=state,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
        )

        self._checkpoints[session_id].append(checkpoint)

        # Keep only last 10 checkpoints per session
        if len(self._checkpoints[session_id]) > 10:
            self._checkpoints[session_id] = self._checkpoints[session_id][-10:]

        logger.debug(
            "Checkpoint created",
            session_id=session_id,
            step_name=step_name,
            checkpoint_id=checkpoint.id,
        )

        return checkpoint

    async def get_checkpoints(self, session_id: str) -> List[Checkpoint]:
        """Get all checkpoints for a session."""
        return self._checkpoints.get(session_id, [])

    async def get_latest_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        """Get the most recent checkpoint for a session."""
        checkpoints = self._checkpoints.get(session_id, [])
        return checkpoints[-1] if checkpoints else None

    async def restore_checkpoint(
        self,
        session_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Restore state from a checkpoint.

        Args:
            session_id: The session
            checkpoint_id: Specific checkpoint (or latest if None)

        Returns:
            The checkpoint state, or None if not found
        """
        checkpoints = self._checkpoints.get(session_id, [])

        if not checkpoints:
            return None

        if checkpoint_id:
            for cp in checkpoints:
                if cp.id == checkpoint_id:
                    return cp.state
            return None

        return checkpoints[-1].state

    # -------------------------------------------------------------------------
    # Approval Gates
    # -------------------------------------------------------------------------

    async def request_approval(
        self,
        session_id: str,
        action_description: str,
        level: ApprovalLevel = ApprovalLevel.REQUIRED,
        details: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
    ) -> bool:
        """
        Request approval for an action.

        Args:
            session_id: The session
            action_description: What action needs approval
            level: How important is approval
            details: Additional details to show user
            timeout_seconds: How long to wait

        Returns:
            True if approved, False otherwise
        """
        if level == ApprovalLevel.NONE:
            return True

        gate = ApprovalGate(
            id=str(uuid.uuid4()),
            session_id=session_id,
            action_description=action_description,
            level=level,
            details=details or {},
        )

        self._approval_gates[gate.id] = gate

        if level == ApprovalLevel.INFORM:
            # Just inform, don't wait
            logger.info(
                "Action notification",
                session_id=session_id,
                action=action_description,
            )
            return True

        # Request interrupt for approval
        await self.request_interrupt(
            session_id=session_id,
            type=InterruptType.APPROVAL,
            message=f"Approval needed: {action_description}",
            options=["Approve", "Reject"],
            data={"gate_id": gate.id, "details": details or {}},
            timeout_seconds=timeout_seconds,
        )

        # Wait for response
        response = await self.wait_for_user(session_id, timeout=timeout_seconds)

        if response.action == InterruptType.TIMEOUT:
            if level == ApprovalLevel.OPTIONAL:
                gate.approved = True  # Default approve for optional
                return True
            gate.approved = False
            return False

        approved = response.selected_option == "Approve"
        gate.approved = approved

        return approved

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_interrupt(self, callback: Callable) -> None:
        """Register callback for interrupt requests."""
        self._interrupt_callbacks.append(callback)

    def on_response(self, callback: Callable) -> None:
        """Register callback for user responses."""
        self._response_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def pause_workflow(self, session_id: str, reason: str = "Paused by user") -> bool:
        """Pause a running workflow."""
        if self._sessions.get(session_id) != WorkflowState.RUNNING:
            return False

        await self.request_interrupt(
            session_id=session_id,
            type=InterruptType.PAUSE,
            message=reason,
        )
        return True

    async def resume_workflow(self, session_id: str) -> bool:
        """Resume a paused workflow."""
        if self._sessions.get(session_id) not in (WorkflowState.PAUSED, WorkflowState.WAITING_APPROVAL):
            return False

        await self.submit_response(
            session_id=session_id,
            action=InterruptType.PAUSE,  # Continue from pause
        )
        return True

    async def abort_workflow(self, session_id: str, reason: str = "Aborted by user") -> bool:
        """Abort a running workflow."""
        await self.submit_response(
            session_id=session_id,
            action=InterruptType.ABORT,
            feedback=reason,
        )
        return True


# =============================================================================
# Singleton and Factory
# =============================================================================

_interrupt_manager: Optional[InterruptManager] = None


def get_interrupt_manager() -> InterruptManager:
    """Get or create InterruptManager singleton."""
    global _interrupt_manager
    if _interrupt_manager is None:
        _interrupt_manager = InterruptManager()
    return _interrupt_manager


# =============================================================================
# Decorators for Agent Methods
# =============================================================================

def interruptible(
    approval_level: ApprovalLevel = ApprovalLevel.NONE,
    checkpoint_before: bool = False,
):
    """
    Decorator to make an agent method interruptible.

    Usage:
        @interruptible(approval_level=ApprovalLevel.OPTIONAL, checkpoint_before=True)
        async def process_documents(self, docs):
            ...
    """
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            session_id = getattr(self, "session_id", None) or kwargs.get("session_id")
            if not session_id:
                return await func(self, *args, **kwargs)

            manager = get_interrupt_manager()

            # Check for pending interrupt
            if await manager.check_interrupt(session_id):
                response = await manager.wait_for_user(session_id, timeout=60)
                if response.action == InterruptType.ABORT:
                    raise WorkflowAborted(response.feedback or "Aborted by user")

            # Create checkpoint if requested
            if checkpoint_before:
                state = {
                    "func": func.__name__,
                    "args": str(args)[:500],
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                }
                await manager.create_checkpoint(
                    session_id=session_id,
                    workflow_id=getattr(self, "workflow_id", "unknown"),
                    step_name=func.__name__,
                    state=state,
                )

            # Request approval if needed
            if approval_level != ApprovalLevel.NONE:
                approved = await manager.request_approval(
                    session_id=session_id,
                    action_description=f"Execute {func.__name__}",
                    level=approval_level,
                )
                if not approved:
                    raise WorkflowAborted(f"Approval denied for {func.__name__}")

            return await func(self, *args, **kwargs)

        return wrapper
    return decorator


class WorkflowAborted(Exception):
    """Raised when a workflow is aborted by user."""
    pass


# Export
__all__ = [
    "InterruptManager",
    "InterruptType",
    "WorkflowState",
    "ApprovalLevel",
    "Checkpoint",
    "InterruptRequest",
    "UserResponse",
    "ApprovalGate",
    "WorkflowAborted",
    "get_interrupt_manager",
    "interruptible",
]
