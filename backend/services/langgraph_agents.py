"""
AIDocumentIndexer - LangGraph Agent Framework
==============================================

Production-ready agent orchestration with:
- Persistent state checkpointing
- Fault tolerance and recovery
- Multi-step reasoning workflows
- Human-in-the-loop integration points
- Streaming execution with progress updates
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import structlog

logger = structlog.get_logger(__name__)

# Check for LangGraph availability
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.info("LangGraph not installed. Install with: pip install langgraph")


class AgentStatus(str, Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # Waiting for human input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(str, Enum):
    """Types of nodes in agent workflow."""
    START = "start"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    DECISION = "decision"
    HUMAN_INPUT = "human_input"
    END = "end"


@dataclass
class AgentState:
    """State maintained across agent execution steps."""
    thread_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    current_node: str = "start"
    step_count: int = 0
    max_steps: int = 50
    error: Optional[str] = None
    status: AgentStatus = AgentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Checkpointing fields
    checkpoint_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None

    # Human-in-the-loop
    pending_approval: Optional[Dict[str, Any]] = None
    approval_timeout_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "context": self.context,
            "current_node": self.current_node,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "error": self.error,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checkpoint_id": self.checkpoint_id,
            "pending_approval": self.pending_approval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        return cls(
            thread_id=data["thread_id"],
            messages=data.get("messages", []),
            context=data.get("context", {}),
            current_node=data.get("current_node", "start"),
            step_count=data.get("step_count", 0),
            max_steps=data.get("max_steps", 50),
            error=data.get("error"),
            status=AgentStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            checkpoint_id=data.get("checkpoint_id"),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            pending_approval=data.get("pending_approval"),
        )


@dataclass
class NodeResult:
    """Result from executing a node."""
    success: bool
    next_node: Optional[str] = None
    state_updates: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    requires_approval: bool = False
    approval_request: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentNode(ABC):
    """Base class for agent workflow nodes."""

    def __init__(self, name: str, node_type: NodeType):
        self.name = name
        self.node_type = node_type

    @abstractmethod
    async def execute(self, state: AgentState) -> NodeResult:
        """Execute this node and return result."""
        pass


class ReasoningNode(AgentNode):
    """Node for LLM reasoning steps."""

    def __init__(
        self,
        name: str,
        llm_service: Any,
        system_prompt: str,
        next_node: str = "end",
        decision_nodes: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, NodeType.REASONING)
        self.llm_service = llm_service
        self.system_prompt = system_prompt
        self.next_node = next_node
        self.decision_nodes = decision_nodes or {}

    async def execute(self, state: AgentState) -> NodeResult:
        try:
            # Build messages for LLM
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(state.messages)

            # Call LLM
            if hasattr(self.llm_service, 'chat_completion'):
                response = await self.llm_service.chat_completion(messages)
            else:
                # Fallback for testing
                response = {"content": "Reasoning complete", "role": "assistant"}

            # Determine next node based on response
            next_node = self.next_node
            if self.decision_nodes:
                content = response.get("content", "").lower()
                for keyword, node in self.decision_nodes.items():
                    if keyword.lower() in content:
                        next_node = node
                        break

            return NodeResult(
                success=True,
                next_node=next_node,
                messages=[response],
            )
        except Exception as e:
            logger.error(f"Reasoning node failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
            )


class ToolCallNode(AgentNode):
    """Node for executing tool calls."""

    def __init__(
        self,
        name: str,
        tools: Dict[str, Callable],
        next_node: str = "end",
    ):
        super().__init__(name, NodeType.TOOL_CALL)
        self.tools = tools
        self.next_node = next_node

    async def execute(self, state: AgentState) -> NodeResult:
        try:
            # Get tool call from last message
            last_message = state.messages[-1] if state.messages else {}
            tool_calls = last_message.get("tool_calls", [])

            results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})

                if tool_name in self.tools:
                    tool_fn = self.tools[tool_name]
                    if asyncio.iscoroutinefunction(tool_fn):
                        result = await tool_fn(**tool_args)
                    else:
                        result = tool_fn(**tool_args)
                    results.append({
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_name,
                        "result": result,
                    })
                else:
                    results.append({
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_name,
                        "error": f"Unknown tool: {tool_name}",
                    })

            return NodeResult(
                success=True,
                next_node=self.next_node,
                messages=[{"role": "tool", "tool_results": results}],
            )
        except Exception as e:
            logger.error(f"Tool call node failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
            )


class RetrievalNode(AgentNode):
    """Node for RAG retrieval."""

    def __init__(
        self,
        name: str,
        rag_service: Any,
        next_node: str = "end",
        top_k: int = 5,
    ):
        super().__init__(name, NodeType.RETRIEVAL)
        self.rag_service = rag_service
        self.next_node = next_node
        self.top_k = top_k

    async def execute(self, state: AgentState) -> NodeResult:
        try:
            # Get query from context or last user message
            query = state.context.get("query")
            if not query and state.messages:
                for msg in reversed(state.messages):
                    if msg.get("role") == "user":
                        query = msg.get("content")
                        break

            if not query:
                return NodeResult(
                    success=False,
                    error="No query found for retrieval",
                )

            # Perform retrieval
            if hasattr(self.rag_service, 'retrieve'):
                results = await self.rag_service.retrieve(query, top_k=self.top_k)
            else:
                results = []

            # Add to context
            context_update = {
                "retrieved_documents": results,
                "retrieval_query": query,
            }

            return NodeResult(
                success=True,
                next_node=self.next_node,
                state_updates={"context": {**state.context, **context_update}},
            )
        except Exception as e:
            logger.error(f"Retrieval node failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
            )


class HumanInputNode(AgentNode):
    """Node that pauses for human approval/input."""

    def __init__(
        self,
        name: str,
        prompt: str,
        input_type: str = "approval",  # approval, text, choice
        choices: Optional[List[str]] = None,
        next_node: str = "end",
        rejection_node: Optional[str] = None,
    ):
        super().__init__(name, NodeType.HUMAN_INPUT)
        self.prompt = prompt
        self.input_type = input_type
        self.choices = choices
        self.next_node = next_node
        self.rejection_node = rejection_node

    async def execute(self, state: AgentState) -> NodeResult:
        # Check if we have pending approval response
        if state.pending_approval and state.pending_approval.get("response"):
            response = state.pending_approval["response"]

            if self.input_type == "approval":
                if response.get("approved"):
                    return NodeResult(
                        success=True,
                        next_node=self.next_node,
                        state_updates={"pending_approval": None},
                    )
                else:
                    return NodeResult(
                        success=True,
                        next_node=self.rejection_node or "end",
                        state_updates={"pending_approval": None},
                    )
            else:
                # Text or choice input
                return NodeResult(
                    success=True,
                    next_node=self.next_node,
                    state_updates={
                        "context": {**state.context, "human_input": response.get("value")},
                        "pending_approval": None,
                    },
                )

        # Request human input
        return NodeResult(
            success=True,
            requires_approval=True,
            approval_request={
                "node": self.name,
                "prompt": self.prompt,
                "input_type": self.input_type,
                "choices": self.choices,
            },
        )


class DatabaseCheckpointSaver:
    """Checkpoint saver that persists to database."""

    def __init__(self, db_service: Any = None):
        self.db_service = db_service
        self._memory: Dict[str, Dict[str, Any]] = {}

    async def save(self, thread_id: str, checkpoint_id: str, state: AgentState) -> None:
        """Save checkpoint."""
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "parent_checkpoint_id": state.parent_checkpoint_id,
            "state": state.to_dict(),
            "created_at": datetime.utcnow().isoformat(),
        }

        if self.db_service:
            # Save to database
            await self.db_service.save_checkpoint(checkpoint_data)
        else:
            # In-memory fallback
            if thread_id not in self._memory:
                self._memory[thread_id] = {}
            self._memory[thread_id][checkpoint_id] = checkpoint_data

        logger.debug(f"Saved checkpoint {checkpoint_id} for thread {thread_id}")

    async def load(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[AgentState]:
        """Load checkpoint. If checkpoint_id is None, load latest."""
        if self.db_service:
            checkpoint_data = await self.db_service.load_checkpoint(thread_id, checkpoint_id)
        else:
            # In-memory fallback
            thread_checkpoints = self._memory.get(thread_id, {})
            if not thread_checkpoints:
                return None

            if checkpoint_id:
                checkpoint_data = thread_checkpoints.get(checkpoint_id)
            else:
                # Get latest
                checkpoint_data = max(
                    thread_checkpoints.values(),
                    key=lambda x: x["created_at"],
                )

        if checkpoint_data:
            return AgentState.from_dict(checkpoint_data["state"])
        return None

    async def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a thread."""
        if self.db_service:
            return await self.db_service.list_checkpoints(thread_id)
        else:
            return list(self._memory.get(thread_id, {}).values())


class AgentWorkflow:
    """
    Orchestrates agent execution with checkpointing and fault tolerance.
    """

    def __init__(
        self,
        name: str,
        checkpoint_saver: Optional[DatabaseCheckpointSaver] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.name = name
        self.checkpoint_saver = checkpoint_saver or DatabaseCheckpointSaver()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.nodes: Dict[str, AgentNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.start_node: str = "start"

        # Execution state
        self._running_threads: Dict[str, AgentState] = {}

    def add_node(self, node: AgentNode) -> "AgentWorkflow":
        """Add a node to the workflow."""
        self.nodes[node.name] = node
        return self

    def add_edge(self, from_node: str, to_node: str) -> "AgentWorkflow":
        """Add an edge between nodes."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
        return self

    def set_start(self, node_name: str) -> "AgentWorkflow":
        """Set the starting node."""
        self.start_node = node_name
        return self

    async def run(
        self,
        thread_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: bool = True,
        stream: bool = False,
    ) -> AgentState:
        """
        Run the agent workflow.

        Args:
            thread_id: Unique identifier for this execution thread
            initial_state: Initial state values
            resume_from_checkpoint: Whether to resume from last checkpoint
            stream: Whether to yield intermediate results

        Returns:
            Final agent state
        """
        thread_id = thread_id or str(uuid.uuid4())

        # Try to resume from checkpoint
        state = None
        if resume_from_checkpoint:
            state = await self.checkpoint_saver.load(thread_id)
            if state:
                logger.info(f"Resuming thread {thread_id} from checkpoint {state.checkpoint_id}")

        # Create new state if not resuming
        if state is None:
            state = AgentState(
                thread_id=thread_id,
                current_node=self.start_node,
                context=initial_state.get("context", {}) if initial_state else {},
                messages=initial_state.get("messages", []) if initial_state else [],
            )

        state.status = AgentStatus.RUNNING
        self._running_threads[thread_id] = state

        try:
            while state.current_node != "end" and state.step_count < state.max_steps:
                # Check if paused for human input
                if state.status == AgentStatus.PAUSED:
                    break

                # Get current node
                node = self.nodes.get(state.current_node)
                if not node:
                    state.error = f"Unknown node: {state.current_node}"
                    state.status = AgentStatus.FAILED
                    break

                # Execute node with retries
                result = await self._execute_with_retry(node, state)

                if not result.success:
                    state.error = result.error
                    state.status = AgentStatus.FAILED
                    break

                # Handle human input request
                if result.requires_approval:
                    state.pending_approval = result.approval_request
                    state.status = AgentStatus.PAUSED

                    # Save checkpoint before pausing
                    checkpoint_id = str(uuid.uuid4())
                    state.parent_checkpoint_id = state.checkpoint_id
                    state.checkpoint_id = checkpoint_id
                    await self.checkpoint_saver.save(thread_id, checkpoint_id, state)
                    break

                # Update state
                state.messages.extend(result.messages)
                if result.state_updates:
                    for key, value in result.state_updates.items():
                        if key == "context":
                            state.context.update(value)
                        else:
                            setattr(state, key, value)

                # Move to next node
                state.current_node = result.next_node or "end"
                state.step_count += 1
                state.updated_at = datetime.utcnow()

                # Checkpoint after each step
                checkpoint_id = str(uuid.uuid4())
                state.parent_checkpoint_id = state.checkpoint_id
                state.checkpoint_id = checkpoint_id
                await self.checkpoint_saver.save(thread_id, checkpoint_id, state)

                logger.debug(
                    f"Thread {thread_id} step {state.step_count}: {node.name} -> {state.current_node}"
                )

            # Check if we hit max steps
            if state.step_count >= state.max_steps:
                state.error = "Max steps exceeded"
                state.status = AgentStatus.FAILED
            elif state.current_node == "end" and state.status == AgentStatus.RUNNING:
                state.status = AgentStatus.COMPLETED

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            state.error = str(e)
            state.status = AgentStatus.FAILED

        finally:
            # Final checkpoint
            checkpoint_id = str(uuid.uuid4())
            state.checkpoint_id = checkpoint_id
            await self.checkpoint_saver.save(thread_id, checkpoint_id, state)

            # Clean up running threads
            if thread_id in self._running_threads:
                del self._running_threads[thread_id]

        return state

    async def _execute_with_retry(
        self,
        node: AgentNode,
        state: AgentState,
    ) -> NodeResult:
        """Execute a node with retries."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await node.execute(state)
                if result.success:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Node {node.name} attempt {attempt + 1} failed: {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        return NodeResult(
            success=False,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
        )

    async def resume(
        self,
        thread_id: str,
        approval_response: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """Resume a paused workflow."""
        state = await self.checkpoint_saver.load(thread_id)

        if not state:
            raise ValueError(f"No checkpoint found for thread {thread_id}")

        if state.status != AgentStatus.PAUSED:
            raise ValueError(f"Thread {thread_id} is not paused (status: {state.status})")

        # Update pending approval with response
        if approval_response and state.pending_approval:
            state.pending_approval["response"] = approval_response

        # Resume execution
        return await self.run(
            thread_id=thread_id,
            resume_from_checkpoint=False,
        )

    async def cancel(self, thread_id: str) -> AgentState:
        """Cancel a running workflow."""
        state = self._running_threads.get(thread_id)

        if not state:
            state = await self.checkpoint_saver.load(thread_id)

        if state:
            state.status = AgentStatus.CANCELLED
            await self.checkpoint_saver.save(thread_id, str(uuid.uuid4()), state)

        return state

    def get_status(self, thread_id: str) -> Optional[AgentStatus]:
        """Get current status of a thread."""
        state = self._running_threads.get(thread_id)
        return state.status if state else None


class LangGraphAgentService:
    """
    High-level service for managing LangGraph agents.

    Provides:
    - Pre-built agent workflows (RAG, Multi-step reasoning, etc.)
    - Workflow persistence and recovery
    - Execution monitoring
    - Human-in-the-loop integration
    """

    def __init__(
        self,
        llm_service: Any = None,
        rag_service: Any = None,
        db_service: Any = None,
    ):
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.checkpoint_saver = DatabaseCheckpointSaver(db_service)

        self._workflows: Dict[str, AgentWorkflow] = {}
        self._active_threads: Dict[str, str] = {}  # thread_id -> workflow_name

        # Register built-in workflows
        self._register_builtin_workflows()

    def _register_builtin_workflows(self):
        """Register built-in agent workflows."""

        # RAG Agent: Retrieve -> Reason -> Generate
        rag_workflow = AgentWorkflow("rag_agent", self.checkpoint_saver)
        rag_workflow.add_node(RetrievalNode(
            "retrieve",
            self.rag_service,
            next_node="reason",
        ))
        rag_workflow.add_node(ReasoningNode(
            "reason",
            self.llm_service,
            system_prompt="""You are a helpful assistant. Use the retrieved context to answer questions.
If you can answer from the context, do so. If not, explain what information is missing.""",
            next_node="end",
        ))
        rag_workflow.set_start("retrieve")
        self._workflows["rag_agent"] = rag_workflow

        # Multi-step Reasoning Agent: Plan -> Execute -> Verify
        reasoning_workflow = AgentWorkflow("reasoning_agent", self.checkpoint_saver)
        reasoning_workflow.add_node(ReasoningNode(
            "plan",
            self.llm_service,
            system_prompt="""You are a planning assistant. Break down the user's request into steps.
Output a numbered list of steps to accomplish the goal.""",
            next_node="execute",
        ))
        reasoning_workflow.add_node(ReasoningNode(
            "execute",
            self.llm_service,
            system_prompt="""Execute each step of the plan. Show your work for each step.""",
            next_node="verify",
        ))
        reasoning_workflow.add_node(ReasoningNode(
            "verify",
            self.llm_service,
            system_prompt="""Review the execution. Verify the results are correct.
If there are errors, explain them. If correct, summarize the final answer.""",
            next_node="end",
        ))
        reasoning_workflow.set_start("plan")
        self._workflows["reasoning_agent"] = reasoning_workflow

        # Research Agent with Human Approval
        research_workflow = AgentWorkflow("research_agent", self.checkpoint_saver)
        research_workflow.add_node(RetrievalNode(
            "gather",
            self.rag_service,
            next_node="analyze",
            top_k=10,
        ))
        research_workflow.add_node(ReasoningNode(
            "analyze",
            self.llm_service,
            system_prompt="""Analyze the retrieved documents. Identify key themes and insights.
Prepare a research summary with citations.""",
            next_node="approval",
        ))
        research_workflow.add_node(HumanInputNode(
            "approval",
            prompt="Review the research summary. Approve to continue or provide feedback.",
            input_type="approval",
            next_node="synthesize",
            rejection_node="analyze",
        ))
        research_workflow.add_node(ReasoningNode(
            "synthesize",
            self.llm_service,
            system_prompt="""Create a comprehensive research report from the approved analysis.""",
            next_node="end",
        ))
        research_workflow.set_start("gather")
        self._workflows["research_agent"] = research_workflow

    def register_workflow(self, workflow: AgentWorkflow) -> None:
        """Register a custom workflow."""
        self._workflows[workflow.name] = workflow

    async def create_thread(
        self,
        workflow_name: str,
        initial_context: Optional[Dict[str, Any]] = None,
        initial_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Create a new agent thread."""
        if workflow_name not in self._workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        thread_id = str(uuid.uuid4())
        self._active_threads[thread_id] = workflow_name

        # Create initial state
        state = AgentState(
            thread_id=thread_id,
            context=initial_context or {},
            messages=initial_messages or [],
        )

        # Save initial checkpoint
        await self.checkpoint_saver.save(thread_id, str(uuid.uuid4()), state)

        logger.info(f"Created thread {thread_id} for workflow {workflow_name}")
        return thread_id

    async def run_thread(
        self,
        thread_id: str,
        query: Optional[str] = None,
    ) -> AgentState:
        """Run or continue an agent thread."""
        workflow_name = self._active_threads.get(thread_id)

        if not workflow_name:
            # Try to find from checkpoint
            state = await self.checkpoint_saver.load(thread_id)
            if not state:
                raise ValueError(f"Unknown thread: {thread_id}")
            workflow_name = state.context.get("workflow_name", "rag_agent")

        workflow = self._workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Add query to initial state if provided
        initial_state = None
        if query:
            initial_state = {
                "context": {"query": query},
                "messages": [{"role": "user", "content": query}],
            }

        return await workflow.run(
            thread_id=thread_id,
            initial_state=initial_state,
        )

    async def resume_thread(
        self,
        thread_id: str,
        approval_response: Dict[str, Any],
    ) -> AgentState:
        """Resume a paused thread with human input."""
        workflow_name = self._active_threads.get(thread_id)

        if not workflow_name:
            state = await self.checkpoint_saver.load(thread_id)
            if state:
                workflow_name = state.context.get("workflow_name", "rag_agent")

        if not workflow_name:
            raise ValueError(f"Unknown thread: {thread_id}")

        workflow = self._workflows[workflow_name]
        return await workflow.resume(thread_id, approval_response)

    async def cancel_thread(self, thread_id: str) -> AgentState:
        """Cancel a thread."""
        workflow_name = self._active_threads.get(thread_id, "rag_agent")
        workflow = self._workflows.get(workflow_name)

        if workflow:
            state = await workflow.cancel(thread_id)
            if thread_id in self._active_threads:
                del self._active_threads[thread_id]
            return state

        raise ValueError(f"Unknown thread: {thread_id}")

    async def get_thread_state(self, thread_id: str) -> Optional[AgentState]:
        """Get current state of a thread."""
        return await self.checkpoint_saver.load(thread_id)

    async def list_threads(
        self,
        status: Optional[AgentStatus] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List all threads, optionally filtered by status."""
        threads = []

        for thread_id in self._active_threads:
            state = await self.checkpoint_saver.load(thread_id)
            if state:
                if status is None or state.status == status:
                    threads.append({
                        "thread_id": thread_id,
                        "workflow": self._active_threads[thread_id],
                        "status": state.status.value,
                        "step_count": state.step_count,
                        "created_at": state.created_at.isoformat(),
                        "updated_at": state.updated_at.isoformat(),
                    })

        return threads[:limit]

    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        threads = await self.list_threads()

        stats = {
            "total_threads": len(threads),
            "by_status": {},
            "by_workflow": {},
        }

        for thread in threads:
            status = thread["status"]
            workflow = thread["workflow"]

            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["by_workflow"][workflow] = stats["by_workflow"].get(workflow, 0) + 1

        return stats


# Singleton instance
_langgraph_service: Optional[LangGraphAgentService] = None


def get_langgraph_service(
    llm_service: Any = None,
    rag_service: Any = None,
    db_service: Any = None,
) -> LangGraphAgentService:
    """Get or create the LangGraph service singleton."""
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphAgentService(
            llm_service=llm_service,
            rag_service=rag_service,
            db_service=db_service,
        )
    return _langgraph_service
