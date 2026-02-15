"""
AIDocumentIndexer - Manager Agent
==================================

Lead orchestrator agent that:
1. Analyzes user requests
2. Decomposes into subtasks
3. Creates execution plans
4. Dispatches to worker agents
5. Monitors progress and handles failures
6. Synthesizes final output

The Manager Agent uses LLM reasoning to understand intent and
create appropriate task decomposition based on the request type.
"""

import json
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.agents.agent_base import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
    TaskType,
    TaskStatus,
    FallbackStrategy,
    PromptTemplate,
    TrajectoryStep,
    _get_language_instruction,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Execution Plan Types
# =============================================================================

class PlanStatus(str, Enum):
    """Status of an execution plan."""
    PENDING = "pending"
    ESTIMATING = "estimating"       # Cost estimation in progress
    AWAITING_APPROVAL = "awaiting_approval"  # Waiting for user cost approval
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PlanStep:
    """Single step in an execution plan."""
    id: str
    agent_type: str  # "generator", "critic", "research", "tool"
    task: AgentTask
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # Step IDs
    result: Optional[AgentResult] = None
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "task_name": self.task.name,
            "task_type": self.task.type.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a user request.

    Contains ordered steps with dependencies, cost estimates,
    and tracks execution progress.
    """
    id: str
    session_id: str
    user_id: str
    user_request: str
    request_type: str  # Classified type: document_generation, qa, analysis, etc.
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    current_step_index: int = 0
    total_estimated_cost_usd: float = 0.0
    total_actual_cost_usd: float = 0.0
    final_output: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    @property
    def completed_steps(self) -> int:
        """Count of completed steps."""
        return sum(1 for s in self.steps if s.status == TaskStatus.COMPLETED)

    @property
    def failed_steps(self) -> int:
        """Count of failed steps."""
        return sum(1 for s in self.steps if s.status == TaskStatus.FAILED)

    @property
    def progress_percentage(self) -> float:
        """Execution progress as percentage."""
        if not self.steps:
            return 0.0
        return (self.completed_steps / len(self.steps)) * 100

    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute (dependencies met)."""
        completed_ids = {s.id for s in self.steps if s.status == TaskStatus.COMPLETED}
        ready = []
        for step in self.steps:
            if step.status != TaskStatus.PENDING:
                continue
            if all(dep_id in completed_ids for dep_id in step.dependencies):
                ready.append(step)
        return ready

    def summary(self) -> str:
        """Generate human-readable summary."""
        step_summaries = []
        for i, step in enumerate(self.steps, 1):
            step_summaries.append(f"{i}. {step.task.name} ({step.agent_type})")
        return f"Plan with {len(self.steps)} steps:\n" + "\n".join(step_summaries)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_request": self.user_request,
            "request_type": self.request_type,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "completed_steps": self.completed_steps,
            "total_steps": len(self.steps),
            "progress_percentage": self.progress_percentage,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "total_actual_cost_usd": self.total_actual_cost_usd,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# Manager Agent
# =============================================================================

# Default prompts for the Manager Agent
MANAGER_SYSTEM_PROMPT = """You are a Task Manager Agent responsible for understanding user requests and creating execution plans.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. The "generator" agent CANNOT access user documents - it only has general LLM knowledge
2. The "research" agent CAN search user's uploaded documents via RAG
3. For ANY request about documents, files, or finding specific information:
   - ALWAYS start with a "research" step to search the user's documents
   - NEVER use "generator" alone when the user asks about their documents
   - Pass research results to subsequent steps via dependencies

Your role:
1. Analyze the user's request to understand their intent
2. Classify the request type (document_generation, qa_with_citations, analysis, comparison, research)
3. Decompose complex requests into ordered subtasks
4. Assign each subtask to the appropriate worker agent

Available worker agents:
- generator: Creates content using LLM knowledge (NO document access - use for writing/creating)
- critic: Evaluates quality and provides feedback
- research: Searches user's uploaded documents and retrieves information (HAS document access via RAG)
- tool: Executes file operations (generate PPTX, DOCX, PDF)
- validator: Cross-validates generated content against source documents to detect hallucinations

Guidelines:
- ALWAYS use "research" first when user mentions: documents, docs, files, list, find, search, uploaded, "in my", "from my"
- Keep plans focused and minimal - don't add unnecessary steps
- Each step should have clear inputs and expected outputs
- Mark dependencies between steps correctly - steps that need document content must depend on research
- Include critique steps for important deliverables

Output your plan as JSON with this structure:
{
    "request_type": "document_generation|qa_with_citations|analysis|comparison|research|other",
    "understanding": "Brief description of what user wants",
    "steps": [
        {
            "name": "Step name",
            "agent": "generator|critic|research|tool",
            "task_type": "generation|evaluation|research|tool_execution",
            "description": "What this step does",
            "depends_on": [0, 1],  // indices of prior steps this depends on
            "expected_output": "What this step produces"
        }
    ]
}"""

MANAGER_TASK_PROMPT = """User Request: {{request}}

{{context}}

Create an execution plan for this request. Consider:
1. What information is needed?
2. What content needs to be generated?
3. Should the output be reviewed/critiqued?
4. What format is expected?

Respond with your JSON plan:"""


class ManagerAgent(BaseAgent):
    """
    Lead orchestrator that coordinates worker agents.

    Responsibilities:
    - Parse user request and understand intent
    - Decompose into subtasks with dependencies
    - Create execution plans
    - Monitor progress and handle failures
    - Synthesize final output from worker results
    """

    # Common decomposition patterns
    DECOMPOSITION_PATTERNS = {
        "document_generation": [
            ("research", "Gather relevant information"),
            ("generator", "Create outline"),
            ("generator", "Draft content"),
            ("validator", "Validate content against sources"),
            ("critic", "Review and score"),
            ("generator", "Revise based on feedback"),
        ],
        "qa_with_citations": [
            ("research", "Retrieve relevant context"),
            ("generator", "Synthesize answer"),
            ("validator", "Verify answer against sources"),
            ("research", "Find supporting citations"),
            ("critic", "Verify accuracy"),
        ],
        "analysis": [
            ("research", "Gather data"),
            ("generator", "Analyze findings"),
            ("validator", "Validate analysis against sources"),
            ("critic", "Evaluate analysis"),
            ("generator", "Summarize conclusions"),
        ],
        "comparison": [
            ("research", "Research item A"),
            ("research", "Research item B"),
            ("generator", "Compare and contrast"),
            ("validator", "Validate comparison against sources"),
            ("critic", "Verify accuracy of comparison"),
        ],
        "research": [
            ("research", "Search documents"),
            ("research", "Search web if needed"),
            ("generator", "Compile findings"),
        ],
    }

    def __init__(
        self,
        config: AgentConfig,
        llm=None,
        prompt_template: Optional[PromptTemplate] = None,
        trajectory_collector=None,
        worker_registry: Optional[Dict[str, BaseAgent]] = None,
    ):
        """
        Initialize Manager Agent.

        Args:
            config: Agent configuration
            llm: LLM instance
            prompt_template: Custom prompt template
            trajectory_collector: Trajectory collector
            worker_registry: Dict mapping agent_type to agent instance
        """
        # Use default prompt if none provided
        if prompt_template is None:
            prompt_template = PromptTemplate(
                id="manager_default",
                version=1,
                system_prompt=MANAGER_SYSTEM_PROMPT,
                task_prompt_template=MANAGER_TASK_PROMPT,
            )

        super().__init__(
            config=config,
            llm=llm,
            prompt_template=prompt_template,
            trajectory_collector=trajectory_collector,
        )

        self.worker_registry = worker_registry or {}

    def register_worker(self, agent_type: str, agent: BaseAgent) -> None:
        """Register a worker agent."""
        self.worker_registry[agent_type] = agent
        logger.info(f"Registered worker agent: {agent_type}")

    async def plan_execution(
        self,
        user_request: str,
        session_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """
        Create execution plan for user request using LLM reasoning.

        Args:
            user_request: The user's request
            session_id: Session UUID
            user_id: User UUID
            context: Additional context

        Returns:
            ExecutionPlan with steps and dependencies
        """
        self.clear_trajectory()
        start_time = time.time()
        context = context or {}

        # Record planning step
        self.record_step(
            action_type="planning_start",
            input_data={"request": user_request},
            output_data={},
        )

        # Build context string
        context_parts = []
        if context.get("documents"):
            context_parts.append(f"Available documents: {context['documents']}")
        if context.get("session_history"):
            context_parts.append(f"Prior conversation context available")
        context_str = "\n".join(context_parts) if context_parts else ""

        # Get language from context (defaults to English)
        language = context.get("language", "en")

        # Build messages for LLM with language support
        messages = self.prompt_template.build_messages(
            task=user_request,
            context=context_str,
            language=language,
        )

        # Invoke LLM to create plan
        try:
            response_text, input_tokens, output_tokens = await self.invoke_llm(
                messages, record=True
            )

            # Parse JSON response
            plan_data = self._parse_plan_response(response_text)

            # Create execution plan (pass options from context for fallback logic)
            plan = self._build_execution_plan(
                plan_data=plan_data,
                user_request=user_request,
                session_id=session_id,
                user_id=user_id,
                options=context.get("options"),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            self.record_step(
                action_type="planning_complete",
                input_data={},
                output_data={
                    "plan_id": plan.id,
                    "step_count": len(plan.steps),
                    "request_type": plan.request_type,
                },
                tokens_used=input_tokens + output_tokens,
                duration_ms=duration_ms,
            )

            logger.info(
                "Created execution plan",
                plan_id=plan.id,
                request_type=plan.request_type,
                steps=len(plan.steps),
            )

            return plan

        except Exception as e:
            self.record_step(
                action_type="planning_failed",
                input_data={},
                output_data={},
                success=False,
                error_message=str(e),
            )
            raise

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into plan data."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.index("```") + 3
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response:
                # Find the JSON object
                json_start = response.index("{")
                # Find matching closing brace
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(response[json_start:], json_start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                json_str = response[json_start:json_end]
            else:
                json_str = response

            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse plan JSON, using fallback: {e}")
            # Fallback to simple plan
            return {
                "request_type": "other",
                "understanding": "Processing request",
                "steps": [
                    {
                        "name": "Process request",
                        "agent": "generator",
                        "task_type": "generation",
                        "description": "Generate response to user request",
                        "depends_on": [],
                        "expected_output": "Response text",
                    }
                ],
            }

    def _build_execution_plan(
        self,
        plan_data: Dict[str, Any],
        user_request: str,
        session_id: str,
        user_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Build ExecutionPlan from parsed plan data."""
        options = options or {}
        steps = []

        # Check if LLM plan already has research step
        has_research = any(
            s.get("agent") == "research"
            for s in plan_data.get("steps", [])
        )

        # FALLBACK: If LLM didn't add research but query needs documents
        # Keywords that indicate user wants to search their documents
        document_keywords = [
            "document", "docs", "file", "list", "find", "search",
            "all the", "in my", "from my", "uploaded", "extract",
            "summarize", "what", "which", "show me"
        ]
        needs_research = any(kw in user_request.lower() for kw in document_keywords)

        # User can also force document search via options
        force_document_search = options.get("search_documents", False)

        # Prepend research step if needed and not present
        if (needs_research or force_document_search) and not has_research:
            logger.warning(
                "LLM plan missing research step for document query, adding fallback",
                user_request=user_request[:100],
            )
            research_task = AgentTask(
                id=str(uuid.uuid4()),
                type=TaskType.RESEARCH,
                name="Search Documents",
                description=f"Search user's uploaded documents for: {user_request}",
                expected_outputs={"output": "Relevant document excerpts"},
                fallback_strategy=FallbackStrategy.RETRY,
                max_retries=2,
            )
            research_step = PlanStep(
                id=str(uuid.uuid4()),
                agent_type="research",
                task=research_task,
                dependencies=[],
            )
            steps.append(research_step)

        # Track if we prepended a research step (for dependency adjustment)
        prepended_research = len(steps) > 0

        for i, step_data in enumerate(plan_data.get("steps", [])):
            # Map task type string to enum
            task_type_str = step_data.get("task_type", "generation")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.GENERATION

            # Create task
            task = AgentTask(
                id=str(uuid.uuid4()),
                type=task_type,
                name=step_data.get("name", f"Step {i+1}"),
                description=step_data.get("description", ""),
                expected_outputs={"output": step_data.get("expected_output", "")},
                fallback_strategy=FallbackStrategy.RETRY,
                max_retries=2,
            )

            # Map dependency indices to step IDs
            depends_on_indices = step_data.get("depends_on", [])
            dependencies = []

            # If we prepended a research step, adjust indices and make first LLM step depend on it
            offset = 1 if prepended_research else 0

            for dep_idx in depends_on_indices:
                # Handle both int and string indices from LLM
                try:
                    idx = int(dep_idx) if isinstance(dep_idx, str) else dep_idx
                    # Adjust index for prepended research step
                    adjusted_idx = idx + offset
                    if 0 <= adjusted_idx < len(steps):
                        dependencies.append(steps[adjusted_idx].id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid dependency index: {dep_idx}")

            # If we prepended research and this is the first step with no dependencies,
            # make it depend on the research step
            if prepended_research and not dependencies and i == 0:
                dependencies.append(steps[0].id)  # Research step is first

            # Create plan step
            plan_step = PlanStep(
                id=str(uuid.uuid4()),
                agent_type=step_data.get("agent", "generator"),
                task=task,
                dependencies=dependencies,
            )
            steps.append(plan_step)

        return ExecutionPlan(
            id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            user_request=user_request,
            request_type=plan_data.get("request_type", "other"),
            steps=steps,
        )

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute plan by dispatching to worker agents.

        Yields progress updates as execution proceeds.

        Args:
            plan: Execution plan to execute
            context: Additional context passed to workers

        Yields:
            Progress updates with step results
        """
        context = context or {}
        plan.status = PlanStatus.EXECUTING
        plan.started_at = datetime.utcnow()
        plan_start_time = time.time()  # Track for trajectory recording

        yield {
            "type": "plan_started",
            "plan_id": plan.id,
            "total_steps": len(plan.steps),
        }

        # Track results for dependent steps
        step_results: Dict[str, AgentResult] = {}

        # Pass original user request to workers so research agents can search
        # with the actual query, not the LLM-rewritten task description
        context = context or {}
        context["user_request"] = plan.user_request

        while True:
            # Get steps ready to execute
            ready_steps = plan.get_ready_steps()

            if not ready_steps:
                # Check if all done or blocked
                pending = [s for s in plan.steps if s.status == TaskStatus.PENDING]
                if not pending:
                    break  # All complete
                else:
                    # Still pending but nothing ready - shouldn't happen
                    logger.error("Plan stuck: pending steps but none ready")
                    plan.status = PlanStatus.FAILED
                    plan.error_message = "Plan execution stuck"
                    break

            # Execute ready steps in PARALLEL if multiple are ready
            # This improves performance when steps don't depend on each other
            if len(ready_steps) > 1:
                logger.info(
                    "Executing steps in parallel",
                    parallel_steps=len(ready_steps),
                    step_names=[s.task.name for s in ready_steps],
                )

                # Emit step_started for all parallel steps
                for step in ready_steps:
                    yield {
                        "type": "step_started",
                        "step_id": step.id,
                        "step_name": step.task.name,
                        "agent_type": step.agent_type,
                        "progress": plan.progress_percentage,
                        "parallel": True,
                    }
                    step.status = TaskStatus.IN_PROGRESS
                    step.started_at = datetime.utcnow()

                # Execute all ready steps in parallel
                import asyncio

                async def execute_step_parallel(step: PlanStep) -> tuple:
                    """Execute a single step and return (step, result, error)."""
                    step_start_time = time.time()
                    try:
                        step_context = dict(context)
                        step_context["dependency_results"] = {
                            dep_id: step_results[dep_id].output
                            for dep_id in step.dependencies
                            if dep_id in step_results
                        }
                        result = await self._execute_step(step, step_context)
                        return (step, result, None, step_start_time)
                    except Exception as e:
                        return (step, None, e, step_start_time)

                # Run all steps in parallel
                parallel_results = await asyncio.gather(
                    *[execute_step_parallel(step) for step in ready_steps],
                    return_exceptions=True,
                )

                # Process results from parallel execution
                for parallel_result in parallel_results:
                    if isinstance(parallel_result, Exception):
                        logger.error(f"Parallel step failed with exception: {parallel_result}")
                        continue

                    step, result, error, step_start_time = parallel_result

                    if error:
                        logger.error(f"Step execution error: {error}", exc_info=True)
                        step.status = TaskStatus.FAILED
                        step.result = AgentResult(
                            task_id=step.task.id,
                            agent_id=self.agent_id,
                            status=TaskStatus.FAILED,
                            output=None,
                            error_message=str(error),
                        )
                        yield {
                            "type": "step_failed",
                            "step_id": step.id,
                            "step_name": step.task.name,
                            "error": str(error),
                        }
                    else:
                        step.result = result
                        step_results[step.id] = result

                        # Record trajectory
                        step_trajectory = getattr(result, 'trajectory_steps', None) or []
                        if step_trajectory and self.trajectory_collector:
                            try:
                                await self.trajectory_collector.record_trajectory_by_type(
                                    session_id=context.get("session_id", plan.session_id),
                                    agent_type=step.agent_type,
                                    task_type=step.task.type.value,
                                    input_summary=step.task.description[:500] if step.task.description else step.task.name,
                                    steps=step_trajectory,
                                    success=result.status == TaskStatus.COMPLETED,
                                    quality_score=result.confidence_score,
                                    error_message=result.error_message,
                                    total_tokens=result.tokens_used or 0,
                                    total_duration_ms=int((time.time() - step_start_time) * 1000),
                                    total_cost_usd=getattr(result, 'cost_usd', 0.0) or 0.0,
                                )
                            except Exception as traj_err:
                                logger.warning(f"Failed to record step trajectory: {traj_err}")

                        if result.is_success:
                            step.status = TaskStatus.COMPLETED
                            step.actual_cost_usd = self._estimate_step_cost(result)
                            plan.total_actual_cost_usd += step.actual_cost_usd

                            full_output = None
                            if result.output:
                                if isinstance(result.output, dict):
                                    full_output = result.output.get("findings", str(result.output))
                                else:
                                    full_output = str(result.output)

                            yield {
                                "type": "step_completed",
                                "step_id": step.id,
                                "step_index": plan.current_step_index,
                                "step_name": step.task.name,
                                "agent_type": step.agent_type,
                                "status": "success",
                                "output_preview": full_output[:200] if full_output else None,
                                "full_output": full_output,
                                "progress": plan.progress_percentage,
                                "parallel": True,
                            }

                            if result.output:
                                output_text = result.output
                                if isinstance(result.output, dict):
                                    output_text = result.output.get("findings", str(result.output))
                                    sources = result.output.get("sources", [])
                                    if sources and isinstance(sources, list) and len(sources) > 0:
                                        if isinstance(sources[0], dict) and "content" in sources[0]:
                                            yield {"type": "sources", "data": sources}
                                yield {
                                    "type": "content",
                                    "data": f"**{step.task.name}**\n\n{output_text}\n\n---\n\n"
                                }
                        else:
                            recovery = await self._handle_step_failure(step, result, plan, context)
                            yield recovery

                    step.completed_at = datetime.utcnow()
                    plan.current_step_index += 1

            else:
                # Single step - execute sequentially (original behavior)
                for step in ready_steps:
                    yield {
                        "type": "step_started",
                        "step_id": step.id,
                        "step_name": step.task.name,
                        "agent_type": step.agent_type,
                        "progress": plan.progress_percentage,
                    }

                    step.status = TaskStatus.IN_PROGRESS
                    step.started_at = datetime.utcnow()

                    try:
                        # Build step context from dependencies
                        step_context = dict(context)
                        step_context["dependency_results"] = {
                            dep_id: step_results[dep_id].output
                            for dep_id in step.dependencies
                            if dep_id in step_results
                        }

                        # Execute step
                        step_start_time = time.time()
                        result = await self._execute_step(step, step_context)
                        step.result = result
                        step_results[step.id] = result

                        # Record trajectory for this step
                        step_trajectory = getattr(result, 'trajectory_steps', None) or []
                        if step_trajectory and self.trajectory_collector:
                            try:
                                await self.trajectory_collector.record_trajectory_by_type(
                                    session_id=context.get("session_id", plan.session_id),
                                    agent_type=step.agent_type,
                                    task_type=step.task.type.value,
                                    input_summary=step.task.description[:500] if step.task.description else step.task.name,
                                    steps=step_trajectory,
                                    success=result.status == TaskStatus.COMPLETED,
                                    quality_score=result.confidence_score,
                                    error_message=result.error_message,
                                    total_tokens=result.tokens_used or 0,
                                    total_duration_ms=int((time.time() - step_start_time) * 1000),
                                    total_cost_usd=getattr(result, 'cost_usd', 0.0) or 0.0,
                                )
                            except Exception as traj_err:
                                logger.warning(f"Failed to record step trajectory: {traj_err}")

                        if result.is_success:
                            step.status = TaskStatus.COMPLETED
                            step.actual_cost_usd = self._estimate_step_cost(result)
                            plan.total_actual_cost_usd += step.actual_cost_usd

                            # Prepare full output for step display
                            full_output = None
                            if result.output:
                                if isinstance(result.output, dict):
                                    # Research agent returns {"findings": ..., "sources": ...}
                                    full_output = result.output.get("findings", str(result.output))
                                else:
                                    full_output = str(result.output)

                            yield {
                                "type": "step_completed",
                                "step_id": step.id,
                                "step_index": plan.current_step_index,
                                "step_name": step.task.name,
                                "agent_type": step.agent_type,
                                "status": "success",
                                "output_preview": full_output[:200] if full_output else None,
                                "full_output": full_output,  # Full output for detailed view
                                "progress": plan.progress_percentage,
                            }

                            # Stream step output as content for real-time display
                            if result.output:
                                output_text = result.output
                                if isinstance(result.output, dict):
                                    # Research agent returns {"findings": ..., "sources": ...}
                                    output_text = result.output.get("findings", str(result.output))

                                    # Emit sources if this was a research step with document sources
                                    sources = result.output.get("sources", [])
                                    if sources and isinstance(sources, list) and len(sources) > 0:
                                        # Only emit if sources have actual document data (not just strings)
                                        if isinstance(sources[0], dict) and "content" in sources[0]:
                                            yield {
                                                "type": "sources",
                                                "data": sources
                                            }
                                yield {
                                    "type": "content",
                                    "data": f"**{step.task.name}**\n\n{output_text}\n\n---\n\n"
                                }
                        else:
                            # Handle failure based on strategy
                            recovery = await self._handle_step_failure(
                                step, result, plan, context
                            )
                            yield recovery

                    except Exception as e:
                        logger.error(f"Step execution error: {e}", exc_info=True)
                        step.status = TaskStatus.FAILED
                        step.result = AgentResult(
                            task_id=step.task.id,
                            agent_id=self.agent_id,
                            status=TaskStatus.FAILED,
                            output=None,
                            error_message=str(e),
                        )

                        yield {
                            "type": "step_failed",
                            "step_id": step.id,
                            "step_name": step.task.name,
                            "error": str(e),
                        }

                    step.completed_at = datetime.utcnow()
                    plan.current_step_index += 1

        # Plan completed
        plan.completed_at = datetime.utcnow()

        if plan.failed_steps == 0:
            plan.status = PlanStatus.COMPLETED
            # Synthesize final output
            plan.final_output = await self._synthesize_output(plan, step_results, context)

            yield {
                "type": "plan_completed",
                "plan_id": plan.id,
                "status": "success",
                "output": plan.final_output,
                "total_cost_usd": plan.total_actual_cost_usd,
            }
        else:
            plan.status = PlanStatus.FAILED
            plan.error_message = f"{plan.failed_steps} steps failed"

            yield {
                "type": "plan_completed",
                "plan_id": plan.id,
                "status": "failed",
                "error": plan.error_message,
                "completed_steps": plan.completed_steps,
                "failed_steps": plan.failed_steps,
            }

        # Record manager agent's overall trajectory
        if self._current_trajectory and self.trajectory_collector:
            try:
                await self.trajectory_collector.record_trajectory_by_type(
                    session_id=plan.session_id,
                    agent_type="manager",
                    task_type="orchestration",
                    input_summary=plan.user_request[:500] if plan.user_request else "Agent execution",
                    steps=self._current_trajectory,
                    success=plan.status == PlanStatus.COMPLETED,
                    quality_score=None,
                    error_message=plan.error_message,
                    total_tokens=sum((r.tokens_used or 0) for r in step_results.values()),
                    total_duration_ms=int((time.time() - plan_start_time) * 1000),
                    total_cost_usd=plan.total_actual_cost_usd or 0.0,
                )
            except Exception as traj_err:
                logger.warning(f"Failed to record manager trajectory: {traj_err}")

    async def _execute_step(
        self,
        step: PlanStep,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Execute a single plan step using appropriate worker."""
        worker = self.worker_registry.get(step.agent_type)

        if not worker:
            logger.warning(f"No worker for {step.agent_type}, using fallback")
            # Fallback: execute with manager's LLM
            return await self._fallback_execute(step, context)

        return await worker.execute_with_retry(step.task, context)

    async def _fallback_execute(
        self,
        step: PlanStep,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Fallback execution when worker not available."""
        start_time = time.time()

        # Build simple prompt
        prompt = f"""Task: {step.task.name}
Description: {step.task.description}

Context from previous steps:
{json.dumps(context.get('dependency_results', {}), indent=2)}

Please complete this task and provide your response."""

        messages = [
            SystemMessage(content="You are a helpful assistant completing a task."),
            HumanMessage(content=prompt),
        ]

        try:
            response_text, input_tokens, output_tokens = await self.invoke_llm(
                messages, record=True
            )

            return AgentResult(
                task_id=step.task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output=response_text,
                tokens_used=input_tokens + output_tokens,
                duration_ms=int((time.time() - start_time) * 1000),
                confidence_score=0.7,  # Lower confidence for fallback
            )

        except Exception as e:
            return AgentResult(
                task_id=step.task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(e),
            )

    async def _handle_step_failure(
        self,
        step: PlanStep,
        result: AgentResult,
        plan: ExecutionPlan,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle step failure based on fallback strategy."""
        strategy = step.task.fallback_strategy

        if strategy == FallbackStrategy.SKIP:
            step.status = TaskStatus.SKIPPED
            return {
                "type": "step_skipped",
                "step_id": step.id,
                "step_name": step.task.name,
                "reason": "Non-critical step skipped after failure",
            }

        elif strategy == FallbackStrategy.ESCALATE:
            step.status = TaskStatus.FAILED
            return {
                "type": "step_escalated",
                "step_id": step.id,
                "step_name": step.task.name,
                "error": result.error_message,
                "requires_attention": True,
            }

        else:  # RETRY or ALTERNATIVE
            step.status = TaskStatus.FAILED
            return {
                "type": "step_failed",
                "step_id": step.id,
                "step_name": step.task.name,
                "error": result.error_message,
            }

    async def _synthesize_output(
        self,
        plan: ExecutionPlan,
        step_results: Dict[str, AgentResult],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synthesize final output from all step results."""
        # Collect outputs from all steps, normalizing dict outputs to strings
        outputs = []
        for step in plan.steps:
            if step.id in step_results and step_results[step.id].output:
                output = step_results[step.id].output
                # Normalize dict outputs to strings for consistent synthesis
                if isinstance(output, dict):
                    # Extract meaningful content from structured outputs
                    if "findings" in output:
                        output = output["findings"]  # Research agent
                    elif "evaluation" in output:
                        output = str(output.get("evaluation", output))  # Critic agent
                    elif "result" in output:
                        output = str(output.get("result", output))  # Tool agent
                    else:
                        output = str(output)
                outputs.append({
                    "step": step.task.name,
                    "output": output,
                })

        if len(outputs) == 1:
            # Single step - return its output directly
            return outputs[0]["output"]

        # Multiple steps - synthesize with clear formatting
        step_outputs_text = "\n\n".join([
            f"=== Step {i+1}: {o['step']} ===\n{o['output']}"
            for i, o in enumerate(outputs)
        ])

        synthesis_prompt = f"""The user asked:
"{plan.user_request}"

The following steps were completed to fulfill this request:

{step_outputs_text}

Please synthesize these outputs into a coherent final response that DIRECTLY addresses what the user asked for.
IMPORTANT: If the user asked to CREATE something (slogan, content, strategy, etc.), make sure your response INCLUDES that creation.
Do not just summarize the research - provide the requested deliverable."""

        # Add language instruction so synthesis respects user's language.
        # For synthesis, don't use auto-detect (its examples confuse small models).
        # Workers already handled language; synthesis just needs a simple constraint.
        language = (context or {}).get("language", "en")
        effective_lang = "en" if language in ("auto", "en", "") else language
        lang_instruction = _get_language_instruction(effective_lang, auto_detect=False)

        system_content = "You are synthesizing multiple agent outputs into a coherent final response for the user."
        if lang_instruction:
            system_content += f"\n{lang_instruction}"

        # Verification instruction — synthesis-specific (not in invoke_llm)
        # Asks the model to trace claims back to step outputs
        enable_verification = (context or {}).get("enable_verification", False)
        if enable_verification:
            synthesis_prompt += """

VERIFICATION: After synthesizing, verify each key claim against the step outputs above.
If a claim cannot be traced to the provided step outputs, mark it as [unverified]."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=synthesis_prompt),
        ]

        # Intelligence grounding + CoT are handled centrally by invoke_llm()
        # (with small-model gating). Only verification is synthesis-specific above.
        try:
            response, _, _ = await self.invoke_llm(
                messages,
                record=True,
                temperature_override=(context or {}).get("temperature_override"),
                intelligence_level=(context or {}).get("intelligence_level"),
                enable_cot=(context or {}).get("enable_cot", False),
            )
            return response
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: concatenate outputs
            return "\n\n".join(o["output"] for o in outputs)

    def _estimate_step_cost(self, result: AgentResult) -> float:
        """Estimate cost of a step from its result."""
        # Simple estimation based on tokens
        # Override with actual pricing in cost_estimator
        tokens = result.tokens_used or (result.input_tokens + result.output_tokens)
        # Rough estimate: $0.01 per 1000 tokens (varies by model)
        return tokens * 0.00001

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute a task (implementation of abstract method).

        For Manager Agent, this creates and executes a plan.
        """
        # Extract request from task or context
        user_request = context.get("user_request") or task.description
        session_id = context.get("session_id", str(uuid.uuid4()))
        user_id = context.get("user_id", str(uuid.uuid4()))

        # Create plan
        plan = await self.plan_execution(
            user_request=user_request,
            session_id=session_id,
            user_id=user_id,
            context=context,
        )

        # Execute plan and collect results
        final_output = None
        async for update in self.execute_plan(plan, context):
            if update.get("type") == "plan_completed":
                final_output = update.get("output")

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED if plan.status == PlanStatus.COMPLETED else TaskStatus.FAILED,
            output=final_output,
            error_message=plan.error_message,
            trajectory_steps=self._current_trajectory,
        )

    async def execute_with_streaming(
        self,
        user_request: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a user request with enhanced streaming support.

        Provides real-time progress updates including:
        - Plan creation progress
        - Step execution status
        - Content chunks as they're generated
        - Source references
        - Final synthesis

        This is ideal for frontend integrations that need real-time feedback.

        Args:
            user_request: The user's request to process
            session_id: Optional session ID
            user_id: Optional user ID
            context: Additional context

        Yields:
            Streaming events with types:
            - "planning": Plan is being created
            - "plan_ready": Plan created with step overview
            - "step_started": Step execution began
            - "step_progress": Intermediate progress (0-100%)
            - "content_chunk": Partial content as it's generated
            - "step_completed": Step finished successfully
            - "step_failed": Step failed with error
            - "sources": Document sources used
            - "synthesis_started": Final synthesis in progress
            - "complete": Final result ready
            - "error": Execution error
        """
        context = context or {}
        session_id = session_id or str(uuid.uuid4())
        user_id = user_id or str(uuid.uuid4())

        try:
            # Emit planning start
            yield {
                "type": "planning",
                "message": "Analyzing request and creating execution plan...",
            }

            # Create plan
            plan = await self.plan_execution(
                user_request=user_request,
                session_id=session_id,
                user_id=user_id,
                context=context,
            )

            # Emit plan ready with overview
            yield {
                "type": "plan_ready",
                "plan_id": plan.id,
                "total_steps": len(plan.steps),
                "steps": [
                    {
                        "id": step.id,
                        "name": step.task.name,
                        "agent_type": step.agent_type,
                        "dependencies": step.dependencies,
                    }
                    for step in plan.steps
                ],
                "estimated_cost_usd": plan.total_estimated_cost_usd,
            }

            # Execute plan with streaming updates
            async for update in self.execute_plan(plan, context):
                event_type = update.get("type", "unknown")

                # Transform standard events to enhanced streaming format
                if event_type == "step_started":
                    yield {
                        "type": "step_started",
                        "step_id": update.get("step_id"),
                        "step_name": update.get("step_name"),
                        "agent_type": update.get("agent_type"),
                        "progress_percent": update.get("progress", 0),
                        "parallel": update.get("parallel", False),
                    }

                elif event_type == "step_completed":
                    # Emit step progress before completion
                    yield {
                        "type": "step_progress",
                        "step_id": update.get("step_id"),
                        "progress_percent": 100,
                    }

                    # Emit content if available
                    full_output = update.get("full_output")
                    if full_output:
                        yield {
                            "type": "content_chunk",
                            "step_id": update.get("step_id"),
                            "step_name": update.get("step_name"),
                            "content": full_output,
                            "is_final": True,
                        }

                    yield {
                        "type": "step_completed",
                        "step_id": update.get("step_id"),
                        "step_name": update.get("step_name"),
                        "agent_type": update.get("agent_type"),
                        "progress_percent": update.get("progress", 0),
                    }

                elif event_type == "step_failed":
                    yield {
                        "type": "step_failed",
                        "step_id": update.get("step_id"),
                        "step_name": update.get("step_name"),
                        "error": update.get("error"),
                    }

                elif event_type == "sources":
                    yield {
                        "type": "sources",
                        "sources": update.get("data", []),
                    }

                elif event_type == "content":
                    yield {
                        "type": "content_chunk",
                        "content": update.get("data", ""),
                        "is_final": False,
                    }

                elif event_type == "plan_completed":
                    # Emit synthesis started
                    yield {
                        "type": "synthesis_started",
                        "message": "Combining results into final response...",
                    }

                    # Emit final result
                    if update.get("status") == "success":
                        yield {
                            "type": "complete",
                            "plan_id": update.get("plan_id"),
                            "output": update.get("output"),
                            "total_cost_usd": update.get("total_cost_usd", 0),
                        }
                    else:
                        yield {
                            "type": "error",
                            "plan_id": update.get("plan_id"),
                            "error": update.get("error"),
                            "completed_steps": update.get("completed_steps", 0),
                            "failed_steps": update.get("failed_steps", 0),
                        }

        except Exception as e:
            logger.error(f"Streaming execution error: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
            }
