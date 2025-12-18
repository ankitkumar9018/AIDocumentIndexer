"""
AIDocumentIndexer - Base Agent Abstractions
============================================

Core abstractions for the multi-agent system:
- AgentTask: Structured task definitions with validation schemas
- AgentResult: Standardized execution results
- BaseAgent: Abstract base class for all agent types

Design principles:
- Structured input/output validation
- Trajectory recording for self-improvement
- Configurable LLM providers per agent
- Graceful failure handling with fallback strategies
"""

import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================

class FallbackStrategy(str, Enum):
    """Strategy when agent execution fails."""
    RETRY = "retry"           # Retry with same prompt
    ESCALATE = "escalate"     # Escalate to manager/user
    SKIP = "skip"             # Skip this task (for non-critical)
    ALTERNATIVE = "alternative"  # Try alternative agent/approach


class TaskStatus(str, Enum):
    """Status of an agent task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of agent tasks."""
    GENERATION = "generation"       # Content creation
    EVALUATION = "evaluation"       # Quality assessment
    RESEARCH = "research"          # Information retrieval
    TOOL_EXECUTION = "tool_execution"  # File ops, exports
    DECOMPOSITION = "decomposition"  # Task breakdown
    SYNTHESIS = "synthesis"         # Combining results


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentConfig:
    """
    Configuration for an agent instance.

    Note: provider_type and model are now optional as agents use
    LLMConfigManager to get the admin-configured provider dynamically.
    These fields are kept for backward compatibility and override cases.
    """
    agent_id: str
    name: str
    description: str
    provider_type: Optional[str] = None  # Uses admin-configured default if None
    model: Optional[str] = None  # Uses admin-configured default if None
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    timeout_seconds: int = 120
    max_retries: int = 3
    tools: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())


@dataclass
class AgentTask:
    """
    Structured task definition for agent execution.

    Each task has:
    - Clear input/output expectations
    - Success criteria for validation
    - Fallback strategy for failures
    - Linked prompt template for versioning
    """
    id: str
    type: TaskType
    name: str
    description: str
    expected_inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    fallback_strategy: FallbackStrategy = FallbackStrategy.RETRY
    max_retries: int = 3
    timeout_seconds: int = 120
    prompt_template_id: Optional[str] = None
    priority: int = 5  # 1 (highest) to 10 (lowest)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.type, str):
            self.type = TaskType(self.type)
        if isinstance(self.fallback_strategy, str):
            self.fallback_strategy = FallbackStrategy(self.fallback_strategy)


@dataclass
class ValidationResult:
    """Result of input/output validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0 confidence score

    @classmethod
    def success(cls, score: float = 1.0) -> "ValidationResult":
        """Create successful validation result."""
        return cls(valid=True, score=score)

    @classmethod
    def failure(cls, errors: List[str], score: float = 0.0) -> "ValidationResult":
        """Create failed validation result."""
        return cls(valid=False, errors=errors, score=score)


@dataclass
class TrajectoryStep:
    """
    Single step in agent execution trajectory.

    Used for recording agent actions for analysis and self-improvement.
    """
    step_id: str
    timestamp: datetime
    agent_id: str
    action_type: str  # "reasoning", "tool_call", "llm_invoke", "output", "error"
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    tokens_used: int = 0
    duration_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.step_id:
            self.step_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


@dataclass
class AgentResult:
    """
    Standardized result from agent execution.

    Contains:
    - Output data
    - Reasoning trace for transparency
    - Tool calls made
    - Performance metrics
    - Confidence score
    """
    task_id: str
    agent_id: str
    status: TaskStatus
    output: Any
    reasoning_trace: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    confidence_score: float = 1.0
    error_message: Optional[str] = None
    trajectory_steps: List[TrajectoryStep] = field(default_factory=list)
    prompt_version_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string status to enum."""
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        # TaskStatus is a (str, Enum) so comparison should work directly
        # But for robustness, also check string value
        if isinstance(self.status, TaskStatus):
            return self.status == TaskStatus.COMPLETED
        # Fallback for string status
        return str(self.status).lower() == "completed"

    @property
    def is_partial(self) -> bool:
        """Check if execution was partially successful."""
        return self.status == TaskStatus.COMPLETED and self.confidence_score < 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "output": self.output,
            "reasoning_trace": self.reasoning_trace,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": self.duration_ms,
            "confidence_score": self.confidence_score,
            "error_message": self.error_message,
            "prompt_version_id": self.prompt_version_id,
            "metadata": self.metadata,
        }


# =============================================================================
# Prompt Template
# =============================================================================

@dataclass
class PromptTemplate:
    """
    Versioned prompt template for agents.

    Supports:
    - System prompt (role, capabilities, constraints)
    - Task prompt with {{variable}} placeholders
    - Few-shot examples
    - Output format specification
    """
    id: str
    version: int
    system_prompt: str
    task_prompt_template: str
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    constitutional_guidelines: List[str] = field(default_factory=list)

    def render(
        self,
        task: str,
        context: str = "",
        format_spec: str = "text",
        **kwargs
    ) -> str:
        """
        Render the task prompt with variables.

        Args:
            task: The specific task description
            context: Additional context
            format_spec: Output format specification
            **kwargs: Additional template variables

        Returns:
            Rendered prompt string
        """
        prompt = self.task_prompt_template

        # Replace standard variables
        prompt = prompt.replace("{{task}}", task)
        prompt = prompt.replace("{{context}}", context)
        prompt = prompt.replace("{{format}}", format_spec)

        # Replace custom variables
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

        # Add few-shot examples if present
        if self.few_shot_examples:
            examples_text = "\n\nExamples:\n"
            for i, example in enumerate(self.few_shot_examples, 1):
                examples_text += f"\nExample {i}:\n"
                examples_text += f"Input: {example.get('input', '')}\n"
                examples_text += f"Output: {example.get('output', '')}\n"
            prompt = prompt + examples_text

        # Add output format if specified
        if self.output_schema:
            prompt += f"\n\nExpected output format:\n{self.output_schema}"

        return prompt

    def build_messages(
        self,
        task: str,
        context: str = "",
        format_spec: str = "text",
        **kwargs
    ) -> List[Any]:
        """
        Build LangChain messages from template.

        Returns:
            List of SystemMessage and HumanMessage
        """
        messages = []

        # System message
        system_content = self.system_prompt
        if self.constitutional_guidelines:
            guidelines = "\n\nGuidelines:\n" + "\n".join(
                f"- {g}" for g in self.constitutional_guidelines
            )
            system_content += guidelines
        messages.append(SystemMessage(content=system_content))

        # User message with rendered task
        user_content = self.render(task, context, format_spec, **kwargs)
        messages.append(HumanMessage(content=user_content))

        return messages


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides:
    - Configurable LLM backend
    - Input/output validation
    - Trajectory recording
    - Prompt template management
    - Graceful error handling

    Subclasses must implement:
    - execute(): Core execution logic
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[BaseChatModel] = None,
        prompt_template: Optional[PromptTemplate] = None,
        trajectory_collector: Optional[Any] = None,  # Will be TrajectoryCollector
    ):
        """
        Initialize agent.

        Args:
            config: Agent configuration
            llm: Optional pre-configured LLM instance
            prompt_template: Optional prompt template
            trajectory_collector: Optional collector for recording execution
        """
        self.config = config
        self._llm = llm
        self.prompt_template = prompt_template
        self.trajectory_collector = trajectory_collector
        self._current_trajectory: List[TrajectoryStep] = []

        logger.info(
            "Agent initialized",
            agent_id=config.agent_id,
            name=config.name,
            provider=config.provider_type,
            model=config.model,
        )

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self.config.agent_id

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name

    async def get_llm(self) -> BaseChatModel:
        """
        Get or create LLM instance using admin-configured provider.

        Uses LLMConfigManager to respect:
        1. Operation-specific config (for "agent" operation)
        2. Default provider set in admin UI
        3. Environment fallback

        This ensures agents use the provider configured by admin,
        not hardcoded OpenAI.
        """
        if self._llm is None:
            from backend.services.llm import EnhancedLLMFactory

            # Get LLM using operation-based config resolution
            # This respects admin-configured default provider
            self._llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="agent",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            logger.debug(
                "Agent LLM initialized",
                agent=self.config.name,
                provider=config.provider_type,
                model=config.model,
                source=config.source,
            )

        return self._llm

    @property
    def llm(self) -> BaseChatModel:
        """
        Get LLM instance (sync property for backward compatibility).

        DEPRECATED: Use get_llm() async method instead.
        This property only works if _llm was already initialized.
        """
        if self._llm is None:
            raise RuntimeError(
                "LLM not initialized. Call 'await get_llm()' first, or use invoke_llm()."
            )
        return self._llm

    async def get_current_prompt(self) -> Optional[PromptTemplate]:
        """
        Get current prompt template, supporting A/B testing traffic split.

        Override this in subclasses to integrate with PromptVersionManager.
        """
        return self.prompt_template

    def validate_inputs(
        self,
        task: AgentTask,
        inputs: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate task inputs against expected schema.

        Args:
            task: Task definition
            inputs: Actual inputs

        Returns:
            ValidationResult with errors if invalid
        """
        errors = []
        warnings = []

        # Check required inputs
        for key, spec in task.expected_inputs.items():
            if spec.get("required", False) and key not in inputs:
                errors.append(f"Missing required input: {key}")
            elif key in inputs:
                # Type validation
                expected_type = spec.get("type")
                if expected_type and not self._check_type(inputs[key], expected_type):
                    errors.append(
                        f"Input '{key}' expected type {expected_type}, "
                        f"got {type(inputs[key]).__name__}"
                    )

        # Check for unexpected inputs
        expected_keys = set(task.expected_inputs.keys())
        actual_keys = set(inputs.keys())
        unexpected = actual_keys - expected_keys
        if unexpected:
            warnings.append(f"Unexpected inputs will be ignored: {unexpected}")

        if errors:
            return ValidationResult.failure(errors, score=0.0)

        score = 1.0 - (len(warnings) * 0.1)  # Small penalty for warnings
        return ValidationResult(valid=True, warnings=warnings, score=max(0.0, score))

    def validate_outputs(
        self,
        task: AgentTask,
        output: Any
    ) -> ValidationResult:
        """
        Validate task outputs against expected schema.

        Args:
            task: Task definition
            output: Actual output

        Returns:
            ValidationResult with errors if invalid
        """
        errors = []
        score = 1.0

        # Basic presence check
        if output is None:
            return ValidationResult.failure(["Output is None"], score=0.0)

        # If output is expected to be dict, validate structure
        if isinstance(task.expected_outputs, dict) and task.expected_outputs:
            if isinstance(output, dict):
                for key, spec in task.expected_outputs.items():
                    # spec might be a dict with "required" key, or just a string description
                    if isinstance(spec, dict) and spec.get("required", False) and key not in output:
                        errors.append(f"Missing required output field: {key}")
                        score -= 0.2
            else:
                # Output is not dict but spec expects dict - only check if spec values are dicts with required=True
                has_required = any(
                    isinstance(spec, dict) and spec.get("required", False)
                    for spec in task.expected_outputs.values()
                )
                if has_required:
                    errors.append("Output should be a dictionary")
                    score = 0.3

        # Check success criteria
        criteria_met = 0
        for criterion in task.success_criteria:
            if self._check_criterion(criterion, output):
                criteria_met += 1

        if task.success_criteria:
            criteria_score = criteria_met / len(task.success_criteria)
            score = (score + criteria_score) / 2

        if errors:
            return ValidationResult(valid=False, errors=errors, score=score)

        return ValidationResult(valid=True, score=score)

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type string."""
        type_map = {
            "string": str,
            "str": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }
        expected = type_map.get(expected_type.lower())
        if expected:
            return isinstance(value, expected)
        return True  # Unknown type, assume valid

    def _check_criterion(self, criterion: str, output: Any) -> bool:
        """
        Check if output meets a success criterion.

        Simple implementation - can be extended for more complex criteria.
        """
        criterion_lower = criterion.lower()

        # Check for non-empty output
        if "non-empty" in criterion_lower or "not empty" in criterion_lower:
            if isinstance(output, str):
                return len(output.strip()) > 0
            elif isinstance(output, (list, dict)):
                return len(output) > 0
            return output is not None

        # Check for minimum length
        if "min_length" in criterion_lower:
            # Parse min_length:100 format
            try:
                min_len = int(criterion.split(":")[-1])
                if isinstance(output, str):
                    return len(output) >= min_len
            except (ValueError, IndexError):
                pass

        # Default: criterion is met if we have output
        return output is not None

    def record_step(
        self,
        action_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        tokens_used: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> TrajectoryStep:
        """
        Record a step in the execution trajectory.

        Args:
            action_type: Type of action (reasoning, tool_call, llm_invoke, etc.)
            input_data: Input to the action
            output_data: Output from the action
            tokens_used: Tokens consumed
            duration_ms: Duration in milliseconds
            success: Whether step succeeded
            error_message: Error message if failed

        Returns:
            Created TrajectoryStep
        """
        step = TrajectoryStep(
            step_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            agent_id=self.agent_id,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )
        self._current_trajectory.append(step)
        return step

    def clear_trajectory(self) -> None:
        """Clear current trajectory for new execution."""
        self._current_trajectory = []

    async def invoke_llm(
        self,
        messages: List[Any],
        record: bool = True
    ) -> tuple[str, int, int]:
        """
        Invoke LLM with messages and optionally record the step.

        Args:
            messages: LangChain messages to send
            record: Whether to record in trajectory

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        start_time = time.time()

        try:
            # Get LLM using admin-configured provider
            llm = await self.get_llm()
            response = await llm.ainvoke(messages)

            # Extract response text
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            # Extract token usage if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
            elif hasattr(response, "response_metadata"):
                meta = response.response_metadata
                if "token_usage" in meta:
                    input_tokens = meta["token_usage"].get("prompt_tokens", 0)
                    output_tokens = meta["token_usage"].get("completion_tokens", 0)

            duration_ms = int((time.time() - start_time) * 1000)

            if record:
                self.record_step(
                    action_type="llm_invoke",
                    input_data={"messages": [str(m) for m in messages]},
                    output_data={"response": response_text[:500]},  # Truncate for storage
                    tokens_used=input_tokens + output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                )

            return response_text, input_tokens, output_tokens

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            if record:
                self.record_step(
                    action_type="llm_invoke",
                    input_data={"messages": [str(m) for m in messages]},
                    output_data={},
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                )

            raise

    @abstractmethod
    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute the agent task.

        Must be implemented by subclasses.

        Args:
            task: Task to execute
            context: Execution context (inputs, dependencies, etc.)

        Returns:
            AgentResult with output and metrics
        """
        pass

    async def execute_with_retry(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute task with automatic retry on failure.

        Uses task's fallback_strategy and max_retries.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            AgentResult (may indicate failure after all retries)
        """
        last_error = None

        for attempt in range(task.max_retries):
            try:
                self.clear_trajectory()
                result = await self.execute(task, context)

                # Debug: log status info
                logger.info(
                    "execute_with_retry status check",
                    status=result.status,
                    status_type=type(result.status).__name__,
                    is_success=result.is_success,
                    confidence=result.confidence_score,
                )

                if result.is_success:
                    return result

                # Partial success - might be acceptable
                if result.confidence_score >= 0.5:
                    logger.warning(
                        "Task completed with low confidence",
                        agent_id=self.agent_id,
                        task_id=task.id,
                        status=result.status,
                        status_type=type(result.status).__name__,
                        confidence=result.confidence_score,
                        attempt=attempt + 1,
                    )
                    return result

                last_error = result.error_message or "Low confidence result"

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Task execution failed, will retry",
                    agent_id=self.agent_id,
                    task_id=task.id,
                    attempt=attempt + 1,
                    max_retries=task.max_retries,
                    error=str(e),
                )

            # Check if we should continue retrying
            if task.fallback_strategy != FallbackStrategy.RETRY:
                break

        # All retries exhausted
        logger.error(
            "Task failed after all retries",
            agent_id=self.agent_id,
            task_id=task.id,
            max_retries=task.max_retries,
            error=last_error,
        )

        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status=TaskStatus.FAILED,
            output=None,
            error_message=f"Failed after {task.max_retries} attempts: {last_error}",
            trajectory_steps=self._current_trajectory,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<{self.__class__.__name__} "
            f"id={self.agent_id} "
            f"name={self.name} "
            f"model={self.config.model}>"
        )
