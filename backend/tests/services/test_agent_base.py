"""
AIDocumentIndexer - Agent Base Tests
=====================================

Unit tests for the base agent abstractions.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

from backend.services.agents.agent_base import (
    AgentTask,
    AgentResult,
    ValidationResult,
    TrajectoryStep,
    BaseAgent,
    TaskType,
    TaskStatus,
    FallbackStrategy,
)
from backend.services.agents.manager_agent import (
    ExecutionPlan,
    PlanStep,
)


# =============================================================================
# AgentTask Tests
# =============================================================================

class TestAgentTask:
    """Tests for AgentTask dataclass."""

    def test_create_task_with_defaults(self):
        """Test creating AgentTask with default values."""
        task = AgentTask(
            id="task-1",
            type=TaskType.GENERATION,
            name="Test Task",
            description="A test task",
            expected_inputs={"input": "str"},
            expected_outputs={"output": "str"},
            success_criteria=["Must produce output"],
            fallback_strategy=FallbackStrategy.RETRY,
        )

        assert task.id == "task-1"
        assert task.type == TaskType.GENERATION
        assert task.max_retries == 3
        assert task.timeout_seconds == 120
        assert task.prompt_template_id is None

    def test_create_task_with_custom_values(self):
        """Test creating AgentTask with custom values."""
        task = AgentTask(
            id="task-2",
            type=TaskType.RESEARCH,
            name="Research Task",
            description="Research something",
            expected_inputs={"query": "str"},
            expected_outputs={"results": "list"},
            success_criteria=["Find relevant info"],
            fallback_strategy=FallbackStrategy.SKIP,
            max_retries=5,
            timeout_seconds=300,
            prompt_template_id="template-123",
        )

        assert task.max_retries == 5
        assert task.timeout_seconds == 300
        assert task.prompt_template_id == "template-123"


# =============================================================================
# AgentResult Tests
# =============================================================================

class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_create_successful_result(self):
        """Test creating a successful AgentResult."""
        result = AgentResult(
            task_id="task-1",
            agent_id="agent-1",
            status=TaskStatus.SUCCESS,
            output={"content": "Generated content"},
            reasoning_trace=["Step 1", "Step 2"],
            tool_calls=[{"tool": "search", "args": {}}],
            tokens_used=500,
            confidence_score=0.95,
            error_message=None,
        )

        assert result.status == TaskStatus.SUCCESS
        assert result.confidence_score == 0.95
        assert result.error_message is None

    def test_create_failed_result(self):
        """Test creating a failed AgentResult."""
        result = AgentResult(
            task_id="task-1",
            agent_id="agent-1",
            status=TaskStatus.FAILED,
            output=None,
            reasoning_trace=["Started", "Failed"],
            tool_calls=[],
            tokens_used=100,
            confidence_score=0.0,
            error_message="Task failed due to timeout",
        )

        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Task failed due to timeout"


# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid ValidationResult."""
        result = ValidationResult(valid=True, errors=[], score=1.0)

        assert result.valid is True
        assert len(result.errors) == 0
        assert result.score == 1.0

    def test_invalid_result(self):
        """Test creating an invalid ValidationResult."""
        result = ValidationResult(
            valid=False,
            errors=["Missing required field", "Invalid format"],
            score=0.3,
        )

        assert result.valid is False
        assert len(result.errors) == 2
        assert result.score == 0.3


# =============================================================================
# ExecutionPlan Tests
# =============================================================================

class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_create_execution_plan(self):
        """Test creating an ExecutionPlan."""
        steps = [
            PlanStep(
                step_id="step-1",
                step_number=1,
                agent_id="research-agent",
                task=AgentTask(
                    id="task-1",
                    type=TaskType.RESEARCH,
                    name="Research",
                    description="Research the topic",
                    expected_inputs={},
                    expected_outputs={},
                    success_criteria=[],
                    fallback_strategy=FallbackStrategy.RETRY,
                ),
                dependencies=[],
                estimated_cost_usd=0.01,
            ),
            PlanStep(
                step_id="step-2",
                step_number=2,
                agent_id="generator-agent",
                task=AgentTask(
                    id="task-2",
                    type=TaskType.GENERATION,
                    name="Generate",
                    description="Generate content",
                    expected_inputs={},
                    expected_outputs={},
                    success_criteria=[],
                    fallback_strategy=FallbackStrategy.RETRY,
                ),
                dependencies=["step-1"],
                estimated_cost_usd=0.05,
            ),
        ]

        plan = ExecutionPlan(
            plan_id=str(uuid4()),
            session_id=str(uuid4()),
            user_id=str(uuid4()),
            user_request="Generate a report",
            steps=steps,
            estimated_total_cost=0.06,
        )

        assert len(plan.steps) == 2
        assert plan.estimated_total_cost == 0.06


# =============================================================================
# BaseAgent Tests (via mock implementation)
# =============================================================================

class MockAgent(BaseAgent):
    """Mock agent for testing BaseAgent functionality."""

    async def execute(self, task: AgentTask, context: dict) -> AgentResult:
        """Mock execute method."""
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            status=TaskStatus.SUCCESS,
            output={"result": "Mock output"},
            reasoning_trace=["Executed mock task"],
            tool_calls=[],
            tokens_used=100,
            confidence_score=0.9,
            error_message=None,
        )


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_create_agent(self):
        """Test creating an agent."""
        agent = MockAgent(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
        )

        assert agent.agent_id == "test-agent"
        assert agent.name == "Test Agent"

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        agent = MockAgent(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
        )

        task = AgentTask(
            id="task-1",
            type=TaskType.GENERATION,
            name="Test",
            description="Test task",
            expected_inputs={"query": "str", "limit": "int"},
            expected_outputs={},
            success_criteria=[],
            fallback_strategy=FallbackStrategy.RETRY,
        )

        inputs = {"query": "test query", "limit": 10}
        result = agent.validate_inputs(task, inputs)

        assert result.valid is True

    def test_validate_inputs_missing(self):
        """Test input validation with missing inputs."""
        agent = MockAgent(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
        )

        task = AgentTask(
            id="task-1",
            type=TaskType.GENERATION,
            name="Test",
            description="Test task",
            expected_inputs={"query": "str", "limit": "int"},
            expected_outputs={},
            success_criteria=[],
            fallback_strategy=FallbackStrategy.RETRY,
        )

        inputs = {"query": "test query"}  # Missing 'limit'
        result = agent.validate_inputs(task, inputs)

        assert result.valid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test execute_with_retry on first attempt success."""
        agent = MockAgent(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
        )

        task = AgentTask(
            id="task-1",
            type=TaskType.GENERATION,
            name="Test",
            description="Test task",
            expected_inputs={},
            expected_outputs={},
            success_criteria=[],
            fallback_strategy=FallbackStrategy.RETRY,
            max_retries=3,
        )

        result = await agent.execute_with_retry(task, {})

        assert result.status == TaskStatus.SUCCESS


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types_exist(self):
        """Test that all expected task types exist."""
        assert TaskType.GENERATION.value == "generation"
        assert TaskType.EVALUATION.value == "evaluation"
        assert TaskType.RESEARCH.value == "research"
        assert TaskType.TOOL_EXECUTION.value == "tool_execution"
        assert TaskType.ORCHESTRATION.value == "orchestration"


class TestFallbackStrategy:
    """Tests for FallbackStrategy enum."""

    def test_fallback_strategies_exist(self):
        """Test that all expected fallback strategies exist."""
        assert FallbackStrategy.RETRY.value == "retry"
        assert FallbackStrategy.SKIP.value == "skip"
        assert FallbackStrategy.ESCALATE.value == "escalate"
        assert FallbackStrategy.ALTERNATIVE.value == "alternative"
