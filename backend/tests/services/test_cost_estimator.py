"""
AIDocumentIndexer - Cost Estimator Tests
=========================================

Unit tests for the agent cost estimation service.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4
from decimal import Decimal

from backend.services.agents.cost_estimator import (
    AgentCostEstimator,
    CostEstimate,
    StepCostEstimate,
    BudgetCheckResult,
    MODEL_PRICING,
)
from backend.services.agents.agent_base import (
    AgentTask,
    TaskType,
    FallbackStrategy,
)
from backend.services.agents.manager_agent import (
    ExecutionPlan,
    PlanStep,
)


# =============================================================================
# AgentCostEstimator Tests
# =============================================================================

class TestAgentCostEstimator:
    """Tests for AgentCostEstimator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = AsyncMock()
        self.estimator = AgentCostEstimator(db=self.mock_db)

    def test_model_pricing_exists(self):
        """Test that model pricing data exists for common models."""
        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING
        assert "claude-3-5-sonnet-20241022" in MODEL_PRICING

    def test_model_pricing_structure(self):
        """Test model pricing structure."""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            # Note: local models have 0 pricing
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0

    def test_get_model_pricing_known(self):
        """Test getting pricing for a known model."""
        pricing = self.estimator.get_model_pricing("gpt-4o")
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00

    def test_get_model_pricing_unknown(self):
        """Test getting pricing for an unknown model returns default."""
        pricing = self.estimator.get_model_pricing("unknown-model")
        assert pricing["input"] == MODEL_PRICING["default"]["input"]
        assert pricing["output"] == MODEL_PRICING["default"]["output"]

    def test_calculate_token_cost_gpt4o(self):
        """Test cost calculation for GPT-4o."""
        input_tokens = 1000
        output_tokens = 500

        cost = self.estimator.calculate_token_cost(
            model="gpt-4o",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # GPT-4o pricing: $2.50/M input, $10.00/M output
        expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_calculate_token_cost_unknown_model(self):
        """Test cost calculation with unknown model uses default."""
        cost = self.estimator.calculate_token_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should return some cost using default pricing
        assert cost > 0

    def test_estimate_step_tokens(self):
        """Test token estimation for different task types."""
        # Generation tasks
        input_tokens, output_tokens = self.estimator.estimate_step_tokens("generation")
        assert input_tokens > 0
        assert output_tokens > 0

        # Research tasks
        input_tokens, output_tokens = self.estimator.estimate_step_tokens("research")
        assert input_tokens > 0
        assert output_tokens > 0

    def test_estimate_step_tokens_with_context(self):
        """Test token estimation with additional context."""
        base_input, base_output = self.estimator.estimate_step_tokens("generation", context_length=0)
        with_context_input, with_context_output = self.estimator.estimate_step_tokens("generation", context_length=1000)

        # Context should add to input tokens
        assert with_context_input > base_input
        # Output should be same
        assert with_context_output == base_output

    @pytest.mark.asyncio
    async def test_estimate_plan_cost(self):
        """Test estimating cost for an execution plan."""
        # Create a simple plan
        steps = [
            PlanStep(
                id="step-1",
                agent_type="research",
                task=AgentTask(
                    id="task-1",
                    type=TaskType.RESEARCH,
                    name="Research",
                    description="Search for documents",
                    expected_inputs={"query": {"type": "str", "required": True}},
                    expected_outputs={"results": {"type": "list"}},
                    success_criteria=[],
                    fallback_strategy=FallbackStrategy.RETRY,
                ),
                dependencies=[],
                estimated_cost_usd=0,
            ),
        ]

        plan = ExecutionPlan(
            id=str(uuid4()),
            session_id=str(uuid4()),
            user_id=str(uuid4()),
            user_request="Find relevant documents",
            request_type="research",
            steps=steps,
            total_estimated_cost_usd=0,
        )

        estimate = await self.estimator.estimate_plan_cost(plan)

        assert isinstance(estimate, CostEstimate)
        assert estimate.total_cost_usd >= 0
        assert len(estimate.steps) == 1

    @pytest.mark.asyncio
    async def test_check_budget_no_db(self):
        """Test budget check with no database returns allowed."""
        estimator_no_db = AgentCostEstimator(db=None)
        user_id = str(uuid4())
        estimated_cost = 0.50

        result = await estimator_no_db.check_budget(user_id, estimated_cost)

        assert isinstance(result, BudgetCheckResult)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_record_actual_cost_no_db(self):
        """Test recording actual cost with no database doesn't error."""
        estimator_no_db = AgentCostEstimator(db=None)
        user_id = str(uuid4())
        cost = 0.25
        operation = "agent_execution"

        # Should not raise an exception
        await estimator_no_db.record_actual_cost(user_id, cost, operation)

        # Verify no exceptions
        assert True


# =============================================================================
# CostEstimate Tests
# =============================================================================

class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_create_cost_estimate(self):
        """Test creating a CostEstimate."""
        steps = [
            StepCostEstimate(
                step_id="step-1",
                step_name="Research Step",
                agent_type="research",
                model="gpt-4o-mini",
                estimated_input_tokens=500,
                estimated_output_tokens=200,
                estimated_cost_usd=0.01,
            ),
            StepCostEstimate(
                step_id="step-2",
                step_name="Generate Step",
                agent_type="generator",
                model="gpt-4o",
                estimated_input_tokens=1000,
                estimated_output_tokens=800,
                estimated_cost_usd=0.05,
            ),
        ]

        estimate = CostEstimate(
            plan_id=str(uuid4()),
            total_cost_usd=0.06,
            steps=steps,
            currency="USD",
        )

        assert estimate.total_cost_usd == 0.06
        assert len(estimate.steps) == 2
        assert estimate.currency == "USD"

    def test_cost_estimate_to_dict(self):
        """Test CostEstimate to_dict serialization."""
        plan_id = str(uuid4())
        estimate = CostEstimate(
            plan_id=plan_id,
            total_cost_usd=0.05,
            steps=[],
            currency="USD",
            confidence=0.9,
        )

        result = estimate.to_dict()

        assert result["plan_id"] == plan_id
        assert result["total_cost_usd"] == 0.05
        assert result["currency"] == "USD"
        assert result["confidence"] == 0.9


# =============================================================================
# StepCostEstimate Tests
# =============================================================================

class TestStepCostEstimate:
    """Tests for StepCostEstimate dataclass."""

    def test_create_step_estimate(self):
        """Test creating a StepCostEstimate."""
        step = StepCostEstimate(
            step_id="step-1",
            step_name="Research",
            agent_type="research",
            model="gpt-4o-mini",
            estimated_input_tokens=500,
            estimated_output_tokens=200,
            estimated_cost_usd=0.01,
        )

        assert step.step_id == "step-1"
        assert step.step_name == "Research"
        assert step.agent_type == "research"
        assert step.model == "gpt-4o-mini"

    def test_step_estimate_to_dict(self):
        """Test StepCostEstimate to_dict serialization."""
        step = StepCostEstimate(
            step_id="step-1",
            step_name="Research",
            agent_type="research",
            model="gpt-4o-mini",
            estimated_input_tokens=500,
            estimated_output_tokens=200,
            estimated_cost_usd=0.01,
        )

        result = step.to_dict()

        assert result["step_id"] == "step-1"
        assert result["step_name"] == "Research"
        assert result["agent_type"] == "research"
        assert result["model"] == "gpt-4o-mini"


# =============================================================================
# BudgetCheckResult Tests
# =============================================================================

class TestBudgetCheckResult:
    """Tests for BudgetCheckResult dataclass."""

    def test_allowed_result(self):
        """Test creating an allowed BudgetCheckResult."""
        result = BudgetCheckResult(
            allowed=True,
            remaining_budget=5.0,
            estimated_cost=0.5,
            reason=None,
        )

        assert result.allowed is True
        assert result.remaining_budget == 5.0
        assert result.reason is None

    def test_denied_result(self):
        """Test creating a denied BudgetCheckResult."""
        result = BudgetCheckResult(
            allowed=False,
            remaining_budget=0.5,
            estimated_cost=2.0,
            reason="Would exceed daily budget",
        )

        assert result.allowed is False
        assert result.reason is not None
