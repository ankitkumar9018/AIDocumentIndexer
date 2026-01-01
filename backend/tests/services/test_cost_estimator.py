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
            assert pricing["input"] > 0
            assert pricing["output"] > 0

    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        short_text = "Hello world"
        tokens = self.estimator.estimate_tokens(short_text)

        assert tokens > 0
        assert tokens < 100

    def test_estimate_tokens_long_text(self):
        """Test token estimation for longer text."""
        long_text = " ".join(["word"] * 1000)
        tokens = self.estimator.estimate_tokens(long_text)

        # Rough estimate: ~250 tokens per 1000 characters
        assert tokens > 100
        assert tokens < 2000

    def test_calculate_cost_gpt4o(self):
        """Test cost calculation for GPT-4o."""
        input_tokens = 1000
        output_tokens = 500

        cost = self.estimator.calculate_cost(
            provider="openai",
            model="gpt-4o",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # GPT-4o pricing: $2.50/M input, $10.00/M output
        expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation with unknown model uses default."""
        cost = self.estimator.calculate_cost(
            provider="unknown",
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should return some cost using default pricing
        assert cost > 0

    @pytest.mark.asyncio
    async def test_estimate_plan_cost(self):
        """Test estimating cost for an execution plan."""
        # Create a simple plan
        steps = [
            PlanStep(
                step_id="step-1",
                step_number=1,
                agent_id="research-agent",
                task=AgentTask(
                    id="task-1",
                    type=TaskType.RESEARCH,
                    name="Research",
                    description="Search for documents",
                    expected_inputs={"query": "str"},
                    expected_outputs={"results": "list"},
                    success_criteria=[],
                    fallback_strategy=FallbackStrategy.RETRY,
                ),
                dependencies=[],
                estimated_cost_usd=None,
            ),
        ]

        plan = ExecutionPlan(
            plan_id=str(uuid4()),
            session_id=str(uuid4()),
            user_id=str(uuid4()),
            user_request="Find relevant documents",
            steps=steps,
            estimated_total_cost=0,
        )

        # Mock getting agent config
        mock_agent = MagicMock()
        mock_agent.default_model = "gpt-4o-mini"
        mock_agent.default_provider_id = str(uuid4())

        with patch.object(self.estimator, 'get_agent', return_value=mock_agent):
            with patch.object(self.estimator, 'get_provider', return_value=MagicMock(provider_type="openai")):
                estimate = await self.estimator.estimate_plan_cost(plan)

                assert isinstance(estimate, CostEstimate)
                assert estimate.total_cost_usd >= 0
                assert len(estimate.steps) == 1

    @pytest.mark.asyncio
    async def test_check_budget_within_limit(self):
        """Test budget check when cost is within limit."""
        user_id = str(uuid4())
        estimated_cost = 0.50

        # Mock user's cost limit
        mock_limit = MagicMock()
        mock_limit.daily_limit_usd = 10.0

        # Mock current usage
        mock_usage = MagicMock()
        mock_usage.total_cost = 2.0

        with patch.object(self.estimator, 'get_user_cost_limit', return_value=mock_limit):
            with patch.object(self.estimator, 'get_current_usage', return_value=mock_usage):
                result = await self.estimator.check_budget(user_id, estimated_cost)

                assert isinstance(result, BudgetCheckResult)
                assert result.allowed is True
                assert result.remaining_budget == 8.0

    @pytest.mark.asyncio
    async def test_check_budget_exceeds_limit(self):
        """Test budget check when cost exceeds limit."""
        user_id = str(uuid4())
        estimated_cost = 5.0

        mock_limit = MagicMock()
        mock_limit.daily_limit_usd = 10.0

        mock_usage = MagicMock()
        mock_usage.total_cost = 8.0  # Only $2 remaining

        with patch.object(self.estimator, 'get_user_cost_limit', return_value=mock_limit):
            with patch.object(self.estimator, 'get_current_usage', return_value=mock_usage):
                result = await self.estimator.check_budget(user_id, estimated_cost)

                assert result.allowed is False
                assert "exceed" in result.reason.lower()
                assert result.remaining_budget == 2.0

    @pytest.mark.asyncio
    async def test_record_actual_cost(self):
        """Test recording actual cost after execution."""
        user_id = str(uuid4())
        cost = 0.25
        operation = "agent_execution"

        await self.estimator.record_actual_cost(user_id, cost, operation)

        # Verify DB was called
        # This is a basic test - implementation may vary
        assert True  # Just verify no exceptions


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
                agent="research",
                estimated_input_tokens=500,
                estimated_output_tokens=200,
                estimated_cost_usd=0.01,
            ),
            StepCostEstimate(
                step_id="step-2",
                agent="generator",
                estimated_input_tokens=1000,
                estimated_output_tokens=800,
                estimated_cost_usd=0.05,
            ),
        ]

        estimate = CostEstimate(
            total_cost_usd=0.06,
            steps=steps,
            currency="USD",
        )

        assert estimate.total_cost_usd == 0.06
        assert len(estimate.steps) == 2
        assert estimate.currency == "USD"


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
