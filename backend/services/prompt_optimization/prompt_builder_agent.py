"""
AIDocumentIndexer - Prompt Builder Agent
=========================================

Background agent for self-improvement of agent prompts.

Capabilities:
1. Analyzes failed trajectories to detect patterns
2. Uses LLM reflection to understand failure causes
3. Generates improved prompt variants using GEPA-style mutations
4. Creates A/B tests for variants
5. Requires human approval before promotion

Trigger conditions:
- Success rate drops below 70%
- Quality score drops below 3.5/5
- Manual admin trigger
- Scheduled daily analysis
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, SystemMessage

from backend.db.models import (
    AgentDefinition,
    AgentPromptVersion,
    AgentTrajectory,
    PromptOptimizationJob,
)
from backend.services.agents.agent_base import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
    TaskStatus,
    PromptTemplate,
)
from backend.services.agents.trajectory_collector import TrajectoryCollector

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class MutationStrategy(str, Enum):
    """Strategies for prompt mutation."""
    REPHRASE_INSTRUCTIONS = "rephrase_instructions"
    ADD_EXAMPLES = "add_examples"
    ADD_GUARDRAILS = "add_guardrails"
    RESTRUCTURE_FORMAT = "restructure_format"
    ADD_CHAIN_OF_THOUGHT = "add_chain_of_thought"
    SIMPLIFY = "simplify"
    ADD_CONSTRAINTS = "add_constraints"


@dataclass
class FailurePattern:
    """Pattern detected in failed trajectories."""
    pattern_id: str
    description: str
    frequency: int  # How many times this pattern occurred
    example_trajectories: List[str]  # IDs of example failed trajectories
    root_causes: List[str]  # Identified root causes
    suggested_fixes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "frequency": self.frequency,
            "example_count": len(self.example_trajectories),
            "root_causes": self.root_causes,
            "suggested_fixes": self.suggested_fixes,
        }


@dataclass
class FailureAnalysis:
    """Complete analysis of agent failures."""
    agent_id: str
    analysis_window_hours: int
    total_trajectories: int
    failed_trajectories: int
    failure_rate: float
    patterns: List[FailurePattern] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "analysis_window_hours": self.analysis_window_hours,
            "total_trajectories": self.total_trajectories,
            "failed_trajectories": self.failed_trajectories,
            "failure_rate": self.failure_rate,
            "patterns": [p.to_dict() for p in self.patterns],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


@dataclass
class PromptMutation:
    """A mutated prompt variant."""
    id: str
    strategy: MutationStrategy
    original_version_id: str
    system_prompt: str
    task_prompt_template: str
    few_shot_examples: List[Dict[str, str]]
    change_description: str
    expected_improvement: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "change_description": self.change_description,
            "expected_improvement": self.expected_improvement,
        }


# =============================================================================
# Prompt Builder Agent
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are a Prompt Analysis Agent that identifies patterns in failed LLM agent executions.

Your task is to analyze failed trajectories and:
1. Identify common failure patterns
2. Determine root causes for each pattern
3. Suggest specific fixes for the prompts

Focus on actionable insights that can improve prompt engineering."""

ANALYSIS_TASK_PROMPT = """Analyze these failed agent trajectories for agent "{{agent_name}}":

Current System Prompt:
{{system_prompt}}

Current Task Template:
{{task_template}}

Failed Trajectories ({{failure_count}} failures):
{{trajectories}}

Analyze the failures and provide:
1. Common patterns you observe
2. Root causes for each pattern
3. Specific prompt improvements to address each issue

Output as JSON:
{
    "patterns": [
        {
            "description": "Pattern description",
            "frequency_estimate": "high/medium/low",
            "root_causes": ["cause1", "cause2"],
            "suggested_fixes": ["fix1", "fix2"]
        }
    ],
    "summary": "Overall analysis summary",
    "recommendations": ["Top recommendation 1", "Top recommendation 2"]
}"""

MUTATION_SYSTEM_PROMPT = """You are a Prompt Engineering Agent that improves LLM prompts based on failure analysis.

Your task is to create improved versions of prompts using specific mutation strategies.
Each mutation should directly address identified issues while maintaining the original intent."""

MUTATION_TASK_PROMPT = """Create an improved prompt variant using the "{{strategy}}" strategy.

Original System Prompt:
{{system_prompt}}

Original Task Template:
{{task_template}}

Failure Analysis:
{{failure_analysis}}

Strategy Description:
{{strategy_description}}

Create an improved version that addresses the identified issues. Output as JSON:
{
    "system_prompt": "Improved system prompt",
    "task_prompt_template": "Improved task template with {{placeholders}}",
    "few_shot_examples": [
        {"input": "example input", "output": "example output"}
    ],
    "change_description": "What was changed and why",
    "expected_improvement": "How this should improve performance"
}"""


class PromptBuilderAgent(BaseAgent):
    """
    Background agent that analyzes performance and generates improved prompts.

    Runs on triggers:
    - Success rate drops below threshold
    - Quality score drops below threshold
    - Manual admin trigger
    - Scheduled daily run
    """

    # Thresholds for triggering optimization
    SUCCESS_RATE_THRESHOLD = 0.7
    QUALITY_SCORE_THRESHOLD = 3.5
    MIN_TRAJECTORIES_FOR_ANALYSIS = 10

    # Strategy descriptions for mutations
    STRATEGY_DESCRIPTIONS = {
        MutationStrategy.REPHRASE_INSTRUCTIONS: (
            "Rephrase unclear or ambiguous instructions to be more specific and direct. "
            "Replace vague language with concrete, actionable directives."
        ),
        MutationStrategy.ADD_EXAMPLES: (
            "Add few-shot examples that demonstrate correct behavior. "
            "Include examples of both successful and unsuccessful patterns to avoid."
        ),
        MutationStrategy.ADD_GUARDRAILS: (
            "Add explicit constraints and guardrails to prevent common errors. "
            "Include what NOT to do as well as what to do."
        ),
        MutationStrategy.RESTRUCTURE_FORMAT: (
            "Restructure the output format specification. "
            "Make the expected format clearer with explicit schema or examples."
        ),
        MutationStrategy.ADD_CHAIN_OF_THOUGHT: (
            "Add step-by-step reasoning instructions. "
            "Guide the model to think through the problem before answering."
        ),
        MutationStrategy.SIMPLIFY: (
            "Simplify the prompt by removing unnecessary complexity. "
            "Focus on core requirements and reduce cognitive load."
        ),
        MutationStrategy.ADD_CONSTRAINTS: (
            "Add specific constraints for edge cases. "
            "Handle boundary conditions and common failure scenarios explicitly."
        ),
    }

    def __init__(
        self,
        config: AgentConfig,
        db: AsyncSession,
        llm=None,
        trajectory_collector: Optional[TrajectoryCollector] = None,
    ):
        """
        Initialize Prompt Builder Agent.

        Args:
            config: Agent configuration
            db: Database session
            llm: Optional LLM instance
            trajectory_collector: For accessing trajectories
        """
        super().__init__(
            config=config,
            llm=llm,
            trajectory_collector=trajectory_collector,
        )
        self.db = db
        self._trajectory_collector = trajectory_collector or TrajectoryCollector(db)

    async def should_trigger_optimization(
        self,
        agent_id: str,
        hours: int = 24,
    ) -> tuple[bool, str]:
        """
        Check if optimization should be triggered for an agent.

        Args:
            agent_id: Agent UUID to check
            hours: Analysis window

        Returns:
            Tuple of (should_trigger, reason)
        """
        # Get agent stats
        stats = await self._trajectory_collector.get_agent_stats(agent_id, hours)

        if stats.get("total_executions", 0) < self.MIN_TRAJECTORIES_FOR_ANALYSIS:
            return False, "Not enough trajectories for analysis"

        success_rate = stats.get("success_rate", 1.0)
        avg_quality = stats.get("avg_quality_score")

        if success_rate < self.SUCCESS_RATE_THRESHOLD:
            return True, f"Success rate ({success_rate:.1%}) below threshold ({self.SUCCESS_RATE_THRESHOLD:.0%})"

        if avg_quality and avg_quality < self.QUALITY_SCORE_THRESHOLD:
            return True, f"Quality score ({avg_quality:.1f}) below threshold ({self.QUALITY_SCORE_THRESHOLD})"

        return False, "Performance within acceptable range"

    async def run_daily_analysis(self) -> List[PromptOptimizationJob]:
        """
        Run daily analysis for all agents and create optimization jobs.

        Returns:
            List of created optimization jobs
        """
        jobs = []

        # Get all active agents
        result = await self.db.execute(
            select(AgentDefinition)
            .where(AgentDefinition.is_active == True)
        )
        agents = result.scalars().all()

        for agent in agents:
            should_trigger, reason = await self.should_trigger_optimization(
                str(agent.id), hours=24
            )

            if should_trigger:
                logger.info(
                    "Triggering optimization for agent",
                    agent_id=str(agent.id),
                    agent_name=agent.name,
                    reason=reason,
                )

                job = await self.create_optimization_job(agent)
                jobs.append(job)

        return jobs

    async def create_optimization_job(
        self,
        agent: AgentDefinition,
        hours: int = 24,
    ) -> PromptOptimizationJob:
        """
        Create an optimization job for an agent.

        Args:
            agent: Agent definition
            hours: Analysis window

        Returns:
            Created job
        """
        # Get baseline stats
        stats = await self._trajectory_collector.get_agent_stats(str(agent.id), hours)

        job = PromptOptimizationJob(
            agent_id=agent.id,
            status="pending",
            analysis_window_hours=hours,
            baseline_success_rate=stats.get("success_rate", 0.0),
        )

        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)

        return job

    async def analyze_failures(
        self,
        agent: AgentDefinition,
        hours: int = 24,
    ) -> FailureAnalysis:
        """
        Analyze failure patterns for an agent using LLM.

        Args:
            agent: Agent definition
            hours: Analysis window

        Returns:
            FailureAnalysis with patterns and recommendations
        """
        # Get failed trajectories
        failed = await self._trajectory_collector.get_failed_trajectories(
            str(agent.id), hours=hours, limit=50
        )

        if not failed:
            return FailureAnalysis(
                agent_id=str(agent.id),
                analysis_window_hours=hours,
                total_trajectories=0,
                failed_trajectories=0,
                failure_rate=0.0,
                summary="No failures to analyze",
            )

        # Get total count for failure rate
        stats = await self._trajectory_collector.get_agent_stats(str(agent.id), hours)

        # Get current prompt
        current_prompt = await self._get_agent_prompt(agent)

        # Format trajectories for analysis
        trajectories_text = self._format_trajectories_for_analysis(failed[:10])

        # Build analysis prompt
        analysis_template = PromptTemplate(
            id="analysis",
            version=1,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            task_prompt_template=ANALYSIS_TASK_PROMPT,
        )

        messages = analysis_template.build_messages(
            task="",
            agent_name=agent.name,
            system_prompt=current_prompt.get("system_prompt", "N/A"),
            task_template=current_prompt.get("task_prompt_template", "N/A"),
            failure_count=len(failed),
            trajectories=trajectories_text,
        )

        # Invoke LLM
        response, _, _ = await self.invoke_llm(messages, record=True)

        # Parse response
        analysis_data = self._parse_analysis_response(response)

        # Build FailureAnalysis
        patterns = []
        for p in analysis_data.get("patterns", []):
            patterns.append(FailurePattern(
                pattern_id=str(uuid.uuid4()),
                description=p.get("description", ""),
                frequency=self._estimate_frequency(p.get("frequency_estimate", "medium")),
                example_trajectories=[str(t.id) for t in failed[:3]],
                root_causes=p.get("root_causes", []),
                suggested_fixes=p.get("suggested_fixes", []),
            ))

        return FailureAnalysis(
            agent_id=str(agent.id),
            analysis_window_hours=hours,
            total_trajectories=stats.get("total_executions", 0),
            failed_trajectories=len(failed),
            failure_rate=1 - stats.get("success_rate", 1.0),
            patterns=patterns,
            summary=analysis_data.get("summary", ""),
            recommendations=analysis_data.get("recommendations", []),
        )

    async def generate_prompt_variants(
        self,
        agent: AgentDefinition,
        failure_analysis: FailureAnalysis,
        num_variants: int = 3,
    ) -> List[PromptMutation]:
        """
        Generate improved prompt variants using GEPA-style mutations.

        Args:
            agent: Agent definition
            failure_analysis: Analysis results
            num_variants: Number of variants to generate

        Returns:
            List of PromptMutation variants
        """
        current_prompt = await self._get_agent_prompt(agent)
        variants = []

        # Select strategies based on failure patterns
        strategies = self._select_strategies(failure_analysis, num_variants)

        mutation_template = PromptTemplate(
            id="mutation",
            version=1,
            system_prompt=MUTATION_SYSTEM_PROMPT,
            task_prompt_template=MUTATION_TASK_PROMPT,
        )

        for strategy in strategies:
            messages = mutation_template.build_messages(
                task="",
                strategy=strategy.value,
                system_prompt=current_prompt.get("system_prompt", ""),
                task_template=current_prompt.get("task_prompt_template", ""),
                failure_analysis=json.dumps(failure_analysis.to_dict(), indent=2),
                strategy_description=self.STRATEGY_DESCRIPTIONS.get(strategy, ""),
            )

            try:
                response, _, _ = await self.invoke_llm(messages, record=True)
                mutation_data = self._parse_mutation_response(response)

                mutation = PromptMutation(
                    id=str(uuid.uuid4()),
                    strategy=strategy,
                    original_version_id=current_prompt.get("version_id", ""),
                    system_prompt=mutation_data.get("system_prompt", ""),
                    task_prompt_template=mutation_data.get("task_prompt_template", ""),
                    few_shot_examples=mutation_data.get("few_shot_examples", []),
                    change_description=mutation_data.get("change_description", ""),
                    expected_improvement=mutation_data.get("expected_improvement", ""),
                )
                variants.append(mutation)

            except Exception as e:
                logger.error(f"Failed to generate variant with {strategy}: {e}")

        return variants

    async def generate_mutation(
        self,
        agent_id: str,
        strategy: MutationStrategy,
        custom_context: Optional[str] = None,
    ) -> Optional[PromptMutation]:
        """
        Generate a single enhanced prompt mutation for the enhance-prompt API.

        This method generates a single improved prompt based on the specified
        strategy, without requiring failure analysis (for manual enhancement).

        Args:
            agent_id: Agent UUID
            strategy: Mutation strategy to apply
            custom_context: Optional custom instructions for enhancement

        Returns:
            PromptMutation or None if generation fails
        """
        # Get current prompt
        current_prompt = await self._get_agent_prompt_by_id(agent_id)
        if not current_prompt:
            logger.error("No prompt found for agent", agent_id=agent_id)
            return None

        # Build mutation prompt
        mutation_template = PromptTemplate(
            id="mutation",
            version=1,
            system_prompt=MUTATION_SYSTEM_PROMPT,
            task_prompt_template=MUTATION_TASK_PROMPT,
        )

        # Create minimal failure analysis for context
        context_info = custom_context or "No specific issues identified. Enhance the prompt for general quality improvement."
        fake_analysis = {
            "patterns": [{"description": context_info}],
            "recommendations": [f"Apply {strategy.value} mutation"],
            "summary": context_info,
        }

        messages = mutation_template.build_messages(
            task="",
            strategy=strategy.value,
            system_prompt=current_prompt.get("system_prompt", ""),
            task_template=current_prompt.get("task_prompt_template", ""),
            failure_analysis=json.dumps(fake_analysis, indent=2),
            strategy_description=self.STRATEGY_DESCRIPTIONS.get(strategy, ""),
        )

        try:
            response, _, _ = await self.invoke_llm(messages, record=True)
            logger.info(
                "LLM response for mutation",
                strategy=strategy.value,
                response_length=len(response) if response else 0,
                response_preview=response[:500] if response else "None",
            )
            mutation_data = self._parse_mutation_response(response)
            logger.info(
                "Parsed mutation data",
                has_system_prompt=bool(mutation_data.get("system_prompt")),
                has_task_template=bool(mutation_data.get("task_prompt_template")),
                keys=list(mutation_data.keys()) if mutation_data else [],
            )

            return PromptMutation(
                id=str(uuid.uuid4()),
                strategy=strategy,
                original_version_id=current_prompt.get("version_id", ""),
                system_prompt=mutation_data.get("system_prompt", ""),
                task_prompt_template=mutation_data.get("task_prompt_template", ""),
                few_shot_examples=mutation_data.get("few_shot_examples", []),
                change_description=mutation_data.get("change_description", "Enhanced prompt"),
                expected_improvement=mutation_data.get("expected_improvement", "Improved clarity and effectiveness"),
            )

        except Exception as e:
            logger.error(f"Failed to generate mutation with {strategy}: {e}", exc_info=True)
            return None

    async def _get_agent_prompt_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current active prompt for agent by ID.

        Args:
            agent_id: Agent UUID string

        Returns:
            Dict with prompt details or None
        """
        result = await self.db.execute(
            select(AgentPromptVersion).where(
                and_(
                    AgentPromptVersion.agent_id == uuid.UUID(agent_id),
                    AgentPromptVersion.is_active == True,
                )
            )
        )
        version = result.scalar_one_or_none()

        if not version:
            return None

        return {
            "version_id": str(version.id),
            "system_prompt": version.system_prompt or "",
            "task_prompt_template": version.task_prompt_template or "",
        }

    async def create_prompt_versions(
        self,
        agent_id: str,
        variants: List[PromptMutation],
    ) -> List[AgentPromptVersion]:
        """
        Create database records for prompt variants.

        Args:
            agent_id: Agent UUID
            variants: List of mutations

        Returns:
            List of created AgentPromptVersion records
        """
        agent_uuid = uuid.UUID(agent_id)

        # Get current max version number
        result = await self.db.execute(
            select(AgentPromptVersion.version_number)
            .where(AgentPromptVersion.agent_id == agent_uuid)
            .order_by(AgentPromptVersion.version_number.desc())
            .limit(1)
        )
        max_version = result.scalar() or 0

        versions = []
        for i, variant in enumerate(variants):
            version = AgentPromptVersion(
                agent_id=agent_uuid,
                version_number=max_version + i + 1,
                system_prompt=variant.system_prompt,
                task_prompt_template=variant.task_prompt_template,
                few_shot_examples=variant.few_shot_examples,
                change_reason=f"Auto-generated: {variant.strategy.value}",
                created_by="prompt_builder",
                is_active=False,  # Not active until A/B tested
                traffic_percentage=0,
                parent_version_id=uuid.UUID(variant.original_version_id) if variant.original_version_id else None,
            )
            self.db.add(version)
            versions.append(version)

        await self.db.commit()

        # Refresh to get IDs
        for v in versions:
            await self.db.refresh(v)

        return versions

    async def request_approval(
        self,
        job: PromptOptimizationJob,
        winning_variant_id: str,
    ) -> None:
        """
        Mark job as awaiting approval.

        Args:
            job: Optimization job
            winning_variant_id: ID of winning variant
        """
        job.status = "awaiting_approval"
        job.winning_variant_id = uuid.UUID(winning_variant_id)

        await self.db.commit()

        logger.info(
            "Prompt optimization awaiting approval",
            job_id=str(job.id),
            agent_id=str(job.agent_id),
            winning_variant_id=winning_variant_id,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_agent_prompt(
        self,
        agent: AgentDefinition,
    ) -> Dict[str, Any]:
        """Get current prompt for an agent."""
        if agent.active_prompt_version_id:
            result = await self.db.execute(
                select(AgentPromptVersion)
                .where(AgentPromptVersion.id == agent.active_prompt_version_id)
            )
            version = result.scalar_one_or_none()
            if version:
                return {
                    "version_id": str(version.id),
                    "system_prompt": version.system_prompt,
                    "task_prompt_template": version.task_prompt_template,
                }

        return {
            "version_id": "",
            "system_prompt": "Default system prompt",
            "task_prompt_template": "{{task}}",
        }

    def _format_trajectories_for_analysis(
        self,
        trajectories: List[AgentTrajectory],
    ) -> str:
        """Format trajectories for LLM analysis."""
        formatted = []
        for t in trajectories:
            steps_summary = []
            for step in (t.trajectory_steps or [])[:5]:
                steps_summary.append(
                    f"  - {step.get('action_type', 'unknown')}: "
                    f"{step.get('error_message', 'no error')}"
                )

            formatted.append(
                f"Trajectory {t.id}:\n"
                f"  Task: {t.task_type}\n"
                f"  Error: {t.error_message or 'N/A'}\n"
                f"  Steps:\n" + "\n".join(steps_summary)
            )

        return "\n\n".join(formatted)

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        try:
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response:
                json_start = response.index("{")
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
        except (json.JSONDecodeError, ValueError):
            return {
                "patterns": [],
                "summary": response[:500],
                "recommendations": [],
            }

    def _parse_mutation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM mutation response."""
        try:
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response:
                json_start = response.index("{")
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
        except (json.JSONDecodeError, ValueError):
            return {}

    def _estimate_frequency(self, frequency_str: str) -> int:
        """Convert frequency string to numeric estimate."""
        mapping = {"high": 10, "medium": 5, "low": 2}
        return mapping.get(frequency_str.lower(), 5)

    def _select_strategies(
        self,
        analysis: FailureAnalysis,
        num_strategies: int,
    ) -> List[MutationStrategy]:
        """Select mutation strategies based on failure patterns."""
        # Default strategies if no specific patterns
        default_strategies = [
            MutationStrategy.REPHRASE_INSTRUCTIONS,
            MutationStrategy.ADD_EXAMPLES,
            MutationStrategy.ADD_GUARDRAILS,
        ]

        if not analysis.patterns:
            return default_strategies[:num_strategies]

        # Map common issues to strategies
        strategies = []

        for pattern in analysis.patterns:
            pattern_lower = pattern.description.lower()

            if "format" in pattern_lower or "output" in pattern_lower:
                strategies.append(MutationStrategy.RESTRUCTURE_FORMAT)
            if "unclear" in pattern_lower or "ambiguous" in pattern_lower:
                strategies.append(MutationStrategy.REPHRASE_INSTRUCTIONS)
            if "wrong" in pattern_lower or "incorrect" in pattern_lower:
                strategies.append(MutationStrategy.ADD_EXAMPLES)
            if "missing" in pattern_lower or "incomplete" in pattern_lower:
                strategies.append(MutationStrategy.ADD_CONSTRAINTS)
            if "reasoning" in pattern_lower or "logic" in pattern_lower:
                strategies.append(MutationStrategy.ADD_CHAIN_OF_THOUGHT)
            if "complex" in pattern_lower or "confused" in pattern_lower:
                strategies.append(MutationStrategy.SIMPLIFY)

        # Deduplicate and limit
        unique_strategies = list(dict.fromkeys(strategies))

        # Fill with defaults if needed
        for s in default_strategies:
            if len(unique_strategies) >= num_strategies:
                break
            if s not in unique_strategies:
                unique_strategies.append(s)

        return unique_strategies[:num_strategies]

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        Execute prompt optimization task.

        This is the main entry point when triggered.
        """
        agent_id = context.get("agent_id")
        if not agent_id:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message="No agent_id provided",
            )

        # Get agent
        result = await self.db.execute(
            select(AgentDefinition)
            .where(AgentDefinition.id == uuid.UUID(agent_id))
        )
        agent = result.scalar_one_or_none()

        if not agent:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=f"Agent {agent_id} not found",
            )

        try:
            # Analyze failures
            analysis = await self.analyze_failures(agent)

            # Generate variants
            variants = await self.generate_prompt_variants(agent, analysis)

            # Create versions
            versions = await self.create_prompt_versions(agent_id, variants)

            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output={
                    "analysis": analysis.to_dict(),
                    "variants_created": len(versions),
                    "variant_ids": [str(v.id) for v in versions],
                },
            )

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}", exc_info=True)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(e),
            )
