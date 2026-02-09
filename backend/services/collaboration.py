"""
AIDocumentIndexer - Multi-LLM Collaboration Service
====================================================

Implements multi-LLM collaboration workflows using LangGraph.
Multiple LLMs work together for higher quality output:
- Generator: Primary LLM creates initial response
- Critic/Reviewer: Second LLM reviews and suggests improvements
- Synthesizer: Combines feedback into final output

LLM provider and model are configured via Admin UI (Operation-Level Config).
Configure the "collaboration" operation in Admin > Settings > LLM Configuration.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict
from uuid import uuid4

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from backend.services.llm import EnhancedLLMFactory

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class CollaborationRole(str, Enum):
    """Roles in multi-LLM collaboration."""
    GENERATOR = "generator"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


class CollaborationMode(str, Enum):
    """Collaboration workflow modes."""
    SINGLE = "single"  # No collaboration, single LLM
    REVIEW = "review"  # Generator + Critic
    FULL = "full"  # Generator + Critic + Synthesizer
    DEBATE = "debate"  # Multiple generators debate, synthesizer decides


class CollaborationStatus(str, Enum):
    """Status of a collaboration session."""
    PENDING = "pending"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a specific model in collaboration."""
    provider: Optional[str] = None   # None = use system-configured default
    model: Optional[str] = None      # None = use provider's default model
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class CollaborationConfig:
    """Configuration for collaboration workflow."""
    mode: CollaborationMode = CollaborationMode.REVIEW
    generator: ModelConfig = field(default_factory=ModelConfig)
    critic: ModelConfig = field(default_factory=lambda: ModelConfig(
        temperature=0.3,  # Lower temperature for more focused critique
    ))
    synthesizer: ModelConfig = field(default_factory=lambda: ModelConfig(
        temperature=0.5,
    ))
    max_iterations: int = 2  # Max review-revise cycles
    enable_cost_tracking: bool = True


@dataclass
class GenerationResult:
    """Result from a single generation step."""
    role: CollaborationRole
    model: str
    content: str
    tokens_used: int = 0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CritiqueResult:
    """Result from critique/review step."""
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    overall_score: float  # 0-10
    should_revise: bool
    raw_content: str


@dataclass
class CollaborationSession:
    """A collaboration session with history."""
    id: str
    user_id: str
    config: CollaborationConfig
    status: CollaborationStatus
    prompt: str
    context: Optional[str]

    # Results from each step
    initial_generation: Optional[GenerationResult] = None
    critiques: List[CritiqueResult] = field(default_factory=list)
    revisions: List[GenerationResult] = field(default_factory=list)
    final_synthesis: Optional[GenerationResult] = None
    final_output: Optional[str] = None

    # Metadata
    total_tokens: int = 0
    total_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# =============================================================================
# LangGraph State
# =============================================================================

class CollaborationState(TypedDict):
    """State for LangGraph collaboration workflow."""
    session_id: str
    prompt: str
    context: Optional[str]
    config: Dict[str, Any]

    # Generation state
    current_draft: str
    iteration: int

    # Results
    generations: List[Dict]
    critiques: List[Dict]

    # Control
    should_continue: bool
    final_output: str
    error: Optional[str]


# =============================================================================
# Prompts
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are a skilled content generator. Create high-quality,
well-structured content based on the user's request. Be thorough, accurate, and engaging.

If context is provided, incorporate relevant information from it.

Focus on:
- Clear structure and organization
- Accurate information
- Engaging writing style
- Complete coverage of the topic"""

CRITIC_SYSTEM_PROMPT = """You are a critical reviewer. Your job is to analyze content
and provide constructive feedback. Be thorough but fair.

Analyze the content for:
1. Accuracy - Is the information correct?
2. Completeness - Are all aspects covered?
3. Clarity - Is it easy to understand?
4. Structure - Is it well-organized?
5. Engagement - Is it interesting to read?

Provide your feedback in this format:
STRENGTHS:
- [List specific strengths]

WEAKNESSES:
- [List specific areas for improvement]

SUGGESTIONS:
- [List actionable suggestions]

OVERALL SCORE: [0-10]
RECOMMEND REVISION: [YES/NO]"""

SYNTHESIZER_SYSTEM_PROMPT = """You are a skilled synthesizer. Your job is to take
the original content and the critique feedback, then produce an improved final version.

Guidelines:
- Address all valid criticisms
- Preserve the strengths of the original
- Implement the suggested improvements
- Maintain consistency and flow
- Produce a polished final output"""


# =============================================================================
# Collaboration Service
# =============================================================================

class CollaborationService:
    """
    Service for multi-LLM collaboration workflows.

    Uses LangGraph to orchestrate multiple LLMs working together
    to produce higher quality output.
    """

    def __init__(self):
        self._sessions: Dict[str, CollaborationSession] = {}
        self._workflows: Dict[CollaborationMode, StateGraph] = {}
        self._initialize_workflows()

    def _initialize_workflows(self):
        """Initialize LangGraph workflows for each mode."""
        # Review workflow: Generator -> Critic -> Synthesizer
        self._workflows[CollaborationMode.REVIEW] = self._build_review_workflow()

        # Full workflow: Generator -> Critic -> Revise -> Critic -> Synthesize
        self._workflows[CollaborationMode.FULL] = self._build_full_workflow()

    def _build_review_workflow(self) -> StateGraph:
        """Build simple review workflow."""
        workflow = StateGraph(CollaborationState)

        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Add edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "critique")
        workflow.add_edge("critique", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _build_full_workflow(self) -> StateGraph:
        """Build full iterative workflow with revision cycles."""
        workflow = StateGraph(CollaborationState)

        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("revise", self._revise_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Add edges
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique",
            self._should_revise,
            {
                "revise": "revise",
                "synthesize": "synthesize",
            }
        )
        workflow.add_edge("revise", "critique")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _generate_node(self, state: CollaborationState) -> CollaborationState:
        """Generate initial content."""
        logger.info("Generating initial content", session_id=state["session_id"])

        try:
            # Get LLM using database-driven configuration
            llm, llm_config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="collaboration",
                user_id=None,  # System-level operation
            )
            model_name = llm_config.model if llm_config else "unknown"

            # Build prompt
            messages = [
                SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            ]

            if state.get("context"):
                messages.append(HumanMessage(content=f"Context:\n{state['context']}\n\nRequest:\n{state['prompt']}"))
            else:
                messages.append(HumanMessage(content=state["prompt"]))

            # Generate
            response = await llm.ainvoke(messages)

            # Update state
            state["current_draft"] = response.content
            state["generations"].append({
                "role": "generator",
                "model": model_name,
                "content": response.content,
                "timestamp": datetime.utcnow().isoformat(),
            })

        except Exception as e:
            logger.error("Generation failed", error=str(e))
            state["error"] = str(e)
            state["should_continue"] = False

        return state

    async def _critique_node(self, state: CollaborationState) -> CollaborationState:
        """Critique the current draft."""
        logger.info("Critiquing content", session_id=state["session_id"], iteration=state["iteration"])

        try:
            config = state["config"]

            # Get LLM using database-driven configuration
            llm, llm_config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="collaboration",
                user_id=None,  # System-level operation
            )
            model_name = llm_config.model if llm_config else "unknown"

            # Build prompt
            messages = [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=f"Please review this content:\n\n{state['current_draft']}"),
            ]

            # Generate critique
            response = await llm.ainvoke(messages)

            # Parse critique
            critique = self._parse_critique(response.content)

            # Update state
            state["critiques"].append({
                "iteration": state["iteration"],
                "model": model_name,
                "content": response.content,
                "score": critique.overall_score,
                "should_revise": critique.should_revise,
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Determine if we should continue revising
            max_iterations = config.get("max_iterations", 2)
            state["should_continue"] = critique.should_revise and state["iteration"] < max_iterations
            state["iteration"] += 1

        except Exception as e:
            logger.error("Critique failed", error=str(e))
            state["error"] = str(e)
            state["should_continue"] = False

        return state

    async def _revise_node(self, state: CollaborationState) -> CollaborationState:
        """Revise content based on critique."""
        logger.info("Revising content", session_id=state["session_id"], iteration=state["iteration"])

        try:
            # Get LLM using database-driven configuration
            llm, llm_config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="collaboration",
                user_id=None,  # System-level operation
            )
            model_name = llm_config.model if llm_config else "unknown"

            # Get latest critique
            latest_critique = state["critiques"][-1] if state["critiques"] else None

            if not latest_critique:
                return state

            # Build prompt
            revision_prompt = f"""Please revise this content based on the feedback:

ORIGINAL CONTENT:
{state['current_draft']}

FEEDBACK:
{latest_critique['content']}

Please produce an improved version that addresses the feedback while maintaining the strengths."""

            messages = [
                SystemMessage(content="You are revising content based on feedback. Address the criticisms while preserving what works well."),
                HumanMessage(content=revision_prompt),
            ]

            # Generate revision
            response = await llm.ainvoke(messages)

            # Update state
            state["current_draft"] = response.content
            state["generations"].append({
                "role": "reviser",
                "model": model_name,
                "content": response.content,
                "iteration": state["iteration"],
                "timestamp": datetime.utcnow().isoformat(),
            })

        except Exception as e:
            logger.error("Revision failed", error=str(e))
            state["error"] = str(e)
            state["should_continue"] = False

        return state

    async def _synthesize_node(self, state: CollaborationState) -> CollaborationState:
        """Synthesize final output."""
        logger.info("Synthesizing final output", session_id=state["session_id"])

        try:
            # Get LLM using database-driven configuration
            llm, llm_config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="collaboration",
                user_id=None,  # System-level operation
            )
            model_name = llm_config.model if llm_config else "unknown"

            # Build synthesis prompt
            all_critiques = "\n\n".join([
                f"Critique {i+1}:\n{c['content']}"
                for i, c in enumerate(state["critiques"])
            ])

            synthesis_prompt = f"""Please produce a polished final version of this content.

CURRENT DRAFT:
{state['current_draft']}

ACCUMULATED FEEDBACK:
{all_critiques}

Produce the best possible final version that incorporates all valuable feedback."""

            messages = [
                SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
                HumanMessage(content=synthesis_prompt),
            ]

            # Generate final output
            response = await llm.ainvoke(messages)

            # Update state
            state["final_output"] = response.content
            state["generations"].append({
                "role": "synthesizer",
                "model": model_name,
                "content": response.content,
                "timestamp": datetime.utcnow().isoformat(),
            })

        except Exception as e:
            logger.error("Synthesis failed", error=str(e))
            state["error"] = str(e)
            # Use current draft as fallback
            state["final_output"] = state["current_draft"]

        return state

    def _should_revise(self, state: CollaborationState) -> str:
        """Determine whether to revise or synthesize."""
        if state.get("error"):
            return "synthesize"

        if state["should_continue"]:
            return "revise"

        return "synthesize"

    def _parse_critique(self, critique_text: str) -> CritiqueResult:
        """Parse critique response into structured format."""
        strengths = []
        weaknesses = []
        suggestions = []
        overall_score = 7.0
        should_revise = False

        lines = critique_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if "STRENGTHS:" in line.upper():
                current_section = "strengths"
            elif "WEAKNESSES:" in line.upper():
                current_section = "weaknesses"
            elif "SUGGESTIONS:" in line.upper():
                current_section = "suggestions"
            elif "OVERALL SCORE:" in line.upper():
                try:
                    score_text = line.split(":")[-1].strip()
                    # Extract numeric score
                    score_text = "".join(c for c in score_text if c.isdigit() or c == ".")
                    if score_text:
                        overall_score = float(score_text)
                except (ValueError, IndexError):
                    pass
            elif "RECOMMEND REVISION:" in line.upper():
                should_revise = "YES" in line.upper()
            elif line.startswith("-") and current_section:
                item = line[1:].strip()
                if current_section == "strengths":
                    strengths.append(item)
                elif current_section == "weaknesses":
                    weaknesses.append(item)
                elif current_section == "suggestions":
                    suggestions.append(item)

        return CritiqueResult(
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            overall_score=overall_score,
            should_revise=should_revise,
            raw_content=critique_text,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def create_session(
        self,
        user_id: str,
        prompt: str,
        context: Optional[str] = None,
        config: Optional[CollaborationConfig] = None,
    ) -> CollaborationSession:
        """Create a new collaboration session."""
        session_id = str(uuid4())

        if config is None:
            config = CollaborationConfig()

        session = CollaborationSession(
            id=session_id,
            user_id=user_id,
            config=config,
            status=CollaborationStatus.PENDING,
            prompt=prompt,
            context=context,
        )

        self._sessions[session_id] = session

        logger.info(
            "Created collaboration session",
            session_id=session_id,
            user_id=user_id,
            mode=config.mode.value,
        )

        return session

    async def run_collaboration(
        self,
        session_id: str,
    ) -> CollaborationSession:
        """Run the collaboration workflow."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        logger.info("Starting collaboration", session_id=session_id, mode=session.config.mode.value)

        # Get appropriate workflow
        if session.config.mode == CollaborationMode.SINGLE:
            # Single LLM, no collaboration
            return await self._run_single(session)

        workflow = self._workflows.get(session.config.mode)
        if not workflow:
            workflow = self._workflows[CollaborationMode.REVIEW]

        # Prepare initial state
        initial_state: CollaborationState = {
            "session_id": session_id,
            "prompt": session.prompt,
            "context": session.context,
            "config": {
                "generator": {
                    "provider": session.config.generator.provider,
                    "model": session.config.generator.model,
                    "temperature": session.config.generator.temperature,
                },
                "critic": {
                    "provider": session.config.critic.provider,
                    "model": session.config.critic.model,
                    "temperature": session.config.critic.temperature,
                },
                "synthesizer": {
                    "provider": session.config.synthesizer.provider,
                    "model": session.config.synthesizer.model,
                    "temperature": session.config.synthesizer.temperature,
                },
                "max_iterations": session.config.max_iterations,
            },
            "current_draft": "",
            "iteration": 0,
            "generations": [],
            "critiques": [],
            "should_continue": True,
            "final_output": "",
            "error": None,
        }

        # Update session status
        session.status = CollaborationStatus.GENERATING

        try:
            # Run workflow
            final_state = await workflow.ainvoke(initial_state)

            # Update session with results
            session.final_output = final_state["final_output"]
            session.status = CollaborationStatus.COMPLETED
            session.completed_at = datetime.utcnow()

            # Store generation history
            for gen in final_state["generations"]:
                if gen["role"] == "generator":
                    if session.initial_generation is None:
                        session.initial_generation = GenerationResult(
                            role=CollaborationRole.GENERATOR,
                            model=gen["model"],
                            content=gen["content"],
                        )
                    else:
                        session.revisions.append(GenerationResult(
                            role=CollaborationRole.GENERATOR,
                            model=gen["model"],
                            content=gen["content"],
                        ))
                elif gen["role"] == "synthesizer":
                    session.final_synthesis = GenerationResult(
                        role=CollaborationRole.SYNTHESIZER,
                        model=gen["model"],
                        content=gen["content"],
                    )

            # Store critiques
            for crit in final_state["critiques"]:
                session.critiques.append(CritiqueResult(
                    strengths=[],
                    weaknesses=[],
                    suggestions=[],
                    overall_score=crit.get("score", 7.0),
                    should_revise=crit.get("should_revise", False),
                    raw_content=crit["content"],
                ))

            if final_state.get("error"):
                session.error_message = final_state["error"]

            logger.info(
                "Collaboration completed",
                session_id=session_id,
                generations=len(final_state["generations"]),
                critiques=len(final_state["critiques"]),
            )

        except Exception as e:
            logger.error("Collaboration failed", session_id=session_id, error=str(e))
            session.status = CollaborationStatus.FAILED
            session.error_message = str(e)
            raise

        return session

    async def _run_single(self, session: CollaborationSession) -> CollaborationSession:
        """Run single LLM mode (no collaboration)."""
        try:
            # Get LLM using database-driven configuration
            llm, llm_config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="collaboration",
                user_id=None,  # System-level operation
            )
            model_name = llm_config.model if llm_config else "unknown"

            messages = [
                SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            ]

            if session.context:
                messages.append(HumanMessage(content=f"Context:\n{session.context}\n\nRequest:\n{session.prompt}"))
            else:
                messages.append(HumanMessage(content=session.prompt))

            response = await llm.ainvoke(messages)

            session.initial_generation = GenerationResult(
                role=CollaborationRole.GENERATOR,
                model=model_name,
                content=response.content,
            )
            session.final_output = response.content
            session.status = CollaborationStatus.COMPLETED
            session.completed_at = datetime.utcnow()

        except Exception as e:
            session.status = CollaborationStatus.FAILED
            session.error_message = str(e)
            raise

        return session

    async def stream_collaboration(
        self,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream collaboration progress updates."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Yield initial status
        yield {
            "type": "status",
            "status": session.status.value,
            "session_id": session_id,
        }

        # Run collaboration with progress updates
        # Currently yields final result only; streaming callbacks can be added
        # by passing an async callback to run_collaboration() for real-time updates

        result = await self.run_collaboration(session_id)

        yield {
            "type": "complete",
            "status": result.status.value,
            "final_output": result.final_output,
            "generations": len(result.revisions) + (1 if result.initial_generation else 0),
            "critiques": len(result.critiques),
        }

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a collaboration session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        user_id: str,
        status: Optional[CollaborationStatus] = None,
    ) -> List[CollaborationSession]:
        """List sessions for a user."""
        sessions = [
            s for s in self._sessions.values()
            if s.user_id == user_id
        ]

        if status:
            sessions = [s for s in sessions if s.status == status]

        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def estimate_cost(
        self,
        prompt: str,
        context: Optional[str] = None,
        config: Optional[CollaborationConfig] = None,
    ) -> Dict[str, Any]:
        """Estimate cost for a collaboration session."""
        if config is None:
            config = CollaborationConfig()

        # Rough token estimation
        prompt_tokens = len(prompt.split()) * 1.3
        context_tokens = len(context.split()) * 1.3 if context else 0

        # Estimated output tokens per step
        output_tokens = 500  # Average assumption

        # Cost per 1K tokens (rough estimates)
        costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "claude-2": {"input": 0.008, "output": 0.024},
        }

        def get_cost(model: str, input_tokens: int, output_tokens: int) -> float:
            model_costs = costs.get(model, costs["gpt-4"])
            return (input_tokens * model_costs["input"] + output_tokens * model_costs["output"]) / 1000

        total_cost = 0.0
        breakdown = []

        # Generator cost
        gen_input = prompt_tokens + context_tokens
        gen_cost = get_cost(config.generator.model, gen_input, output_tokens)
        total_cost += gen_cost
        breakdown.append({"step": "generate", "cost": gen_cost})

        if config.mode != CollaborationMode.SINGLE:
            # Critic cost
            crit_input = output_tokens  # Reviews the generated content
            crit_cost = get_cost(config.critic.model, crit_input, 300)
            total_cost += crit_cost
            breakdown.append({"step": "critique", "cost": crit_cost})

            # Synthesizer cost
            synth_input = output_tokens + 300  # Draft + critique
            synth_cost = get_cost(config.synthesizer.model, synth_input, output_tokens)
            total_cost += synth_cost
            breakdown.append({"step": "synthesize", "cost": synth_cost})

            if config.mode == CollaborationMode.FULL:
                # Additional revision cycles
                for i in range(config.max_iterations - 1):
                    total_cost += crit_cost + gen_cost
                    breakdown.append({"step": f"revision_{i+1}", "cost": crit_cost + gen_cost})

        return {
            "estimated_cost": round(total_cost, 4),
            "currency": "USD",
            "breakdown": breakdown,
            "mode": config.mode.value,
            "models": {
                "generator": config.generator.model,
                "critic": config.critic.model,
                "synthesizer": config.synthesizer.model,
            },
        }


# =============================================================================
# Module-level singleton and helpers
# =============================================================================

_collaboration_service: Optional[CollaborationService] = None


def get_collaboration_service() -> CollaborationService:
    """Get the collaboration service singleton."""
    global _collaboration_service
    if _collaboration_service is None:
        _collaboration_service = CollaborationService()
    return _collaboration_service
