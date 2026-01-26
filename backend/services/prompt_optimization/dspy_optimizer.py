"""
AIDocumentIndexer - DSPy Prompt Optimizer
=========================================

Phase 93: Core DSPy compilation pipeline.

Bridges the system's LLM configuration to DSPy, runs optimization
(BootstrapFewShot / MIPROv2), and exports results to PromptVersionManager
for A/B testing and deployment.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from backend.services.prompt_optimization.dspy_signatures import (
    DSPY_AVAILABLE as SIGS_AVAILABLE,
    MODULE_REGISTRY,
    get_module,
)
from backend.services.prompt_optimization.dspy_example_collector import (
    DSPyExampleCollector,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DSPyOptimizationResult:
    """Result of a DSPy optimization run."""
    signature_name: str
    optimizer_used: str
    num_training_examples: int
    num_dev_examples: int
    baseline_score: float
    optimized_score: float
    improvement_pct: float
    compiled_instructions: str
    compiled_demos: List[Dict[str, str]] = field(default_factory=list)
    prompt_version_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature_name": self.signature_name,
            "optimizer_used": self.optimizer_used,
            "num_training_examples": self.num_training_examples,
            "num_dev_examples": self.num_dev_examples,
            "baseline_score": round(self.baseline_score, 4),
            "optimized_score": round(self.optimized_score, 4),
            "improvement_pct": round(self.improvement_pct, 2),
            "compiled_instructions": self.compiled_instructions[:500],
            "compiled_demos_count": len(self.compiled_demos),
            "prompt_version_id": self.prompt_version_id,
            "error": self.error,
        }


# =============================================================================
# DSPy LM Adapter
# =============================================================================

class DSPyLMAdapter:
    """Bridges the system's LLM configuration to DSPy's LM interface."""

    @staticmethod
    async def from_system_config(db: Optional[AsyncSession] = None) -> Any:
        """
        Create a dspy.LM from the system's configured LLM provider.

        Reads provider/model from admin settings or environment,
        then creates the corresponding dspy.LM instance.

        Args:
            db: Optional database session for reading settings

        Returns:
            dspy.LM instance
        """
        if not DSPY_AVAILABLE:
            raise ImportError("dspy-ai is required")

        import os

        # Read LLM config from environment (settings DB adds complexity here)
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")

        # Build DSPy model string (DSPy uses litellm format internally)
        if provider == "openai":
            model_string = model  # e.g., "gpt-4o-mini"
        elif provider == "ollama":
            model_string = f"ollama_chat/{model}"
        elif provider == "anthropic":
            model_string = f"anthropic/{model}"
        elif provider == "vllm":
            model_string = f"openai/{model}"
        else:
            model_string = f"{provider}/{model}"

        # Set API keys
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = None

        if provider == "ollama":
            api_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        elif provider == "vllm":
            api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
            api_key = os.getenv("VLLM_API_KEY", "dummy")

        kwargs = {
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key

        lm = dspy.LM(model_string, **kwargs)

        logger.info(
            "Created DSPy LM from system config",
            provider=provider,
            model=model_string,
        )

        return lm


# =============================================================================
# DSPy Optimizer
# =============================================================================

class DSPyOptimizer:
    """
    Core DSPy optimization pipeline.

    Collects training examples, runs BootstrapFewShot or MIPROv2 optimization,
    and exports results to the PromptVersionManager for A/B testing.
    """

    def __init__(self, db: AsyncSession, lm: Optional[Any] = None):
        """
        Initialize optimizer.

        Args:
            db: Database session for reading training data
            lm: Optional dspy.LM instance (auto-configured if not provided)
        """
        self.db = db
        self._lm = lm
        self._collector = DSPyExampleCollector(db)

    async def _ensure_lm(self) -> Any:
        """Ensure dspy.LM is configured."""
        if self._lm is None:
            self._lm = await DSPyLMAdapter.from_system_config(self.db)
        return self._lm

    async def optimize(
        self,
        signature_name: str,
        optimizer_type: str = "bootstrap_few_shot",
        max_examples: int = 50,
        num_threads: int = 4,
        agent_id: Optional[str] = None,
    ) -> DSPyOptimizationResult:
        """
        Run DSPy optimization for a specific signature.

        Args:
            signature_name: Which signature to optimize (e.g., "rag_answer")
            optimizer_type: "bootstrap_few_shot" or "miprov2"
            max_examples: Maximum training examples to use
            num_threads: Parallelism for evaluation
            agent_id: Optional agent ID for prompt version tracking

        Returns:
            DSPyOptimizationResult with scores and compiled state
        """
        if not DSPY_AVAILABLE:
            return DSPyOptimizationResult(
                signature_name=signature_name,
                optimizer_used=optimizer_type,
                num_training_examples=0,
                num_dev_examples=0,
                baseline_score=0,
                optimized_score=0,
                improvement_pct=0,
                compiled_instructions="",
                error="dspy-ai not installed",
            )

        logger.info(
            "Starting DSPy optimization",
            signature=signature_name,
            optimizer=optimizer_type,
            max_examples=max_examples,
        )

        try:
            # 1. Configure LM
            lm = await self._ensure_lm()
            dspy.configure(lm=lm)

            # 2. Collect training data
            trainset, devset = await self._collector.build_dataset(
                signature_name=signature_name,
                max_examples=max_examples,
            )

            if len(trainset) < 5:
                return DSPyOptimizationResult(
                    signature_name=signature_name,
                    optimizer_used=optimizer_type,
                    num_training_examples=len(trainset),
                    num_dev_examples=len(devset),
                    baseline_score=0,
                    optimized_score=0,
                    improvement_pct=0,
                    compiled_instructions="",
                    error=f"Insufficient training data: {len(trainset)} examples (minimum 5)",
                )

            # 3. Create module and metric
            module = get_module(signature_name)
            metric = self._build_metric(signature_name)

            # 4. Evaluate baseline
            baseline_score = await asyncio.to_thread(
                self._evaluate, module, devset, metric, num_threads
            )

            # 5. Run optimizer
            compiled_module = await asyncio.to_thread(
                self._run_optimizer,
                module,
                trainset,
                devset,
                metric,
                optimizer_type,
                num_threads,
            )

            # 6. Evaluate optimized module
            optimized_score = await asyncio.to_thread(
                self._evaluate, compiled_module, devset, metric, num_threads
            )

            # 7. Extract compiled state
            instructions, demos = self._extract_compiled_state(compiled_module)

            # 8. Calculate improvement
            improvement = (
                ((optimized_score - baseline_score) / baseline_score * 100)
                if baseline_score > 0 else 0
            )

            result = DSPyOptimizationResult(
                signature_name=signature_name,
                optimizer_used=optimizer_type,
                num_training_examples=len(trainset),
                num_dev_examples=len(devset),
                baseline_score=baseline_score,
                optimized_score=optimized_score,
                improvement_pct=improvement,
                compiled_instructions=instructions,
                compiled_demos=demos,
            )

            # 9. Export to prompt version manager (if agent_id provided)
            if agent_id and improvement > 0:
                version_id = await self._export_to_prompt_version(
                    result, agent_id
                )
                result.prompt_version_id = version_id

            logger.info(
                "DSPy optimization completed",
                signature=signature_name,
                baseline=round(baseline_score, 4),
                optimized=round(optimized_score, 4),
                improvement=f"{improvement:.1f}%",
                examples=len(trainset),
            )

            return result

        except Exception as e:
            logger.error("DSPy optimization failed", error=str(e))
            return DSPyOptimizationResult(
                signature_name=signature_name,
                optimizer_used=optimizer_type,
                num_training_examples=0,
                num_dev_examples=0,
                baseline_score=0,
                optimized_score=0,
                improvement_pct=0,
                compiled_instructions="",
                error=str(e),
            )

    def _build_metric(self, signature_name: str) -> Callable:
        """
        Build an evaluation metric function for a signature.

        Returns a callable(example, prediction, trace=None) -> float.
        """
        if signature_name == "rag_answer":
            return self._rag_answer_metric
        elif signature_name == "query_expansion":
            return self._query_expansion_metric
        elif signature_name in ("react_reasoning", "query_decomposition"):
            return self._reasoning_metric
        elif signature_name == "answer_synthesis":
            return self._synthesis_metric
        else:
            return self._default_metric

    @staticmethod
    def _rag_answer_metric(example, prediction, trace=None) -> float:
        """Metric for RAG answer generation: checks answer quality."""
        if not hasattr(prediction, 'answer') or not prediction.answer:
            return 0.0

        score = 0.0
        answer = prediction.answer.strip()

        # Length check (non-trivial answer)
        if len(answer) > 20:
            score += 0.3

        # Citation check (contains source references)
        if "[" in answer and "]" in answer:
            score += 0.3

        # Relevance check (overlaps with expected answer if available)
        if hasattr(example, 'answer') and example.answer:
            expected_words = set(example.answer.lower().split())
            actual_words = set(answer.lower().split())
            overlap = len(expected_words & actual_words) / max(len(expected_words), 1)
            score += 0.4 * min(overlap * 2, 1.0)  # Scale up, cap at 0.4
        else:
            # No expected answer: give partial credit for substantive response
            score += 0.2

        return min(score, 1.0)

    @staticmethod
    def _query_expansion_metric(example, prediction, trace=None) -> float:
        """Metric for query expansion: checks diversity and relevance."""
        if not hasattr(prediction, 'expanded_queries') or not prediction.expanded_queries:
            return 0.0

        try:
            queries = json.loads(prediction.expanded_queries)
            if not isinstance(queries, list):
                return 0.2

            score = 0.0
            # Generated expected number of variations
            expected = getattr(example, 'num_variations', 3)
            if len(queries) >= expected:
                score += 0.4

            # Queries are diverse (not all identical)
            unique = len(set(str(q).lower() for q in queries))
            if unique >= len(queries) * 0.8:
                score += 0.3

            # Queries relate to original
            original = getattr(example, 'original_query', '').lower()
            if original:
                related = sum(
                    1 for q in queries
                    if any(w in str(q).lower() for w in original.split()[:3])
                )
                score += 0.3 * min(related / max(len(queries), 1), 1.0)

            return min(score, 1.0)
        except (json.JSONDecodeError, TypeError):
            return 0.1

    @staticmethod
    def _reasoning_metric(example, prediction, trace=None) -> float:
        """Metric for ReAct reasoning: checks action validity."""
        score = 0.0

        if hasattr(prediction, 'thought') and prediction.thought:
            score += 0.3

        if hasattr(prediction, 'action') and prediction.action:
            valid_actions = {"search", "graph", "summarize", "compare", "answer"}
            if prediction.action.lower().strip() in valid_actions:
                score += 0.4
            else:
                score += 0.1

        if hasattr(prediction, 'action_input') and prediction.action_input:
            score += 0.3

        return min(score, 1.0)

    @staticmethod
    def _synthesis_metric(example, prediction, trace=None) -> float:
        """Metric for answer synthesis: comprehensive answer check."""
        if not hasattr(prediction, 'answer') or not prediction.answer:
            return 0.0

        answer = prediction.answer.strip()
        score = 0.0

        if len(answer) > 50:
            score += 0.3
        if "[" in answer:  # Citations
            score += 0.2

        # Check it addresses the query
        query = getattr(example, 'query', '')
        if query:
            query_words = set(query.lower().split()[:5])
            answer_words = set(answer.lower().split())
            if query_words & answer_words:
                score += 0.3

        # Coherence heuristic: sentence count
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.2

        return min(score, 1.0)

    @staticmethod
    def _default_metric(example, prediction, trace=None) -> float:
        """Default metric: checks that output fields are non-empty."""
        if not prediction:
            return 0.0
        non_empty = sum(
            1 for k, v in prediction.items()
            if v and str(v).strip()
        ) if hasattr(prediction, 'items') else 0.5
        return min(non_empty / 3.0, 1.0)

    def _evaluate(
        self,
        module,
        devset: List,
        metric: Callable,
        num_threads: int = 4,
    ) -> float:
        """Evaluate a module on the dev set."""
        if not devset:
            return 0.0

        evaluator = dspy.Evaluate(
            devset=devset,
            metric=metric,
            num_threads=num_threads,
            display_progress=False,
            return_outputs=False,
        )
        return evaluator(module)

    def _run_optimizer(
        self,
        module,
        trainset: List,
        devset: List,
        metric: Callable,
        optimizer_type: str,
        num_threads: int = 4,
    ):
        """Run DSPy optimizer to compile the module."""
        if optimizer_type == "miprov2":
            optimizer = dspy.MIPROv2(
                metric=metric,
                num_threads=num_threads,
                max_bootstrapped_demos=4,
                max_labeled_demos=8,
            )
            return optimizer.compile(
                module,
                trainset=trainset,
                requires_permission_to_run=False,
            )
        else:
            # Default: BootstrapFewShot
            optimizer = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=8,
                max_rounds=1,
            )
            return optimizer.compile(
                module,
                trainset=trainset,
            )

    @staticmethod
    def _extract_compiled_state(compiled_module) -> tuple:
        """
        Extract optimized instructions and demonstrations from a compiled module.

        Returns:
            Tuple of (instructions_str, demos_list)
        """
        instructions = ""
        demos = []

        try:
            # Extract from module's predictors
            for name, predictor in compiled_module.named_predictors():
                # Get instruction/prefix if optimized
                if hasattr(predictor, 'signature'):
                    sig = predictor.signature
                    if hasattr(sig, 'instructions'):
                        instructions = str(sig.instructions)

                # Get bootstrapped demos
                if hasattr(predictor, 'demos') and predictor.demos:
                    for demo in predictor.demos:
                        demo_dict = {}
                        if hasattr(demo, 'items'):
                            demo_dict = dict(demo.items())
                        elif hasattr(demo, '__dict__'):
                            demo_dict = {
                                k: str(v) for k, v in demo.__dict__.items()
                                if not k.startswith('_')
                            }
                        if demo_dict:
                            demos.append(demo_dict)

        except Exception as e:
            logger.warning("Failed to extract compiled state", error=str(e))

        return instructions, demos

    async def _export_to_prompt_version(
        self,
        result: DSPyOptimizationResult,
        agent_id: str,
    ) -> Optional[str]:
        """
        Export optimization result to PromptVersionManager for A/B testing.

        Args:
            result: Optimization result with compiled state
            agent_id: Agent ID to create version for

        Returns:
            Created prompt version ID, or None
        """
        try:
            from backend.services.prompt_optimization.prompt_version_manager import (
                PromptVersionManager,
            )

            manager = PromptVersionManager(self.db)

            version = await manager.create_version(
                agent_id=agent_id,
                system_prompt=result.compiled_instructions or "DSPy-optimized prompt",
                change_reason=(
                    f"DSPy {result.optimizer_used}: "
                    f"{result.improvement_pct:.1f}% improvement "
                    f"(score {result.baseline_score:.2f} → {result.optimized_score:.2f})"
                ),
                created_by="dspy_optimizer",
                few_shot_examples=result.compiled_demos if result.compiled_demos else None,
            )

            if version:
                version_id = str(version.id) if hasattr(version, 'id') else str(version)
                logger.info(
                    "Exported DSPy result to prompt version",
                    version_id=version_id,
                    agent_id=agent_id,
                )
                return version_id

        except Exception as e:
            logger.warning(
                "Failed to export to prompt version",
                error=str(e),
                agent_id=agent_id,
            )

        return None
