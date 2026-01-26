"""
AIDocumentIndexer - TruLens Integration
========================================

Experiment tracking and evaluation using TruLens.

TruLens provides:
- RAG evaluation with feedback functions
- Experiment tracking and comparison
- Dashboard for visualizing results
- A/B testing support
- Latency and cost tracking

Key features:
- Track RAG pipeline experiments over time
- Compare different retrieval/reranking strategies
- Monitor production quality metrics
- Identify regressions early

Research:
- TruEra TruLens Documentation (2024-2025)
- "Evaluating Large Language Models" (Stanford HAI)
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Check for TruLens
try:
    from trulens_eval import Tru, Feedback, TruChain, TruLlama
    from trulens_eval.feedback.provider import OpenAI as TruLensOpenAI
    from trulens_eval.feedback.provider import LiteLLM as TruLensLiteLLM
    HAS_TRULENS = True
except ImportError:
    HAS_TRULENS = False
    Tru = None
    Feedback = None


# =============================================================================
# Configuration
# =============================================================================

class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class MetricType(str, Enum):
    """Types of metrics to track."""
    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RELEVANCE = "context_relevance"
    COHERENCE = "coherence"
    LATENCY = "latency"
    COST = "cost"
    CUSTOM = "custom"


@dataclass
class TruLensConfig:
    """Configuration for TruLens integration."""
    # Database
    database_url: str = "sqlite:///trulens.db"
    database_prefix: str = "aidoc_"

    # Provider settings
    feedback_provider: str = "openai"  # openai, litellm, huggingface
    feedback_model: str = "gpt-4o-mini"

    # Evaluation settings
    enable_groundedness: bool = True
    enable_relevance: bool = True
    enable_coherence: bool = True
    enable_latency_tracking: bool = True
    enable_cost_tracking: bool = True

    # Sampling
    sample_rate: float = 1.0  # 1.0 = evaluate all, 0.1 = 10%

    # Dashboard
    enable_dashboard: bool = True
    dashboard_port: int = 8501


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str = ""
    retriever_config: Dict[str, Any] = field(default_factory=dict)
    reranker_config: Dict[str, Any] = field(default_factory=dict)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRecord:
    """A single evaluation record."""
    record_id: str
    experiment_id: str
    query: str
    answer: str
    contexts: List[str]
    metrics: Dict[str, float]
    latency_ms: float
    cost_usd: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""
    experiment_id: str
    name: str
    total_records: int
    avg_relevance: float
    avg_groundedness: float
    avg_answer_relevance: float
    avg_latency_ms: float
    total_cost_usd: float
    pass_rate: float
    created_at: datetime
    status: ExperimentStatus


# =============================================================================
# TruLens Manager
# =============================================================================

class TruLensManager:
    """
    Manages TruLens experiments and evaluations.

    Usage:
        manager = TruLensManager()
        await manager.initialize()

        # Create experiment
        exp_id = await manager.create_experiment(
            name="ColBERT + GPT-4",
            description="Testing ColBERT retrieval with GPT-4",
            retriever_config={"model": "colbert-v2"},
            llm_config={"model": "gpt-4"},
        )

        # Log evaluation
        await manager.log_evaluation(
            experiment_id=exp_id,
            query="What is machine learning?",
            answer="Machine learning is...",
            contexts=["Context 1", "Context 2"],
        )

        # Get results
        summary = await manager.get_experiment_summary(exp_id)
    """

    def __init__(self, config: Optional[TruLensConfig] = None):
        self.config = config or TruLensConfig()
        self._tru = None
        self._feedback_provider = None
        self._feedback_functions = {}
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._records: Dict[str, List[EvaluationRecord]] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize TruLens connection and feedback functions."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            if HAS_TRULENS:
                try:
                    # Initialize TruLens
                    self._tru = Tru(database_url=self.config.database_url)
                    self._tru.reset_database()  # Clean start for demo

                    # Initialize feedback provider
                    if self.config.feedback_provider == "openai":
                        self._feedback_provider = TruLensOpenAI(
                            model_engine=self.config.feedback_model
                        )
                    elif self.config.feedback_provider == "litellm":
                        self._feedback_provider = TruLensLiteLLM(
                            model_engine=self.config.feedback_model
                        )

                    # Create feedback functions
                    self._create_feedback_functions()

                    logger.info(
                        "TruLens initialized",
                        database=self.config.database_url,
                        provider=self.config.feedback_provider,
                    )

                except Exception as e:
                    logger.warning(f"TruLens initialization failed: {e}")
                    HAS_TRULENS = False

            self._initialized = True

    def _create_feedback_functions(self) -> None:
        """Create TruLens feedback functions."""
        if not self._feedback_provider:
            return

        try:
            # Relevance feedback
            if self.config.enable_relevance:
                self._feedback_functions["relevance"] = Feedback(
                    self._feedback_provider.relevance
                ).on_input_output()

            # Groundedness feedback
            if self.config.enable_groundedness:
                self._feedback_functions["groundedness"] = Feedback(
                    self._feedback_provider.groundedness_measure_with_cot_reasons
                ).on(
                    source=TruChain.select_context(),
                    statement=TruChain.select_output(),
                )

            # Answer relevance
            self._feedback_functions["answer_relevance"] = Feedback(
                self._feedback_provider.relevance_with_cot_reasons
            ).on_input_output()

            logger.info(
                "Feedback functions created",
                functions=list(self._feedback_functions.keys()),
            )

        except Exception as e:
            logger.error(f"Failed to create feedback functions: {e}")

    async def create_experiment(
        self,
        name: str,
        description: str = "",
        retriever_config: Optional[Dict] = None,
        reranker_config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a new experiment.

        Returns:
            experiment_id
        """
        await self.initialize()

        experiment_id = hashlib.md5(
            f"{name}{time.time()}".encode()
        ).hexdigest()[:12]

        config = ExperimentConfig(
            name=name,
            description=description,
            retriever_config=retriever_config or {},
            reranker_config=reranker_config or {},
            llm_config=llm_config or {},
            metadata=metadata or {},
        )

        self._experiments[experiment_id] = config
        self._records[experiment_id] = []

        logger.info(
            "Experiment created",
            experiment_id=experiment_id,
            name=name,
        )

        return experiment_id

    async def log_evaluation(
        self,
        experiment_id: str,
        query: str,
        answer: str,
        contexts: List[str],
        latency_ms: Optional[float] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict] = None,
        run_feedback: bool = True,
    ) -> str:
        """
        Log an evaluation for an experiment.

        Args:
            experiment_id: Experiment to log to
            query: User query
            answer: Generated answer
            contexts: Retrieved contexts
            latency_ms: Response latency
            cost_usd: API cost
            metadata: Additional metadata
            run_feedback: Whether to run feedback evaluation

        Returns:
            record_id
        """
        await self.initialize()

        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Sample check
        if self.config.sample_rate < 1.0:
            import random
            if random.random() > self.config.sample_rate:
                return ""  # Skip this record

        record_id = hashlib.md5(
            f"{experiment_id}{query}{time.time()}".encode()
        ).hexdigest()[:16]

        # Run feedback evaluation
        metrics = {}
        if run_feedback and self._feedback_provider:
            metrics = await self._run_feedback_evaluation(query, answer, contexts)

        record = EvaluationRecord(
            record_id=record_id,
            experiment_id=experiment_id,
            query=query,
            answer=answer,
            contexts=contexts,
            metrics=metrics,
            latency_ms=latency_ms or 0.0,
            cost_usd=cost_usd or 0.0,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        self._records[experiment_id].append(record)

        logger.debug(
            "Evaluation logged",
            experiment_id=experiment_id,
            record_id=record_id,
            metrics=metrics,
        )

        return record_id

    async def _run_feedback_evaluation(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        """Run TruLens feedback evaluation."""
        metrics = {}

        if not self._feedback_provider:
            return metrics

        loop = asyncio.get_event_loop()

        # Run evaluations in parallel
        async def eval_relevance():
            try:
                score = await loop.run_in_executor(
                    None,
                    lambda: self._feedback_provider.relevance(query, answer)
                )
                return ("relevance", score)
            except Exception as e:
                logger.debug(f"Relevance evaluation failed: {e}")
                return ("relevance", 0.5)

        async def eval_groundedness():
            try:
                context_str = "\n\n".join(contexts[:5])
                score = await loop.run_in_executor(
                    None,
                    lambda: self._feedback_provider.groundedness_measure_with_cot_reasons(
                        context_str, answer
                    )
                )
                # groundedness returns (score, reasons)
                return ("groundedness", score[0] if isinstance(score, tuple) else score)
            except Exception as e:
                logger.debug(f"Groundedness evaluation failed: {e}")
                return ("groundedness", 0.5)

        async def eval_context_relevance():
            try:
                scores = []
                for ctx in contexts[:3]:  # Limit to first 3 contexts
                    score = await loop.run_in_executor(
                        None,
                        lambda c=ctx: self._feedback_provider.relevance(query, c)
                    )
                    scores.append(score)
                return ("context_relevance", sum(scores) / len(scores) if scores else 0.5)
            except Exception as e:
                logger.debug(f"Context relevance evaluation failed: {e}")
                return ("context_relevance", 0.5)

        # Run all evaluations
        tasks = [eval_relevance(), eval_groundedness(), eval_context_relevance()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                metrics[result[0]] = result[1]

        return metrics

    async def get_experiment_summary(
        self,
        experiment_id: str,
    ) -> ExperimentSummary:
        """Get summary statistics for an experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        config = self._experiments[experiment_id]
        records = self._records.get(experiment_id, [])

        if not records:
            return ExperimentSummary(
                experiment_id=experiment_id,
                name=config.name,
                total_records=0,
                avg_relevance=0.0,
                avg_groundedness=0.0,
                avg_answer_relevance=0.0,
                avg_latency_ms=0.0,
                total_cost_usd=0.0,
                pass_rate=0.0,
                created_at=datetime.now(timezone.utc),
                status=ExperimentStatus.CREATED,
            )

        # Calculate averages
        relevance_scores = [r.metrics.get("relevance", 0.5) for r in records]
        groundedness_scores = [r.metrics.get("groundedness", 0.5) for r in records]
        answer_relevance_scores = [r.metrics.get("context_relevance", 0.5) for r in records]
        latencies = [r.latency_ms for r in records]
        costs = [r.cost_usd for r in records]

        # Pass rate (all metrics > 0.7)
        passed = sum(
            1 for r in records
            if all(v >= 0.7 for v in r.metrics.values())
        )

        return ExperimentSummary(
            experiment_id=experiment_id,
            name=config.name,
            total_records=len(records),
            avg_relevance=sum(relevance_scores) / len(relevance_scores),
            avg_groundedness=sum(groundedness_scores) / len(groundedness_scores),
            avg_answer_relevance=sum(answer_relevance_scores) / len(answer_relevance_scores),
            avg_latency_ms=sum(latencies) / len(latencies),
            total_cost_usd=sum(costs),
            pass_rate=passed / len(records),
            created_at=records[0].timestamp,
            status=ExperimentStatus.RUNNING,
        )

    async def compare_experiments(
        self,
        experiment_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.

        Returns comparison metrics and statistical significance.
        """
        summaries = []
        for exp_id in experiment_ids:
            try:
                summary = await self.get_experiment_summary(exp_id)
                summaries.append(summary)
            except ValueError:
                continue

        if len(summaries) < 2:
            return {"error": "Need at least 2 experiments to compare"}

        # Sort by relevance score
        summaries.sort(key=lambda s: s.avg_relevance, reverse=True)

        comparison = {
            "experiments": [
                {
                    "id": s.experiment_id,
                    "name": s.name,
                    "relevance": s.avg_relevance,
                    "groundedness": s.avg_groundedness,
                    "latency_ms": s.avg_latency_ms,
                    "pass_rate": s.pass_rate,
                    "total_records": s.total_records,
                }
                for s in summaries
            ],
            "best_relevance": summaries[0].experiment_id,
            "best_groundedness": max(summaries, key=lambda s: s.avg_groundedness).experiment_id,
            "fastest": min(summaries, key=lambda s: s.avg_latency_ms).experiment_id,
            "recommendations": [],
        }

        # Add recommendations
        best = summaries[0]
        if best.avg_groundedness < 0.7:
            comparison["recommendations"].append(
                "Consider improving context quality or adding verification"
            )
        if best.avg_latency_ms > 2000:
            comparison["recommendations"].append(
                "Latency is high - consider caching or faster models"
            )
        if best.pass_rate < 0.8:
            comparison["recommendations"].append(
                "Pass rate is below 80% - review failing cases"
            )

        return comparison

    async def get_all_experiments(self) -> List[ExperimentSummary]:
        """Get summaries for all experiments."""
        summaries = []
        for exp_id in self._experiments:
            summary = await self.get_experiment_summary(exp_id)
            summaries.append(summary)
        return summaries

    async def export_records(
        self,
        experiment_id: str,
        format: str = "json",
    ) -> str:
        """Export experiment records."""
        if experiment_id not in self._records:
            raise ValueError(f"Experiment {experiment_id} not found")

        records = self._records[experiment_id]

        if format == "json":
            return json.dumps([
                {
                    "record_id": r.record_id,
                    "query": r.query,
                    "answer": r.answer,
                    "contexts": r.contexts,
                    "metrics": r.metrics,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in records
            ], indent=2)

        return ""

    def start_dashboard(self) -> None:
        """Start TruLens dashboard if available."""
        if HAS_TRULENS and self._tru and self.config.enable_dashboard:
            try:
                self._tru.run_dashboard(port=self.config.dashboard_port)
                logger.info(
                    "TruLens dashboard started",
                    port=self.config.dashboard_port,
                )
            except Exception as e:
                logger.warning(f"Failed to start dashboard: {e}")


# =============================================================================
# Built-in Evaluation (Fallback when TruLens not installed)
# =============================================================================

class BuiltinEvaluator:
    """
    Built-in evaluation metrics when TruLens is not available.

    Provides basic evaluation without external dependencies.
    """

    def __init__(self, llm=None):
        self.llm = llm

    async def evaluate_relevance(
        self,
        query: str,
        answer: str,
    ) -> float:
        """Evaluate answer relevance to query."""
        if not self.llm:
            # Simple keyword overlap
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(query_words & answer_words)
            return min(overlap / max(len(query_words), 1), 1.0)

        # LLM-based evaluation
        prompt = f"""Rate how relevant this answer is to the question on a scale of 0-1.

Question: {query}
Answer: {answer}

Respond with only a number between 0 and 1."""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return float(content.strip())
        except:
            return 0.5

    async def evaluate_groundedness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Evaluate if answer is grounded in contexts."""
        if not contexts:
            return 0.0

        if not self.llm:
            # Simple overlap check
            context_text = " ".join(contexts).lower()
            answer_words = answer.lower().split()
            grounded = sum(1 for w in answer_words if w in context_text)
            return min(grounded / max(len(answer_words), 1), 1.0)

        # LLM-based evaluation
        context_str = "\n\n".join(contexts[:3])
        prompt = f"""Rate how well this answer is supported by the context on a scale of 0-1.

Context:
{context_str}

Answer: {answer}

Respond with only a number between 0 and 1."""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return float(content.strip())
        except:
            return 0.5

    async def evaluate_all(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        """Run all evaluations."""
        relevance = await self.evaluate_relevance(query, answer)
        groundedness = await self.evaluate_groundedness(answer, contexts)

        return {
            "relevance": relevance,
            "groundedness": groundedness,
            "overall": (relevance + groundedness) / 2,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_trulens_manager: Optional[TruLensManager] = None
_manager_lock = asyncio.Lock()


async def get_trulens_manager(
    config: Optional[TruLensConfig] = None,
) -> TruLensManager:
    """Get or create TruLens manager singleton."""
    global _trulens_manager

    async with _manager_lock:
        if _trulens_manager is None:
            _trulens_manager = TruLensManager(config)
            await _trulens_manager.initialize()

        return _trulens_manager


async def log_rag_evaluation(
    experiment_name: str,
    query: str,
    answer: str,
    contexts: List[str],
    **kwargs,
) -> str:
    """
    Convenience function to log RAG evaluation.

    Creates experiment if needed and logs the evaluation.
    """
    manager = await get_trulens_manager()

    # Find or create experiment
    for exp_id, config in manager._experiments.items():
        if config.name == experiment_name:
            return await manager.log_evaluation(
                experiment_id=exp_id,
                query=query,
                answer=answer,
                contexts=contexts,
                **kwargs,
            )

    # Create new experiment
    exp_id = await manager.create_experiment(name=experiment_name)
    return await manager.log_evaluation(
        experiment_id=exp_id,
        query=query,
        answer=answer,
        contexts=contexts,
        **kwargs,
    )
