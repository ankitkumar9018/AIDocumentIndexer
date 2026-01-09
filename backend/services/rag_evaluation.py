"""
AIDocumentIndexer - RAG Evaluation Framework
=============================================

Implements RAGAS-inspired metrics for systematic RAG quality evaluation:
1. Context Relevance: How relevant is retrieved context to the query?
2. Faithfulness: Is the answer grounded in the context?
3. Answer Relevance: Does the answer address the question?
4. Context Recall: Did we retrieve all needed information?

Also provides benchmarking utilities for continuous quality monitoring.

Reference: RAGAS (https://github.com/explodinggradients/ragas)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RAGASMetrics:
    """RAGAS evaluation metrics for a single response."""
    context_relevance: float  # 0-1: How relevant is retrieved context?
    faithfulness: float       # 0-1: Is answer grounded in context?
    answer_relevance: float   # 0-1: Does answer address the question?
    context_recall: float     # 0-1: Did we retrieve all needed info?
    overall_score: float      # Weighted average

    # Additional metrics
    answer_similarity: Optional[float] = None  # Similarity to ground truth
    context_precision: Optional[float] = None  # Precision of retrieved chunks

    # Metadata
    evaluation_time_ms: Optional[float] = None
    evaluator_model: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result for a RAG query."""
    query: str
    answer: str
    contexts: List[str]
    metrics: RAGASMetrics
    ground_truth: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark suite."""
    test_count: int
    individual_results: List[EvaluationResult]
    aggregate_metrics: Dict[str, float]
    passing_rate: float  # Percentage meeting thresholds
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RAGEvaluator:
    """
    Evaluate RAG responses using RAGAS-inspired metrics.

    Uses LLM-as-judge for nuanced evaluation of context relevance,
    faithfulness, and answer quality.
    """

    def __init__(
        self,
        llm=None,
        embedding_service=None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the RAG evaluator.

        Args:
            llm: LangChain LLM for evaluation prompts
            embedding_service: For similarity-based metrics
            weights: Custom weights for metrics (default: balanced)
        """
        self.llm = llm
        self.embedding_service = embedding_service
        self.weights = weights or {
            "context_relevance": 0.25,
            "faithfulness": 0.35,
            "answer_relevance": 0.30,
            "context_recall": 0.10,
        }

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a RAG response.

        Args:
            query: The user's query
            answer: The generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional ground truth answer

        Returns:
            EvaluationResult with metrics and analysis
        """
        start_time = datetime.utcnow()
        issues = []
        suggestions = []

        # Evaluate individual metrics
        context_relevance = await self._evaluate_context_relevance(query, contexts)
        faithfulness = await self._evaluate_faithfulness(answer, contexts)
        answer_relevance = await self._evaluate_answer_relevance(query, answer)

        # Context recall only if ground truth available
        if ground_truth:
            context_recall = await self._evaluate_context_recall(contexts, ground_truth)
        else:
            context_recall = 0.0

        # Calculate weighted overall score
        weights = self.weights.copy()
        if not ground_truth:
            # Redistribute context_recall weight
            total_other = sum(v for k, v in weights.items() if k != "context_recall")
            weights = {
                k: v / total_other if k != "context_recall" else 0
                for k, v in weights.items()
            }

        overall = (
            weights["context_relevance"] * context_relevance +
            weights["faithfulness"] * faithfulness +
            weights["answer_relevance"] * answer_relevance +
            weights["context_recall"] * context_recall
        )

        # Generate issues and suggestions
        if context_relevance < 0.5:
            issues.append("Low context relevance - retrieved chunks may not be relevant")
            suggestions.append("Consider refining search parameters or query expansion")

        if faithfulness < 0.6:
            issues.append("Potential hallucination - answer may not be fully supported")
            suggestions.append("Review answer against source documents")

        if answer_relevance < 0.5:
            issues.append("Answer may not fully address the question")
            suggestions.append("Consider clarifying the query or using more specific terms")

        # Calculate evaluation time
        eval_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        metrics = RAGASMetrics(
            context_relevance=context_relevance,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_recall=context_recall,
            overall_score=overall,
            evaluation_time_ms=eval_time,
            evaluator_model=getattr(self.llm, 'model_name', 'unknown') if self.llm else None,
        )

        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            metrics=metrics,
            ground_truth=ground_truth,
            issues=issues,
            suggestions=suggestions,
        )

    async def _evaluate_context_relevance(
        self,
        query: str,
        contexts: List[str],
    ) -> float:
        """
        Evaluate how relevant the retrieved context is to the query.

        Score 0-1: What fraction of the context is useful for answering?
        """
        if not contexts:
            return 0.0

        if not self.llm:
            # Fallback to keyword overlap
            return self._keyword_relevance(query, contexts)

        combined_context = "\n---\n".join(contexts[:5])  # Limit for token efficiency

        prompt = f"""Rate how relevant the following context passages are for
answering the question. Consider:
- Does the context contain information that helps answer the question?
- Is the information specific and useful, not just tangentially related?

Question: {query}

Context:
{combined_context}

Rate the overall relevance from 0 to 10, where:
- 10: Highly relevant - context directly addresses the question
- 7: Mostly relevant - context contains useful information
- 5: Partially relevant - some useful information mixed with irrelevant
- 3: Marginally relevant - only tangentially related
- 0: Not relevant - context doesn't help answer the question

Provide ONLY a number from 0-10:"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            score = float(content.strip()) / 10
            return max(0, min(1, score))
        except Exception as e:
            logger.warning("Context relevance evaluation failed", error=str(e))
            return 0.5

    async def _evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """
        Evaluate if the answer is grounded in the context.

        Score 0-1: What fraction of claims are supported by context?
        """
        if not contexts or not answer:
            return 0.0

        if not self.llm:
            return self._keyword_overlap_score(answer, contexts)

        combined_context = "\n---\n".join(contexts[:5])

        prompt = f"""Check if the answer is fully supported by the given context.
For each claim in the answer, verify it appears in or can be inferred from the context.

Context:
{combined_context}

Answer: {answer}

Rate faithfulness from 0 to 10, where:
- 10: Every claim is directly supported by the context
- 7: Most claims are supported, with minor inferences
- 5: About half the claims are supported
- 3: Few claims are supported, significant unsupported content
- 0: Answer contradicts context or has no support

Provide ONLY a number from 0-10:"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            score = float(content.strip()) / 10
            return max(0, min(1, score))
        except Exception as e:
            logger.warning("Faithfulness evaluation failed", error=str(e))
            return 0.5

    async def _evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        Evaluate if the answer addresses the question.

        Score 0-1: How well does the answer address what was asked?
        """
        if not answer:
            return 0.0

        if not self.llm:
            return self._keyword_overlap_score(query, [answer])

        prompt = f"""Rate how well the answer addresses the question.
Consider:
- Does the answer directly respond to what was asked?
- Is the answer complete, or does it miss key aspects?
- Is the answer focused on the question, or does it go off-topic?

Question: {query}

Answer: {answer}

Rate answer relevance from 0 to 10, where:
- 10: Directly and completely answers the question
- 7: Answers the main question with minor gaps
- 5: Partially addresses the question
- 3: Tangentially related but doesn't really answer
- 0: Does not address the question at all

Provide ONLY a number from 0-10:"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            score = float(content.strip()) / 10
            return max(0, min(1, score))
        except Exception as e:
            logger.warning("Answer relevance evaluation failed", error=str(e))
            return 0.5

    async def _evaluate_context_recall(
        self,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """
        Evaluate if context contains the ground truth information.

        Score 0-1: What fraction of ground truth is in the context?
        """
        if not contexts or not ground_truth:
            return 0.0

        if self.embedding_service:
            try:
                # Use embedding similarity
                gt_embedding = await self.embedding_service.embed_text_async(ground_truth)
                combined = " ".join(contexts)
                ctx_embedding = await self.embedding_service.embed_text_async(combined)

                # Cosine similarity
                similarity = np.dot(gt_embedding, ctx_embedding) / (
                    np.linalg.norm(gt_embedding) * np.linalg.norm(ctx_embedding)
                )
                return float(max(0, similarity))
            except Exception as e:
                logger.warning("Embedding-based recall failed", error=str(e))

        # Fallback to keyword overlap
        return self._keyword_overlap_score(ground_truth, contexts)

    def _keyword_relevance(self, query: str, contexts: List[str]) -> float:
        """Simple keyword-based relevance scoring."""
        query_words = set(query.lower().split())
        context_words = set(" ".join(contexts).lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words & context_words)
        return min(1.0, overlap / len(query_words))

    def _keyword_overlap_score(self, text: str, contexts: List[str]) -> float:
        """Calculate keyword overlap between text and contexts."""
        text_words = set(text.lower().split())
        context_words = set(" ".join(contexts).lower().split())

        if not text_words:
            return 0.0

        overlap = len(text_words & context_words)
        return min(1.0, overlap / len(text_words))


class RAGBenchmark:
    """
    Run systematic RAG benchmarks.

    Evaluates RAG performance on a test set and provides
    aggregate metrics for quality monitoring.
    """

    def __init__(
        self,
        evaluator: RAGEvaluator,
        rag_service=None,
        passing_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            evaluator: RAGEvaluator instance
            rag_service: RAG service with query() method
            passing_thresholds: Minimum scores to "pass" (default: 0.6)
        """
        self.evaluator = evaluator
        self.rag_service = rag_service
        self.passing_thresholds = passing_thresholds or {
            "context_relevance": 0.6,
            "faithfulness": 0.7,
            "answer_relevance": 0.6,
            "overall_score": 0.65,
        }

    async def run_benchmark(
        self,
        test_cases: List[Dict[str, Any]],
        session=None,
    ) -> BenchmarkResult:
        """
        Run benchmark on a set of test cases.

        Args:
            test_cases: List of {query, ground_truth?, context?}
            session: Optional database session for RAG queries

        Returns:
            BenchmarkResult with aggregate metrics
        """
        start_time = datetime.utcnow()
        results = []
        passing_count = 0

        for case in test_cases:
            query = case.get("query", "")
            ground_truth = case.get("ground_truth")

            # Get RAG response if service available
            if self.rag_service and session:
                try:
                    response = await self.rag_service.query(query, session)
                    answer = response.content
                    contexts = [s.full_content or s.snippet for s in response.sources]
                except Exception as e:
                    logger.warning(f"RAG query failed: {e}")
                    answer = ""
                    contexts = case.get("contexts", [])
            else:
                answer = case.get("answer", "")
                contexts = case.get("contexts", [])

            # Evaluate
            result = await self.evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            results.append(result)

            # Check if passing
            if self._is_passing(result.metrics):
                passing_count += 1

        # Calculate aggregate metrics
        aggregate = self._calculate_aggregates(results)
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        logger.info(
            "Benchmark complete",
            test_count=len(test_cases),
            passing_rate=passing_count / len(test_cases) if test_cases else 0,
            overall_score=aggregate.get("overall_score", 0),
        )

        return BenchmarkResult(
            test_count=len(test_cases),
            individual_results=results,
            aggregate_metrics=aggregate,
            passing_rate=passing_count / len(test_cases) if test_cases else 0,
            duration_ms=duration,
        )

    def _is_passing(self, metrics: RAGASMetrics) -> bool:
        """Check if metrics meet passing thresholds."""
        for metric, threshold in self.passing_thresholds.items():
            value = getattr(metrics, metric, None)
            if value is not None and value < threshold:
                return False
        return True

    def _calculate_aggregates(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}

        metrics_names = [
            "context_relevance",
            "faithfulness",
            "answer_relevance",
            "context_recall",
            "overall_score",
        ]

        aggregates = {}
        for name in metrics_names:
            values = [
                getattr(r.metrics, name)
                for r in results
                if getattr(r.metrics, name, None) is not None
            ]
            if values:
                aggregates[name] = sum(values) / len(values)
                aggregates[f"{name}_std"] = float(np.std(values))
                aggregates[f"{name}_min"] = min(values)
                aggregates[f"{name}_max"] = max(values)

        return aggregates


class EvaluationTracker:
    """Track evaluation metrics over time for monitoring."""

    def __init__(self, max_history: int = 1000):
        """
        Initialize tracker.

        Args:
            max_history: Maximum evaluation results to keep
        """
        self.max_history = max_history
        self._history: List[EvaluationResult] = []

    def record(self, result: EvaluationResult):
        """Record an evaluation result."""
        self._history.append(result)

        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_recent_metrics(
        self,
        hours: int = 24,
    ) -> Dict[str, float]:
        """Get aggregate metrics for recent evaluations."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [r for r in self._history if r.timestamp > cutoff]

        if not recent:
            return {}

        return {
            "count": len(recent),
            "avg_overall": sum(r.metrics.overall_score for r in recent) / len(recent),
            "avg_faithfulness": sum(r.metrics.faithfulness for r in recent) / len(recent),
            "avg_relevance": sum(r.metrics.answer_relevance for r in recent) / len(recent),
        }

    def get_trend(
        self,
        metric: str = "overall_score",
        periods: int = 7,
        period_hours: int = 24,
    ) -> List[float]:
        """Get trend of a metric over time periods."""
        trends = []
        now = datetime.utcnow()

        for i in range(periods):
            start = now - timedelta(hours=(i + 1) * period_hours)
            end = now - timedelta(hours=i * period_hours)

            period_results = [
                r for r in self._history
                if start <= r.timestamp < end
            ]

            if period_results:
                avg = sum(getattr(r.metrics, metric) for r in period_results) / len(period_results)
                trends.append(avg)
            else:
                trends.append(None)

        return list(reversed(trends))


# =============================================================================
# Convenience Functions
# =============================================================================

async def evaluate_rag_response(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    llm=None,
    embedding_service=None,
) -> EvaluationResult:
    """
    Convenience function to evaluate a RAG response.

    Args:
        query: User's query
        answer: Generated answer
        contexts: Retrieved context chunks
        ground_truth: Optional expected answer
        llm: Optional LLM for evaluation
        embedding_service: Optional embedding service

    Returns:
        EvaluationResult
    """
    evaluator = RAGEvaluator(llm=llm, embedding_service=embedding_service)
    return await evaluator.evaluate(query, answer, contexts, ground_truth)


def get_quality_level(score: float) -> str:
    """
    Get quality level description from score.

    Args:
        score: 0-1 score

    Returns:
        "excellent", "good", "fair", or "poor"
    """
    if score >= 0.8:
        return "excellent"
    elif score >= 0.65:
        return "good"
    elif score >= 0.5:
        return "fair"
    else:
        return "poor"


def format_metrics_report(metrics: RAGASMetrics) -> str:
    """
    Format metrics as a readable report.

    Args:
        metrics: RAGASMetrics to format

    Returns:
        Formatted string report
    """
    quality = get_quality_level(metrics.overall_score)

    lines = [
        f"RAG Quality Report ({quality.upper()})",
        "=" * 40,
        f"Overall Score:      {metrics.overall_score:.2f}",
        f"Context Relevance:  {metrics.context_relevance:.2f}",
        f"Faithfulness:       {metrics.faithfulness:.2f}",
        f"Answer Relevance:   {metrics.answer_relevance:.2f}",
    ]

    if metrics.context_recall > 0:
        lines.append(f"Context Recall:     {metrics.context_recall:.2f}")

    if metrics.evaluation_time_ms:
        lines.append(f"Evaluation Time:    {metrics.evaluation_time_ms:.1f}ms")

    return "\n".join(lines)
