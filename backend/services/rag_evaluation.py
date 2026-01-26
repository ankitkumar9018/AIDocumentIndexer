"""
AIDocumentIndexer - RAG Evaluation Framework
=============================================

Implements RAGAS-inspired metrics for systematic RAG quality evaluation:
1. Context Relevance: How relevant is retrieved context to the query?
2. Faithfulness: Is the answer grounded in the context?
3. Answer Relevance: Does the answer address the question?
4. Context Recall: Did we retrieve all needed information?

Also includes DeepEval integration (2025) for self-explaining metrics:
- Provides detailed reasoning for each metric score
- Better interpretability for debugging and improvement
- Automatic identification of failure modes

References:
- RAGAS: https://github.com/explodinggradients/ragas
- DeepEval: https://github.com/confident-ai/deepeval
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


# =============================================================================
# DeepEval Integration (2025)
# =============================================================================

@dataclass
class DeepEvalMetric:
    """Single DeepEval metric with self-explaining reasoning."""
    name: str
    score: float  # 0-1
    passed: bool
    reason: str  # Self-explaining reason for the score
    threshold: float = 0.5


@dataclass
class DeepEvalResult:
    """Complete DeepEval result with self-explaining metrics."""
    query: str
    answer: str
    contexts: List[str]
    metrics: List[DeepEvalMetric]
    overall_passed: bool
    failure_modes: List[str]
    improvement_suggestions: List[str]
    evaluation_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_score(self) -> float:
        """Calculate overall score from individual metrics."""
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)


class DeepEvalEvaluator:
    """
    DeepEval-style evaluator with self-explaining metrics.

    Provides detailed reasoning for each metric, making it easier
    to understand why a RAG response succeeded or failed.
    """

    # Evaluation prompt templates with self-explanation
    FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Answer: {answer}

Evaluate faithfulness by checking:
1. Are all claims in the answer supported by the context?
2. Does the answer contradict any information in the context?
3. Does the answer make claims not found in the context?

Provide your evaluation in this EXACT format:
SCORE: [0-10 where 10 is perfectly faithful]
VERDICT: [PASS if score >= 7, else FAIL]
REASON: [1-2 sentences explaining your score, citing specific examples]
CLAIMS_UNSUPPORTED: [List any unsupported claims, or "None"]"""

    ANSWER_RELEVANCY_PROMPT = """You are evaluating whether an answer is relevant to the question.

Question: {query}

Answer: {answer}

Evaluate answer relevancy by checking:
1. Does the answer directly address the question?
2. Is the answer complete or does it miss key aspects?
3. Does the answer include irrelevant information?

Provide your evaluation in this EXACT format:
SCORE: [0-10 where 10 is perfectly relevant]
VERDICT: [PASS if score >= 7, else FAIL]
REASON: [1-2 sentences explaining your score]
MISSING_ASPECTS: [List any aspects of the question not addressed, or "None"]"""

    CONTEXT_RELEVANCY_PROMPT = """You are evaluating whether the retrieved context is relevant to answering the question.

Question: {query}

Context:
{context}

Evaluate context relevancy by checking:
1. Does the context contain information needed to answer the question?
2. How much of the context is actually useful vs. noise?
3. Is critical information missing from the context?

Provide your evaluation in this EXACT format:
SCORE: [0-10 where 10 is perfectly relevant context]
VERDICT: [PASS if score >= 6, else FAIL]
REASON: [1-2 sentences explaining your score]
USEFUL_PERCENTAGE: [Estimated % of context that is useful]"""

    HALLUCINATION_PROMPT = """You are checking for hallucinations in an answer.

Context (the only source of truth):
{context}

Answer to check: {answer}

A hallucination is when the answer states something as fact that:
1. Is not supported by the context
2. Contradicts the context
3. Makes up specific details (names, numbers, dates) not in the context

Provide your evaluation in this EXACT format:
SCORE: [0-10 where 10 means NO hallucinations]
VERDICT: [PASS if score >= 8, else FAIL]
REASON: [1-2 sentences explaining your findings]
HALLUCINATIONS_FOUND: [List specific hallucinations, or "None"]"""

    def __init__(
        self,
        llm=None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize DeepEval evaluator.

        Args:
            llm: LangChain LLM for evaluations
            thresholds: Custom thresholds for each metric
        """
        self.llm = llm
        self.thresholds = thresholds or {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_relevancy": 0.6,
            "hallucination": 0.8,
        }

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> DeepEvalResult:
        """
        Evaluate a RAG response with self-explaining metrics.

        Args:
            query: User's question
            answer: Generated answer
            contexts: Retrieved context chunks

        Returns:
            DeepEvalResult with detailed reasoning
        """
        start_time = datetime.utcnow()
        metrics = []
        failure_modes = []
        suggestions = []

        combined_context = "\n---\n".join(contexts[:5])

        # Evaluate all metrics in parallel if LLM available
        if self.llm:
            results = await self._evaluate_all_metrics(query, answer, combined_context)
        else:
            results = self._fallback_evaluation(query, answer, contexts)

        for metric_name, result in results.items():
            threshold = self.thresholds.get(metric_name, 0.7)
            passed = result["score"] >= threshold

            metrics.append(DeepEvalMetric(
                name=metric_name,
                score=result["score"],
                passed=passed,
                reason=result["reason"],
                threshold=threshold,
            ))

            if not passed:
                failure_modes.append(f"{metric_name}: {result['reason']}")

                # Generate improvement suggestions based on failure mode
                if metric_name == "faithfulness":
                    suggestions.append("Consider adding citation markers or reducing claims beyond context")
                elif metric_name == "answer_relevancy":
                    suggestions.append("Rephrase to more directly address the question")
                elif metric_name == "context_relevancy":
                    suggestions.append("Improve retrieval with query expansion or reranking")
                elif metric_name == "hallucination":
                    suggestions.append("Add strict grounding constraints or fact verification")

        eval_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        overall_passed = all(m.passed for m in metrics)

        return DeepEvalResult(
            query=query,
            answer=answer,
            contexts=contexts,
            metrics=metrics,
            overall_passed=overall_passed,
            failure_modes=failure_modes,
            improvement_suggestions=suggestions,
            evaluation_time_ms=eval_time,
        )

    async def _evaluate_all_metrics(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all metrics using LLM."""
        import asyncio

        prompts = {
            "faithfulness": self.FAITHFULNESS_PROMPT.format(
                context=context[:8000], answer=answer
            ),
            "answer_relevancy": self.ANSWER_RELEVANCY_PROMPT.format(
                query=query, answer=answer
            ),
            "context_relevancy": self.CONTEXT_RELEVANCY_PROMPT.format(
                query=query, context=context[:8000]
            ),
            "hallucination": self.HALLUCINATION_PROMPT.format(
                context=context[:8000], answer=answer
            ),
        }

        async def evaluate_single(name: str, prompt: str) -> tuple:
            try:
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                parsed = self._parse_evaluation_response(content)
                return name, parsed
            except Exception as e:
                logger.warning(f"DeepEval {name} failed: {e}")
                return name, {"score": 0.5, "reason": f"Evaluation failed: {e}"}

        tasks = [evaluate_single(name, prompt) for name, prompt in prompts.items()]
        results = await asyncio.gather(*tasks)

        return dict(results)

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response."""
        lines = response.strip().split('\n')
        result = {"score": 0.5, "reason": "Could not parse response"}

        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip()
                    score = float(score_text.split()[0]) / 10.0
                    result["score"] = max(0, min(1, score))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASON:"):
                result["reason"] = line.replace("REASON:", "").strip()

        return result

    def _fallback_evaluation(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Keyword-based fallback when LLM is unavailable."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())

        # Simple overlap-based metrics
        answer_query_overlap = len(query_words & answer_words) / max(len(query_words), 1)
        answer_context_overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        context_query_overlap = len(query_words & context_words) / max(len(query_words), 1)

        return {
            "faithfulness": {
                "score": min(1.0, answer_context_overlap * 1.5),
                "reason": f"Keyword overlap with context: {answer_context_overlap:.0%}",
            },
            "answer_relevancy": {
                "score": min(1.0, answer_query_overlap * 1.5),
                "reason": f"Keyword overlap with query: {answer_query_overlap:.0%}",
            },
            "context_relevancy": {
                "score": min(1.0, context_query_overlap * 1.5),
                "reason": f"Context covers {context_query_overlap:.0%} of query terms",
            },
            "hallucination": {
                "score": min(1.0, answer_context_overlap * 1.2),
                "reason": f"Based on keyword grounding: {answer_context_overlap:.0%}",
            },
        }


async def evaluate_with_deepeval(
    query: str,
    answer: str,
    contexts: List[str],
    llm=None,
) -> DeepEvalResult:
    """
    Convenience function for DeepEval evaluation.

    Args:
        query: User's question
        answer: Generated answer
        contexts: Retrieved contexts
        llm: Optional LLM for evaluation

    Returns:
        DeepEvalResult with self-explaining metrics
    """
    evaluator = DeepEvalEvaluator(llm=llm)
    return await evaluator.evaluate(query, answer, contexts)


def format_deepeval_report(result: DeepEvalResult) -> str:
    """
    Format DeepEval result as readable report.

    Args:
        result: DeepEvalResult to format

    Returns:
        Formatted report string
    """
    status = "PASSED" if result.overall_passed else "FAILED"
    lines = [
        f"DeepEval Report ({status})",
        "=" * 50,
        f"Overall Score: {result.overall_score:.2f}",
        "",
        "Metrics:",
    ]

    for metric in result.metrics:
        status_icon = "[PASS]" if metric.passed else "[FAIL]"
        lines.append(f"  {status_icon} {metric.name}: {metric.score:.2f}")
        lines.append(f"         Reason: {metric.reason}")

    if result.failure_modes:
        lines.append("")
        lines.append("Failure Modes:")
        for mode in result.failure_modes:
            lines.append(f"  - {mode}")

    if result.improvement_suggestions:
        lines.append("")
        lines.append("Suggestions:")
        for suggestion in result.improvement_suggestions:
            lines.append(f"  - {suggestion}")

    lines.append("")
    lines.append(f"Evaluation Time: {result.evaluation_time_ms:.1f}ms")

    return "\n".join(lines)


# =============================================================================
# Native DeepEval Library Integration (2025-2026)
# =============================================================================

class DeepEvalLibraryIntegration:
    """
    Integration with the official DeepEval library.

    Provides access to DeepEval's comprehensive metric suite:
    - G-Eval: LLM-based evaluation with chain-of-thought
    - Summarization metrics
    - Bias and toxicity detection
    - JSON correctness validation
    - Custom metric definitions

    Falls back to built-in evaluation when library not installed.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_async: bool = True,
    ):
        """
        Initialize DeepEval integration.

        Args:
            model: Model to use for evaluation
            use_async: Whether to use async evaluation
        """
        self.model = model
        self.use_async = use_async
        self._deepeval_available = False
        self._metrics = {}
        self._initialize()

    def _initialize(self):
        """Check if DeepEval library is available."""
        try:
            import deepeval
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualRelevancyMetric,
                HallucinationMetric,
            )
            self._deepeval_available = True

            # Initialize metrics
            self._metrics = {
                "faithfulness": FaithfulnessMetric(
                    model=self.model,
                    threshold=0.7,
                ),
                "answer_relevancy": AnswerRelevancyMetric(
                    model=self.model,
                    threshold=0.7,
                ),
                "contextual_relevancy": ContextualRelevancyMetric(
                    model=self.model,
                    threshold=0.6,
                ),
                "hallucination": HallucinationMetric(
                    model=self.model,
                    threshold=0.8,
                ),
            }
            logger.info("DeepEval library initialized", model=self.model)

        except ImportError:
            logger.info("DeepEval library not installed, using built-in evaluation")
            self._deepeval_available = False

    @property
    def is_available(self) -> bool:
        """Check if DeepEval library is available."""
        return self._deepeval_available

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> DeepEvalResult:
        """
        Evaluate using DeepEval library.

        Args:
            query: User's question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional expected answer

        Returns:
            DeepEvalResult with library-powered metrics
        """
        start_time = datetime.utcnow()

        if not self._deepeval_available:
            # Fall back to built-in evaluator
            evaluator = DeepEvalEvaluator()
            return await evaluator.evaluate(query, answer, contexts)

        try:
            from deepeval.test_case import LLMTestCase

            # Create test case
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=contexts,
                expected_output=ground_truth,
            )

            metrics_results = []
            failure_modes = []
            suggestions = []

            # Evaluate each metric
            for metric_name, metric in self._metrics.items():
                try:
                    if self.use_async:
                        await metric.a_measure(test_case)
                    else:
                        metric.measure(test_case)

                    passed = metric.is_successful()
                    score = metric.score if hasattr(metric, 'score') else (1.0 if passed else 0.0)
                    reason = metric.reason if hasattr(metric, 'reason') else "Evaluated by DeepEval"

                    metrics_results.append(DeepEvalMetric(
                        name=metric_name,
                        score=score,
                        passed=passed,
                        reason=reason,
                        threshold=metric.threshold,
                    ))

                    if not passed:
                        failure_modes.append(f"{metric_name}: {reason}")
                        suggestions.append(self._get_suggestion(metric_name))

                except Exception as e:
                    logger.warning(f"DeepEval metric {metric_name} failed", error=str(e))
                    metrics_results.append(DeepEvalMetric(
                        name=metric_name,
                        score=0.5,
                        passed=False,
                        reason=f"Evaluation error: {str(e)}",
                        threshold=0.7,
                    ))

            eval_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return DeepEvalResult(
                query=query,
                answer=answer,
                contexts=contexts,
                metrics=metrics_results,
                overall_passed=all(m.passed for m in metrics_results),
                failure_modes=failure_modes,
                improvement_suggestions=suggestions,
                evaluation_time_ms=eval_time,
            )

        except Exception as e:
            logger.error("DeepEval evaluation failed", error=str(e))
            # Fall back to built-in
            evaluator = DeepEvalEvaluator()
            return await evaluator.evaluate(query, answer, contexts)

    def _get_suggestion(self, metric_name: str) -> str:
        """Get improvement suggestion for a failed metric."""
        suggestions = {
            "faithfulness": "Add stricter grounding constraints or citation requirements",
            "answer_relevancy": "Improve answer generation to focus on the question",
            "contextual_relevancy": "Enhance retrieval with reranking or query expansion",
            "hallucination": "Implement fact verification or constrain generation",
            "bias": "Review prompts and training data for potential biases",
            "toxicity": "Add content filtering or safety guardrails",
        }
        return suggestions.get(metric_name, "Review and improve the RAG pipeline")

    async def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[DeepEvalResult]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of {query, answer, contexts, ground_truth?}

        Returns:
            List of DeepEvalResult
        """
        import asyncio

        tasks = [
            self.evaluate(
                query=case.get("query", ""),
                answer=case.get("answer", ""),
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth"),
            )
            for case in test_cases
        ]

        return await asyncio.gather(*tasks)

    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        if self._deepeval_available:
            return list(self._metrics.keys())
        return ["faithfulness", "answer_relevancy", "context_relevancy", "hallucination"]


# =============================================================================
# G-Eval Implementation (2025)
# =============================================================================

class GEvalMetric:
    """
    G-Eval: LLM-based evaluation with chain-of-thought.

    Based on: "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"

    Features:
    - Chain-of-thought prompting for nuanced evaluation
    - Form-filling paradigm for structured output
    - Better alignment with human judgment than traditional metrics
    """

    GEVAL_PROMPT = """You will be given a task to evaluate. Follow these steps:

1. Read the evaluation criteria carefully
2. Think step-by-step about how well the response meets the criteria
3. Provide your reasoning
4. Give a score from 1-5

Task: Evaluate the {metric_name} of a RAG response.

Evaluation Criteria:
{criteria}

Question: {query}

Retrieved Context:
{context}

Response to Evaluate: {answer}

Now evaluate step-by-step:

Step 1 - Understanding: What is the question asking for?

Step 2 - Analysis: {analysis_prompt}

Step 3 - Reasoning: Based on your analysis, explain your evaluation.

Step 4 - Score: Provide a score from 1-5 where:
1 = Very Poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

FORMAT YOUR RESPONSE AS:
REASONING: [Your step-by-step reasoning]
SCORE: [1-5]"""

    CRITERIA = {
        "faithfulness": """
The response should be faithful to the retrieved context:
- All claims should be supported by the context
- No contradictions with the context
- No invented facts or hallucinations
""",
        "coherence": """
The response should be coherent and well-structured:
- Logical flow of ideas
- Clear and understandable language
- Appropriate level of detail
""",
        "relevance": """
The response should be relevant to the question:
- Directly addresses what was asked
- Doesn't include unnecessary information
- Complete answer to the question
""",
        "fluency": """
The response should be fluent and natural:
- Grammatically correct
- Natural language flow
- Appropriate tone
""",
    }

    ANALYSIS_PROMPTS = {
        "faithfulness": "Check each claim in the response against the context. Are they supported?",
        "coherence": "Analyze the structure and flow of the response. Is it logical?",
        "relevance": "Compare the question and response. Does it address what was asked?",
        "fluency": "Read the response aloud mentally. Does it sound natural?",
    }

    def __init__(
        self,
        llm=None,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize G-Eval.

        Args:
            llm: LLM for evaluation
            metrics: Which metrics to evaluate (default: all)
        """
        self.llm = llm
        self.metrics = metrics or list(self.CRITERIA.keys())

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate using G-Eval methodology.

        Returns:
            Dict with score and reasoning for each metric
        """
        if not self.llm:
            logger.warning("G-Eval requires LLM, returning default scores")
            return {m: {"score": 0.6, "reasoning": "LLM not available"} for m in self.metrics}

        combined_context = "\n---\n".join(contexts[:5])
        results = {}

        import asyncio

        async def evaluate_metric(metric_name: str) -> tuple:
            prompt = self.GEVAL_PROMPT.format(
                metric_name=metric_name,
                criteria=self.CRITERIA[metric_name],
                query=query,
                context=combined_context[:6000],
                answer=answer,
                analysis_prompt=self.ANALYSIS_PROMPTS[metric_name],
            )

            try:
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                return metric_name, self._parse_geval_response(content)
            except Exception as e:
                logger.warning(f"G-Eval {metric_name} failed", error=str(e))
                return metric_name, {"score": 0.6, "reasoning": f"Evaluation failed: {e}"}

        tasks = [evaluate_metric(m) for m in self.metrics]
        metric_results = await asyncio.gather(*tasks)

        return dict(metric_results)

    def _parse_geval_response(self, response: str) -> Dict[str, Any]:
        """Parse G-Eval response."""
        result = {"score": 0.6, "reasoning": "Could not parse response"}

        lines = response.strip().split('\n')

        for i, line in enumerate(lines):
            if line.startswith("REASONING:"):
                # Get all text until SCORE
                reasoning_lines = [line.replace("REASONING:", "").strip()]
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("SCORE:"):
                        break
                    reasoning_lines.append(lines[j])
                result["reasoning"] = " ".join(reasoning_lines)

            elif line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip()
                    score = float(score_text.split()[0])
                    result["score"] = (score - 1) / 4  # Convert 1-5 to 0-1
                except (ValueError, IndexError):
                    pass

        return result


# =============================================================================
# Factory Functions
# =============================================================================

_deepeval_integration: Optional[DeepEvalLibraryIntegration] = None
_geval_metric: Optional[GEvalMetric] = None


def get_deepeval_integration(
    model: str = "gpt-4o-mini",
) -> DeepEvalLibraryIntegration:
    """Get or create DeepEval library integration."""
    global _deepeval_integration
    if _deepeval_integration is None:
        _deepeval_integration = DeepEvalLibraryIntegration(model=model)
    return _deepeval_integration


def get_geval_metric(llm=None) -> GEvalMetric:
    """Get or create G-Eval metric."""
    global _geval_metric
    if _geval_metric is None:
        _geval_metric = GEvalMetric(llm=llm)
    return _geval_metric


async def comprehensive_evaluation(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    llm=None,
    use_deepeval_library: bool = True,
    use_geval: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation using all available methods.

    Args:
        query: User's question
        answer: Generated answer
        contexts: Retrieved contexts
        ground_truth: Optional expected answer
        llm: LLM for evaluation
        use_deepeval_library: Whether to try DeepEval library
        use_geval: Whether to include G-Eval metrics

    Returns:
        Combined evaluation results
    """
    results = {}

    # RAGAS-style evaluation
    ragas_evaluator = RAGEvaluator(llm=llm)
    ragas_result = await ragas_evaluator.evaluate(query, answer, contexts, ground_truth)
    results["ragas"] = {
        "metrics": {
            "context_relevance": ragas_result.metrics.context_relevance,
            "faithfulness": ragas_result.metrics.faithfulness,
            "answer_relevance": ragas_result.metrics.answer_relevance,
            "context_recall": ragas_result.metrics.context_recall,
            "overall": ragas_result.metrics.overall_score,
        },
        "issues": ragas_result.issues,
        "suggestions": ragas_result.suggestions,
    }

    # DeepEval-style evaluation
    if use_deepeval_library:
        deepeval_integration = get_deepeval_integration()
        deepeval_result = await deepeval_integration.evaluate(query, answer, contexts, ground_truth)
    else:
        deepeval_evaluator = DeepEvalEvaluator(llm=llm)
        deepeval_result = await deepeval_evaluator.evaluate(query, answer, contexts)

    results["deepeval"] = {
        "metrics": {m.name: m.score for m in deepeval_result.metrics},
        "overall_passed": deepeval_result.overall_passed,
        "overall_score": deepeval_result.overall_score,
        "failure_modes": deepeval_result.failure_modes,
        "suggestions": deepeval_result.improvement_suggestions,
    }

    # G-Eval evaluation
    if use_geval and llm:
        geval = get_geval_metric(llm=llm)
        geval_results = await geval.evaluate(query, answer, contexts)
        results["geval"] = {
            "metrics": {k: v["score"] for k, v in geval_results.items()},
            "reasoning": {k: v["reasoning"] for k, v in geval_results.items()},
        }

    # Calculate combined score
    all_scores = []
    for eval_type, eval_data in results.items():
        if "metrics" in eval_data:
            all_scores.extend(eval_data["metrics"].values())

    results["combined_score"] = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return results
