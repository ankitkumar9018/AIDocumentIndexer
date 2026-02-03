"""
AIDocumentIndexer - Langfuse Observability Service
===================================================

LLM observability, tracing, and evaluation using Langfuse.

Features:
- Request/response tracing
- Cost tracking per model
- Latency monitoring
- Quality evaluation
- User feedback collection
- A/B testing support
- Prompt versioning
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from functools import wraps
import hashlib
import json

import structlog

logger = structlog.get_logger(__name__)

# Check if langfuse is available
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.info("Langfuse not installed. Install with: pip install langfuse")


class TraceType(str, Enum):
    """Types of traces."""
    CHAT = "chat"
    RAG = "rag"
    EMBEDDING = "embedding"
    AGENT = "agent"
    TOOL = "tool"
    EVALUATION = "evaluation"


class FeedbackType(str, Enum):
    """Types of user feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 scale
    COMMENT = "comment"
    CORRECTION = "correction"


@dataclass
class TraceContext:
    """Context for a trace."""
    trace_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_type: TraceType = TraceType.CHAT
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


@dataclass
class GenerationMetrics:
    """Metrics for a generation."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of quality evaluation."""
    trace_id: str
    score: float  # 0-1
    category: str
    reasoning: str
    evaluator: str  # "human" or model name
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangfuseObservabilityService:
    """
    LLM observability service using Langfuse.

    Provides comprehensive tracing, monitoring, and evaluation
    for all LLM interactions in the system.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.sample_rate = sample_rate

        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                    host=host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                logger.info("Langfuse client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
                self.enabled = False
                self.client = None
        else:
            self.client = None

        # Cost tracking per model (USD per 1K tokens)
        self._model_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        }

        # Local metrics for when Langfuse is not available
        self._local_metrics: List[Dict[str, Any]] = []
        self._active_traces: Dict[str, TraceContext] = {}

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        trace_type: TraceType = TraceType.CHAT,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
    ):
        """
        Context manager for tracing operations.

        Usage:
            async with langfuse.trace("chat", user_id="123") as trace:
                # Do LLM operations
                trace.add_generation(...)
        """
        trace_id = self._generate_trace_id()
        context = TraceContext(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            trace_type=trace_type,
            metadata=metadata or {},
        )

        self._active_traces[trace_id] = context

        trace_obj = None
        if self.enabled and self.client and self._should_sample():
            try:
                trace_obj = self.client.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                    input=input_data,
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")

        try:
            yield TraceHandle(self, trace_id, trace_obj)
        finally:
            # Complete trace
            if trace_obj:
                try:
                    latency = (time.time() - context.start_time) * 1000
                    trace_obj.update(
                        output={"status": "completed"},
                        metadata={"latency_ms": latency, **context.metadata},
                    )
                except Exception as e:
                    logger.warning(f"Failed to complete Langfuse trace: {e}")

            del self._active_traces[trace_id]

    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: Union[str, List[Dict[str, str]]],
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GenerationMetrics:
        """
        Log a generation (LLM call).

        Args:
            trace_id: Parent trace ID
            name: Name of the generation
            model: Model used
            prompt: Prompt sent
            completion: Completion received
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            latency_ms: Latency in milliseconds
            metadata: Additional metadata

        Returns:
            GenerationMetrics with cost calculation
        """
        # Calculate cost
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        metrics = GenerationMetrics(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            success=True,
        )

        if self.enabled and self.client:
            try:
                self.client.generation(
                    trace_id=trace_id,
                    name=name,
                    model=model,
                    input=prompt,
                    output=completion,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    metadata={
                        "latency_ms": latency_ms,
                        "cost_usd": cost,
                        **(metadata or {}),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log Langfuse generation: {e}")

        # Store locally for analytics
        self._local_metrics.append({
            "trace_id": trace_id,
            "type": "generation",
            "model": model,
            "tokens": prompt_tokens + completion_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return metrics

    def log_span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a span (sub-operation) within a trace."""
        if self.enabled and self.client:
            try:
                self.client.span(
                    trace_id=trace_id,
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata={
                        "latency_ms": latency_ms,
                        **(metadata or {}),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log Langfuse span: {e}")

    def log_event(
        self,
        trace_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an event within a trace."""
        if self.enabled and self.client:
            try:
                self.client.event(
                    trace_id=trace_id,
                    name=name,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to log Langfuse event: {e}")

    def log_feedback(
        self,
        trace_id: str,
        feedback_type: FeedbackType,
        value: Union[bool, int, str],
        user_id: Optional[str] = None,
        comment: Optional[str] = None,
    ):
        """
        Log user feedback for a trace.

        Args:
            trace_id: Trace to provide feedback for
            feedback_type: Type of feedback
            value: Feedback value (bool for thumbs, int for rating, str for correction)
            user_id: User providing feedback
            comment: Optional comment
        """
        score_value = None
        if feedback_type == FeedbackType.THUMBS_UP:
            score_value = 1.0
        elif feedback_type == FeedbackType.THUMBS_DOWN:
            score_value = 0.0
        elif feedback_type == FeedbackType.RATING:
            score_value = float(value) / 5.0 if isinstance(value, (int, float)) else 0.5

        if self.enabled and self.client:
            try:
                self.client.score(
                    trace_id=trace_id,
                    name=feedback_type.value,
                    value=score_value if score_value is not None else 0.5,
                    comment=comment or str(value),
                )
            except Exception as e:
                logger.warning(f"Failed to log Langfuse feedback: {e}")

        # Store locally
        self._local_metrics.append({
            "trace_id": trace_id,
            "type": "feedback",
            "feedback_type": feedback_type.value,
            "value": value,
            "score": score_value,
            "user_id": user_id,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def evaluate(
        self,
        trace_id: str,
        evaluator: str = "default",
        criteria: Optional[Dict[str, str]] = None,
    ) -> EvaluationResult:
        """
        Run evaluation on a trace.

        Args:
            trace_id: Trace to evaluate
            evaluator: Evaluator to use (model name or "human")
            criteria: Evaluation criteria

        Returns:
            EvaluationResult with score and reasoning
        """
        # Default criteria
        if criteria is None:
            criteria = {
                "relevance": "Is the response relevant to the query?",
                "accuracy": "Is the information accurate?",
                "helpfulness": "Is the response helpful?",
            }

        # In production, this would call an LLM to evaluate
        # For now, return placeholder
        result = EvaluationResult(
            trace_id=trace_id,
            score=0.8,
            category="quality",
            reasoning="Automated evaluation placeholder",
            evaluator=evaluator,
            metadata={"criteria": criteria},
        )

        if self.enabled and self.client:
            try:
                self.client.score(
                    trace_id=trace_id,
                    name="evaluation",
                    value=result.score,
                    comment=result.reasoning,
                )
            except Exception as e:
                logger.warning(f"Failed to log Langfuse evaluation: {e}")

        return result

    def get_prompt(
        self,
        name: str,
        version: Optional[int] = None,
        label: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a prompt from Langfuse prompt management.

        Args:
            name: Prompt name
            version: Specific version (optional)
            label: Label like "production" or "staging" (optional)

        Returns:
            Prompt template string or None
        """
        if not self.enabled or not self.client:
            return None

        try:
            prompt = self.client.get_prompt(
                name=name,
                version=version,
                label=label,
            )
            return prompt.prompt if prompt else None
        except Exception as e:
            logger.warning(f"Failed to get Langfuse prompt: {e}")
            return None

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for a generation."""
        # Normalize model name
        model_lower = model.lower()
        costs = None

        for model_key, model_costs in self._model_costs.items():
            if model_key in model_lower:
                costs = model_costs
                break

        if not costs:
            # Default costs for unknown models
            costs = {"input": 0.001, "output": 0.002}

        input_cost = (prompt_tokens / 1000) * costs["input"]
        output_cost = (completion_tokens / 1000) * costs["output"]

        return round(input_cost + output_cost, 6)

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        content = f"{time.time()}_{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _should_sample(self) -> bool:
        """Determine if this request should be sampled."""
        import random
        return random.random() < self.sample_rate

    async def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get analytics from local metrics.

        In production, this would query Langfuse API.
        """
        metrics = self._local_metrics

        if start_date:
            metrics = [m for m in metrics if m.get("timestamp", "") >= start_date.isoformat()]
        if end_date:
            metrics = [m for m in metrics if m.get("timestamp", "") <= end_date.isoformat()]

        generations = [m for m in metrics if m.get("type") == "generation"]
        feedback = [m for m in metrics if m.get("type") == "feedback"]

        total_tokens = sum(m.get("tokens", 0) for m in generations)
        total_cost = sum(m.get("cost", 0) for m in generations)
        avg_latency = (
            sum(m.get("latency_ms", 0) for m in generations) / len(generations)
            if generations else 0
        )

        positive_feedback = sum(1 for f in feedback if f.get("score", 0) >= 0.5)
        feedback_rate = positive_feedback / len(feedback) if feedback else 0

        return {
            "total_requests": len(generations),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "feedback_count": len(feedback),
            "positive_feedback_rate": round(feedback_rate, 2),
            "by_model": self._aggregate_by_model(generations),
        }

    def _aggregate_by_model(self, generations: List[Dict]) -> Dict[str, Dict]:
        """Aggregate metrics by model."""
        by_model: Dict[str, Dict] = {}

        for gen in generations:
            model = gen.get("model", "unknown")
            if model not in by_model:
                by_model[model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0,
                    "avg_latency_ms": 0,
                }

            by_model[model]["requests"] += 1
            by_model[model]["tokens"] += gen.get("tokens", 0)
            by_model[model]["cost"] += gen.get("cost", 0)

        # Calculate averages
        for model in by_model:
            if by_model[model]["requests"] > 0:
                model_gens = [g for g in generations if g.get("model") == model]
                by_model[model]["avg_latency_ms"] = round(
                    sum(g.get("latency_ms", 0) for g in model_gens) / len(model_gens), 2
                )
                by_model[model]["cost"] = round(by_model[model]["cost"], 4)

        return by_model

    def flush(self):
        """Flush any pending data to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse: {e}")


class TraceHandle:
    """Handle for interacting with an active trace."""

    def __init__(
        self,
        service: LangfuseObservabilityService,
        trace_id: str,
        trace_obj: Optional[Any],
    ):
        self.service = service
        self.trace_id = trace_id
        self._trace_obj = trace_obj

    def log_generation(self, **kwargs) -> GenerationMetrics:
        """Log a generation within this trace."""
        return self.service.log_generation(trace_id=self.trace_id, **kwargs)

    def log_span(self, **kwargs):
        """Log a span within this trace."""
        self.service.log_span(trace_id=self.trace_id, **kwargs)

    def log_event(self, **kwargs):
        """Log an event within this trace."""
        self.service.log_event(trace_id=self.trace_id, **kwargs)

    def set_output(self, output: Any):
        """Set the trace output."""
        if self._trace_obj:
            try:
                self._trace_obj.update(output=output)
            except Exception:
                pass


# Singleton instance
_langfuse_service: Optional[LangfuseObservabilityService] = None


def get_langfuse_service() -> LangfuseObservabilityService:
    """Get or create the Langfuse service singleton."""
    global _langfuse_service
    if _langfuse_service is None:
        _langfuse_service = LangfuseObservabilityService()
    return _langfuse_service


# Decorator for easy tracing
def traced(
    name: Optional[str] = None,
    trace_type: TraceType = TraceType.CHAT,
):
    """
    Decorator to trace a function.

    Usage:
        @traced("my_function")
        async def my_function(query: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            service = get_langfuse_service()
            func_name = name or func.__name__

            async with service.trace(func_name, trace_type) as trace:
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    trace.log_span(
                        name=f"{func_name}_execution",
                        latency_ms=(time.time() - start) * 1000,
                        output_data={"status": "success"},
                    )
                    return result
                except Exception as e:
                    trace.log_event(
                        name="error",
                        metadata={"error": str(e)},
                    )
                    raise

        return wrapper
    return decorator
