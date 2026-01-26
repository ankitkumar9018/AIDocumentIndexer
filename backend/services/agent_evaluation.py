"""
AIDocumentIndexer - Agent Evaluation & Personalization
=======================================================

Phase 23B: Implements agent evaluation metrics and personalization system.

Evaluation Metrics (based on 2025 Agent benchmarks):
- Pass^k: Reliability across k trials (target: >95% at k=3)
- Progress Rate: Task completion advancement (target: >80%)
- Invocation Accuracy: Correct tool/knowledge retrieval (target: >90%)
- Hallucination Rate: False information generation (target: <5%)

Personalization System:
- User preference learning from interactions
- Communication style adaptation
- Expertise level detection
- Topic interest modeling
- Feedback loop for continuous improvement

Based on:
- SWE-agent evaluation framework
- LlamaIndex Agent benchmarks
- Multi-agent personalization systems (ICLR 2025)
"""

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import uuid

import structlog

from backend.services.agent_memory import AgentMemory, MemoryType

logger = structlog.get_logger(__name__)


# =============================================================================
# Evaluation Data Classes
# =============================================================================

class MetricType(str, Enum):
    """Types of evaluation metrics."""
    PASS_K = "pass_k"
    PROGRESS_RATE = "progress_rate"
    INVOCATION_ACCURACY = "invocation_accuracy"
    HALLUCINATION_RATE = "hallucination_rate"
    RESPONSE_QUALITY = "response_quality"
    LATENCY = "latency"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class TrialResult:
    """Result of a single evaluation trial."""
    trial_id: str
    query: str
    response: str
    ground_truth: Optional[str] = None
    passed: bool = False
    score: float = 0.0
    latency_ms: float = 0.0
    hallucination_detected: bool = False
    retrieved_docs: int = 0
    correct_retrievals: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "query": self.query,
            "passed": self.passed,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "hallucination_detected": self.hallucination_detected,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvaluationResult:
    """Aggregate evaluation results."""
    agent_id: str
    evaluation_id: str
    trials: List[TrialResult]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def pass_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.passed) / len(self.trials)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "evaluation_id": self.evaluation_id,
            "trial_count": len(self.trials),
            "pass_rate": self.pass_rate,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PassKResult:
    """Result of Pass^k evaluation."""
    k: int
    pass_rate: float  # Probability of at least one pass in k trials
    trials_per_query: int
    total_queries: int
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


# =============================================================================
# Agent Evaluator
# =============================================================================

class AgentEvaluator:
    """
    Evaluates agent performance using multiple metrics.

    Implements:
    - Pass^k reliability metric
    - Progress rate for multi-step tasks
    - Hallucination detection
    - Retrieval accuracy
    """

    def __init__(
        self,
        agent_id: str,
        llm_service=None,
        ground_truth_checker=None,
    ):
        self.agent_id = agent_id
        self.llm = llm_service
        self.ground_truth_checker = ground_truth_checker
        self._trial_history: List[TrialResult] = []
        self._redis = None

    async def initialize(self):
        """Initialize evaluator and load history."""
        try:
            from backend.services.redis_client import get_redis_client
            self._redis = await get_redis_client()
            await self._load_history()
        except Exception as e:
            logger.warning(f"Could not initialize evaluator storage: {e}")

    async def _load_history(self):
        """Load evaluation history from storage."""
        if not self._redis:
            return

        try:
            key = f"agent:evaluation:{self.agent_id}"
            data = await self._redis.get(key)
            if data:
                parsed = json.loads(data)
                self._trial_history = [
                    TrialResult(**{
                        **t,
                        "timestamp": datetime.fromisoformat(t["timestamp"])
                    })
                    for t in parsed.get("trials", [])[-1000:]  # Keep last 1000
                ]
        except Exception as e:
            logger.error(f"Failed to load evaluation history: {e}")

    async def _save_history(self):
        """Save evaluation history to storage."""
        if not self._redis:
            return

        try:
            key = f"agent:evaluation:{self.agent_id}"
            data = {
                "trials": [t.to_dict() for t in self._trial_history[-1000:]],
            }
            await self._redis.set(key, json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to save evaluation history: {e}")

    async def record_trial(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None,
        latency_ms: float = 0.0,
        retrieved_docs: int = 0,
        correct_retrievals: int = 0,
        metadata: Optional[Dict] = None,
    ) -> TrialResult:
        """
        Record a single trial result.

        Args:
            query: Input query
            response: Agent response
            ground_truth: Expected answer (if available)
            latency_ms: Response latency
            retrieved_docs: Number of documents retrieved
            correct_retrievals: Number of correctly retrieved documents

        Returns:
            TrialResult with evaluation
        """
        trial = TrialResult(
            trial_id=str(uuid.uuid4()),
            query=query,
            response=response,
            ground_truth=ground_truth,
            latency_ms=latency_ms,
            retrieved_docs=retrieved_docs,
            correct_retrievals=correct_retrievals,
            metadata=metadata or {},
        )

        # Evaluate the trial
        await self._evaluate_trial(trial)

        self._trial_history.append(trial)
        await self._save_history()

        return trial

    async def _evaluate_trial(self, trial: TrialResult):
        """Evaluate a single trial."""
        # Check against ground truth if available
        if trial.ground_truth and self.ground_truth_checker:
            trial.passed, trial.score = await self.ground_truth_checker(
                trial.response,
                trial.ground_truth,
            )
        elif trial.ground_truth:
            # Simple exact/fuzzy match
            trial.passed = self._fuzzy_match(trial.response, trial.ground_truth)
            trial.score = 1.0 if trial.passed else 0.0
        else:
            # No ground truth - use heuristics
            trial.passed = len(trial.response) > 20  # Non-trivial response
            trial.score = 0.5

        # Check for hallucination
        if self.llm:
            trial.hallucination_detected = await self._detect_hallucination(trial)

    def _fuzzy_match(self, response: str, ground_truth: str) -> bool:
        """Check if response matches ground truth (fuzzy)."""
        response_lower = response.lower()
        truth_lower = ground_truth.lower()

        # Check if key parts of ground truth appear in response
        truth_words = set(truth_lower.split())
        response_words = set(response_lower.split())

        # If ground truth is short, require high overlap
        if len(truth_words) <= 5:
            overlap = len(truth_words & response_words) / len(truth_words)
            return overlap >= 0.8

        # For longer ground truth, be more lenient
        overlap = len(truth_words & response_words) / len(truth_words)
        return overlap >= 0.5 or truth_lower in response_lower

    async def _detect_hallucination(self, trial: TrialResult) -> bool:
        """Detect if response contains hallucinations."""
        if not self.llm:
            return False

        try:
            prompt = f"""Analyze if this response contains any hallucinations or false claims not supported by typical knowledge.

Question: {trial.query}
Response: {trial.response}

Does this response contain any obvious hallucinations, false claims, or made-up information?
Answer with just YES or NO."""

            result = await self.llm.generate(prompt)
            return "YES" in result.upper()

        except Exception as e:
            logger.warning(f"Hallucination detection failed: {e}")
            return False

    async def compute_pass_k(
        self,
        k: int = 3,
        query_groups: Optional[Dict[str, List[TrialResult]]] = None,
    ) -> PassKResult:
        """
        Compute Pass^k metric.

        Pass^k measures the probability that at least one of k trials
        passes for each query. Higher is better.

        Args:
            k: Number of trials to consider
            query_groups: Trials grouped by query (auto-groups if not provided)

        Returns:
            PassKResult with pass rate
        """
        if query_groups is None:
            # Group trials by query
            query_groups = defaultdict(list)
            for trial in self._trial_history:
                query_groups[trial.query].append(trial)

        if not query_groups:
            return PassKResult(k=k, pass_rate=0.0, trials_per_query=0, total_queries=0)

        # For each query, check if at least one of k trials passed
        passed_queries = 0
        total_queries = 0

        for query, trials in query_groups.items():
            if len(trials) < k:
                continue

            total_queries += 1
            # Check last k trials
            recent_trials = sorted(trials, key=lambda t: t.timestamp, reverse=True)[:k]
            if any(t.passed for t in recent_trials):
                passed_queries += 1

        pass_rate = passed_queries / total_queries if total_queries > 0 else 0.0

        # Compute confidence interval (Wilson score)
        ci = self._wilson_confidence_interval(passed_queries, total_queries)

        return PassKResult(
            k=k,
            pass_rate=pass_rate,
            trials_per_query=k,
            total_queries=total_queries,
            confidence_interval=ci,
        )

    def _wilson_confidence_interval(
        self,
        successes: int,
        trials: int,
        z: float = 1.96,  # 95% confidence
    ) -> Tuple[float, float]:
        """Compute Wilson confidence interval."""
        if trials == 0:
            return (0.0, 0.0)

        p = successes / trials
        denominator = 1 + z**2 / trials
        centre = p + z**2 / (2 * trials)
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials)

        lower = (centre - spread) / denominator
        upper = (centre + spread) / denominator

        return (max(0.0, lower), min(1.0, upper))

    async def compute_progress_rate(
        self,
        task_id: Optional[str] = None,
    ) -> float:
        """
        Compute progress rate for multi-step tasks.

        Progress rate = completed_steps / total_steps
        """
        # Filter trials for task
        relevant = self._trial_history
        if task_id:
            relevant = [t for t in relevant if t.metadata.get("task_id") == task_id]

        if not relevant:
            return 0.0

        # Count completed steps
        completed = sum(1 for t in relevant if t.passed)
        return completed / len(relevant)

    async def compute_hallucination_rate(
        self,
        time_window: Optional[timedelta] = None,
    ) -> float:
        """
        Compute hallucination rate.

        Args:
            time_window: Only consider trials in this window

        Returns:
            Hallucination rate (lower is better)
        """
        trials = self._trial_history

        if time_window:
            cutoff = datetime.utcnow() - time_window
            trials = [t for t in trials if t.timestamp >= cutoff]

        if not trials:
            return 0.0

        hallucinations = sum(1 for t in trials if t.hallucination_detected)
        return hallucinations / len(trials)

    async def compute_invocation_accuracy(self) -> float:
        """
        Compute retrieval/invocation accuracy.

        Accuracy = correct_retrievals / total_retrievals
        """
        total_retrieved = sum(t.retrieved_docs for t in self._trial_history)
        correct_retrieved = sum(t.correct_retrievals for t in self._trial_history)

        if total_retrieved == 0:
            return 0.0

        return correct_retrieved / total_retrieved

    async def get_full_evaluation(self) -> EvaluationResult:
        """Get comprehensive evaluation results."""
        pass_k_result = await self.compute_pass_k(k=3)
        progress_rate = await self.compute_progress_rate()
        hallucination_rate = await self.compute_hallucination_rate()
        invocation_accuracy = await self.compute_invocation_accuracy()

        # Compute average metrics
        avg_latency = (
            sum(t.latency_ms for t in self._trial_history) / len(self._trial_history)
            if self._trial_history else 0.0
        )
        avg_score = (
            sum(t.score for t in self._trial_history) / len(self._trial_history)
            if self._trial_history else 0.0
        )

        return EvaluationResult(
            agent_id=self.agent_id,
            evaluation_id=str(uuid.uuid4()),
            trials=self._trial_history[-100:],  # Last 100 trials
            metrics={
                "pass_k_3": pass_k_result.pass_rate,
                "progress_rate": progress_rate,
                "hallucination_rate": hallucination_rate,
                "invocation_accuracy": invocation_accuracy,
                "avg_latency_ms": avg_latency,
                "avg_score": avg_score,
                "total_trials": len(self._trial_history),
            },
        )


# =============================================================================
# Personalization System
# =============================================================================

class ExpertiseLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CommunicationStyle(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    CASUAL = "casual"


@dataclass
class UserPreferences:
    """Learned user preferences."""
    user_id: str
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    communication_style: CommunicationStyle = CommunicationStyle.DETAILED
    preferred_topics: List[str] = field(default_factory=list)
    disliked_patterns: List[str] = field(default_factory=list)
    response_length_preference: str = "medium"  # short, medium, long
    use_examples: bool = True
    use_analogies: bool = True
    language: str = "en"
    timezone: Optional[str] = None
    custom_preferences: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "expertise_level": self.expertise_level.value,
            "communication_style": self.communication_style.value,
            "preferred_topics": self.preferred_topics,
            "disliked_patterns": self.disliked_patterns,
            "response_length_preference": self.response_length_preference,
            "use_examples": self.use_examples,
            "use_analogies": self.use_analogies,
            "language": self.language,
            "timezone": self.timezone,
            "custom_preferences": self.custom_preferences,
            "confidence_scores": self.confidence_scores,
            "last_updated": self.last_updated.isoformat(),
        }

    def to_prompt_instructions(self) -> str:
        """Convert preferences to prompt instructions."""
        instructions = []

        # Expertise level
        if self.expertise_level == ExpertiseLevel.BEGINNER:
            instructions.append("Use simple language and explain technical terms.")
        elif self.expertise_level == ExpertiseLevel.EXPERT:
            instructions.append("You can use technical jargon freely. The user is an expert.")

        # Communication style
        if self.communication_style == CommunicationStyle.CONCISE:
            instructions.append("Keep responses brief and to the point.")
        elif self.communication_style == CommunicationStyle.DETAILED:
            instructions.append("Provide comprehensive explanations.")
        elif self.communication_style == CommunicationStyle.TECHNICAL:
            instructions.append("Use precise technical language and include details.")

        # Response length
        if self.response_length_preference == "short":
            instructions.append("Aim for responses under 100 words.")
        elif self.response_length_preference == "long":
            instructions.append("Feel free to provide thorough, detailed responses.")

        # Examples and analogies
        if self.use_examples:
            instructions.append("Include relevant examples when helpful.")
        if self.use_analogies:
            instructions.append("Use analogies to explain complex concepts.")

        return "\n".join(instructions)


@dataclass
class InteractionFeedback:
    """Feedback from a user interaction."""
    interaction_id: str
    query: str
    response: str
    rating: Optional[int] = None  # 1-5
    explicit_feedback: Optional[str] = None
    follow_up_questions: int = 0
    time_to_response: float = 0.0
    response_was_edited: bool = False
    signals: Dict[str, Any] = field(default_factory=dict)


class PersonalizationService:
    """
    Learns and applies user preferences for personalized responses.

    Features:
    - Preference learning from interactions
    - Expertise level detection
    - Communication style adaptation
    - Topic interest modeling
    - Continuous improvement via feedback
    """

    def __init__(
        self,
        user_id: str,
        agent_id: str,
        memory: Optional[AgentMemory] = None,
        llm_service=None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.memory = memory
        self.llm = llm_service
        self.preferences = UserPreferences(user_id=user_id)
        self._interaction_history: List[InteractionFeedback] = []
        self._redis = None
        self._initialized = False

    async def initialize(self):
        """Initialize and load preferences."""
        if self._initialized:
            return

        try:
            from backend.services.redis_client import get_redis_client
            self._redis = await get_redis_client()
            await self._load_preferences()
        except Exception as e:
            logger.warning(f"Could not initialize personalization storage: {e}")

        self._initialized = True

    async def _load_preferences(self):
        """Load preferences from storage."""
        if not self._redis:
            return

        try:
            key = f"user:preferences:{self.user_id}:{self.agent_id}"
            data = await self._redis.get(key)
            if data:
                parsed = json.loads(data)
                self.preferences = UserPreferences(
                    user_id=self.user_id,
                    expertise_level=ExpertiseLevel(parsed.get("expertise_level", "intermediate")),
                    communication_style=CommunicationStyle(parsed.get("communication_style", "detailed")),
                    preferred_topics=parsed.get("preferred_topics", []),
                    disliked_patterns=parsed.get("disliked_patterns", []),
                    response_length_preference=parsed.get("response_length_preference", "medium"),
                    use_examples=parsed.get("use_examples", True),
                    use_analogies=parsed.get("use_analogies", True),
                    language=parsed.get("language", "en"),
                    timezone=parsed.get("timezone"),
                    custom_preferences=parsed.get("custom_preferences", {}),
                    confidence_scores=parsed.get("confidence_scores", {}),
                    last_updated=datetime.fromisoformat(parsed.get("last_updated", datetime.utcnow().isoformat())),
                )
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")

    async def _save_preferences(self):
        """Save preferences to storage."""
        if not self._redis:
            return

        try:
            key = f"user:preferences:{self.user_id}:{self.agent_id}"
            await self._redis.set(key, json.dumps(self.preferences.to_dict()))
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

    async def record_interaction(
        self,
        query: str,
        response: str,
        rating: Optional[int] = None,
        explicit_feedback: Optional[str] = None,
        follow_up_questions: int = 0,
        response_was_edited: bool = False,
    ) -> InteractionFeedback:
        """
        Record an interaction for preference learning.

        Args:
            query: User query
            response: Agent response
            rating: Explicit rating (1-5)
            explicit_feedback: Text feedback
            follow_up_questions: Number of follow-ups needed
            response_was_edited: Whether user edited the response

        Returns:
            InteractionFeedback record
        """
        feedback = InteractionFeedback(
            interaction_id=str(uuid.uuid4()),
            query=query,
            response=response,
            rating=rating,
            explicit_feedback=explicit_feedback,
            follow_up_questions=follow_up_questions,
            response_was_edited=response_was_edited,
        )

        self._interaction_history.append(feedback)

        # Learn from this interaction
        await self._learn_from_interaction(feedback)

        return feedback

    async def _learn_from_interaction(self, feedback: InteractionFeedback):
        """Learn preferences from an interaction."""
        # Detect expertise level from query complexity
        expertise = await self._detect_expertise(feedback.query)
        if expertise:
            self._update_preference_with_confidence(
                "expertise_level",
                expertise,
                confidence=0.3,
            )

        # Learn from explicit feedback
        if feedback.explicit_feedback:
            await self._process_explicit_feedback(feedback.explicit_feedback)

        # Learn from implicit signals
        if feedback.follow_up_questions > 2:
            # User needed many follow-ups - response might be too brief or unclear
            self._update_preference_with_confidence(
                "response_length_preference",
                "long",
                confidence=0.2,
            )

        if feedback.response_was_edited:
            # User edited response - might prefer different style
            self._update_preference_with_confidence(
                "communication_style",
                CommunicationStyle.CONCISE,
                confidence=0.1,
            )

        # Extract topics from query
        topics = self._extract_topics(feedback.query)
        for topic in topics:
            if topic not in self.preferences.preferred_topics:
                self.preferences.preferred_topics.append(topic)
                # Keep only recent topics
                self.preferences.preferred_topics = self.preferences.preferred_topics[-20:]

        self.preferences.last_updated = datetime.utcnow()
        await self._save_preferences()

    async def _detect_expertise(self, query: str) -> Optional[ExpertiseLevel]:
        """Detect user expertise from query complexity."""
        # Simple heuristics
        technical_terms = [
            "api", "algorithm", "implementation", "architecture",
            "database", "optimization", "framework", "integration",
        ]

        query_lower = query.lower()
        technical_count = sum(1 for term in technical_terms if term in query_lower)

        if technical_count >= 3:
            return ExpertiseLevel.EXPERT
        elif technical_count >= 1:
            return ExpertiseLevel.ADVANCED
        elif len(query.split()) < 5 and "how" in query_lower:
            return ExpertiseLevel.BEGINNER

        return None

    async def _process_explicit_feedback(self, feedback: str):
        """Process explicit user feedback."""
        feedback_lower = feedback.lower()

        # Detect preferences from feedback
        if any(word in feedback_lower for word in ["shorter", "brief", "concise"]):
            self._update_preference_with_confidence(
                "response_length_preference",
                "short",
                confidence=0.5,
            )
            self.preferences.communication_style = CommunicationStyle.CONCISE

        if any(word in feedback_lower for word in ["more detail", "elaborate", "explain more"]):
            self._update_preference_with_confidence(
                "response_length_preference",
                "long",
                confidence=0.5,
            )
            self.preferences.communication_style = CommunicationStyle.DETAILED

        if "example" in feedback_lower:
            self.preferences.use_examples = "no example" not in feedback_lower

        if "technical" in feedback_lower:
            self.preferences.communication_style = CommunicationStyle.TECHNICAL

    def _update_preference_with_confidence(
        self,
        preference_name: str,
        value: Any,
        confidence: float,
    ):
        """Update a preference with weighted confidence."""
        current_confidence = self.preferences.confidence_scores.get(preference_name, 0.0)
        new_confidence = min(1.0, current_confidence + confidence)

        # Only update if new confidence is higher
        if new_confidence > current_confidence:
            setattr(self.preferences, preference_name, value)
            self.preferences.confidence_scores[preference_name] = new_confidence

    def _extract_topics(self, query: str) -> List[str]:
        """Extract topic keywords from query."""
        # Simple keyword extraction
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "how",
            "why", "when", "where", "who", "which", "can", "could", "would",
            "should", "do", "does", "did", "have", "has", "had", "be", "been",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
        }

        words = query.lower().split()
        topics = [
            w for w in words
            if len(w) > 3 and w not in stop_words
        ]

        return topics[:3]

    def get_personalized_prompt_prefix(self) -> str:
        """Get personalized system prompt prefix."""
        return f"""User Preferences:
{self.preferences.to_prompt_instructions()}

Remember these preferences when responding to this user."""

    async def get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations based on learned preferences."""
        return {
            "suggested_topics": self.preferences.preferred_topics[:5],
            "expertise_level": self.preferences.expertise_level.value,
            "communication_style": self.preferences.communication_style.value,
            "use_examples": self.preferences.use_examples,
            "confidence": sum(self.preferences.confidence_scores.values()) / max(len(self.preferences.confidence_scores), 1),
        }


# =============================================================================
# Factory Functions
# =============================================================================

async def get_agent_evaluator(
    agent_id: str,
    llm_service=None,
) -> AgentEvaluator:
    """Get or create agent evaluator."""
    evaluator = AgentEvaluator(
        agent_id=agent_id,
        llm_service=llm_service,
    )
    await evaluator.initialize()
    return evaluator


async def get_personalization_service(
    user_id: str,
    agent_id: str,
    memory: Optional[AgentMemory] = None,
    llm_service=None,
) -> PersonalizationService:
    """Get or create personalization service."""
    service = PersonalizationService(
        user_id=user_id,
        agent_id=agent_id,
        memory=memory,
        llm_service=llm_service,
    )
    await service.initialize()
    return service
