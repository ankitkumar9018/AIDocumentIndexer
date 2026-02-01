"""
AIDocumentIndexer - RAG Sufficiency Detection
=============================================

Phase 31: Implements context sufficiency detection based on ICLR 2025 research.

This module detects when the LLM has enough context to provide an accurate answer,
helping reduce hallucinations by:
1. Adding a sufficiency check before generation
2. Retrieving more context or re-ranking when insufficient
3. Tuning abstention threshold with confidence signals

References:
- "Deeper Insights into RAG: The Role of Sufficient Context" (ICLR 2025)
- https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/
"""

import os
import structlog
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger(__name__)


class SufficiencyLevel(Enum):
    """Context sufficiency levels."""
    INSUFFICIENT = "insufficient"      # Need more context
    PARTIAL = "partial"                # Some info available, may be incomplete
    SUFFICIENT = "sufficient"          # Enough context for accurate answer
    HIGHLY_CONFIDENT = "highly_confident"  # Very high confidence in context


@dataclass
class SufficiencyResult:
    """Result of context sufficiency check."""
    level: SufficiencyLevel
    confidence: float  # 0.0 to 1.0
    reasoning: str
    should_abstain: bool
    suggested_action: str  # "proceed", "retrieve_more", "rerank", "abstain"
    coverage_score: float  # How much of the query is covered by context
    relevance_score: float  # How relevant the retrieved context is
    missing_information: list = None  # List of information that would improve context

    def __post_init__(self):
        if self.missing_information is None:
            self.missing_information = []


class SufficiencyChecker:
    """
    RAG Context Sufficiency Checker.

    Determines whether retrieved context is sufficient for the LLM to
    generate an accurate answer, reducing hallucinations.

    Key features:
    - Query coverage analysis
    - Context relevance scoring
    - Confidence calibration
    - Abstention recommendations
    """

    def __init__(
        self,
        abstention_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        min_relevant_chunks: int = 1,
        use_llm_verification: bool = True,
    ):
        """
        Initialize sufficiency checker.

        Args:
            abstention_threshold: Below this confidence, recommend abstention
            confidence_threshold: Above this, context is considered sufficient
            min_relevant_chunks: Minimum chunks needed for sufficiency
            use_llm_verification: Use LLM for semantic sufficiency check
        """
        self.abstention_threshold = abstention_threshold
        self.confidence_threshold = confidence_threshold
        self.min_relevant_chunks = min_relevant_chunks
        self.use_llm_verification = use_llm_verification

        logger.info(
            "Initialized SufficiencyChecker",
            abstention_threshold=abstention_threshold,
            confidence_threshold=confidence_threshold,
            use_llm_verification=use_llm_verification,
        )

    async def check_sufficiency(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
    ) -> SufficiencyResult:
        """
        Check if retrieved context is sufficient for answering the query.

        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks with scores
            query_embedding: Optional pre-computed query embedding

        Returns:
            SufficiencyResult with assessment and recommendations
        """
        # Quick checks first
        if not retrieved_chunks:
            return SufficiencyResult(
                level=SufficiencyLevel.INSUFFICIENT,
                confidence=0.0,
                reasoning="No context chunks retrieved",
                should_abstain=True,
                suggested_action="retrieve_more",
                coverage_score=0.0,
                relevance_score=0.0,
            )

        # Calculate relevance score from chunk scores
        relevance_score = self._calculate_relevance_score(retrieved_chunks)

        # Calculate query coverage
        coverage_score = self._calculate_coverage_score(query, retrieved_chunks)

        # Combine scores for overall confidence
        confidence = self._calculate_confidence(
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            num_chunks=len(retrieved_chunks),
        )

        # Determine sufficiency level
        level = self._determine_level(confidence)

        # Decide on action
        should_abstain, suggested_action = self._decide_action(
            level=level,
            confidence=confidence,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            level=level,
            confidence=confidence,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            num_chunks=len(retrieved_chunks),
        )

        # Optional: LLM-based semantic verification
        if self.use_llm_verification and level in (
            SufficiencyLevel.PARTIAL,
            SufficiencyLevel.INSUFFICIENT,
        ):
            llm_result = await self._llm_verify_sufficiency(query, retrieved_chunks)
            if llm_result:
                # Adjust confidence based on LLM assessment
                confidence = (confidence + llm_result["confidence"]) / 2
                level = self._determine_level(confidence)
                reasoning += f" LLM assessment: {llm_result['reasoning']}"

        result = SufficiencyResult(
            level=level,
            confidence=confidence,
            reasoning=reasoning,
            should_abstain=should_abstain,
            suggested_action=suggested_action,
            coverage_score=coverage_score,
            relevance_score=relevance_score,
        )

        logger.debug(
            "Sufficiency check complete",
            level=level.value,
            confidence=round(confidence, 3),
            should_abstain=should_abstain,
            action=suggested_action,
        )

        return result

    def _calculate_relevance_score(
        self,
        chunks: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate overall relevance from chunk scores.

        Uses weighted average favoring top chunks.
        """
        if not chunks:
            return 0.0

        # Extract scores
        scores = []
        for chunk in chunks:
            score = chunk.get("score", chunk.get("similarity", 0.5))
            scores.append(float(score))

        if not scores:
            return 0.5  # Default if no scores

        # Weight top chunks more heavily
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        weight_total = sum(weights)

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _calculate_coverage_score(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate how much of the query is covered by context.

        Simple heuristic based on keyword overlap.
        """
        # Extract query terms
        query_terms = set(query.lower().split())
        query_terms = {t for t in query_terms if len(t) > 2}  # Filter short words

        if not query_terms:
            return 1.0  # Empty query is "covered"

        # Combine all chunk content
        all_content = " ".join(
            chunk.get("content", chunk.get("text", "")).lower()
            for chunk in chunks
        )
        content_terms = set(all_content.split())

        # Calculate overlap
        overlap = query_terms.intersection(content_terms)
        coverage = len(overlap) / len(query_terms) if query_terms else 0.0

        return coverage

    def _calculate_confidence(
        self,
        relevance_score: float,
        coverage_score: float,
        num_chunks: int,
    ) -> float:
        """
        Calculate overall confidence score.

        Combines relevance, coverage, and chunk count.
        """
        # Weight factors
        relevance_weight = 0.5
        coverage_weight = 0.3
        chunk_weight = 0.2

        # Chunk factor (diminishing returns after min_relevant_chunks)
        chunk_factor = min(1.0, num_chunks / max(self.min_relevant_chunks, 1))

        confidence = (
            relevance_weight * relevance_score +
            coverage_weight * coverage_score +
            chunk_weight * chunk_factor
        )

        return min(1.0, max(0.0, confidence))

    def _determine_level(self, confidence: float) -> SufficiencyLevel:
        """Determine sufficiency level from confidence score."""
        if confidence >= 0.85:
            return SufficiencyLevel.HIGHLY_CONFIDENT
        elif confidence >= self.confidence_threshold:
            return SufficiencyLevel.SUFFICIENT
        elif confidence >= self.abstention_threshold:
            return SufficiencyLevel.PARTIAL
        else:
            return SufficiencyLevel.INSUFFICIENT

    def _decide_action(
        self,
        level: SufficiencyLevel,
        confidence: float,
        relevance_score: float,
        coverage_score: float,
    ) -> Tuple[bool, str]:
        """
        Decide whether to abstain and suggest action.

        Returns:
            Tuple of (should_abstain, suggested_action)
        """
        if level == SufficiencyLevel.HIGHLY_CONFIDENT:
            return False, "proceed"

        if level == SufficiencyLevel.SUFFICIENT:
            return False, "proceed"

        if level == SufficiencyLevel.PARTIAL:
            # Partial context - decide based on which score is low
            if relevance_score < 0.5:
                return False, "rerank"  # Try reranking for better relevance
            elif coverage_score < 0.5:
                return False, "retrieve_more"  # Need more coverage
            else:
                return False, "proceed"  # Cautious proceed

        # Insufficient
        if confidence < self.abstention_threshold:
            return True, "abstain"
        else:
            return False, "retrieve_more"

    def _generate_reasoning(
        self,
        level: SufficiencyLevel,
        confidence: float,
        relevance_score: float,
        coverage_score: float,
        num_chunks: int,
    ) -> str:
        """Generate human-readable reasoning for the assessment."""
        parts = [f"Confidence: {confidence:.1%}"]

        if relevance_score < 0.5:
            parts.append("Low relevance in retrieved context")
        elif relevance_score > 0.8:
            parts.append("Highly relevant context found")

        if coverage_score < 0.5:
            parts.append("Query terms not well covered")
        elif coverage_score > 0.8:
            parts.append("Good query coverage")

        if num_chunks < self.min_relevant_chunks:
            parts.append(f"Only {num_chunks} chunks (need {self.min_relevant_chunks}+)")

        return ". ".join(parts) + "."

    async def _llm_verify_sufficiency(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to semantically verify context sufficiency.

        This is an optional deeper check using the LLM itself.
        """
        try:
            from backend.services.llm import get_llm_service

            llm = get_llm_service()

            # Prepare context
            context = "\n\n".join(
                chunk.get("content", chunk.get("text", ""))[:500]
                for chunk in chunks[:5]  # Limit to top 5 chunks
            )

            prompt = f"""Analyze if this context is sufficient to answer the question.

Question: {query}

Context:
{context}

Respond with ONLY a JSON object:
{{"sufficient": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            response = await llm.generate(prompt, max_tokens=150)

            # Parse response
            import json
            # Find JSON in response
            response_text = response.get("text", response) if isinstance(response, dict) else str(response)
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])
                return {
                    "sufficient": result.get("sufficient", False),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                }

        except Exception as e:
            logger.debug(f"LLM sufficiency verification failed: {e}")

        return None


class AdaptiveSufficiencyChecker(SufficiencyChecker):
    """
    Adaptive sufficiency checker that learns from feedback.

    Adjusts thresholds based on:
    - User feedback (thumbs up/down)
    - Query success rates
    - Domain-specific patterns
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feedback_history: List[Dict] = []
        self._domain_thresholds: Dict[str, float] = {}

    def record_feedback(
        self,
        query: str,
        result: SufficiencyResult,
        was_accurate: bool,
        domain: Optional[str] = None,
    ):
        """
        Record feedback to improve future predictions.

        Args:
            query: Original query
            result: Sufficiency result that was used
            was_accurate: Whether the final answer was accurate
            domain: Optional domain classification
        """
        self._feedback_history.append({
            "confidence": result.confidence,
            "level": result.level.value,
            "should_abstain": result.should_abstain,
            "was_accurate": was_accurate,
            "domain": domain,
        })

        # Adjust thresholds based on feedback
        self._adjust_thresholds()

    def _adjust_thresholds(self):
        """Adjust thresholds based on accumulated feedback."""
        if len(self._feedback_history) < 10:
            return  # Need minimum feedback

        # Calculate error rate at current threshold
        recent = self._feedback_history[-100:]  # Last 100 feedbacks

        # False positives: Proceeded when should have abstained
        false_positives = [
            f for f in recent
            if not f["should_abstain"] and not f["was_accurate"]
        ]

        # False negatives: Abstained when could have answered
        false_negatives = [
            f for f in recent
            if f["should_abstain"] and f["was_accurate"]
        ]

        fp_rate = len(false_positives) / len(recent) if recent else 0
        fn_rate = len(false_negatives) / len(recent) if recent else 0

        # Adjust abstention threshold
        if fp_rate > 0.1:  # Too many false positives
            self.abstention_threshold = min(0.5, self.abstention_threshold + 0.02)
            logger.info(f"Raised abstention threshold to {self.abstention_threshold}")
        elif fn_rate > 0.1:  # Too many false negatives
            self.abstention_threshold = max(0.1, self.abstention_threshold - 0.02)
            logger.info(f"Lowered abstention threshold to {self.abstention_threshold}")


# Singleton instance
_sufficiency_checker: Optional[SufficiencyChecker] = None


def get_sufficiency_checker(adaptive: bool = False) -> SufficiencyChecker:
    """Get or create sufficiency checker singleton."""
    global _sufficiency_checker

    if _sufficiency_checker is None:
        if adaptive:
            _sufficiency_checker = AdaptiveSufficiencyChecker(
                abstention_threshold=float(os.getenv("SUFFICIENCY_ABSTENTION_THRESHOLD", "0.3")),
                confidence_threshold=float(os.getenv("SUFFICIENCY_CONFIDENCE_THRESHOLD", "0.6")),
                use_llm_verification=os.getenv("SUFFICIENCY_LLM_VERIFY", "true").lower() == "true",
            )
        else:
            _sufficiency_checker = SufficiencyChecker(
                abstention_threshold=float(os.getenv("SUFFICIENCY_ABSTENTION_THRESHOLD", "0.3")),
                confidence_threshold=float(os.getenv("SUFFICIENCY_CONFIDENCE_THRESHOLD", "0.6")),
                use_llm_verification=os.getenv("SUFFICIENCY_LLM_VERIFY", "true").lower() == "true",
            )

    return _sufficiency_checker


async def check_context_sufficiency(
    query: str,
    chunks: List[Dict[str, Any]],
) -> SufficiencyResult:
    """
    Convenience function to check context sufficiency.

    Args:
        query: User's question
        chunks: Retrieved context chunks

    Returns:
        SufficiencyResult with assessment
    """
    checker = get_sufficiency_checker()
    return await checker.check_sufficiency(query, chunks)
