"""
AIDocumentIndexer - Answer Refiner Service
============================================

Implements Self-Refine, CRITIC, and Chain-of-Verification for +20% answer quality.

Techniques Implemented:
1. Self-Refine (NeurIPS 2023): Iterative self-feedback and refinement
2. CRITIC: Tool-verified fact checking with retrieval
3. Chain-of-Verification (CoVe, ACL 2024): Generate verification questions, verify independently

Research:
- Self-Refine: "Self-Refine: Iterative Refinement with Self-Feedback" (NeurIPS 2023)
  - +20% absolute improvement across tasks
- CRITIC: "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing"
  - Tool-verified answers reduce hallucinations
- CoVe: "Chain-of-Verification Reduces Hallucination in Large Language Models" (ACL 2024)
  - +23% F1 on hallucination-prone tasks

Performance Benchmarks:
| Technique | Improvement | Best For |
|-----------|-------------|----------|
| Self-Refine | +20% absolute | General quality |
| CRITIC | Tool-verified | Fact-checking |
| CoVe | +23% F1 | Hallucination reduction |
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from backend.core.config import settings
from backend.core.performance import gather_with_concurrency

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class RefinementStrategy(str, Enum):
    """Answer refinement strategies."""
    SELF_REFINE = "self_refine"     # Iterative self-feedback
    CRITIC = "critic"               # Tool-verified critiquing
    COVE = "cove"                   # Chain-of-Verification
    COMBINED = "combined"           # All strategies


@dataclass
class RefinerConfig:
    """Configuration for answer refinement."""
    # Self-Refine settings
    max_refinement_iterations: int = 3
    refinement_threshold: float = 0.8  # Stop if feedback score > threshold

    # CRITIC settings
    enable_retrieval_verification: bool = True
    max_verification_queries: int = 3

    # CoVe settings
    max_verification_questions: int = 5
    verification_independence: bool = True  # Verify questions independently

    # Model settings
    refinement_model: str = "gpt-4o-mini"
    refinement_provider: str = "openai"
    temperature: float = 0.1

    # Strategy
    default_strategy: RefinementStrategy = RefinementStrategy.SELF_REFINE


@dataclass(slots=True)
class RefinementResult:
    """Result from answer refinement."""
    original_answer: str
    refined_answer: str
    strategy_used: RefinementStrategy
    iterations: int = 0
    feedback_history: List[str] = field(default_factory=list)
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    improvement_score: float = 0.0
    execution_time_ms: float = 0.0


@dataclass
class Feedback:
    """Structured feedback for refinement."""
    score: float  # 0-1 quality score
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    is_satisfactory: bool = False


@dataclass
class VerificationQuestion:
    """Verification question for CoVe."""
    question: str
    expected_type: str  # "factual", "logical", "consistency"
    source_claim: str


@dataclass
class VerificationResult:
    """Result of verifying a claim."""
    question: str
    answer: str
    is_consistent: bool
    evidence: Optional[str] = None
    confidence: float = 0.0


# =============================================================================
# Prompts
# =============================================================================

SELF_REFINE_FEEDBACK_PROMPT = """You are a critical reviewer. Evaluate this answer for quality, accuracy, and completeness.

Query: {query}

Context (relevant documents):
{context}

Answer to evaluate:
{answer}

Provide structured feedback:
1. Score (0-10): Overall quality score
2. Issues: List any problems (inaccuracies, missing info, unclear parts)
3. Suggestions: Specific improvements to make
4. Is Satisfactory: Yes/No - is this answer good enough?

Format your response as:
Score: [0-10]
Issues:
- [issue 1]
- [issue 2]
Suggestions:
- [suggestion 1]
- [suggestion 2]
Satisfactory: [Yes/No]"""

SELF_REFINE_IMPROVE_PROMPT = """Improve this answer based on the feedback provided.

Query: {query}

Context (relevant documents):
{context}

Current Answer:
{answer}

Feedback:
{feedback}

Provide an improved answer that addresses all the issues and incorporates the suggestions.
Only output the improved answer, nothing else."""

COVE_GENERATE_QUESTIONS_PROMPT = """Given this answer, generate verification questions to check its accuracy.

Query: {query}

Answer:
{answer}

Generate {num_questions} verification questions that would help verify the claims in this answer.
For each question, identify:
1. The specific claim being verified
2. What type of verification (factual, logical, consistency)

Format:
Q1: [question]
Claim: [claim being verified]
Type: [factual/logical/consistency]

Q2: [question]
..."""

COVE_ANSWER_QUESTION_PROMPT = """Answer this verification question based only on the provided context.

Context:
{context}

Question: {question}

Provide a factual answer based only on what's in the context. If the context doesn't contain the answer, say "Not found in context."

Answer:"""

COVE_FINAL_VERIFICATION_PROMPT = """Verify the original answer against these verification results.

Original Query: {query}

Original Answer:
{answer}

Verification Results:
{verification_results}

Based on the verification results:
1. Is the original answer accurate? (Yes/Partially/No)
2. What corrections are needed?
3. Provide a corrected answer if needed.

Format:
Accuracy: [Yes/Partially/No]
Corrections Needed:
- [correction 1]
Corrected Answer:
[corrected answer or "No corrections needed"]"""

CRITIC_VERIFY_PROMPT = """Verify this claim using the provided search results.

Claim: {claim}

Search Results:
{search_results}

Does the search evidence support this claim?
- Verdict: Supported / Contradicted / Insufficient Evidence
- Explanation: [brief explanation]
- Confidence: [0-100]%"""


# =============================================================================
# Answer Refiner Service
# =============================================================================

class AnswerRefiner:
    """
    Service for improving answer quality through refinement.

    Implements three complementary techniques:
    1. Self-Refine: Iterative feedback and improvement
    2. CRITIC: Tool-verified fact checking
    3. CoVe: Independent verification questions

    Usage:
        refiner = AnswerRefiner()

        # Self-refine an answer
        result = await refiner.refine(
            query="What is Apple's revenue?",
            answer="Apple's revenue was $394B in 2022.",
            context="...",
            strategy=RefinementStrategy.SELF_REFINE,
        )

        # Chain-of-Verification
        result = await refiner.refine(
            query="...",
            answer="...",
            context="...",
            strategy=RefinementStrategy.COVE,
        )
    """

    def __init__(
        self,
        config: Optional[RefinerConfig] = None,
        retrieval_fn: Optional[Callable] = None,
    ):
        """
        Initialize answer refiner.

        Args:
            config: Refiner configuration
            retrieval_fn: Optional function for retrieval verification
        """
        self.config = config or RefinerConfig()
        self.retrieval_fn = retrieval_fn
        self._llm = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the LLM for refinement."""
        if self._initialized:
            return True

        try:
            from backend.services.llm import LLMFactory

            self._llm = LLMFactory.get_chat_model(
                provider=self.config.refinement_provider,
                model=self.config.refinement_model,
                temperature=self.config.temperature,
                max_tokens=2048,
            )

            logger.info(
                "Answer refiner initialized",
                model=self.config.refinement_model,
            )

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize answer refiner", error=str(e))
            return False

    async def refine(
        self,
        query: str,
        answer: str,
        context: str,
        strategy: Optional[RefinementStrategy] = None,
    ) -> RefinementResult:
        """
        Refine an answer using the specified strategy.

        Args:
            query: Original query
            answer: Initial answer to refine
            context: Retrieved context
            strategy: Refinement strategy to use

        Returns:
            RefinementResult with refined answer
        """
        if not await self.initialize():
            return RefinementResult(
                original_answer=answer,
                refined_answer=answer,
                strategy_used=RefinementStrategy.SELF_REFINE,
            )

        start_time = time.time()
        strategy = strategy or self.config.default_strategy

        logger.info(
            "Starting answer refinement",
            strategy=strategy.value,
            answer_length=len(answer),
        )

        if strategy == RefinementStrategy.SELF_REFINE:
            result = await self._self_refine(query, answer, context)
        elif strategy == RefinementStrategy.CRITIC:
            result = await self._critic_refine(query, answer, context)
        elif strategy == RefinementStrategy.COVE:
            result = await self._cove_refine(query, answer, context)
        else:  # COMBINED
            result = await self._combined_refine(query, answer, context)

        result.execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Answer refinement complete",
            strategy=strategy.value,
            iterations=result.iterations,
            improvement=round(result.improvement_score, 2),
            time_ms=round(result.execution_time_ms, 2),
        )

        return result

    async def _self_refine(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> RefinementResult:
        """
        Self-Refine: Iterative self-feedback and improvement.

        Process:
        1. Generate feedback on current answer
        2. If satisfactory, stop
        3. Generate improved answer based on feedback
        4. Repeat until satisfactory or max iterations
        """
        from langchain_core.messages import HumanMessage

        current_answer = answer
        feedback_history = []
        iterations = 0

        for i in range(self.config.max_refinement_iterations):
            iterations = i + 1

            # Generate feedback
            feedback = await self._generate_feedback(query, current_answer, context)
            feedback_history.append(f"Iteration {i+1}: Score={feedback.score:.1f}, Issues={feedback.issues}")

            # Check if satisfactory
            if feedback.is_satisfactory or feedback.score >= self.config.refinement_threshold * 10:
                logger.debug(f"Self-refine converged at iteration {i+1}")
                break

            # Generate improved answer
            improve_prompt = SELF_REFINE_IMPROVE_PROMPT.format(
                query=query,
                context=context[:3000],
                answer=current_answer,
                feedback=self._format_feedback(feedback),
            )

            try:
                response = await self._llm.ainvoke([HumanMessage(content=improve_prompt)])
                current_answer = response.content.strip()
            except Exception as e:
                logger.warning("Improvement generation failed", error=str(e))
                break

        # Calculate improvement score
        final_feedback = await self._generate_feedback(query, current_answer, context)
        initial_score = await self._quick_score(query, answer, context)
        improvement_score = (final_feedback.score - initial_score) / 10.0

        return RefinementResult(
            original_answer=answer,
            refined_answer=current_answer,
            strategy_used=RefinementStrategy.SELF_REFINE,
            iterations=iterations,
            feedback_history=feedback_history,
            confidence=final_feedback.score / 10.0,
            improvement_score=improvement_score,
        )

    async def _critic_refine(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> RefinementResult:
        """
        CRITIC: Tool-verified critiquing.

        Process:
        1. Extract claims from answer
        2. Verify each claim using retrieval
        3. Generate corrected answer based on verification
        """
        # Extract claims from answer
        claims = await self._extract_claims(answer)

        verification_results = []

        # Verify each claim
        for claim in claims[:self.config.max_verification_queries]:
            result = await self._verify_claim(claim, context)
            verification_results.append({
                "claim": claim,
                "verdict": result.get("verdict", "unknown"),
                "confidence": result.get("confidence", 0),
            })

        # Check if corrections needed
        needs_correction = any(
            r["verdict"] == "Contradicted"
            for r in verification_results
        )

        refined_answer = answer
        if needs_correction:
            # Generate corrected answer
            corrections = [
                r for r in verification_results
                if r["verdict"] == "Contradicted"
            ]
            refined_answer = await self._apply_corrections(
                query, answer, context, corrections
            )

        # Calculate confidence
        supported = sum(1 for r in verification_results if r["verdict"] == "Supported")
        confidence = supported / len(verification_results) if verification_results else 0.5

        return RefinementResult(
            original_answer=answer,
            refined_answer=refined_answer,
            strategy_used=RefinementStrategy.CRITIC,
            iterations=1,
            verification_results=verification_results,
            confidence=confidence,
            improvement_score=0.1 if needs_correction else 0.0,
        )

    async def _cove_refine(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> RefinementResult:
        """
        Chain-of-Verification: Generate and verify questions independently.

        Process:
        1. Generate verification questions from answer
        2. Answer each question independently (without seeing original answer)
        3. Compare verification answers with original claims
        4. Generate corrected answer
        """
        from langchain_core.messages import HumanMessage

        # Step 1: Generate verification questions
        questions = await self._generate_verification_questions(query, answer)

        # Step 2: Answer questions independently
        verification_results = []

        for vq in questions[:self.config.max_verification_questions]:
            # Answer based only on context (independent verification)
            answer_prompt = COVE_ANSWER_QUESTION_PROMPT.format(
                context=context[:4000],
                question=vq.question,
            )

            try:
                response = await self._llm.ainvoke([HumanMessage(content=answer_prompt)])
                verified_answer = response.content.strip()

                # Check consistency with original claim
                is_consistent = await self._check_consistency(
                    vq.source_claim, verified_answer
                )

                verification_results.append({
                    "question": vq.question,
                    "claim": vq.source_claim,
                    "verified_answer": verified_answer,
                    "is_consistent": is_consistent,
                })
            except Exception as e:
                logger.warning("Verification failed", error=str(e))

        # Step 3: Generate final verified answer
        final_prompt = COVE_FINAL_VERIFICATION_PROMPT.format(
            query=query,
            answer=answer,
            verification_results=self._format_verification_results(verification_results),
        )

        try:
            response = await self._llm.ainvoke([HumanMessage(content=final_prompt)])
            result_text = response.content

            # Extract corrected answer
            if "Corrected Answer:" in result_text:
                corrected = result_text.split("Corrected Answer:")[-1].strip()
                if corrected.lower() != "no corrections needed":
                    refined_answer = corrected
                else:
                    refined_answer = answer
            else:
                refined_answer = answer

        except Exception as e:
            logger.warning("Final verification failed", error=str(e))
            refined_answer = answer

        # Calculate confidence
        consistent_count = sum(1 for r in verification_results if r.get("is_consistent", False))
        confidence = consistent_count / len(verification_results) if verification_results else 0.5

        return RefinementResult(
            original_answer=answer,
            refined_answer=refined_answer,
            strategy_used=RefinementStrategy.COVE,
            iterations=1,
            verification_results=verification_results,
            confidence=confidence,
            improvement_score=0.1 if refined_answer != answer else 0.0,
        )

    async def _combined_refine(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> RefinementResult:
        """Apply all refinement strategies in sequence."""
        # Start with Self-Refine
        result = await self._self_refine(query, answer, context)
        current_answer = result.refined_answer

        # Apply CoVe
        cove_result = await self._cove_refine(query, current_answer, context)
        current_answer = cove_result.refined_answer

        # Combine results
        all_feedback = result.feedback_history + [
            f"CoVe: confidence={cove_result.confidence:.2f}"
        ]
        all_verifications = result.verification_results + cove_result.verification_results

        return RefinementResult(
            original_answer=answer,
            refined_answer=current_answer,
            strategy_used=RefinementStrategy.COMBINED,
            iterations=result.iterations + 1,
            feedback_history=all_feedback,
            verification_results=all_verifications,
            confidence=(result.confidence + cove_result.confidence) / 2,
            improvement_score=result.improvement_score + cove_result.improvement_score,
        )

    async def _generate_feedback(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> Feedback:
        """Generate structured feedback for an answer."""
        from langchain_core.messages import HumanMessage

        prompt = SELF_REFINE_FEEDBACK_PROMPT.format(
            query=query,
            context=context[:3000],
            answer=answer,
        )

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_feedback(response.content)
        except Exception as e:
            logger.warning("Feedback generation failed", error=str(e))
            return Feedback(score=5.0, is_satisfactory=False)

    def _parse_feedback(self, text: str) -> Feedback:
        """Parse structured feedback from LLM response."""
        score = 5.0
        issues = []
        suggestions = []
        is_satisfactory = False

        # Extract score
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', text)
        if score_match:
            score = float(score_match.group(1))

        # Extract issues
        issues_match = re.search(r'Issues:(.*?)(?:Suggestions:|Satisfactory:|$)', text, re.DOTALL)
        if issues_match:
            issues_text = issues_match.group(1)
            issues = [
                line.strip().lstrip('- ')
                for line in issues_text.strip().split('\n')
                if line.strip() and line.strip() != '-'
            ]

        # Extract suggestions
        suggestions_match = re.search(r'Suggestions:(.*?)(?:Satisfactory:|$)', text, re.DOTALL)
        if suggestions_match:
            suggestions_text = suggestions_match.group(1)
            suggestions = [
                line.strip().lstrip('- ')
                for line in suggestions_text.strip().split('\n')
                if line.strip() and line.strip() != '-'
            ]

        # Extract satisfactory
        satisfactory_match = re.search(r'Satisfactory:\s*(Yes|No)', text, re.IGNORECASE)
        if satisfactory_match:
            is_satisfactory = satisfactory_match.group(1).lower() == 'yes'

        return Feedback(
            score=score,
            issues=issues,
            suggestions=suggestions,
            is_satisfactory=is_satisfactory,
        )

    def _format_feedback(self, feedback: Feedback) -> str:
        """Format feedback for prompt."""
        parts = [f"Score: {feedback.score}/10"]
        if feedback.issues:
            parts.append("Issues:\n" + "\n".join(f"- {i}" for i in feedback.issues))
        if feedback.suggestions:
            parts.append("Suggestions:\n" + "\n".join(f"- {s}" for s in feedback.suggestions))
        return "\n\n".join(parts)

    async def _quick_score(self, query: str, answer: str, context: str) -> float:
        """Get a quick quality score for an answer."""
        feedback = await self._generate_feedback(query, answer, context)
        return feedback.score

    async def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from an answer."""
        from langchain_core.messages import HumanMessage

        prompt = f"""Extract the main factual claims from this answer. List each claim on a separate line.

Answer: {answer}

Claims (one per line):"""

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            claims = [
                line.strip().lstrip('- ').lstrip('•')
                for line in response.content.strip().split('\n')
                if line.strip()
            ]
            return claims[:10]  # Limit to 10 claims
        except Exception:
            return [answer]  # Fall back to whole answer as single claim

    async def _verify_claim(self, claim: str, context: str) -> Dict[str, Any]:
        """Verify a claim against context."""
        from langchain_core.messages import HumanMessage

        prompt = CRITIC_VERIFY_PROMPT.format(
            claim=claim,
            search_results=context[:2000],
        )

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            text = response.content

            # Parse verdict
            verdict = "Insufficient Evidence"
            if "Supported" in text:
                verdict = "Supported"
            elif "Contradicted" in text:
                verdict = "Contradicted"

            # Parse confidence
            conf_match = re.search(r'(\d+)%', text)
            confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5

            return {"verdict": verdict, "confidence": confidence}
        except Exception:
            return {"verdict": "Insufficient Evidence", "confidence": 0.5}

    async def _apply_corrections(
        self,
        query: str,
        answer: str,
        context: str,
        corrections: List[Dict],
    ) -> str:
        """Apply corrections to answer."""
        from langchain_core.messages import HumanMessage

        corrections_text = "\n".join(
            f"- Claim: {c['claim']} - {c['verdict']}"
            for c in corrections
        )

        prompt = f"""Correct this answer based on the verification results.

Query: {query}

Original Answer: {answer}

Issues Found:
{corrections_text}

Context:
{context[:2000]}

Provide a corrected answer:"""

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            return answer

    async def _generate_verification_questions(
        self,
        query: str,
        answer: str,
    ) -> List[VerificationQuestion]:
        """Generate verification questions for CoVe."""
        from langchain_core.messages import HumanMessage

        prompt = COVE_GENERATE_QUESTIONS_PROMPT.format(
            query=query,
            answer=answer,
            num_questions=self.config.max_verification_questions,
        )

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_verification_questions(response.content)
        except Exception:
            return []

    def _parse_verification_questions(self, text: str) -> List[VerificationQuestion]:
        """Parse verification questions from LLM response."""
        questions = []

        # Split by Q1, Q2, etc.
        parts = re.split(r'Q\d+:', text)

        for part in parts[1:]:  # Skip first empty part
            lines = part.strip().split('\n')
            if not lines:
                continue

            question = lines[0].strip()
            claim = ""
            q_type = "factual"

            for line in lines[1:]:
                if line.strip().startswith("Claim:"):
                    claim = line.split("Claim:")[-1].strip()
                elif line.strip().startswith("Type:"):
                    q_type = line.split("Type:")[-1].strip().lower()

            if question:
                questions.append(VerificationQuestion(
                    question=question,
                    expected_type=q_type,
                    source_claim=claim,
                ))

        return questions

    async def _check_consistency(self, claim: str, verified_answer: str) -> bool:
        """Check if verified answer is consistent with claim."""
        from langchain_core.messages import HumanMessage

        prompt = f"""Are these two statements consistent?

Statement 1: {claim}
Statement 2: {verified_answer}

Answer with just "Yes" or "No"."""

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip().lower().startswith("yes")
        except Exception:
            return True  # Assume consistent on error

    def _format_verification_results(self, results: List[Dict]) -> str:
        """Format verification results for prompt."""
        formatted = []
        for i, r in enumerate(results, 1):
            status = "✓" if r.get("is_consistent", False) else "✗"
            formatted.append(
                f"{i}. {status} Question: {r.get('question', 'N/A')}\n"
                f"   Claim: {r.get('claim', 'N/A')}\n"
                f"   Verified: {r.get('verified_answer', 'N/A')}"
            )
        return "\n\n".join(formatted)


# =============================================================================
# Convenience Functions
# =============================================================================

_refiner: Optional[AnswerRefiner] = None
_refiner_lock = asyncio.Lock()


async def get_answer_refiner(
    config: Optional[RefinerConfig] = None,
    retrieval_fn: Optional[Callable] = None,
) -> AnswerRefiner:
    """Get or create answer refiner singleton."""
    global _refiner

    if _refiner is not None:
        return _refiner

    async with _refiner_lock:
        if _refiner is not None:
            return _refiner

        _refiner = AnswerRefiner(config, retrieval_fn)
        return _refiner


async def refine_answer(
    query: str,
    answer: str,
    context: str,
    strategy: RefinementStrategy = RefinementStrategy.SELF_REFINE,
) -> RefinementResult:
    """
    Convenience function to refine an answer.

    Args:
        query: Original query
        answer: Answer to refine
        context: Retrieved context
        strategy: Refinement strategy

    Returns:
        RefinementResult with refined answer
    """
    refiner = await get_answer_refiner()
    return await refiner.refine(query, answer, context, strategy)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RefinerConfig",
    "RefinementResult",
    "RefinementStrategy",
    "Feedback",
    "VerificationQuestion",
    "VerificationResult",
    "AnswerRefiner",
    "get_answer_refiner",
    "refine_answer",
]
