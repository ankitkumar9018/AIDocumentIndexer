"""
AIDocumentIndexer - Multi-Model Ensemble Voting
================================================

Cross-verify answers from multiple models to improve accuracy.
Particularly useful for fact-checking and reducing hallucinations.

Strategies:
1. Majority voting - Most common answer wins
2. Confidence weighted - Weight by model confidence
3. Consensus detection - Flag disagreements
4. Best-of-N - Pick highest confidence answer
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import asyncio
import re
import json
import time

import structlog

from backend.services.llm import LLMFactory, llm_config
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class VotingStrategy(str, Enum):
    """Voting strategies for ensemble."""
    MAJORITY = "majority"           # Most common answer
    CONFIDENCE = "confidence"       # Weighted by confidence
    CONSENSUS = "consensus"         # Require agreement
    BEST_OF_N = "best_of_n"         # Highest confidence
    SYNTHESIS = "synthesis"         # LLM synthesizes all answers


@dataclass
class ModelAnswer:
    """Answer from a single model."""
    model: str
    provider: str
    answer: str
    confidence: float
    reasoning: str = ""
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EnsembleResult:
    """Result from ensemble voting."""
    query: str
    final_answer: str
    confidence: float
    strategy: VotingStrategy
    model_answers: List[ModelAnswer]
    agreement_level: str  # "full", "partial", "none"
    disagreements: List[str]
    total_latency_ms: int
    models_used: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "strategy": self.strategy.value,
            "model_answers": [a.to_dict() for a in self.model_answers],
            "agreement_level": self.agreement_level,
            "disagreements": self.disagreements,
            "total_latency_ms": self.total_latency_ms,
            "models_used": self.models_used,
        }


class EnsembleVotingService:
    """
    Multi-model ensemble for answer verification.

    Uses multiple LLMs to:
    1. Generate answers in parallel
    2. Detect disagreements
    3. Vote on the best answer
    4. Improve overall accuracy
    """

    ANSWER_PROMPT = """Answer the following question. Be concise and factual.

Question: {question}

Context (if provided):
{context}

Provide your answer and rate your confidence (0-100%).

Format:
ANSWER: [your answer]
CONFIDENCE: [0-100]
REASONING: [brief explanation]"""

    SYNTHESIS_PROMPT = """You are synthesizing answers from multiple AI models.

Question: {question}

Model answers:
{answers}

Analyze these answers and provide:
1. The most accurate synthesized answer
2. Points of agreement
3. Points of disagreement
4. Your confidence in the final answer

Format your response as JSON:
{{
    "final_answer": "synthesized answer",
    "agreements": ["list of agreed points"],
    "disagreements": ["list of disagreements"],
    "confidence": 0-100,
    "reasoning": "explanation"
}}"""

    AGREEMENT_PROMPT = """Compare these answers to determine if they agree:

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Do they fundamentally agree? Respond with:
AGREE: yes/no/partial
EXPLANATION: [brief explanation]"""

    def __init__(self):
        """Initialize the ensemble voting service."""
        # Default models to use (can be overridden)
        # Default to system-configured provider; ensemble can use multiple providers
        _default = llm_config.default_provider
        self.default_models = [
            {"provider": _default, "model": None},  # System default
        ]

    async def query(
        self,
        question: str,
        context: str = "",
        models: List[Dict[str, str]] = None,
        strategy: VotingStrategy = VotingStrategy.CONFIDENCE,
        min_agreement: float = 0.6,
    ) -> EnsembleResult:
        """
        Query multiple models and vote on the best answer.

        Args:
            question: The question to answer
            context: Optional context for RAG
            models: List of model configs [{provider, model}]
            strategy: Voting strategy to use
            min_agreement: Minimum agreement threshold for consensus

        Returns:
            EnsembleResult with voted answer
        """
        start_time = time.time()
        models = models or self.default_models

        logger.info(
            "Starting ensemble query",
            model_count=len(models),
            strategy=strategy.value,
        )

        # Query all models in parallel
        tasks = [
            self._query_model(question, context, m["provider"], m["model"])
            for m in models
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful answers
        model_answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "Model query failed",
                    model=models[i],
                    error=str(result),
                )
            elif result:
                model_answers.append(result)

        if not model_answers:
            raise ValueError("All model queries failed")

        # Detect agreement level
        agreement_level, disagreements = await self._analyze_agreement(
            question, model_answers
        )

        # Vote based on strategy
        final_answer, confidence = await self._vote(
            question, model_answers, strategy, context
        )

        total_latency = int((time.time() - start_time) * 1000)

        result = EnsembleResult(
            query=question,
            final_answer=final_answer,
            confidence=confidence,
            strategy=strategy,
            model_answers=model_answers,
            agreement_level=agreement_level,
            disagreements=disagreements,
            total_latency_ms=total_latency,
            models_used=[f"{a.provider}/{a.model}" for a in model_answers],
        )

        logger.info(
            "Ensemble query complete",
            agreement=agreement_level,
            confidence=confidence,
            models_succeeded=len(model_answers),
            latency_ms=total_latency,
        )

        return result

    async def _query_model(
        self,
        question: str,
        context: str,
        provider: str,
        model: str,
    ) -> Optional[ModelAnswer]:
        """Query a single model."""
        start_time = time.time()

        try:
            llm = LLMFactory.get_chat_model(
                provider=provider,
                model=model,
                temperature=0.3,
                max_tokens=1024,
            )

            prompt = self.ANSWER_PROMPT.format(
                question=question,
                context=context or "No additional context provided.",
            )

            response = await llm.ainvoke(prompt)
            content = response.content

            # Parse response
            answer, confidence, reasoning = self._parse_answer(content)

            latency = int((time.time() - start_time) * 1000)

            return ModelAnswer(
                model=model,
                provider=provider,
                answer=answer,
                confidence=confidence,
                reasoning=reasoning,
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(
                "Model query error",
                provider=provider,
                model=model,
                error=str(e),
            )
            return None

    def _parse_answer(self, content: str) -> Tuple[str, float, str]:
        """Parse model answer from formatted response."""
        answer = ""
        confidence = 0.7
        reasoning = ""

        # Extract ANSWER
        answer_match = re.search(r'ANSWER:\s*(.+?)(?=CONFIDENCE:|REASONING:|$)', content, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Fallback: use entire content
            answer = content.strip()

        # Extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', content, re.IGNORECASE)
        if conf_match:
            confidence = min(100, max(0, int(conf_match.group(1)))) / 100

        # Extract REASONING
        reason_match = re.search(r'REASONING:\s*(.+?)$', content, re.DOTALL | re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return answer, confidence, reasoning

    async def _analyze_agreement(
        self,
        question: str,
        answers: List[ModelAnswer],
    ) -> Tuple[str, List[str]]:
        """Analyze agreement level between answers."""
        if len(answers) < 2:
            return "full", []

        disagreements = []
        agreement_count = 0
        total_pairs = 0

        # Compare pairs of answers
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                total_pairs += 1
                agrees, explanation = await self._check_agreement(
                    question, answers[i].answer, answers[j].answer
                )

                if agrees == "yes":
                    agreement_count += 1
                elif agrees == "no":
                    disagreements.append(
                        f"{answers[i].model} vs {answers[j].model}: {explanation}"
                    )
                else:  # partial
                    agreement_count += 0.5

        agreement_ratio = agreement_count / total_pairs if total_pairs > 0 else 1

        if agreement_ratio >= 0.9:
            return "full", disagreements
        elif agreement_ratio >= 0.5:
            return "partial", disagreements
        else:
            return "none", disagreements

    async def _check_agreement(
        self,
        question: str,
        answer1: str,
        answer2: str,
    ) -> Tuple[str, str]:
        """Check if two answers agree."""
        # Simple heuristic first
        if answer1.lower().strip() == answer2.lower().strip():
            return "yes", "Identical answers"

        # Use LLM to compare
        try:
            llm = LLMFactory.get_chat_model(
                provider=settings.DEFAULT_LLM_PROVIDER,
                model=settings.DEFAULT_CHAT_MODEL,
                temperature=0.1,
                max_tokens=256,
            )

            prompt = self.AGREEMENT_PROMPT.format(
                question=question,
                answer1=answer1[:500],
                answer2=answer2[:500],
            )

            response = await llm.ainvoke(prompt)
            content = response.content

            # Parse agreement
            agree_match = re.search(r'AGREE:\s*(yes|no|partial)', content, re.IGNORECASE)
            if agree_match:
                agrees = agree_match.group(1).lower()
            else:
                agrees = "partial"

            # Parse explanation
            exp_match = re.search(r'EXPLANATION:\s*(.+?)$', content, re.DOTALL | re.IGNORECASE)
            explanation = exp_match.group(1).strip() if exp_match else ""

            return agrees, explanation

        except Exception as e:
            logger.warning("Agreement check failed", error=str(e))
            return "partial", "Could not verify"

    async def _vote(
        self,
        question: str,
        answers: List[ModelAnswer],
        strategy: VotingStrategy,
        context: str = "",
    ) -> Tuple[str, float]:
        """Vote on the best answer."""
        if strategy == VotingStrategy.BEST_OF_N:
            # Pick highest confidence
            best = max(answers, key=lambda a: a.confidence)
            return best.answer, best.confidence

        elif strategy == VotingStrategy.CONFIDENCE:
            # Weight by confidence
            total_weight = sum(a.confidence for a in answers)
            if total_weight == 0:
                return answers[0].answer, answers[0].confidence

            # Pick answer with highest weighted support
            weighted_scores = {}
            for a in answers:
                key = a.answer[:100]  # Normalize by truncating
                weighted_scores[key] = weighted_scores.get(key, 0) + a.confidence

            best_key = max(weighted_scores, key=weighted_scores.get)
            best_answer = next(a for a in answers if a.answer[:100] == best_key)
            avg_confidence = weighted_scores[best_key] / len(answers)

            return best_answer.answer, avg_confidence

        elif strategy == VotingStrategy.MAJORITY:
            # Simple majority voting
            answer_counts = {}
            for a in answers:
                key = a.answer[:100]
                answer_counts[key] = answer_counts.get(key, 0) + 1

            best_key = max(answer_counts, key=answer_counts.get)
            best_answer = next(a for a in answers if a.answer[:100] == best_key)
            majority_ratio = answer_counts[best_key] / len(answers)

            return best_answer.answer, majority_ratio

        elif strategy == VotingStrategy.CONSENSUS:
            # Require agreement, or synthesize
            if all(a.answer[:100] == answers[0].answer[:100] for a in answers):
                avg_conf = sum(a.confidence for a in answers) / len(answers)
                return answers[0].answer, avg_conf
            else:
                # Fall back to synthesis
                return await self._synthesize(question, answers, context)

        elif strategy == VotingStrategy.SYNTHESIS:
            return await self._synthesize(question, answers, context)

        else:
            return answers[0].answer, answers[0].confidence

    async def _synthesize(
        self,
        question: str,
        answers: List[ModelAnswer],
        context: str = "",
    ) -> Tuple[str, float]:
        """Synthesize answers using an LLM."""
        try:
            llm = LLMFactory.get_chat_model(
                provider=settings.DEFAULT_LLM_PROVIDER,
                model=settings.DEFAULT_CHAT_MODEL,
                temperature=0.3,
                max_tokens=1024,
            )

            # Format answers
            answers_text = "\n\n".join([
                f"Model: {a.model}\nAnswer: {a.answer}\nConfidence: {int(a.confidence * 100)}%"
                for a in answers
            ])

            prompt = self.SYNTHESIS_PROMPT.format(
                question=question,
                answers=answers_text,
            )

            response = await llm.ainvoke(prompt)
            content = response.content

            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("final_answer", answers[0].answer),
                    result.get("confidence", 70) / 100,
                )
            else:
                return content, 0.7

        except Exception as e:
            logger.error("Synthesis failed", error=str(e))
            # Fallback to highest confidence
            best = max(answers, key=lambda a: a.confidence)
            return best.answer, best.confidence


# Singleton instance
_ensemble_voting: Optional[EnsembleVotingService] = None


def get_ensemble_voting_service() -> EnsembleVotingService:
    """Get or create the ensemble voting service singleton."""
    global _ensemble_voting
    if _ensemble_voting is None:
        _ensemble_voting = EnsembleVotingService()
    return _ensemble_voting
