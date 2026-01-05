"""
AIDocumentIndexer - Corrective RAG (CRAG)
==========================================

Implements Corrective Retrieval-Augmented Generation for improved
retrieval quality. CRAG evaluates retrieved documents for relevance
and can trigger fallback mechanisms when results are poor.

Reference: https://arxiv.org/abs/2401.15884

Features:
- Relevance evaluation for retrieved documents
- Automatic query refinement when results are weak
- Web search fallback for unsupported queries
- Confidence scoring for responses
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from backend.services.vectorstore import SearchResult

logger = structlog.get_logger(__name__)


class RelevanceLevel(str, Enum):
    """Document relevance classification."""
    CORRECT = "correct"      # Highly relevant to query
    AMBIGUOUS = "ambiguous"  # Partially relevant, may contain useful info
    INCORRECT = "incorrect"  # Not relevant to query


@dataclass
class RelevanceEvaluation:
    """Result of relevance evaluation for a document."""
    chunk_id: str
    document_id: str
    relevance: RelevanceLevel
    score: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class CRAGResult:
    """Result from CRAG processing."""
    original_results: List[SearchResult]
    filtered_results: List[SearchResult]
    evaluations: List[RelevanceEvaluation]
    action_taken: str  # "use_as_is", "filtered", "refined_query", "web_fallback", "low_confidence"
    confidence: float
    needs_web_search: bool
    refined_query: Optional[str] = None  # Set when action_taken == "refined_query"


class CorrectiveRAG:
    """
    Corrective RAG implementation for improved retrieval quality.

    CRAG workflow:
    1. Retrieve initial documents
    2. Evaluate relevance of each document
    3. Classify results: correct, ambiguous, incorrect
    4. Take action based on classification:
       - Mostly correct: Use results as-is
       - Mixed: Filter to correct + some ambiguous
       - Mostly incorrect: Refine query or trigger web search
    """

    def __init__(
        self,
        relevance_threshold_high: float = 0.7,
        relevance_threshold_low: float = 0.3,
        min_correct_docs: int = 2,
        enable_web_fallback: bool = False,
        enable_query_refinement: bool = True,
    ):
        """
        Initialize CRAG.

        Args:
            relevance_threshold_high: Score above this is "correct"
            relevance_threshold_low: Score below this is "incorrect"
            min_correct_docs: Minimum correct docs before triggering fallback
            enable_web_fallback: Enable web search for poor results
            enable_query_refinement: Enable query refinement for poor results
        """
        self.relevance_threshold_high = relevance_threshold_high
        self.relevance_threshold_low = relevance_threshold_low
        self.min_correct_docs = min_correct_docs
        self.enable_web_fallback = enable_web_fallback
        self.enable_query_refinement = enable_query_refinement

    async def process(
        self,
        query: str,
        search_results: List[SearchResult],
        llm: Optional[Any] = None,
    ) -> CRAGResult:
        """
        Process search results through CRAG pipeline.

        Args:
            query: Original user query
            search_results: Initial retrieval results
            llm: Optional LLM for relevance evaluation (uses heuristics if None)

        Returns:
            CRAGResult with filtered results and action taken
        """
        if not search_results:
            return CRAGResult(
                original_results=[],
                filtered_results=[],
                evaluations=[],
                action_taken="no_results",
                confidence=0.0,
                needs_web_search=self.enable_web_fallback,
            )

        # Evaluate relevance of each result
        if llm:
            evaluations = await self._evaluate_with_llm(query, search_results, llm)
        else:
            evaluations = self._evaluate_heuristic(query, search_results)

        # Classify results
        correct = [e for e in evaluations if e.relevance == RelevanceLevel.CORRECT]
        ambiguous = [e for e in evaluations if e.relevance == RelevanceLevel.AMBIGUOUS]
        incorrect = [e for e in evaluations if e.relevance == RelevanceLevel.INCORRECT]

        logger.info(
            "CRAG evaluation complete",
            query_preview=query[:50],
            correct=len(correct),
            ambiguous=len(ambiguous),
            incorrect=len(incorrect),
        )

        # Determine action based on classification
        if len(correct) >= self.min_correct_docs:
            # Good results - use correct + some ambiguous
            filtered_ids = set(e.chunk_id for e in correct)
            filtered_ids.update(e.chunk_id for e in ambiguous[:2])  # Add up to 2 ambiguous

            filtered_results = [r for r in search_results if r.chunk_id in filtered_ids]
            action = "filtered" if len(filtered_results) < len(search_results) else "use_as_is"
            needs_web = False
            confidence = sum(e.score for e in correct) / len(correct)

        elif len(correct) + len(ambiguous) >= self.min_correct_docs:
            # Mixed results - use all correct and ambiguous
            filtered_ids = set(e.chunk_id for e in correct + ambiguous)
            filtered_results = [r for r in search_results if r.chunk_id in filtered_ids]
            action = "filtered"
            needs_web = False
            confidence = sum(e.score for e in correct + ambiguous) / len(correct + ambiguous)

        else:
            # Poor results - try query refinement first, then consider web fallback
            refined_query = None

            # Try query refinement if enabled and LLM is available
            if self.enable_query_refinement and llm and incorrect:
                try:
                    refined_query = await self.refine_query(
                        original_query=query,
                        poor_results=[r for r in search_results if r.chunk_id in set(e.chunk_id for e in incorrect)],
                        llm=llm,
                    )
                    # If we got a different query, signal for re-search
                    if refined_query and refined_query.lower().strip() != query.lower().strip():
                        # Return with refined_query action - caller should re-search
                        filtered_ids = set(e.chunk_id for e in correct + ambiguous)
                        filtered_results = [r for r in search_results if r.chunk_id in filtered_ids]
                        return CRAGResult(
                            original_results=search_results,
                            filtered_results=filtered_results,
                            evaluations=evaluations,
                            action_taken="refined_query",
                            confidence=0.3,
                            needs_web_search=False,
                            refined_query=refined_query,
                        )
                except Exception as e:
                    logger.warning("Query refinement failed in CRAG process", error=str(e))

            # If refinement didn't help, consider web fallback
            if self.enable_web_fallback:
                # Keep some results but flag for web search
                filtered_ids = set(e.chunk_id for e in correct + ambiguous[:3])
                filtered_results = [r for r in search_results if r.chunk_id in filtered_ids]
                action = "web_fallback"
                needs_web = True
                confidence = 0.3
            else:
                # No fallback - return what we have with low confidence
                filtered_ids = set(e.chunk_id for e in correct + ambiguous)
                filtered_results = [r for r in search_results if r.chunk_id in filtered_ids]
                action = "low_confidence"
                needs_web = False
                confidence = 0.3 if filtered_results else 0.0

        return CRAGResult(
            original_results=search_results,
            filtered_results=filtered_results,
            evaluations=evaluations,
            action_taken=action,
            confidence=min(1.0, confidence),
            needs_web_search=needs_web,
            refined_query=None,
        )

    def _evaluate_heuristic(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[RelevanceEvaluation]:
        """
        Evaluate relevance using heuristics (no LLM).

        Uses similarity scores and keyword overlap as signals.
        """
        evaluations = []
        query_words = set(query.lower().split())

        for result in results:
            # Use similarity score as primary signal
            sim_score = result.similarity_score

            # Boost for keyword overlap
            content_words = set(result.content.lower().split()[:100])
            keyword_overlap = len(query_words & content_words) / max(len(query_words), 1)

            # Combined score
            combined_score = (sim_score * 0.7) + (keyword_overlap * 0.3)

            # Classify
            if combined_score >= self.relevance_threshold_high:
                relevance = RelevanceLevel.CORRECT
            elif combined_score >= self.relevance_threshold_low:
                relevance = RelevanceLevel.AMBIGUOUS
            else:
                relevance = RelevanceLevel.INCORRECT

            evaluations.append(RelevanceEvaluation(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                relevance=relevance,
                score=combined_score,
                reasoning=f"Similarity: {sim_score:.2f}, Keyword overlap: {keyword_overlap:.2f}",
            ))

        return evaluations

    async def _evaluate_with_llm(
        self,
        query: str,
        results: List[SearchResult],
        llm: Any,
    ) -> List[RelevanceEvaluation]:
        """
        Evaluate relevance using LLM.

        More accurate but slower and costs tokens.
        """
        evaluations = []

        # Evaluate in parallel batches
        batch_size = 5
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            batch_evals = await asyncio.gather(*[
                self._evaluate_single_with_llm(query, result, llm)
                for result in batch
            ])
            evaluations.extend(batch_evals)

        return evaluations

    async def _evaluate_single_with_llm(
        self,
        query: str,
        result: SearchResult,
        llm: Any,
    ) -> RelevanceEvaluation:
        """Evaluate a single result using LLM."""
        prompt = f"""Evaluate if this document passage is relevant to answering the query.

Query: {query}

Document passage:
{result.content[:1000]}

Rate the relevance on a scale of 0-10 where:
- 8-10: Highly relevant, directly answers the query
- 4-7: Partially relevant, contains some useful information
- 0-3: Not relevant to the query

Respond with ONLY a JSON object:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""

        try:
            from langchain_core.messages import HumanMessage

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse JSON response
            import json
            # Try to extract JSON from response
            try:
                # Handle potential markdown code blocks
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                data = json.loads(content.strip())
            except json.JSONDecodeError:
                # Fallback to heuristic
                return self._evaluate_heuristic(query, [result])[0]

            score = float(data.get("score", 5)) / 10.0
            reasoning = data.get("reasoning", "LLM evaluation")

            # Classify
            if score >= self.relevance_threshold_high:
                relevance = RelevanceLevel.CORRECT
            elif score >= self.relevance_threshold_low:
                relevance = RelevanceLevel.AMBIGUOUS
            else:
                relevance = RelevanceLevel.INCORRECT

            return RelevanceEvaluation(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                relevance=relevance,
                score=score,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning("LLM evaluation failed, using heuristic", error=str(e))
            return self._evaluate_heuristic(query, [result])[0]

    async def refine_query(
        self,
        original_query: str,
        poor_results: List[SearchResult],
        llm: Any,
    ) -> str:
        """
        Refine query when initial results are poor.

        Args:
            original_query: Original user query
            poor_results: Results that were classified as incorrect
            llm: LLM for query refinement

        Returns:
            Refined query string
        """
        # Get sample of poor results for context
        sample_content = "\n\n".join([
            r.content[:200] for r in poor_results[:3]
        ])

        prompt = f"""The following query returned irrelevant results. Rewrite the query to be more specific and likely to match relevant documents.

Original query: {original_query}

Sample of retrieved (irrelevant) content:
{sample_content}

Provide a refined query that:
1. Uses more specific terms
2. Avoids ambiguous words
3. Includes synonyms or related terms

Respond with ONLY the refined query text, nothing else."""

        try:
            from langchain_core.messages import HumanMessage

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            refined = response.content.strip()

            # Validate it's reasonable
            if len(refined) > 10 and len(refined) < 500:
                logger.info(
                    "Query refined",
                    original=original_query[:50],
                    refined=refined[:50],
                )
                return refined

        except Exception as e:
            logger.warning("Query refinement failed", error=str(e))

        return original_query


# Singleton instance
_crag_instance: Optional[CorrectiveRAG] = None


def get_corrective_rag() -> CorrectiveRAG:
    """Get or create the CRAG singleton."""
    global _crag_instance
    if _crag_instance is None:
        _crag_instance = CorrectiveRAG()
    return _crag_instance
