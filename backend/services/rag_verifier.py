"""
AIDocumentIndexer - RAG Verifier Service
=========================================

Self-RAG / Answer Verification service that:
1. Verifies retrieved chunks are relevant to the query
2. Checks if the generated answer is grounded in the sources
3. Calculates confidence scores for responses

This reduces hallucinations and improves answer quality.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import asyncio
import structlog

from langchain_core.documents import Document

from backend.services.llm import LLMFactory, LLMConfig
from backend.services.embeddings import compute_similarity, EmbeddingService

logger = structlog.get_logger(__name__)


class VerificationLevel(str, Enum):
    """Verification intensity levels."""
    NONE = "none"  # No verification (fastest)
    QUICK = "quick"  # Embedding-based relevance only
    STANDARD = "standard"  # + LLM relevance check
    THOROUGH = "thorough"  # + Answer grounding verification


@dataclass
class ChunkRelevance:
    """Relevance assessment for a single chunk."""
    chunk_id: str
    content_snippet: str
    relevance_score: float  # 0-1, how relevant to query (may be RRF score for ranking)
    similarity_score: float  # 0-1, original vector cosine similarity for display/confidence
    is_relevant: bool  # Passes threshold
    reasoning: Optional[str] = None  # LLM explanation


@dataclass
class AnswerGrounding:
    """Assessment of how well an answer is grounded in sources."""
    is_grounded: bool  # Answer is supported by sources
    grounding_score: float  # 0-1, degree of support
    unsupported_claims: List[str] = field(default_factory=list)  # Claims not in sources
    supporting_sources: List[str] = field(default_factory=list)  # Chunk IDs that support claims
    reasoning: str = ""


@dataclass
class VerificationResult:
    """Complete verification result for a RAG response."""
    # Overall confidence
    confidence_score: float  # 0-1, overall confidence
    confidence_level: str  # "high", "medium", "low"

    # Chunk relevance
    relevant_chunks: List[ChunkRelevance]
    num_relevant: int
    num_filtered: int  # Chunks filtered out as irrelevant

    # Answer grounding (if thorough verification)
    grounding: Optional[AnswerGrounding] = None

    # Verification metadata
    verification_level: VerificationLevel = VerificationLevel.STANDARD
    processing_time_ms: float = 0.0


@dataclass
class VerifierConfig:
    """Configuration for RAG verifier."""
    level: VerificationLevel = VerificationLevel.STANDARD

    # Relevance thresholds
    # Note: For hybrid search with RRF, scores are typically 0.01-0.03 (1/(k+rank) where k=60)
    # For cosine similarity, scores are 0-1. We use a low default to support both.
    embedding_relevance_threshold: float = 0.005  # For RRF/embedding scores
    llm_relevance_threshold: float = 0.6  # For LLM-assessed relevance

    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5

    # LLM settings for verification
    verification_model: str = "gpt-4o-mini"  # Cost-effective model
    verification_temperature: float = 0.0  # Deterministic


class RAGVerifier:
    """
    Verifier for RAG responses to reduce hallucinations.

    Performs:
    1. Relevance filtering: Removes irrelevant retrieved chunks
    2. Confidence scoring: Calculates confidence based on retrieval quality
    3. Answer grounding: Checks if answer is supported by sources (optional)
    """

    RELEVANCE_PROMPT = """Assess the relevance of this text passage to the given query.

Query: {query}

Passage:
{passage}

Rate the relevance on a scale of 0-10:
- 0-2: Not relevant at all
- 3-4: Slightly related but doesn't answer the query
- 5-6: Partially relevant, some useful information
- 7-8: Relevant, contains answer to query
- 9-10: Highly relevant, directly answers the query

Respond with ONLY a JSON object:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""

    BATCH_RELEVANCE_PROMPT = """Assess the relevance of each text passage to the given query.

Query: {query}

{passages}

For EACH passage, rate the relevance on a scale of 0-10:
- 0-2: Not relevant at all
- 3-4: Slightly related but doesn't answer the query
- 5-6: Partially relevant, some useful information
- 7-8: Relevant, contains answer to query
- 9-10: Highly relevant, directly answers the query

Respond with ONLY a JSON object with scores for each passage number:
{{"1": {{"score": <0-10>, "reasoning": "<brief>"}}, "2": {{"score": <0-10>, "reasoning": "<brief>"}}, ...}}"""

    GROUNDING_PROMPT = """Analyze if this answer is grounded in the provided sources.

Question: {question}

Answer to verify:
{answer}

Source passages:
{sources}

Analyze the answer and identify:
1. Is the answer fully supported by the sources?
2. What claims in the answer are NOT supported by any source?
3. Which sources support which claims?

Respond with ONLY a JSON object:
{{
  "is_grounded": true/false,
  "grounding_score": <0.0-1.0>,
  "unsupported_claims": ["claim1", "claim2"],
  "supporting_evidence": ["source1 supports X", "source2 supports Y"],
  "reasoning": "<explanation>"
}}"""

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize RAG verifier."""
        self.config = config or VerifierConfig()
        self._embedding_service = embedding_service
        self._llm = None

        logger.info(
            "Initialized RAG verifier",
            level=self.config.level.value,
            model=self.config.verification_model,
        )

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get or create embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def llm(self):
        """Get or create LLM for verification."""
        if self._llm is None:
            self._llm = LLMFactory.get_chat_model(
                provider="openai",
                model=self.config.verification_model,
                temperature=self.config.verification_temperature,
            )
        return self._llm

    async def verify(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        answer: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify retrieved documents and optionally the generated answer.

        Args:
            query: The user's query
            retrieved_docs: List of (Document, score) tuples from retrieval
            answer: Generated answer to verify (optional, for grounding check)

        Returns:
            VerificationResult with relevance and confidence scores
        """
        import time
        start_time = time.time()

        # Handle empty docs case
        if not retrieved_docs:
            return VerificationResult(
                confidence_score=0.0,
                confidence_level="low",
                relevant_chunks=[],
                num_relevant=0,
                num_filtered=0,
                verification_level=self.config.level,
                processing_time_ms=0.0,
            )

        if self.config.level == VerificationLevel.NONE:
            # No verification, pass through with default scores
            return self._create_passthrough_result(retrieved_docs)

        logger.debug(
            "Starting verification",
            level=self.config.level.value,
            num_docs=len(retrieved_docs),
        )

        # Step 1: Assess relevance of each chunk
        chunk_relevances = await self._assess_relevance(query, retrieved_docs)

        # Filter to relevant chunks
        relevant_chunks = [cr for cr in chunk_relevances if cr.is_relevant]
        filtered_count = len(chunk_relevances) - len(relevant_chunks)

        # Step 2: Calculate confidence score
        confidence_score = self._calculate_confidence(relevant_chunks, len(retrieved_docs))
        confidence_level = self._get_confidence_level(confidence_score)

        # Step 3: Verify answer grounding (if thorough level and answer provided)
        grounding = None
        if self.config.level == VerificationLevel.THOROUGH and answer:
            grounding = await self._verify_grounding(
                question=query,
                answer=answer,
                sources=[(doc, score) for (doc, score), cr in zip(retrieved_docs, chunk_relevances) if cr.is_relevant],
            )
            # Adjust confidence based on grounding
            if grounding and not grounding.is_grounded:
                confidence_score *= 0.5  # Reduce confidence for ungrounded answers
                confidence_level = self._get_confidence_level(confidence_score)

        processing_time = (time.time() - start_time) * 1000

        return VerificationResult(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            relevant_chunks=chunk_relevances,
            num_relevant=len(relevant_chunks),
            num_filtered=filtered_count,
            grounding=grounding,
            verification_level=self.config.level,
            processing_time_ms=processing_time,
        )

    async def _assess_relevance(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
    ) -> List[ChunkRelevance]:
        """Assess relevance of each document to the query."""
        relevances = []

        # Quick mode: embedding similarity only (no LLM calls)
        if self.config.level == VerificationLevel.QUICK:
            for doc, retrieval_score in docs:
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                content = doc.page_content[:500]
                similarity = doc.metadata.get("similarity_score", retrieval_score)
                is_relevant = retrieval_score >= self.config.embedding_relevance_threshold

                relevances.append(ChunkRelevance(
                    chunk_id=chunk_id,
                    content_snippet=content[:200],
                    relevance_score=min(retrieval_score, 1.0),
                    similarity_score=similarity,
                    is_relevant=is_relevant,
                ))
            return relevances

        # Standard/Thorough: Use BATCH LLM assessment (single LLM call for all chunks)
        # This is 5-10x more efficient than individual calls
        passages = [doc.page_content for doc, _ in docs]

        try:
            batch_scores = await self._batch_llm_relevance_check(query, passages)

            for i, (doc, retrieval_score) in enumerate(docs):
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                content = doc.page_content[:500]
                similarity = doc.metadata.get("similarity_score", retrieval_score)

                if i < len(batch_scores):
                    llm_score, reasoning = batch_scores[i]
                else:
                    llm_score, reasoning = 0.5, "No score returned"

                combined_score = (retrieval_score + llm_score) / 2
                is_relevant = combined_score >= self.config.llm_relevance_threshold

                relevances.append(ChunkRelevance(
                    chunk_id=chunk_id,
                    content_snippet=content[:200],
                    relevance_score=combined_score,
                    similarity_score=similarity,
                    is_relevant=is_relevant,
                    reasoning=reasoning,
                ))

        except Exception as e:
            logger.warning("Batch LLM relevance check failed, falling back to retrieval scores", error=str(e))
            # Fall back to retrieval scores only
            for doc, retrieval_score in docs:
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                content = doc.page_content[:500]
                similarity = doc.metadata.get("similarity_score", retrieval_score)

                relevances.append(ChunkRelevance(
                    chunk_id=chunk_id,
                    content_snippet=content[:200],
                    relevance_score=min(retrieval_score, 1.0),
                    similarity_score=similarity,
                    is_relevant=retrieval_score >= self.config.embedding_relevance_threshold,
                ))

        return relevances

    async def _llm_relevance_check(
        self,
        query: str,
        passage: str,
    ) -> Tuple[float, str]:
        """Use LLM to assess passage relevance to query."""
        import json

        prompt = self.RELEVANCE_PROMPT.format(
            query=query,
            passage=passage[:1000],  # Limit passage length
        )

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            result = json.loads(content)
            score = result.get("score", 5) / 10.0  # Normalize to 0-1
            reasoning = result.get("reasoning", "")

            return score, reasoning

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM relevance response")
            return 0.5, "Failed to parse response"
        except Exception as e:
            logger.error("LLM relevance check error", error=str(e))
            raise

    async def _batch_llm_relevance_check(
        self,
        query: str,
        passages: List[str],
    ) -> List[Tuple[float, str]]:
        """
        Use LLM to assess relevance of multiple passages in a single call.

        This is 5-10x more efficient than individual calls for multiple passages.

        Args:
            query: The user's query
            passages: List of passage texts to assess

        Returns:
            List of (score, reasoning) tuples, one per passage
        """
        import json

        if not passages:
            return []

        # Format passages for the prompt
        passages_text = "\n\n".join([
            f"[Passage {i+1}]:\n{passage[:800]}"  # Limit each passage
            for i, passage in enumerate(passages)
        ])

        prompt = self.BATCH_RELEVANCE_PROMPT.format(
            query=query,
            passages=passages_text,
        )

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            result = json.loads(content)

            # Extract scores in order
            scores = []
            for i in range(len(passages)):
                key = str(i + 1)
                if key in result:
                    item = result[key]
                    score = item.get("score", 5) / 10.0  # Normalize to 0-1
                    reasoning = item.get("reasoning", "")
                    scores.append((score, reasoning))
                else:
                    # Missing entry, use default
                    scores.append((0.5, "No assessment provided"))

            logger.debug(
                "Batch relevance check complete",
                num_passages=len(passages),
                num_scores=len(scores),
            )

            return scores

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse batch LLM relevance response", error=str(e))
            # Return default scores for all
            return [(0.5, "Failed to parse response")] * len(passages)
        except Exception as e:
            logger.error("Batch LLM relevance check error", error=str(e))
            # Return default scores
            return [(0.5, f"Error: {str(e)}")] * len(passages)

    async def _verify_grounding(
        self,
        question: str,
        answer: str,
        sources: List[Tuple[Document, float]],
    ) -> AnswerGrounding:
        """Verify that the answer is grounded in the source documents."""
        import json

        # Format sources for prompt
        sources_text = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content[:500]}"
            for i, (doc, _) in enumerate(sources)
        ])

        prompt = self.GROUNDING_PROMPT.format(
            question=question,
            answer=answer,
            sources=sources_text,
        )

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            result = json.loads(content)

            return AnswerGrounding(
                is_grounded=result.get("is_grounded", True),
                grounding_score=result.get("grounding_score", 0.5),
                unsupported_claims=result.get("unsupported_claims", []),
                supporting_sources=result.get("supporting_evidence", []),
                reasoning=result.get("reasoning", ""),
            )

        except Exception as e:
            logger.error("Grounding verification failed", error=str(e))
            return AnswerGrounding(
                is_grounded=True,  # Assume grounded on failure
                grounding_score=0.5,
                reasoning=f"Verification failed: {str(e)}",
            )

    def _calculate_confidence(
        self,
        relevant_chunks: List[ChunkRelevance],
        total_retrieved: int,
    ) -> float:
        """Calculate overall confidence score."""
        if not relevant_chunks:
            return 0.0

        # Factors:
        # 1. Average similarity score of relevant chunks (use original vector similarity, not RRF)
        # similarity_score is 0-1 (cosine similarity), while relevance_score may be tiny RRF scores
        avg_similarity = sum(cr.similarity_score for cr in relevant_chunks) / len(relevant_chunks)

        # 2. Ratio of relevant to total retrieved
        relevance_ratio = len(relevant_chunks) / max(total_retrieved, 1)

        # 3. Number of relevant sources (more sources = higher confidence)
        source_bonus = min(len(relevant_chunks) / 3, 1.0) * 0.2  # Up to 0.2 bonus for 3+ sources

        # Weighted combination
        confidence = (avg_similarity * 0.5) + (relevance_ratio * 0.3) + source_bonus

        return min(confidence, 1.0)

    def _get_confidence_level(self, score: float) -> str:
        """Convert confidence score to level string."""
        if score >= self.config.high_confidence_threshold:
            return "high"
        elif score >= self.config.medium_confidence_threshold:
            return "medium"
        return "low"

    def _create_passthrough_result(
        self,
        docs: List[Tuple[Document, float]],
    ) -> VerificationResult:
        """Create passthrough result for no-verification mode."""
        chunk_relevances = [
            ChunkRelevance(
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                content_snippet=doc.page_content[:200],
                relevance_score=min(score, 1.0),
                similarity_score=doc.metadata.get("similarity_score", score),  # Original vector similarity
                is_relevant=True,
            )
            for doc, score in docs
        ]

        return VerificationResult(
            confidence_score=0.7,  # Default confidence
            confidence_level="medium",
            relevant_chunks=chunk_relevances,
            num_relevant=len(docs),
            num_filtered=0,
            verification_level=VerificationLevel.NONE,
        )

    def filter_by_relevance(
        self,
        docs: List[Tuple[Document, float]],
        verification_result: VerificationResult,
    ) -> List[Tuple[Document, float]]:
        """
        Filter documents to only include relevant ones based on verification.

        Args:
            docs: Original retrieved documents
            verification_result: Result from verify()

        Returns:
            Filtered list of relevant documents
        """
        relevant_chunk_ids = {
            cr.chunk_id for cr in verification_result.relevant_chunks
            if cr.is_relevant
        }

        return [
            (doc, score) for doc, score in docs
            if doc.metadata.get("chunk_id", "unknown") in relevant_chunk_ids
        ]


# Convenience functions
def get_verifier(
    level: VerificationLevel = VerificationLevel.STANDARD,
) -> RAGVerifier:
    """Get a configured RAG verifier instance."""
    config = VerifierConfig(level=level)
    return RAGVerifier(config)


async def verify_and_filter(
    query: str,
    docs: List[Tuple[Document, float]],
    level: VerificationLevel = VerificationLevel.QUICK,
) -> Tuple[List[Tuple[Document, float]], VerificationResult]:
    """
    Convenience function to verify and filter documents.

    Args:
        query: User query
        docs: Retrieved documents
        level: Verification level

    Returns:
        Tuple of (filtered_docs, verification_result)
    """
    verifier = get_verifier(level)
    result = await verifier.verify(query, docs)
    filtered = verifier.filter_by_relevance(docs, result)
    return filtered, result
