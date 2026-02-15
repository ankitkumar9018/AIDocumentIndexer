"""
AIDocumentIndexer - Self-RAG (Self-Reflective RAG)
====================================================

Implements Self-Reflective Retrieval-Augmented Generation for improved
response quality. Self-RAG verifies generated responses against source
documents to detect and correct hallucinations.

Reference: https://arxiv.org/abs/2310.11511

Features:
- Response-to-source verification
- Claim extraction and validation
- Hallucination detection
- Automatic response regeneration when confidence is low
- Source coverage analysis
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from backend.services.vectorstore import SearchResult

logger = structlog.get_logger(__name__)


class SupportLevel(str, Enum):
    """Level of support for a claim from sources."""
    FULLY_SUPPORTED = "fully_supported"      # Direct evidence in sources
    PARTIALLY_SUPPORTED = "partially_supported"  # Some evidence
    NOT_SUPPORTED = "not_supported"          # No evidence (potential hallucination)
    CONTRADICTED = "contradicted"            # Sources say opposite


@dataclass
class ClaimAnalysis:
    """Analysis of a single claim from the response."""
    claim: str
    support_level: SupportLevel
    confidence: float  # 0.0 to 1.0
    supporting_sources: List[str] = field(default_factory=list)  # chunk_ids
    evidence: str = ""
    is_hallucination: bool = False


@dataclass
class SelfRAGResult:
    """Result of Self-RAG verification."""
    original_response: str
    verified_response: Optional[str]  # Regenerated if needed
    claims: List[ClaimAnalysis]
    overall_confidence: float
    hallucination_count: int
    supported_claim_ratio: float
    needs_regeneration: bool
    regeneration_feedback: str = ""
    source_coverage: float = 0.0  # How much of the sources were used


class SelfRAG:
    """
    Self-Reflective RAG implementation for response verification.

    Self-RAG workflow:
    1. Generate initial response using retrieved documents
    2. Extract factual claims from response
    3. Verify each claim against source documents
    4. Calculate confidence and detect hallucinations
    5. If confidence is low, regenerate with focused retrieval
    """

    def __init__(
        self,
        support_threshold: float = 0.6,
        min_supported_ratio: float = 0.7,
        max_hallucinations: int = 2,
        enable_regeneration: bool = True,
        max_regeneration_attempts: int = 2,
    ):
        """
        Initialize Self-RAG.

        Args:
            support_threshold: Minimum confidence for a claim to be "supported"
            min_supported_ratio: Minimum ratio of supported claims for valid response
            max_hallucinations: Maximum allowed hallucinations before regeneration
            enable_regeneration: Whether to regenerate responses with issues
            max_regeneration_attempts: Maximum regeneration attempts
        """
        self.support_threshold = support_threshold
        self.min_supported_ratio = min_supported_ratio
        self.max_hallucinations = max_hallucinations
        self.enable_regeneration = enable_regeneration
        self.max_regeneration_attempts = max_regeneration_attempts

    async def verify_response(
        self,
        response: str,
        sources: List[SearchResult],
        query: str,
        llm: Optional[Any] = None,
    ) -> SelfRAGResult:
        """
        Verify a generated response against source documents.

        Args:
            response: Generated response to verify
            sources: Source documents used for generation
            query: Original user query
            llm: LLM for claim extraction and verification

        Returns:
            SelfRAGResult with verification details
        """
        if not response or not sources:
            return SelfRAGResult(
                original_response=response,
                verified_response=None,
                claims=[],
                overall_confidence=0.0,
                hallucination_count=0,
                supported_claim_ratio=0.0,
                needs_regeneration=True,
                regeneration_feedback="No response or sources to verify",
            )

        # Step 1: Extract claims from response
        if llm:
            claims_text = await self._extract_claims_with_llm(response, llm)
        else:
            claims_text = self._extract_claims_heuristic(response)

        if not claims_text:
            # No claims extracted - response might be just acknowledgment
            return SelfRAGResult(
                original_response=response,
                verified_response=None,
                claims=[],
                overall_confidence=0.8,
                hallucination_count=0,
                supported_claim_ratio=1.0,
                needs_regeneration=False,
            )

        # Step 2: Verify each claim against sources
        if llm:
            claim_analyses = await self._verify_claims_with_llm(
                claims_text, sources, llm
            )
        else:
            claim_analyses = self._verify_claims_heuristic(claims_text, sources)

        # Step 3: Calculate metrics
        supported_count = sum(
            1 for c in claim_analyses
            if c.support_level in (SupportLevel.FULLY_SUPPORTED, SupportLevel.PARTIALLY_SUPPORTED)
        )
        hallucination_count = sum(
            1 for c in claim_analyses
            if c.support_level == SupportLevel.NOT_SUPPORTED or c.is_hallucination
        )
        contradicted_count = sum(
            1 for c in claim_analyses
            if c.support_level == SupportLevel.CONTRADICTED
        )

        total_claims = len(claim_analyses)
        supported_ratio = supported_count / total_claims if total_claims > 0 else 1.0

        # Calculate overall confidence
        if total_claims > 0:
            avg_confidence = sum(c.confidence for c in claim_analyses) / total_claims
        else:
            avg_confidence = 0.8

        # Adjust confidence for hallucinations and contradictions
        if contradicted_count > 0:
            avg_confidence *= 0.5
        if hallucination_count > self.max_hallucinations:
            avg_confidence *= 0.7

        # Step 4: Determine if regeneration needed
        needs_regeneration = (
            supported_ratio < self.min_supported_ratio or
            hallucination_count > self.max_hallucinations or
            contradicted_count > 0
        )

        # Generate feedback for regeneration
        regeneration_feedback = ""
        if needs_regeneration:
            issues = []
            if hallucination_count > 0:
                hallucinated_claims = [
                    c.claim for c in claim_analyses
                    if c.support_level == SupportLevel.NOT_SUPPORTED
                ]
                issues.append(f"Unsupported claims: {hallucinated_claims[:3]}")
            if contradicted_count > 0:
                contradicted_claims = [
                    c.claim for c in claim_analyses
                    if c.support_level == SupportLevel.CONTRADICTED
                ]
                issues.append(f"Contradicted claims: {contradicted_claims[:2]}")
            regeneration_feedback = "; ".join(issues)

        # Calculate source coverage
        used_source_ids = set()
        for claim in claim_analyses:
            used_source_ids.update(claim.supporting_sources)
        source_coverage = len(used_source_ids) / len(sources) if sources else 0.0

        logger.info(
            "Self-RAG verification complete",
            total_claims=total_claims,
            supported=supported_count,
            hallucinations=hallucination_count,
            contradicted=contradicted_count,
            confidence=avg_confidence,
            needs_regeneration=needs_regeneration,
        )

        return SelfRAGResult(
            original_response=response,
            verified_response=None,  # Will be set if regeneration happens
            claims=claim_analyses,
            overall_confidence=avg_confidence,
            hallucination_count=hallucination_count,
            supported_claim_ratio=supported_ratio,
            needs_regeneration=needs_regeneration,
            regeneration_feedback=regeneration_feedback,
            source_coverage=source_coverage,
        )

    async def verify_and_regenerate(
        self,
        response: str,
        sources: List[SearchResult],
        query: str,
        llm: Any,
        generation_callback: Any = None,
    ) -> SelfRAGResult:
        """
        Verify response and regenerate if needed.

        Args:
            response: Initial generated response
            sources: Source documents
            query: Original user query
            llm: LLM for verification and regeneration
            generation_callback: Async function to generate new response

        Returns:
            SelfRAGResult with potentially regenerated response
        """
        result = await self.verify_response(response, sources, query, llm)

        if not result.needs_regeneration or not self.enable_regeneration:
            return result

        if not generation_callback:
            logger.warning("Regeneration needed but no callback provided")
            return result

        # Attempt regeneration with feedback
        for attempt in range(self.max_regeneration_attempts):
            logger.info(
                "Attempting response regeneration",
                attempt=attempt + 1,
                feedback_preview=result.regeneration_feedback[:100],
            )

            # Generate new response with focused instructions
            regeneration_prompt = self._build_regeneration_prompt(
                query=query,
                original_response=response,
                feedback=result.regeneration_feedback,
                sources=sources,
            )

            try:
                new_response = await generation_callback(
                    query=regeneration_prompt,
                    sources=sources,
                )

                # Verify the new response
                new_result = await self.verify_response(
                    new_response, sources, query, llm
                )

                # Check if improved
                if (
                    new_result.overall_confidence > result.overall_confidence or
                    new_result.hallucination_count < result.hallucination_count
                ):
                    new_result.verified_response = new_response
                    new_result.original_response = response
                    return new_result

            except Exception as e:
                logger.warning(
                    "Regeneration attempt failed",
                    attempt=attempt + 1,
                    error=str(e),
                )

        # Return original result if regeneration didn't improve
        return result

    def _extract_claims_heuristic(self, response: str) -> List[str]:
        """
        Extract claims from response using heuristics.

        Looks for:
        - Sentences with factual statements
        - Bullet points
        - Numbered items
        """
        claims = []

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', response)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            # Skip questions
            if sentence.endswith('?'):
                continue

            # Skip meta-statements
            skip_patterns = [
                r'^(I think|I believe|In my opinion|Perhaps|Maybe)',
                r'^(According to|Based on|As mentioned)',
                r'^(Here are|Let me|I\'ll)',
            ]
            if any(re.match(p, sentence, re.I) for p in skip_patterns):
                continue

            # Look for factual indicators
            factual_indicators = [
                r'\b(is|are|was|were|has|have|had)\b',
                r'\b(\d+%|\d+ percent)\b',
                r'\b(always|never|every|all|none)\b',
                r'\b(increase|decrease|grow|decline)\b',
            ]
            if any(re.search(p, sentence, re.I) for p in factual_indicators):
                claims.append(sentence)

        return claims[:10]  # Limit to 10 claims

    async def _extract_claims_with_llm(
        self,
        response: str,
        llm: Any,
    ) -> List[str]:
        """Extract claims using LLM for better accuracy."""
        prompt = f"""Extract the main factual claims from this text. A claim is a specific, verifiable statement of fact.

TEXT:
{response[:2000]}

Return ONLY a JSON array of claim strings. Include:
- Specific facts (numbers, dates, names)
- Causal relationships
- Comparisons or rankings
- Definitions or classifications

Example output: ["Claim 1", "Claim 2", "Claim 3"]

JSON array of claims:"""

        try:
            from langchain_core.messages import HumanMessage

            result = await llm.ainvoke([HumanMessage(content=prompt)])
            content = result.content.strip()

            # Parse JSON array
            if content.startswith('['):
                claims = json.loads(content)
                return claims[:10]

            # Try to extract array from response (non-greedy to avoid spanning multiple arrays)
            for match in re.finditer(r'\[.*?\]', content, re.DOTALL):
                try:
                    claims = json.loads(match.group())
                    if isinstance(claims, list):
                        return claims[:10]
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.warning("LLM claim extraction failed", error=str(e))

        # Fallback to heuristic
        return self._extract_claims_heuristic(response)

    def _verify_claims_heuristic(
        self,
        claims: List[str],
        sources: List[SearchResult],
    ) -> List[ClaimAnalysis]:
        """Verify claims using text matching heuristics."""
        analyses = []

        # Build source text index
        source_texts = {
            s.chunk_id: s.content.lower()
            for s in sources
        }
        all_source_text = " ".join(source_texts.values())

        for claim in claims:
            claim_lower = claim.lower()
            claim_words = set(re.findall(r'\b\w+\b', claim_lower))
            claim_words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}

            # Check word overlap with sources
            supporting_sources = []
            max_overlap = 0.0

            for chunk_id, source_text in source_texts.items():
                source_words = set(re.findall(r'\b\w+\b', source_text))
                overlap = len(claim_words & source_words) / max(len(claim_words), 1)

                if overlap > 0.5:
                    supporting_sources.append(chunk_id)
                    max_overlap = max(max_overlap, overlap)

            # Determine support level
            if max_overlap >= 0.8:
                support_level = SupportLevel.FULLY_SUPPORTED
                confidence = 0.9
            elif max_overlap >= 0.5:
                support_level = SupportLevel.PARTIALLY_SUPPORTED
                confidence = 0.6
            else:
                support_level = SupportLevel.NOT_SUPPORTED
                confidence = 0.3

            analyses.append(ClaimAnalysis(
                claim=claim,
                support_level=support_level,
                confidence=confidence,
                supporting_sources=supporting_sources,
                evidence=f"Word overlap: {max_overlap:.2f}",
                is_hallucination=support_level == SupportLevel.NOT_SUPPORTED,
            ))

        return analyses

    async def _verify_claims_with_llm(
        self,
        claims: List[str],
        sources: List[SearchResult],
        llm: Any,
    ) -> List[ClaimAnalysis]:
        """Verify claims using LLM for better accuracy."""
        # Prepare source context
        source_context = "\n\n".join([
            f"[Source {i+1} - {s.chunk_id}]:\n{s.content[:500]}"
            for i, s in enumerate(sources[:5])  # Limit sources for prompt size
        ])

        # Verify claims in batches
        analyses = []
        batch_size = 3

        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]
            batch_analyses = await asyncio.gather(*[
                self._verify_single_claim_llm(claim, source_context, sources, llm)
                for claim in batch
            ])
            analyses.extend(batch_analyses)

        return analyses

    async def _verify_single_claim_llm(
        self,
        claim: str,
        source_context: str,
        sources: List[SearchResult],
        llm: Any,
    ) -> ClaimAnalysis:
        """Verify a single claim using LLM."""
        prompt = f"""Verify if this claim is supported by the source documents.

CLAIM: {claim}

SOURCE DOCUMENTS:
{source_context}

Evaluate the claim and respond with JSON:
{{
    "support_level": "fully_supported" | "partially_supported" | "not_supported" | "contradicted",
    "confidence": 0.0-1.0,
    "evidence": "Quote or reference from sources that supports/contradicts",
    "source_ids": ["chunk_id if claim is supported"]
}}

JSON response:"""

        try:
            from langchain_core.messages import HumanMessage

            result = await llm.ainvoke([HumanMessage(content=prompt)])
            content = result.content.strip()

            # Parse JSON
            if '{' in content:
                start = content.index('{')
                end = content.rindex('}') + 1
                data = json.loads(content[start:end])

                support_str = data.get("support_level", "not_supported")
                support_level = SupportLevel(support_str)

                return ClaimAnalysis(
                    claim=claim,
                    support_level=support_level,
                    confidence=float(data.get("confidence", 0.5)),
                    supporting_sources=data.get("source_ids", []),
                    evidence=data.get("evidence", ""),
                    is_hallucination=support_level == SupportLevel.NOT_SUPPORTED,
                )

        except Exception as e:
            logger.warning("LLM claim verification failed", claim=claim[:50], error=str(e))

        # Fallback to heuristic for this claim
        return self._verify_claims_heuristic([claim], sources)[0]

    def _build_regeneration_prompt(
        self,
        query: str,
        original_response: str,
        feedback: str,
        sources: List[SearchResult],
    ) -> str:
        """Build prompt for regenerating response with feedback."""
        source_summary = "\n".join([
            f"- {s.document_name}: {s.content[:200]}..."
            for s in sources[:5]
        ])

        return f"""Please answer the following question using ONLY information from the provided sources.

IMPORTANT: Your previous response had issues: {feedback}

Please regenerate a response that:
1. Only includes information directly from the sources
2. Does not add any facts not present in the sources
3. Clearly indicates when information is uncertain

QUESTION: {query}

SOURCES:
{source_summary}

Provide a factual answer based only on the sources above:"""


# Singleton instance
_self_rag_instance: Optional[SelfRAG] = None


def get_self_rag(
    support_threshold: float = 0.6,
    min_supported_ratio: float = 0.7,
    enable_regeneration: bool = True,
) -> SelfRAG:
    """Get or create the Self-RAG singleton."""
    global _self_rag_instance
    if _self_rag_instance is None:
        _self_rag_instance = SelfRAG(
            support_threshold=support_threshold,
            min_supported_ratio=min_supported_ratio,
            enable_regeneration=enable_regeneration,
        )
    return _self_rag_instance
