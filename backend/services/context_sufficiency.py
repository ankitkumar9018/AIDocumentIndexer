"""
AIDocumentIndexer - Context Sufficiency Checker
================================================

Validates whether retrieved contexts are sufficient to answer a query.
Research shows this reduces hallucinations by 25-40% (Hallucination Mitigation Survey 2025).

Key features:
1. Coverage scoring - measures how well contexts cover the query
2. Conflict detection - identifies contradictory information in sources
3. Gap identification - highlights missing aspects the LLM might hallucinate about
4. Confidence recommendations - proceed vs retrieve more
5. Abstention threshold tuning via settings
6. "I don't know" response generation for insufficient contexts

Reference: https://arxiv.org/html/2510.24476v1
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import structlog
import re

logger = structlog.get_logger(__name__)


@dataclass
class ContextSufficiencyResult:
    """Result of context sufficiency check."""
    is_sufficient: bool
    coverage_score: float  # 0-1, how well contexts cover the query
    has_conflicts: bool  # Whether sources contain conflicting information
    conflicts: List[Dict[str, Any]] = field(default_factory=list)  # List of detected conflicts
    missing_aspects: List[str] = field(default_factory=list)  # Aspects query asks about but contexts don't cover
    recommendation: str = "proceed"  # "proceed", "retrieve_more", "warn_user", "abstain"
    confidence_level: str = "high"  # "high", "medium", "low"
    explanation: str = ""  # Human-readable explanation
    should_abstain: bool = False  # Whether to respond with "I don't know"
    abstention_response: Optional[str] = None  # Pre-generated "I don't know" response

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "is_sufficient": self.is_sufficient,
            "coverage_score": self.coverage_score,
            "has_conflicts": self.has_conflicts,
            "conflicts": self.conflicts,
            "missing_aspects": self.missing_aspects,
            "recommendation": self.recommendation,
            "confidence_level": self.confidence_level,
            "explanation": self.explanation,
            "should_abstain": self.should_abstain,
            "abstention_response": self.abstention_response,
        }


# Cached settings
_sufficiency_settings: Optional[Dict[str, Any]] = None


async def _get_sufficiency_settings() -> Dict[str, Any]:
    """Get context sufficiency settings from database."""
    global _sufficiency_settings

    if _sufficiency_settings is not None:
        return _sufficiency_settings

    try:
        from backend.services.settings import get_settings_service

        settings = get_settings_service()
        coverage_threshold = await settings.get_setting("rag.context_sufficiency_threshold")
        abstention_threshold = await settings.get_setting("rag.abstention_threshold")
        enable_abstention = await settings.get_setting("rag.enable_abstention")
        conflict_detection = await settings.get_setting("rag.conflict_detection_enabled")

        _sufficiency_settings = {
            "coverage_threshold": coverage_threshold if coverage_threshold is not None else 0.5,
            "abstention_threshold": abstention_threshold if abstention_threshold is not None else 0.3,
            "enable_abstention": enable_abstention if enable_abstention is not None else True,
            "conflict_detection": conflict_detection if conflict_detection is not None else True,
        }

        return _sufficiency_settings
    except Exception as e:
        logger.debug("Could not load context sufficiency settings, using defaults", error=str(e))
        return {
            "coverage_threshold": 0.5,
            "abstention_threshold": 0.3,
            "enable_abstention": True,
            "conflict_detection": True,
        }


def invalidate_sufficiency_settings():
    """Invalidate cached settings (call after settings change)."""
    global _sufficiency_settings
    _sufficiency_settings = None


class ContextSufficiencyChecker:
    """
    Checks if retrieved contexts are sufficient to answer a query.

    Uses multiple heuristics:
    1. Keyword overlap - simple but effective baseline
    2. Semantic similarity - embedding-based coverage
    3. Entity matching - key entities from query found in context
    4. Conflict detection - contradictory statements
    """

    def __init__(
        self,
        coverage_threshold: float = 0.5,
        conflict_threshold: float = 0.8,
        use_llm_for_conflicts: bool = False,  # Set True to use LLM for conflict detection
    ):
        """
        Initialize context sufficiency checker.

        Args:
            coverage_threshold: Minimum coverage score to consider context sufficient (0-1)
            conflict_threshold: Similarity threshold for potential conflicts (0-1)
            use_llm_for_conflicts: Whether to use LLM for advanced conflict detection
        """
        self.coverage_threshold = coverage_threshold
        self.conflict_threshold = conflict_threshold
        self.use_llm_for_conflicts = use_llm_for_conflicts
        self._embedding_service = None

    async def _get_embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from backend.services.embeddings import get_embedding_service
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    async def check_sufficiency(
        self,
        query: str,
        contexts: List[str],
        context_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> ContextSufficiencyResult:
        """
        Check if retrieved contexts are sufficient to answer the query.

        Args:
            query: The user's query
            contexts: List of retrieved context strings
            context_metadata: Optional metadata for each context (document name, etc.)

        Returns:
            ContextSufficiencyResult with coverage score and recommendations
        """
        # Load settings
        settings = await _get_sufficiency_settings()
        coverage_threshold = settings.get("coverage_threshold", self.coverage_threshold)
        abstention_threshold = settings.get("abstention_threshold", 0.3)
        enable_abstention = settings.get("enable_abstention", True)

        if not contexts:
            abstention_response = None
            should_abstain = False
            if enable_abstention:
                should_abstain = True
                abstention_response = self._generate_abstention_response(
                    query, [], "No relevant documents found."
                )

            return ContextSufficiencyResult(
                is_sufficient=False,
                coverage_score=0.0,
                has_conflicts=False,
                missing_aspects=self._extract_query_aspects(query),
                recommendation="abstain" if should_abstain else "retrieve_more",
                confidence_level="low",
                explanation="No relevant documents found for this query.",
                should_abstain=should_abstain,
                abstention_response=abstention_response,
            )

        # Run checks in parallel for efficiency
        coverage_task = self._calculate_coverage(query, contexts)
        conflict_task = self._detect_conflicts(contexts, context_metadata)
        gaps_task = self._identify_gaps(query, contexts)

        coverage_score, keyword_coverage = await coverage_task
        conflicts, has_conflicts = await conflict_task
        missing_aspects = await gaps_task

        # Determine overall sufficiency using settings threshold
        is_sufficient = coverage_score >= coverage_threshold and not has_conflicts

        # Determine if we should abstain (below abstention threshold)
        should_abstain = False
        abstention_response = None
        if enable_abstention and coverage_score < abstention_threshold:
            should_abstain = True
            abstention_response = self._generate_abstention_response(
                query, missing_aspects, f"Low coverage ({coverage_score:.0%})"
            )

        # Determine confidence level
        if coverage_score >= 0.65 and not has_conflicts:
            confidence_level = "high"
        elif coverage_score >= 0.35 or (coverage_score >= 0.25 and keyword_coverage >= 0.5):
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Generate recommendation
        if should_abstain:
            recommendation = "abstain"
        elif is_sufficient:
            recommendation = "proceed"
        elif has_conflicts:
            recommendation = "warn_user"
        else:
            recommendation = "retrieve_more"

        # Generate explanation
        explanation = self._generate_explanation(
            coverage_score, keyword_coverage, has_conflicts, missing_aspects, len(contexts)
        )

        return ContextSufficiencyResult(
            is_sufficient=is_sufficient,
            coverage_score=coverage_score,
            has_conflicts=has_conflicts,
            conflicts=conflicts,
            missing_aspects=missing_aspects,
            recommendation=recommendation,
            confidence_level=confidence_level,
            explanation=explanation,
            should_abstain=should_abstain,
            abstention_response=abstention_response,
        )

    async def _calculate_coverage(
        self,
        query: str,
        contexts: List[str],
    ) -> Tuple[float, float]:
        """
        Calculate how well contexts cover the query.

        Uses keyword coverage and semantic similarity, with a scaling adjustment
        because embedding similarity between a question and its answer text
        typically maxes out at 0.5-0.7 even when the content is perfectly relevant.

        Returns:
            Tuple of (combined_coverage, keyword_coverage)
        """
        # 1. Keyword-based coverage (fast baseline)
        query_keywords = self._extract_keywords(query)
        combined_context = " ".join(contexts).lower()

        if query_keywords:
            keywords_found = sum(1 for kw in query_keywords if kw.lower() in combined_context)
            keyword_coverage = keywords_found / len(query_keywords)
        else:
            keyword_coverage = 0.5  # Default if no keywords extracted

        # 2. Semantic coverage (embedding similarity)
        try:
            embedding_service = await self._get_embedding_service()

            # Embed query and contexts
            query_embedding = await embedding_service.embed_query(query)

            # Calculate similarity with each context
            similarities = []
            for context in contexts[:10]:  # Limit to first 10 for efficiency
                context_embedding = await embedding_service.embed_query(context[:1000])  # Truncate long contexts
                similarity = self._cosine_similarity(query_embedding, context_embedding)
                similarities.append(similarity)

            # Use max similarity as coverage (at least one context should be highly relevant)
            raw_semantic = max(similarities) if similarities else 0.0

            # Scale semantic similarity: question-to-answer similarity typically
            # peaks at 0.5-0.7 even for perfect matches. Rescale so that 0.4+ maps
            # to a more meaningful range: 0.4 → 0.6, 0.55 → 0.8, 0.7+ → 1.0
            if raw_semantic >= 0.4:
                semantic_coverage = min(1.0, 0.6 + (raw_semantic - 0.4) * (0.4 / 0.3))
            else:
                semantic_coverage = raw_semantic * 1.5  # Below 0.4 stays low

            # Also boost if multiple contexts are relevant (breadth signal)
            if len(similarities) >= 3:
                avg_sim = sum(similarities) / len(similarities)
                if avg_sim > 0.3:
                    # Multiple relevant contexts = higher confidence
                    breadth_bonus = min(0.1, (avg_sim - 0.3) * 0.5)
                    semantic_coverage = min(1.0, semantic_coverage + breadth_bonus)

            # Combine semantic and keyword coverage
            combined_coverage = 0.6 * semantic_coverage + 0.4 * keyword_coverage

        except Exception as e:
            logger.warning("Semantic coverage calculation failed, using keyword only", error=str(e))
            combined_coverage = keyword_coverage
            semantic_coverage = keyword_coverage

        return combined_coverage, keyword_coverage

    async def _detect_conflicts(
        self,
        contexts: List[str],
        context_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Detect conflicting information in retrieved contexts.

        Returns:
            Tuple of (list of conflicts, has_conflicts bool)
        """
        conflicts = []

        if len(contexts) < 2:
            return conflicts, False

        # Extract numerical statements for comparison
        numerical_statements = []
        for i, context in enumerate(contexts):
            numbers = self._extract_numerical_statements(context)
            for num_statement in numbers:
                numerical_statements.append({
                    "context_idx": i,
                    "statement": num_statement["statement"],
                    "value": num_statement["value"],
                    "entity": num_statement.get("entity", ""),
                    "source": context_metadata[i].get("document_name", f"Source {i+1}") if context_metadata else f"Source {i+1}",
                })

        # Compare numerical statements for conflicts
        for i, stmt1 in enumerate(numerical_statements):
            for stmt2 in numerical_statements[i+1:]:
                # Check if they're about the same entity but have different values
                if (
                    stmt1["entity"] and stmt2["entity"]
                    and self._entities_match(stmt1["entity"], stmt2["entity"])
                    and stmt1["value"] != stmt2["value"]
                ):
                    conflicts.append({
                        "type": "numerical_conflict",
                        "source1": stmt1["source"],
                        "source2": stmt2["source"],
                        "statement1": stmt1["statement"],
                        "statement2": stmt2["statement"],
                        "description": f"Different values for '{stmt1['entity']}': {stmt1['value']} vs {stmt2['value']}",
                    })

        # Also check for semantic contradictions if LLM is enabled
        if self.use_llm_for_conflicts and len(conflicts) == 0 and len(contexts) >= 2:
            llm_conflicts = await self._detect_semantic_contradictions(contexts, context_metadata)
            conflicts.extend(llm_conflicts)

        has_conflicts = len(conflicts) > 0
        return conflicts, has_conflicts

    async def _detect_semantic_contradictions(
        self,
        contexts: List[str],
        context_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to detect semantic contradictions between sources.

        This catches contradictions that aren't numerical, like:
        - "The project was successful" vs "The project failed to meet goals"
        - "Company X acquired Company Y" vs "Company Y acquired Company X"

        Args:
            contexts: List of context passages
            context_metadata: Metadata for each context

        Returns:
            List of detected semantic contradictions
        """
        from backend.services.llm import LLMFactory

        if len(contexts) < 2:
            return []

        # Build comparison prompt
        sources_text = ""
        for i, ctx in enumerate(contexts[:5]):  # Limit to 5 sources
            source_name = "Unknown"
            if context_metadata and i < len(context_metadata):
                source_name = context_metadata[i].get("document_name", f"Source {i+1}")
            sources_text += f"\n[{source_name}]:\n{ctx[:500]}\n"

        prompt = f"""Analyze these source passages for factual contradictions.

SOURCES:
{sources_text}

Look for statements that directly contradict each other, such as:
- Conflicting dates, names, or facts
- Opposite claims about the same topic
- Mutually exclusive statements

If you find contradictions, list them in this JSON format:
```json
[
  {{
    "type": "semantic_conflict",
    "source1": "Source name 1",
    "source2": "Source name 2",
    "statement1": "First conflicting claim",
    "statement2": "Second conflicting claim",
    "description": "Brief explanation of the contradiction"
  }}
]
```

If no contradictions found, return: []

CONTRADICTIONS:"""

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.1,  # Low temp for factual analysis
                max_tokens=1024,
            )

            response = await llm.ainvoke(prompt)
            output = response.content

            # Parse JSON response
            import json
            import re

            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                conflicts = json.loads(json_match.group())
                # Validate structure
                valid_conflicts = []
                for c in conflicts:
                    if all(k in c for k in ["type", "source1", "source2", "description"]):
                        valid_conflicts.append(c)
                return valid_conflicts[:5]  # Limit to 5 conflicts

        except Exception as e:
            logger.warning("LLM contradiction detection failed", error=str(e))

        return []

    async def _identify_gaps(
        self,
        query: str,
        contexts: List[str],
    ) -> List[str]:
        """
        Identify aspects of the query that contexts don't cover.

        Returns:
            List of missing aspects
        """
        missing_aspects = []

        # Extract query aspects (key topics/questions)
        query_aspects = self._extract_query_aspects(query)
        combined_context = " ".join(contexts).lower()

        for aspect in query_aspects:
            # Check if aspect is covered in any context
            aspect_keywords = self._extract_keywords(aspect)
            coverage = sum(1 for kw in aspect_keywords if kw.lower() in combined_context)

            # If less than half of keywords found, consider it a gap
            if aspect_keywords and coverage < len(aspect_keywords) * 0.5:
                missing_aspects.append(aspect)

        return missing_aspects[:5]  # Limit to top 5 gaps

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him",
            "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves",
            "and", "but", "if", "or", "because", "until", "while",
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]

        return keywords[:20]  # Limit to top 20 keywords

    def _extract_query_aspects(self, query: str) -> List[str]:
        """Extract distinct aspects/topics from a query."""
        aspects = []

        # Split by conjunctions and question words
        parts = re.split(r'\band\b|\bor\b|\?|,', query)

        for part in parts:
            part = part.strip()
            if len(part) > 5:  # Skip very short parts
                aspects.append(part)

        # If no split happened, treat whole query as one aspect
        if not aspects:
            aspects = [query]

        return aspects

    def _extract_numerical_statements(self, text: str) -> List[Dict[str, Any]]:
        """Extract statements containing numbers from text."""
        statements = []

        # Pattern: entity + number + optional unit
        patterns = [
            r'(\w+(?:\s+\w+)?)\s+(?:is|was|are|were|has|have|had|cost|costs|costed|totaled?|amounted?\s+to)\s+\$?([\d,]+(?:\.\d+)?)\s*(%|percent|dollars?|million|billion|k|K)?',
            r'([\d,]+(?:\.\d+)?)\s*(%|percent|dollars?|million|billion|k|K)?\s+(?:for|of)\s+(\w+(?:\s+\w+)?)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    statements.append({
                        "statement": match.group(0),
                        "entity": groups[0] if groups[0] else "",
                        "value": groups[1] if len(groups) > 1 else groups[0],
                    })

        return statements

    def _entities_match(self, entity1: str, entity2: str) -> bool:
        """Check if two entity names refer to the same thing."""
        e1 = entity1.lower().strip()
        e2 = entity2.lower().strip()

        # Exact match
        if e1 == e2:
            return True

        # One contains the other
        if e1 in e2 or e2 in e1:
            return True

        # High word overlap
        words1 = set(e1.split())
        words2 = set(e2.split())
        if words1 and words2:
            overlap = len(words1 & words2) / min(len(words1), len(words2))
            if overlap >= 0.5:
                return True

        return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _generate_explanation(
        self,
        coverage_score: float,
        keyword_coverage: float,
        has_conflicts: bool,
        missing_aspects: List[str],
        num_contexts: int,
    ) -> str:
        """Generate human-readable explanation of sufficiency check."""
        parts = []

        # Coverage explanation
        if coverage_score >= 0.7:
            parts.append(f"Found {num_contexts} relevant sources with high coverage ({coverage_score:.0%}).")
        elif coverage_score >= 0.4:
            parts.append(f"Found {num_contexts} sources with moderate coverage ({coverage_score:.0%}).")
        else:
            parts.append(f"Limited coverage ({coverage_score:.0%}) from {num_contexts} sources.")

        # Conflict warning
        if has_conflicts:
            parts.append("Warning: Sources contain conflicting information.")

        # Missing aspects
        if missing_aspects:
            aspects_str = ", ".join(missing_aspects[:3])
            if len(missing_aspects) > 3:
                aspects_str += f", and {len(missing_aspects) - 3} more"
            parts.append(f"May not fully cover: {aspects_str}.")

        return " ".join(parts)

    def _generate_abstention_response(
        self,
        query: str,
        missing_aspects: List[str],
        reason: str,
    ) -> str:
        """
        Generate an "I don't know" response when context is insufficient.

        This implements calibrated abstention - the model acknowledges when
        it doesn't have enough information rather than hallucinating.

        Args:
            query: The user's original query
            missing_aspects: Aspects of the query not covered by context
            reason: Brief reason for abstention

        Returns:
            A helpful "I don't know" response
        """
        # Base response acknowledging uncertainty
        response_parts = [
            "I don't have enough information in the available documents to fully answer your question."
        ]

        # Add context about what's missing
        if missing_aspects:
            if len(missing_aspects) == 1:
                response_parts.append(
                    f"Specifically, I couldn't find information about: {missing_aspects[0]}."
                )
            else:
                aspects_list = ", ".join(missing_aspects[:3])
                if len(missing_aspects) > 3:
                    aspects_list += f" (and {len(missing_aspects) - 3} more topics)"
                response_parts.append(
                    f"The documents don't appear to cover: {aspects_list}."
                )

        # Suggest next steps
        response_parts.append(
            "You might want to try uploading more relevant documents, "
            "or rephrasing your question to focus on what is covered in the existing documents."
        )

        return " ".join(response_parts)


# Singleton instance
_context_sufficiency_checker: Optional[ContextSufficiencyChecker] = None


def get_context_sufficiency_checker() -> ContextSufficiencyChecker:
    """Get or create the singleton context sufficiency checker."""
    global _context_sufficiency_checker
    if _context_sufficiency_checker is None:
        _context_sufficiency_checker = ContextSufficiencyChecker()
    return _context_sufficiency_checker
