"""
AIDocumentIndexer - Query Decomposition Service
================================================

Decomposes complex multi-step queries into simpler sub-queries
for improved RAG retrieval on complex questions.

Example:
  Input: "Compare the marketing strategies of Company A and B, and explain which is better for startups"
  Output: [
    "What are Company A's marketing strategies?",
    "What are Company B's marketing strategies?",
    "What marketing strategies work best for startups?"
  ]
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    original_query: str
    sub_queries: List[str]
    query_type: str  # "simple", "comparison", "multi_step", "aggregation"
    requires_synthesis: bool = False  # Whether results need to be combined
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryDecomposerConfig:
    """Configuration for query decomposition."""
    enabled: bool = True
    max_sub_queries: int = 5
    min_query_complexity: int = 10  # Min words to consider decomposition
    model: str = "gpt-4o-mini"  # Fast model for decomposition
    use_heuristics: bool = True  # Try rule-based decomposition first


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.

    Uses a combination of:
    1. Heuristic rules for common patterns
    2. LLM-based decomposition for complex cases
    """

    # Comparison patterns
    COMPARISON_PATTERNS = [
        r'\bcompare\b.*\band\b',
        r'\bdifference[s]?\s+between\b',
        r'\bvs\.?\b',
        r'\bversus\b',
        r'\bcompared\s+to\b',
        r'\bbetter\s+than\b',
        r'\bworse\s+than\b',
    ]

    # Multi-step patterns
    MULTI_STEP_PATTERNS = [
        r'\band\s+then\b',
        r'\bafter\s+that\b',
        r'\bfirst.*then\b',
        r'\bexplain.*and.*describe\b',
        r'\bwhat.*and.*how\b',
        r'\bwhy.*and.*what\b',
    ]

    # Aggregation patterns
    AGGREGATION_PATTERNS = [
        r'\ball\s+the\b',
        r'\blist\s+all\b',
        r'\beverything\s+about\b',
        r'\bsummarize\b.*\band\b',
        r'\boverview\s+of\b.*\band\b',
    ]

    def __init__(self, config: Optional[QueryDecomposerConfig] = None):
        self.config = config or QueryDecomposerConfig()
        self._llm = None

    def _get_llm(self):
        """Lazy load LLM for decomposition."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.config.model,
                temperature=0,
                max_tokens=500,
            )
        return self._llm

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query based on patterns."""
        query_lower = query.lower()

        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower):
                return "comparison"

        for pattern in self.MULTI_STEP_PATTERNS:
            if re.search(pattern, query_lower):
                return "multi_step"

        for pattern in self.AGGREGATION_PATTERNS:
            if re.search(pattern, query_lower):
                return "aggregation"

        return "simple"

    def _heuristic_decompose(self, query: str) -> Optional[DecomposedQuery]:
        """Try to decompose using heuristic rules."""
        query_type = self._detect_query_type(query)

        if query_type == "simple":
            return None  # No decomposition needed

        sub_queries = []

        if query_type == "comparison":
            # Extract entities being compared
            # Pattern: "Compare X and Y" or "X vs Y"
            match = re.search(
                r'compare\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+?)(?:\s+and|\s+in|\s+for|[,.]|$)',
                query.lower()
            )
            if match:
                entity1, entity2 = match.groups()
                # Generate sub-queries for each entity
                base_topic = re.sub(
                    r'compare\s+.+?(?:and|vs\.?|versus|with)\s+.+?(?:\s+|$)',
                    '',
                    query.lower()
                ).strip()

                if base_topic:
                    sub_queries = [
                        f"What are {entity1.strip()}'s {base_topic}?",
                        f"What are {entity2.strip()}'s {base_topic}?",
                    ]
                else:
                    sub_queries = [
                        f"What is {entity1.strip()}?",
                        f"What is {entity2.strip()}?",
                    ]

        elif query_type == "multi_step":
            # Split on conjunctions
            parts = re.split(r'\s+and\s+then\s+|\s+and\s+also\s+|\.\s+Then\s+', query, flags=re.IGNORECASE)
            if len(parts) > 1:
                sub_queries = [p.strip() + "?" if not p.strip().endswith("?") else p.strip() for p in parts if p.strip()]

        elif query_type == "aggregation":
            # For aggregation queries, we might want to break into categories
            # This is harder to do heuristically, so we'll let LLM handle it
            return None

        if sub_queries and len(sub_queries) > 1:
            return DecomposedQuery(
                original_query=query,
                sub_queries=sub_queries[:self.config.max_sub_queries],
                query_type=query_type,
                requires_synthesis=True,
            )

        return None

    async def _llm_decompose(self, query: str) -> DecomposedQuery:
        """Use LLM to decompose complex queries."""
        prompt = f"""Analyze this query and break it down into simpler sub-queries that can be answered independently.

Query: "{query}"

If the query is simple and doesn't need decomposition, return just the original query.
If it's complex, return 2-5 simpler sub-queries.

Respond in JSON format:
{{
    "query_type": "simple|comparison|multi_step|aggregation",
    "needs_decomposition": true|false,
    "sub_queries": ["query1", "query2", ...],
    "synthesis_needed": true|false
}}

Only return the JSON, no other text."""

        try:
            llm = self._get_llm()
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            # Parse JSON response
            # Handle markdown code blocks
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)

            result = json.loads(content)

            if not result.get("needs_decomposition", False):
                return DecomposedQuery(
                    original_query=query,
                    sub_queries=[query],
                    query_type="simple",
                    requires_synthesis=False,
                )

            return DecomposedQuery(
                original_query=query,
                sub_queries=result.get("sub_queries", [query])[:self.config.max_sub_queries],
                query_type=result.get("query_type", "multi_step"),
                requires_synthesis=result.get("synthesis_needed", True),
            )

        except Exception as e:
            logger.warning("LLM decomposition failed, using original query", error=str(e))
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                query_type="simple",
                requires_synthesis=False,
            )

    async def decompose(self, query: str) -> DecomposedQuery:
        """
        Decompose a query into sub-queries if needed.

        Args:
            query: The original user query

        Returns:
            DecomposedQuery with sub-queries
        """
        if not self.config.enabled:
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                query_type="simple",
                requires_synthesis=False,
            )

        # Check if query is complex enough to warrant decomposition
        word_count = len(query.split())
        if word_count < self.config.min_query_complexity:
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                query_type="simple",
                requires_synthesis=False,
            )

        # Try heuristic decomposition first (faster, no API call)
        if self.config.use_heuristics:
            result = self._heuristic_decompose(query)
            if result:
                logger.info(
                    "Query decomposed using heuristics",
                    original=query,
                    sub_queries=result.sub_queries,
                    query_type=result.query_type,
                )
                return result

        # Fall back to LLM decomposition
        result = await self._llm_decompose(query)
        logger.info(
            "Query decomposed using LLM",
            original=query,
            sub_queries=result.sub_queries,
            query_type=result.query_type,
        )
        return result

    async def synthesize_results(
        self,
        decomposed: DecomposedQuery,
        sub_results: List[str],
    ) -> str:
        """
        Synthesize results from multiple sub-queries into a coherent answer.

        Args:
            decomposed: The decomposed query info
            sub_results: Results from each sub-query

        Returns:
            Synthesized answer
        """
        if not decomposed.requires_synthesis or len(sub_results) <= 1:
            return sub_results[0] if sub_results else ""

        prompt = f"""The user asked: "{decomposed.original_query}"

This was broken into sub-questions with the following answers:

{chr(10).join([f"Q{i+1}: {decomposed.sub_queries[i]}{chr(10)}A{i+1}: {result}" for i, result in enumerate(sub_results)])}

Please synthesize these answers into a single coherent response that directly addresses the original question.
Focus on providing a clear, comprehensive answer."""

        try:
            llm = self._get_llm()
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning("Synthesis failed, concatenating results", error=str(e))
            return "\n\n".join(sub_results)


# Singleton instance
_query_decomposer: Optional[QueryDecomposer] = None


def get_query_decomposer() -> QueryDecomposer:
    """Get the query decomposer singleton."""
    global _query_decomposer
    if _query_decomposer is None:
        _query_decomposer = QueryDecomposer()
    return _query_decomposer
