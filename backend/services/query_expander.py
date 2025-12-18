"""
AIDocumentIndexer - Query Expansion Service
============================================

Generates query variations/paraphrases to improve RAG retrieval accuracy.
Research shows query expansion can improve retrieval accuracy by 8-12%.

Features:
- LLM-based query paraphrase generation
- Synonym expansion
- Aspect-based expansion (breaks complex queries into sub-queries)
- All features are optional via configuration
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import structlog

from langchain_core.messages import HumanMessage, SystemMessage

logger = structlog.get_logger(__name__)


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion."""

    # Master toggle
    enabled: bool = False

    # Number of query variations to generate
    expansion_count: int = 2  # Default: generate 2 additional variations

    # Model settings
    model: str = "gpt-4o-mini"  # Cost-effective model
    temperature: float = 0.7  # Higher temp for more diverse variations
    max_tokens: int = 200

    # Expansion strategies
    enable_paraphrase: bool = True  # Generate paraphrased versions
    enable_synonyms: bool = True  # Replace key terms with synonyms
    enable_decomposition: bool = False  # Break complex queries into sub-queries (more expensive)

    # Provider config
    provider: str = "openai"

    # Caching
    cache_expansions: bool = True  # Cache expanded queries to avoid re-processing


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    expansion_type: str = "paraphrase"
    tokens_used: int = 0
    model: str = ""


QUERY_EXPANSION_PROMPT = """You are a search query expansion assistant. Your task is to generate
alternative versions of the user's query that might help find relevant documents.

Generate {count} alternative queries that:
1. Capture the same intent as the original query
2. Use different words, synonyms, or phrasing
3. Might match different document formulations
4. Are still concise and search-friendly

Original query: {query}

Respond with ONLY the alternative queries, one per line, without numbering or explanation.
"""


QUERY_DECOMPOSITION_PROMPT = """You are a search query analyst. The user's query is complex and
might benefit from being broken down into sub-queries.

Break the following query into 2-3 simpler, focused sub-queries that together cover the
original intent. Each sub-query should search for a specific aspect.

Original query: {query}

Respond with ONLY the sub-queries, one per line, without numbering or explanation.
"""


class QueryExpander:
    """
    Query expansion service for improved RAG retrieval.

    Generates query variations to increase recall while maintaining precision.
    Uses LLM to generate semantically similar paraphrases.
    """

    def __init__(self, config: Optional[QueryExpansionConfig] = None):
        """
        Initialize the query expander.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or QueryExpansionConfig()
        self._llm = None
        self._expansion_cache: Dict[str, List[str]] = {}

        logger.info(
            "QueryExpander initialized",
            enabled=self.config.enabled,
            expansion_count=self.config.expansion_count,
            model=self.config.model,
        )

    async def _get_llm(self):
        """Get or create LLM instance for query expansion."""
        if self._llm is None:
            try:
                from backend.services.llm import EnhancedLLMFactory

                self._llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                    operation="query_expansion",
                    track_usage=True,
                )
            except Exception as e:
                logger.warning(
                    "Failed to get LLM from factory, using direct import",
                    error=str(e),
                )
                # Fallback to direct import
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

        return self._llm

    def should_expand(self, query: str) -> bool:
        """
        Check if a query should be expanded.

        Args:
            query: The search query

        Returns:
            True if query expansion is enabled and query is suitable
        """
        if not self.config.enabled:
            return False

        # Don't expand very short queries (less than 3 words)
        if len(query.split()) < 3:
            logger.debug("Query too short for expansion", query=query)
            return False

        # Don't expand very long queries (already specific)
        if len(query.split()) > 30:
            logger.debug("Query too long for expansion", query=query)
            return False

        return True

    async def expand_query(
        self,
        query: str,
        expansion_type: Optional[str] = None,
    ) -> ExpandedQuery:
        """
        Expand a query into multiple variations.

        Args:
            query: Original search query
            expansion_type: Type of expansion ("paraphrase", "decomposition", or None for default)

        Returns:
            ExpandedQuery with original and expanded queries
        """
        if not self.config.enabled:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[],
                expansion_type="none",
            )

        # Check cache first
        cache_key = f"{query}:{expansion_type or 'default'}"
        if self.config.cache_expansions and cache_key in self._expansion_cache:
            logger.debug("Using cached expansion", query=query)
            return ExpandedQuery(
                original_query=query,
                expanded_queries=self._expansion_cache[cache_key],
                expansion_type=expansion_type or "paraphrase",
                tokens_used=0,
                model=self.config.model,
            )

        logger.info(
            "Expanding query",
            query=query,
            expansion_type=expansion_type,
        )

        # Determine expansion strategy
        if expansion_type == "decomposition" and self.config.enable_decomposition:
            expanded = await self._decompose_query(query)
            exp_type = "decomposition"
        else:
            expanded = await self._paraphrase_query(query)
            exp_type = "paraphrase"

        # Cache the results
        if self.config.cache_expansions:
            self._expansion_cache[cache_key] = expanded

        # Estimate tokens used
        tokens_used = (len(query) + sum(len(e) for e in expanded)) // 4

        logger.info(
            "Query expanded",
            original=query,
            expansions=len(expanded),
            tokens_used=tokens_used,
        )

        return ExpandedQuery(
            original_query=query,
            expanded_queries=expanded,
            expansion_type=exp_type,
            tokens_used=tokens_used,
            model=self.config.model,
        )

    async def _paraphrase_query(self, query: str) -> List[str]:
        """
        Generate paraphrased versions of the query.

        Args:
            query: Original query

        Returns:
            List of paraphrased queries
        """
        llm = await self._get_llm()

        prompt = QUERY_EXPANSION_PROMPT.format(
            count=self.config.expansion_count,
            query=query,
        )

        try:
            messages = [
                SystemMessage(content="You are a helpful search query expansion assistant."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse response - each line is a query variation
            expanded = [
                line.strip()
                for line in content.strip().split('\n')
                if line.strip() and len(line.strip()) > 5
            ]

            # Limit to configured count
            return expanded[:self.config.expansion_count]

        except Exception as e:
            logger.error("Failed to expand query", error=str(e))
            return []

    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: Original complex query

        Returns:
            List of sub-queries
        """
        llm = await self._get_llm()

        prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)

        try:
            messages = [
                SystemMessage(content="You are a helpful query analyst."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse response - each line is a sub-query
            sub_queries = [
                line.strip()
                for line in content.strip().split('\n')
                if line.strip() and len(line.strip()) > 5
            ]

            return sub_queries[:3]  # Max 3 sub-queries

        except Exception as e:
            logger.error("Failed to decompose query", error=str(e))
            return []

    async def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Expand query by replacing key terms with synonyms.

        This is a lighter-weight alternative to full LLM expansion.

        Args:
            query: Original query

        Returns:
            List of synonym-expanded queries
        """
        if not self.config.enable_synonyms:
            return []

        # Common search term synonyms (can be extended)
        SYNONYMS = {
            "find": ["search", "locate", "discover", "identify"],
            "show": ["display", "present", "list", "reveal"],
            "how": ["what way", "method", "approach"],
            "create": ["make", "build", "generate", "develop"],
            "delete": ["remove", "erase", "eliminate"],
            "update": ["modify", "change", "edit", "revise"],
            "get": ["retrieve", "obtain", "fetch", "acquire"],
            "error": ["issue", "problem", "bug", "failure"],
            "fix": ["resolve", "solve", "repair", "correct"],
            "best": ["optimal", "top", "recommended", "ideal"],
            "quick": ["fast", "rapid", "efficient", "speedy"],
        }

        expanded = []
        query_lower = query.lower()

        for word, synonyms in SYNONYMS.items():
            if word in query_lower:
                for syn in synonyms[:2]:  # Use first 2 synonyms only
                    new_query = query_lower.replace(word, syn)
                    if new_query != query_lower:
                        expanded.append(new_query)

        return expanded[:self.config.expansion_count]

    def get_all_queries(self, expansion_result: ExpandedQuery) -> List[str]:
        """
        Get all queries (original + expanded) for search.

        Args:
            expansion_result: Result from expand_query()

        Returns:
            List containing original query followed by expanded queries
        """
        all_queries = [expansion_result.original_query]
        all_queries.extend(expansion_result.expanded_queries)
        return all_queries

    def clear_cache(self):
        """Clear the expansion cache."""
        self._expansion_cache.clear()
        logger.info("Query expansion cache cleared")


# Singleton instance
_query_expander: Optional[QueryExpander] = None


def get_query_expander(config: Optional[QueryExpansionConfig] = None) -> QueryExpander:
    """
    Get or create the query expander singleton.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        QueryExpander singleton instance
    """
    global _query_expander
    if _query_expander is None:
        _query_expander = QueryExpander(config)
    return _query_expander
