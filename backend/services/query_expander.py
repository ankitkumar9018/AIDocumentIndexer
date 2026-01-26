"""
AIDocumentIndexer - Query Expansion Service
============================================

Generates query variations/paraphrases to improve RAG retrieval accuracy.
Research shows query expansion can improve retrieval accuracy by 8-12%.

Features:
- LLM-based query paraphrase generation
- Synonym expansion
- Aspect-based expansion (breaks complex queries into sub-queries)
- Multi-query rewriting (2025): perspective-based query generation
- Step-back prompting: abstract queries for better conceptual matches
- Reciprocal Rank Fusion: merge results from multiple query variations
- All features are optional via configuration

References:
- RAG-Fusion: https://arxiv.org/abs/2402.03367
- Step-Back Prompting: https://arxiv.org/abs/2310.06117
"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import structlog

from langchain_core.messages import HumanMessage, SystemMessage

logger = structlog.get_logger(__name__)


# =============================================================================
# Phase 69: LRU Cache with TTL for Query Expansion
# =============================================================================

class TTLCache:
    """
    LRU cache with TTL expiry for query expansion results.

    Features:
    - O(1) get/set operations
    - Automatic TTL expiry
    - LRU eviction when max size reached
    - Thread-safe for async usage
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, returns None if expired or not found."""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        """Set item in cache with current timestamp."""
        # Remove if exists (to update position)
        if key in self._cache:
            del self._cache[key]

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Count valid (non-expired) entries
        current_time = time.time()
        valid_count = sum(
            1 for _, (_, ts) in self._cache.items()
            if current_time - ts <= self.ttl_seconds
        )
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


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

    # Multi-query rewriting (2025)
    enable_multi_query: bool = True  # Generate perspective-based variations
    enable_step_back: bool = True  # Generate abstracted versions for conceptual matching
    multi_query_count: int = 3  # Number of perspective variations

    # Provider config
    provider: str = "openai"

    # Caching (Phase 69: Enhanced with TTL and size limits)
    cache_expansions: bool = True  # Cache expanded queries to avoid re-processing
    cache_max_size: int = 1000  # Maximum number of cached expansions
    cache_ttl_seconds: int = 3600  # Cache TTL in seconds (1 hour)

    # Reciprocal Rank Fusion
    rrf_k: int = 60  # RRF constant (default: 60)


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    expansion_type: str = "paraphrase"
    tokens_used: int = 0
    model: str = ""


@dataclass
class MultiQueryResult:
    """Result of multi-query rewriting with perspective variations."""
    original_query: str
    perspective_queries: List[str] = field(default_factory=list)
    step_back_query: Optional[str] = None
    decomposed_queries: List[str] = field(default_factory=list)
    all_queries: List[str] = field(default_factory=list)
    tokens_used: int = 0

    def get_unique_queries(self) -> List[str]:
        """Get all unique queries for retrieval."""
        seen = set()
        unique = []
        for q in self.all_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        return unique


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


# Multi-Query Rewriting Prompts (2025)

MULTI_QUERY_PERSPECTIVES_PROMPT = """You are an AI language model assistant. Your task is to generate {count}
different versions of the given user question to retrieve relevant documents from a vector database.

By generating multiple perspectives on the user question, your goal is to help overcome some
limitations of distance-based similarity search.

Generate alternative questions that:
1. Approach the topic from different angles
2. Use different terminology or technical depth levels
3. Focus on different aspects of the question
4. Would match different types of relevant documents

Original question: {query}

Provide {count} alternative questions, one per line, without numbering:"""


STEP_BACK_PROMPT = """You are an expert at abstraction. Given a specific question, generate a more
general "step-back" question that would help retrieve background knowledge useful for answering
the original question.

The step-back question should:
1. Be more general/abstract than the original
2. Cover broader concepts or principles
3. Help find foundational information

Original question: {query}

Provide ONLY the step-back question, nothing else:"""


HYPOTHETICAL_DOCUMENT_PROMPT = """Given a question, generate a short hypothetical document passage
that would perfectly answer this question. This passage will be used to find similar real documents.

Question: {query}

Write a 2-3 sentence hypothetical answer that would appear in an ideal document:"""


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
        # Phase 69: Enhanced cache with TTL and size limits
        self._expansion_cache = TTLCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            "QueryExpander initialized",
            enabled=self.config.enabled,
            expansion_count=self.config.expansion_count,
            model=self.config.model,
            cache_max_size=self.config.cache_max_size,
            cache_ttl=self.config.cache_ttl_seconds,
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

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring (Phase 69).

        Returns:
            Dict with cache hits, misses, hit rate, and TTLCache stats
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total_requests,
            **self._expansion_cache.stats(),
        }

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

        # Check cache first (Phase 69: Enhanced TTL cache)
        cache_key = f"{query}:{expansion_type or 'default'}"
        if self.config.cache_expansions:
            cached = self._expansion_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                logger.debug(
                    "Using cached expansion",
                    query=query,
                    cache_hits=self._cache_hits,
                )
                return ExpandedQuery(
                    original_query=query,
                    expanded_queries=cached,
                    expansion_type=expansion_type or "paraphrase",
                    tokens_used=0,
                    model=self.config.model,
                )
            self._cache_misses += 1

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

        # Cache the results (Phase 69: Enhanced TTL cache)
        if self.config.cache_expansions:
            self._expansion_cache.set(cache_key, expanded)

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

    # =========================================================================
    # Multi-Query Rewriting (2025)
    # =========================================================================

    async def multi_query_rewrite(
        self,
        query: str,
        include_step_back: Optional[bool] = None,
        include_decomposition: Optional[bool] = None,
    ) -> MultiQueryResult:
        """
        Generate multiple query perspectives for improved retrieval.

        This implements RAG-Fusion style multi-query rewriting with:
        - Perspective variations (different angles on the same question)
        - Step-back queries (abstracted for conceptual matching)
        - Query decomposition (for complex multi-part questions)

        Args:
            query: Original user query
            include_step_back: Override config for step-back generation
            include_decomposition: Override config for decomposition

        Returns:
            MultiQueryResult with all generated queries
        """
        use_step_back = include_step_back if include_step_back is not None else self.config.enable_step_back
        use_decomposition = include_decomposition if include_decomposition is not None else self.config.enable_decomposition

        # Check cache (Phase 69: Enhanced TTL cache)
        cache_key = f"multi:{query}:{use_step_back}:{use_decomposition}"
        if self.config.cache_expansions:
            cached = self._expansion_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return MultiQueryResult(
                    original_query=query,
                    perspective_queries=cached.get("perspectives", []),
                    step_back_query=cached.get("step_back"),
                    decomposed_queries=cached.get("decomposed", []),
                    all_queries=cached.get("all", [query]),
                    tokens_used=0,
                )
            self._cache_misses += 1

        logger.info(
            "Multi-query rewriting",
            query=query[:100],
            step_back=use_step_back,
            decomposition=use_decomposition,
        )

        # Generate all variations in parallel
        async def noop_none():
            return None

        async def noop_list():
            return []

        tasks = [self._generate_perspective_queries(query)]

        if use_step_back:
            tasks.append(self._generate_step_back_query(query))
        else:
            tasks.append(noop_none())

        if use_decomposition and self._should_decompose(query):
            tasks.append(self._decompose_query(query))
        else:
            tasks.append(noop_list())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        perspective_queries = results[0] if not isinstance(results[0], Exception) else []
        step_back_query = results[1] if not isinstance(results[1], Exception) else None
        decomposed_queries = results[2] if not isinstance(results[2], Exception) else []

        # Combine all queries
        all_queries = [query]  # Original first
        all_queries.extend(perspective_queries)
        if step_back_query:
            all_queries.append(step_back_query)
        all_queries.extend(decomposed_queries)

        # Estimate tokens
        tokens_used = sum(len(q) for q in all_queries) // 4

        result = MultiQueryResult(
            original_query=query,
            perspective_queries=perspective_queries,
            step_back_query=step_back_query,
            decomposed_queries=decomposed_queries,
            all_queries=all_queries,
            tokens_used=tokens_used,
        )

        # Cache result (Phase 69: Enhanced TTL cache)
        if self.config.cache_expansions:
            self._expansion_cache.set(cache_key, {
                "perspectives": perspective_queries,
                "step_back": step_back_query,
                "decomposed": decomposed_queries,
                "all": all_queries,
            })

        logger.info(
            "Multi-query rewriting complete",
            total_queries=len(all_queries),
            perspectives=len(perspective_queries),
            has_step_back=step_back_query is not None,
            decomposed=len(decomposed_queries),
        )

        return result

    async def _generate_perspective_queries(self, query: str) -> List[str]:
        """Generate perspective-based query variations."""
        if not self.config.enable_multi_query:
            return []

        llm = await self._get_llm()

        prompt = MULTI_QUERY_PERSPECTIVES_PROMPT.format(
            count=self.config.multi_query_count,
            query=query,
        )

        try:
            messages = [
                SystemMessage(content="You are a helpful search query reformulation assistant."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            perspectives = [
                line.strip()
                for line in content.strip().split('\n')
                if line.strip() and len(line.strip()) > 10
            ]

            return perspectives[:self.config.multi_query_count]

        except Exception as e:
            logger.error("Failed to generate perspective queries", error=str(e))
            return []

    async def _generate_step_back_query(self, query: str) -> Optional[str]:
        """Generate a step-back abstracted query."""
        llm = await self._get_llm()

        prompt = STEP_BACK_PROMPT.format(query=query)

        try:
            messages = [
                SystemMessage(content="You are an expert at generating abstract, high-level questions."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            step_back = content.strip().split('\n')[0].strip()
            return step_back if len(step_back) > 10 else None

        except Exception as e:
            logger.error("Failed to generate step-back query", error=str(e))
            return None

    async def generate_hypothetical_document(self, query: str) -> Optional[str]:
        """
        Generate a hypothetical document for HyDE-style retrieval.

        Creates an idealized answer that can be embedded and used
        to find similar real documents.

        Args:
            query: User's question

        Returns:
            Hypothetical document passage, or None on failure
        """
        llm = await self._get_llm()

        prompt = HYPOTHETICAL_DOCUMENT_PROMPT.format(query=query)

        try:
            messages = [
                SystemMessage(content="You are a helpful assistant that writes concise, factual passages."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            return content.strip() if len(content.strip()) > 20 else None

        except Exception as e:
            logger.error("Failed to generate hypothetical document", error=str(e))
            return None

    def _should_decompose(self, query: str) -> bool:
        """Check if query is complex enough to benefit from decomposition."""
        # Decompose if query has multiple parts or is long
        indicators = [
            " and " in query.lower(),
            " or " in query.lower(),
            len(query.split()) > 15,
            "?" in query and query.count("?") > 1,
            "also" in query.lower(),
            "additionally" in query.lower(),
        ]
        return any(indicators)

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: List[List[Dict[str, Any]]],
        k: int = 60,
        id_key: str = "chunk_id",
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple result lists using Reciprocal Rank Fusion.

        RRF is a simple but effective method for combining ranked lists
        from different queries or retrieval methods.

        Formula: score = sum(1 / (k + rank)) across all lists

        Args:
            result_lists: List of ranked result lists (each result has id_key)
            k: Constant to prevent high ranks from dominating (default: 60)
            id_key: Key to identify unique results

        Returns:
            Merged list sorted by RRF score
        """
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                doc_id = result.get(id_key, str(rank))

                # RRF formula
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

                # Keep the result with best original rank
                if doc_id not in result_map:
                    result_map[doc_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build merged results with RRF scores
        merged = []
        for doc_id in sorted_ids:
            result = result_map[doc_id].copy()
            result["rrf_score"] = rrf_scores[doc_id]
            merged.append(result)

        return merged


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
