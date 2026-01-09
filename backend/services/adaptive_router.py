"""
AIDocumentIndexer - Adaptive RAG Router
========================================

Routes queries to optimal retrieval strategies based on complexity analysis.
Enables efficient resource usage by matching query complexity with appropriate
retrieval methods.

Strategies:
- DIRECT: Simple vector search for factual lookups
- HYBRID: Vector + BM25 for moderate complexity
- TWO_STAGE: Coarse then fine retrieval for aggregation
- AGENTIC: Full reasoning with tools for complex queries
- GRAPH_ENHANCED: Knowledge graph + vectors for entity relations
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

import structlog

logger = structlog.get_logger(__name__)


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Direct factual lookup
    MODERATE = "moderate"  # Multi-hop or comparative
    COMPLEX = "complex"    # Requires reasoning/aggregation


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    DIRECT = "direct"           # Simple vector search
    HYBRID = "hybrid"           # Vector + BM25
    TWO_STAGE = "two_stage"     # Coarse then fine retrieval
    AGENTIC = "agentic"         # Full reasoning with tools
    GRAPH_ENHANCED = "graph"    # Knowledge graph + vectors


@dataclass
class RoutingDecision:
    """Result of routing analysis."""
    complexity: QueryComplexity
    strategy: RetrievalStrategy
    top_k: int
    use_reranking: bool
    use_hyde: bool
    use_rag_fusion: bool
    use_step_back: bool
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Patterns for detecting query characteristics
MULTI_HOP_PATTERNS = [
    r"what .+ of .+ that",
    r"who .+ when .+ was",
    r"find .+ related to .+ and",
    r"which .+ has .+ and .+",
    r"compare .+ with .+",
]

AGGREGATION_PATTERNS = [
    r"total\s+\w+",
    r"sum\s+of",
    r"how\s+much\s+\w+\s+spent",
    r"average\s+\w+",
    r"combined\s+\w+",
    r"aggregate\s+\w+",
    r"count\s+(all|the|how\s+many)",
    r"\d+\s+(months?|years?|quarters?)",
]

REASONING_KEYWORDS = ["why", "how", "explain", "reason", "because", "cause"]
TEMPORAL_KEYWORDS = ["when", "before", "after", "during", "since", "until"]
COMPARISON_KEYWORDS = ["compare", "difference", "versus", "vs", "better", "worse", "more", "less"]
RELATION_KEYWORDS = ["relationship", "connected", "between", "related to", "linked", "associated"]


class AdaptiveRouter:
    """Route queries to optimal retrieval strategies based on complexity."""

    def __init__(
        self,
        llm=None,
        query_classifier=None,
        enable_llm_routing: bool = False,
    ):
        """
        Initialize the adaptive router.

        Args:
            llm: Optional LLM for advanced routing decisions
            query_classifier: Optional query classifier service
            enable_llm_routing: Whether to use LLM for complex routing
        """
        self.llm = llm
        self.query_classifier = query_classifier
        self.enable_llm_routing = enable_llm_routing

    async def route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Analyze query and determine optimal retrieval strategy.

        Args:
            query: The user's query
            context: Optional context (conversation history, user preferences, etc.)

        Returns:
            RoutingDecision with strategy and parameters
        """
        context = context or {}

        # Analyze complexity indicators
        complexity_indicators = self._analyze_complexity(query)

        # Get query intent if classifier available
        intent = None
        if self.query_classifier:
            try:
                classification = await self.query_classifier.classify(query)
                intent = classification.intent
                complexity_indicators["classified_intent"] = intent
            except Exception as e:
                logger.warning("Query classification failed", error=str(e))

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(complexity_indicators)

        # Determine routing decision
        decision = self._make_routing_decision(
            query=query,
            complexity_score=complexity_score,
            indicators=complexity_indicators,
            context=context,
        )

        logger.info(
            "Query routed",
            query_preview=query[:100],
            complexity=decision.complexity.value,
            strategy=decision.strategy.value,
            top_k=decision.top_k,
            confidence=decision.confidence,
        )

        return decision

    def _analyze_complexity(self, query: str) -> Dict[str, bool]:
        """Analyze query for complexity indicators."""
        query_lower = query.lower()

        indicators = {
            "multi_hop": self._detect_multi_hop(query_lower),
            "aggregation": self._detect_aggregation(query_lower),
            "comparison": self._detect_comparison(query_lower),
            "reasoning": self._detect_reasoning(query_lower),
            "temporal": self._detect_temporal(query_lower),
            "entity_relations": self._detect_entity_relations(query_lower),
            "long_query": len(query.split()) > 15,
            "question_chain": query.count("?") > 1,
        }

        return indicators

    def _detect_multi_hop(self, query: str) -> bool:
        """Detect if query requires multi-hop reasoning."""
        return any(re.search(p, query) for p in MULTI_HOP_PATTERNS)

    def _detect_aggregation(self, query: str) -> bool:
        """Detect if query requires numerical aggregation."""
        return any(re.search(p, query) for p in AGGREGATION_PATTERNS)

    def _detect_comparison(self, query: str) -> bool:
        """Detect if query involves comparison."""
        return any(kw in query for kw in COMPARISON_KEYWORDS)

    def _detect_reasoning(self, query: str) -> bool:
        """Detect if query requires reasoning/explanation."""
        return any(kw in query for kw in REASONING_KEYWORDS)

    def _detect_temporal(self, query: str) -> bool:
        """Detect if query has temporal aspects."""
        return any(kw in query for kw in TEMPORAL_KEYWORDS)

    def _detect_entity_relations(self, query: str) -> bool:
        """Check if query involves entity relationships."""
        return any(kw in query for kw in RELATION_KEYWORDS)

    def _calculate_complexity_score(self, indicators: Dict[str, bool]) -> int:
        """Calculate overall complexity score from indicators."""
        # Weight different indicators
        weights = {
            "multi_hop": 2,
            "aggregation": 2,
            "comparison": 1,
            "reasoning": 1,
            "temporal": 1,
            "entity_relations": 2,
            "long_query": 1,
            "question_chain": 1,
        }

        score = sum(
            weights.get(key, 1)
            for key, value in indicators.items()
            if value and key in weights
        )

        return score

    def _make_routing_decision(
        self,
        query: str,
        complexity_score: int,
        indicators: Dict[str, bool],
        context: Dict[str, Any],
    ) -> RoutingDecision:
        """Make final routing decision based on analysis."""
        reasoning_parts = []

        # Determine complexity level
        if complexity_score == 0:
            complexity = QueryComplexity.SIMPLE
            reasoning_parts.append("Simple factual query detected")
        elif complexity_score <= 3:
            complexity = QueryComplexity.MODERATE
            reasoning_parts.append(f"Moderate complexity (score: {complexity_score})")
        else:
            complexity = QueryComplexity.COMPLEX
            reasoning_parts.append(f"Complex query requiring advanced processing (score: {complexity_score})")

        # Determine strategy based on complexity and specific indicators
        if complexity == QueryComplexity.SIMPLE:
            strategy = RetrievalStrategy.DIRECT
            top_k = 5
            use_reranking = False
            use_hyde = False
            use_rag_fusion = False
            use_step_back = False
            reasoning_parts.append("Using direct vector search for efficiency")

        elif complexity == QueryComplexity.MODERATE:
            strategy = RetrievalStrategy.HYBRID
            top_k = 10
            use_reranking = True
            use_hyde = indicators.get("reasoning", False)
            use_rag_fusion = indicators.get("comparison", False)
            use_step_back = False

            if use_hyde:
                reasoning_parts.append("HyDE enabled for reasoning query")
            if use_rag_fusion:
                reasoning_parts.append("RAG-Fusion enabled for comprehensive coverage")

        else:  # COMPLEX
            top_k = 20
            use_reranking = True
            use_hyde = True
            use_rag_fusion = True
            use_step_back = indicators.get("reasoning", False)

            # Choose specific strategy based on indicators
            if indicators.get("aggregation"):
                strategy = RetrievalStrategy.TWO_STAGE
                reasoning_parts.append("Two-stage retrieval for aggregation query")
            elif indicators.get("entity_relations"):
                strategy = RetrievalStrategy.GRAPH_ENHANCED
                reasoning_parts.append("Graph-enhanced retrieval for entity relationships")
            elif indicators.get("multi_hop"):
                strategy = RetrievalStrategy.AGENTIC
                reasoning_parts.append("Agentic retrieval for multi-hop reasoning")
            else:
                strategy = RetrievalStrategy.AGENTIC
                reasoning_parts.append("Agentic retrieval for complex query handling")

            if use_step_back:
                reasoning_parts.append("Step-back prompting enabled")

        # Apply context overrides
        if context.get("force_strategy"):
            strategy = RetrievalStrategy(context["force_strategy"])
            reasoning_parts.append(f"Strategy forced to {strategy.value}")

        if context.get("max_sources"):
            top_k = min(top_k, context["max_sources"])

        # Calculate confidence
        confidence = self._calculate_confidence(complexity_score, indicators)

        return RoutingDecision(
            complexity=complexity,
            strategy=strategy,
            top_k=top_k,
            use_reranking=use_reranking,
            use_hyde=use_hyde,
            use_rag_fusion=use_rag_fusion,
            use_step_back=use_step_back,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts),
            metadata={
                "indicators": indicators,
                "complexity_score": complexity_score,
            },
        )

    def _calculate_confidence(
        self,
        complexity_score: int,
        indicators: Dict[str, bool],
    ) -> float:
        """Calculate confidence in routing decision."""
        # Base confidence decreases with complexity
        base_confidence = 0.9

        # Reduce confidence for higher complexity (harder to route correctly)
        complexity_penalty = min(0.3, complexity_score * 0.05)

        # Increase confidence if clear indicators present
        clear_indicators = sum([
            indicators.get("aggregation", False),
            indicators.get("entity_relations", False),
            indicators.get("multi_hop", False),
        ])

        indicator_bonus = min(0.15, clear_indicators * 0.05)

        confidence = base_confidence - complexity_penalty + indicator_bonus

        return max(0.5, min(1.0, confidence))


class RoutingAnalytics:
    """Track and analyze routing decisions for optimization."""

    def __init__(self):
        self._decisions: List[Dict[str, Any]] = []

    def log_decision(
        self,
        query: str,
        decision: RoutingDecision,
        result_quality: Optional[float] = None,
    ):
        """Log a routing decision for analysis."""
        self._decisions.append({
            "query": query,
            "complexity": decision.complexity.value,
            "strategy": decision.strategy.value,
            "confidence": decision.confidence,
            "result_quality": result_quality,
        })

    def get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of strategies used."""
        distribution = {}
        for d in self._decisions:
            strategy = d["strategy"]
            distribution[strategy] = distribution.get(strategy, 0) + 1
        return distribution

    def get_average_quality_by_strategy(self) -> Dict[str, float]:
        """Get average result quality per strategy."""
        quality_sums = {}
        quality_counts = {}

        for d in self._decisions:
            if d["result_quality"] is not None:
                strategy = d["strategy"]
                quality_sums[strategy] = quality_sums.get(strategy, 0) + d["result_quality"]
                quality_counts[strategy] = quality_counts.get(strategy, 0) + 1

        return {
            strategy: quality_sums[strategy] / quality_counts[strategy]
            for strategy in quality_sums
            if quality_counts[strategy] > 0
        }


# Singleton instance
_router_instance: Optional[AdaptiveRouter] = None


def get_adaptive_router(
    llm=None,
    query_classifier=None,
) -> AdaptiveRouter:
    """Get or create the adaptive router singleton."""
    global _router_instance

    if _router_instance is None:
        _router_instance = AdaptiveRouter(
            llm=llm,
            query_classifier=query_classifier,
        )

    return _router_instance


async def route_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    llm=None,
    query_classifier=None,
) -> RoutingDecision:
    """
    Convenience function to route a query.

    Args:
        query: The user's query
        context: Optional context
        llm: Optional LLM for advanced routing
        query_classifier: Optional query classifier

    Returns:
        RoutingDecision
    """
    router = get_adaptive_router(llm=llm, query_classifier=query_classifier)
    return await router.route(query, context)
