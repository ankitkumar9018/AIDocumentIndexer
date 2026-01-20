"""
AIDocumentIndexer - Query Intent Classifier
============================================

Classifies search queries by intent to optimize retrieval strategy.
Different query types benefit from different vector vs keyword weights.

Enhanced with:
- MMR (Maximal Marginal Relevance) for source diversity
- Chain-of-Thought prompting for complex queries
- Prompt template selection based on intent
- Knowledge graph enhancement recommendations

Query Types:
- FACTUAL: Looking for specific facts, names, dates, numbers → favor keyword
- CONCEPTUAL: Abstract concepts, explanations, understanding → favor vector
- COMPARATIVE: Comparing items, "vs", "difference between" → balanced
- NAVIGATIONAL: Looking for specific documents → favor keyword
- PROCEDURAL: "How to" questions → balanced with slight vector preference

Research shows dynamic weighting can improve retrieval precision by 10-15%.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
import structlog

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    """Query intent categories."""
    FACTUAL = "factual"          # What is X? Who is Y? When did Z?
    CONCEPTUAL = "conceptual"    # Explain X, understanding, meaning
    COMPARATIVE = "comparative"  # X vs Y, difference between, compare
    NAVIGATIONAL = "navigational"  # Find document X, specific file
    PROCEDURAL = "procedural"    # How to X, steps to, process for
    EXPLORATORY = "exploratory"  # Broad exploration, overview
    AGGREGATION = "aggregation"  # Total X, sum of, average, calculations
    UNKNOWN = "unknown"          # Fallback


@dataclass
class QueryClassification:
    """Result of query classification."""
    query: str
    intent: QueryIntent
    confidence: float  # 0-1 confidence in classification
    vector_weight: float  # Recommended vector weight (0-1)
    keyword_weight: float  # Recommended keyword weight (0-1)
    reasoning: str  # Brief explanation of classification

    # Enhanced retrieval options
    use_mmr: bool = False  # Use Maximal Marginal Relevance for diversity
    diversity_weight: float = 0.0  # MMR diversity weight (0.0 to 0.5)
    use_cot: bool = False  # Use Chain-of-Thought prompting
    suggested_top_k: int = 8  # Suggested number of results
    similarity_threshold: float = 0.40  # Minimum similarity score
    use_kg_enhancement: bool = False  # Use knowledge graph enrichment
    prompt_template: str = "default"  # Template to use (factual, comparison, etc.)
    matched_patterns: List[str] = field(default_factory=list)  # Patterns that matched


# Pattern-based classification rules
INTENT_PATTERNS = {
    QueryIntent.FACTUAL: [
        r"^(what|who|when|where|which)\s+(is|are|was|were|did)\s+",
        r"^(what'?s|who'?s)\s+",
        r"\b(name|date|number|amount|price|cost|size|year)\b",
        r"^(tell me|give me)\s+(the|a)\s+",
        r"\b(exactly|specifically|precisely)\b",
        r"^(define|definition of)\s+",
        r"\bin\s+\d{4}\b",  # Year references
        r"\b\d+\s*(dollars?|euros?|pounds?|%|percent)\b",  # Numbers/amounts
    ],
    QueryIntent.CONCEPTUAL: [
        r"^(explain|describe|understand|meaning|concept)\s+",
        r"^(why|how come)\s+",
        r"\b(theory|concept|principle|idea|philosophy)\b",
        r"\b(significance|importance|impact|implications?)\b",
        r"\b(in general|broadly|overall|fundamentally)\b",
        r"^(what does.*mean)",
        r"\b(understanding|comprehension)\b",
    ],
    QueryIntent.COMPARATIVE: [
        r"\bvs\.?\s+",
        r"\bversus\b",
        r"\b(compare|comparison|comparing)\b",
        r"\b(difference|differ|different)\s+(between|from)\b",
        r"\b(better|worse|superior|inferior)\s+(than|to)\b",
        r"\b(pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)\b",
        r"\b(alternative|instead of|or)\b",
        r"\b(similar|similarity|like)\s+(to|with)\b",
    ],
    QueryIntent.NAVIGATIONAL: [
        r"^(find|show|get|retrieve|open)\s+(me\s+)?(the\s+)?",
        r"\b(document|file|report|presentation|spreadsheet)\s+(named?|called|titled)\b",
        r"^(where|locate)\s+",
        r"\.(pdf|docx?|xlsx?|pptx?|txt|csv)\b",
        r"\b(folder|directory)\b",
        r"^(go to|navigate to|open)\s+",
    ],
    QueryIntent.PROCEDURAL: [
        r"^(how (to|do|can|should)|steps? to|guide to|tutorial)\s+",
        r"\b(process|procedure|workflow|method|approach)\s+(for|to)\b",
        r"^(can i|how do i|what'?s the (best )?way to)\s+",
        r"\b(implement|create|build|develop|set up|configure)\b",
        r"\b(instructions?|guidelines?)\b",
        r"^(walk me through|show me how)\s+",
    ],
    QueryIntent.EXPLORATORY: [
        r"^(tell me about|what do you know about|overview of)\s+",
        r"\b(everything|all|overview|summary|summarize)\b",
        r"^(list|enumerate)\s+",
        r"\b(topics?|areas?|subjects?)\s+(related|about)\b",
        r"^(what can you tell me)\s+",
    ],
    QueryIntent.AGGREGATION: [
        r"\b(total|sum|aggregate|combined|cumulative)\s+",
        r"\b(how much|how many)\s+(did|was|were|has|have|is|are)\s+",
        r"\b(average|avg|mean)\s+(of|for)?\s*",
        r"\b(spending|revenue|cost|expense|income|profit|budget)\s+",
        r"\b(calculate|computation|add up)\s+",
        r"\b(maximum|minimum|highest|lowest)\s+(of|for)?\s*",
        r"\b(count|number of)\s+\w+\s+(in|from|for|during)\s+",
        r"\b(last|past|previous)\s+\d+\s+(months?|years?|quarters?|weeks?|days?)",
        r"\b(q[1-4]|quarter)\s*\d*\s+(spending|revenue|cost|total)",
        r"\b(fy|fiscal\s+year)\s*\d*",
        r"\ball\s+(spending|payments?|transactions?|expenses?|costs?)\b",
    ],
}

# Default weights for each intent type
INTENT_WEIGHTS = {
    QueryIntent.FACTUAL: (0.5, 0.5),       # Balanced, slight keyword for exact matches
    QueryIntent.CONCEPTUAL: (0.8, 0.2),    # Strong vector for semantic understanding
    QueryIntent.COMPARATIVE: (0.6, 0.4),   # Slightly favor vector for nuance
    QueryIntent.NAVIGATIONAL: (0.3, 0.7),  # Strong keyword for specific names/titles
    QueryIntent.PROCEDURAL: (0.7, 0.3),    # Favor vector for process understanding
    QueryIntent.EXPLORATORY: (0.75, 0.25), # Strong vector for broad exploration
    QueryIntent.AGGREGATION: (0.6, 0.4),   # Balanced - need both semantic and keyword for numerical extraction
    QueryIntent.UNKNOWN: (0.7, 0.3),       # Default: slight vector preference
}

# Enhanced retrieval configuration for each intent
# PHASE 15: Optimized top_k values for 60-70% context window utilization (research-backed 2026)
# Research shows: RAG with focused context (8-12 chunks) outperforms naive 200K context
# "Lost in the Middle" problem intensifies with longer contexts
INTENT_RETRIEVAL_CONFIG = {
    QueryIntent.FACTUAL: {
        "use_mmr": False,
        "diversity_weight": 0.0,
        "use_cot": False,
        "suggested_top_k": 6,  # Reduced from 8 for optimal 60-70% utilization
        "similarity_threshold": 0.40,
        "prompt_template": "factual",
    },
    QueryIntent.CONCEPTUAL: {
        "use_mmr": True,
        "diversity_weight": 0.2,
        "use_cot": True,  # Complex concepts benefit from step-by-step reasoning
        "suggested_top_k": 8,  # Reduced from 10
        "similarity_threshold": 0.35,
        "prompt_template": "analytical",
    },
    QueryIntent.COMPARATIVE: {
        "use_mmr": True,  # Need diverse sources for comparison
        "diversity_weight": 0.4,
        "use_cot": False,
        "suggested_top_k": 10,  # Reduced from 12 - optimal for comparisons
        "similarity_threshold": 0.35,
        "prompt_template": "comparison",
    },
    QueryIntent.NAVIGATIONAL: {
        "use_mmr": False,
        "diversity_weight": 0.0,
        "use_cot": False,
        "suggested_top_k": 5,  # Already optimal - precise navigation
        "similarity_threshold": 0.45,
        "prompt_template": "factual",
    },
    QueryIntent.PROCEDURAL: {
        "use_mmr": True,
        "diversity_weight": 0.2,
        "use_cot": True,  # Step-by-step processes benefit from CoT
        "suggested_top_k": 8,  # Reduced from 10
        "similarity_threshold": 0.35,
        "prompt_template": "analytical",
    },
    QueryIntent.EXPLORATORY: {
        "use_mmr": True,  # Need diverse sources for overview
        "diversity_weight": 0.3,
        "use_cot": False,
        "suggested_top_k": 12,  # Reduced from 15 - still comprehensive
        "similarity_threshold": 0.30,
        "prompt_template": "summary",
    },
    QueryIntent.AGGREGATION: {
        "use_mmr": False,
        "diversity_weight": 0.0,
        "use_cot": False,
        "suggested_top_k": 15,  # Reduced from 20 - still broad coverage
        "similarity_threshold": 0.30,
        "prompt_template": "list",
    },
    QueryIntent.UNKNOWN: {
        "use_mmr": False,
        "diversity_weight": 0.0,
        "use_cot": False,
        "suggested_top_k": 6,  # Reduced from 8 for optimal default
        "similarity_threshold": 0.40,
        "prompt_template": "default",
    },
}

# Patterns that suggest knowledge graph enhancement would help
KG_ENHANCEMENT_PATTERNS = [
    r"\brelat(?:ed|ionship)\b",
    r"\bconnect(?:ed|ion)\b",
    r"\bwork(?:ed|s)?\s+(?:with|for|at|on)\b",
    r"\bwho\s+(?:is|are|was|were)\b.*\b(?:at|in|from)\b",
    r"\binvolv(?:ed|es)\b",
    r"\bassociat(?:ed|ion)\b",
    r"\baffili(?:ated|ation)\b",
    r"\bpartner(?:ed|ship)?\b",
    r"\bcollaborat(?:ed|ion|or)\b",
    r"\bmentioned\b.*\bwith\b",
    r"\bpeople\b.*\b(?:at|in|from)\b",
    r"\bteam\b.*\b(?:at|in|from)\b",
]


class QueryClassifier:
    """
    Classifies search queries by intent for optimal retrieval strategy.

    Uses pattern matching with optional LLM fallback for ambiguous queries.
    """

    def __init__(
        self,
        use_llm_fallback: bool = False,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize the query classifier.

        Args:
            use_llm_fallback: Whether to use LLM for ambiguous queries
            confidence_threshold: Minimum confidence for pattern-based classification
        """
        self.use_llm_fallback = use_llm_fallback
        self.confidence_threshold = confidence_threshold
        self._compiled_patterns = self._compile_patterns()

        logger.info(
            "QueryClassifier initialized",
            use_llm_fallback=use_llm_fallback,
            confidence_threshold=confidence_threshold,
        )

    def _compile_patterns(self) -> dict:
        """Pre-compile regex patterns for efficiency."""
        compiled = {}
        for intent, patterns in INTENT_PATTERNS.items():
            compiled[intent] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]
        return compiled

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query by intent.

        Args:
            query: The search query to classify

        Returns:
            QueryClassification with intent, weights, and enhanced retrieval config
        """
        query = query.strip()

        if not query:
            config = INTENT_RETRIEVAL_CONFIG[QueryIntent.UNKNOWN]
            return QueryClassification(
                query=query,
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                vector_weight=0.7,
                keyword_weight=0.3,
                reasoning="Empty query",
                use_mmr=config["use_mmr"],
                diversity_weight=config["diversity_weight"],
                use_cot=config["use_cot"],
                suggested_top_k=config["suggested_top_k"],
                similarity_threshold=config["similarity_threshold"],
                use_kg_enhancement=False,
                prompt_template=config["prompt_template"],
                matched_patterns=[],
            )

        # Count pattern matches for each intent
        intent_scores = {}
        matched_patterns_dict = {}

        for intent, patterns in self._compiled_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern.search(query):
                    matches.append(pattern.pattern)

            if matches:
                intent_scores[intent] = len(matches)
                matched_patterns_dict[intent] = matches

        # Determine best intent
        if intent_scores:
            best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
            max_score = intent_scores[best_intent]

            # Calculate confidence based on match ratio and uniqueness
            confidence = min(1.0, max_score / 3)  # Cap at 1.0, 3+ matches = high confidence

            # Reduce confidence if multiple intents matched equally
            if len([s for s in intent_scores.values() if s == max_score]) > 1:
                confidence *= 0.7

            # Get weights for this intent
            vec_weight, kw_weight = INTENT_WEIGHTS[best_intent]

            # Adjust weights based on query characteristics
            vec_weight, kw_weight = self._adjust_weights_for_query(
                query, best_intent, vec_weight, kw_weight
            )

            reasoning = f"Matched {max_score} pattern(s) for {best_intent.value}"

            # Get enhanced retrieval config
            config = INTENT_RETRIEVAL_CONFIG.get(
                best_intent, INTENT_RETRIEVAL_CONFIG[QueryIntent.UNKNOWN]
            )

            # Check if KG enhancement would help
            use_kg = self._should_use_kg_enhancement(query)

            return QueryClassification(
                query=query,
                intent=best_intent,
                confidence=confidence,
                vector_weight=vec_weight,
                keyword_weight=kw_weight,
                reasoning=reasoning,
                use_mmr=config["use_mmr"],
                diversity_weight=config["diversity_weight"],
                use_cot=config["use_cot"],
                suggested_top_k=config["suggested_top_k"],
                similarity_threshold=config["similarity_threshold"],
                use_kg_enhancement=use_kg,
                prompt_template=config["prompt_template"],
                matched_patterns=matched_patterns_dict.get(best_intent, []),
            )

        # No patterns matched - use heuristics
        return self._classify_by_heuristics(query)

    def _should_use_kg_enhancement(self, query: str) -> bool:
        """
        Determine if knowledge graph enhancement would help this query.

        Args:
            query: The query string

        Returns:
            True if KG enhancement is recommended
        """
        for pattern in KG_ENHANCEMENT_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _adjust_weights_for_query(
        self,
        query: str,
        intent: QueryIntent,
        vec_weight: float,
        kw_weight: float,
    ) -> Tuple[float, float]:
        """
        Fine-tune weights based on specific query characteristics.

        Args:
            query: The query string
            intent: Detected intent
            vec_weight: Base vector weight
            kw_weight: Base keyword weight

        Returns:
            Adjusted (vector_weight, keyword_weight) tuple
        """
        query_lower = query.lower()

        # Quoted phrases → boost keyword weight (exact match expected)
        if '"' in query or "'" in query:
            kw_weight = min(1.0, kw_weight + 0.2)
            vec_weight = max(0.0, vec_weight - 0.2)

        # Very short queries (1-2 words) → boost keyword
        word_count = len(query.split())
        if word_count <= 2:
            kw_weight = min(1.0, kw_weight + 0.15)
            vec_weight = max(0.0, vec_weight - 0.15)

        # Long queries (7+ words) → boost vector (more semantic context)
        elif word_count >= 7:
            vec_weight = min(1.0, vec_weight + 0.1)
            kw_weight = max(0.0, kw_weight - 0.1)

        # Contains technical terms/acronyms → boost keyword
        if re.search(r'\b[A-Z]{2,}\b', query):  # Acronyms
            kw_weight = min(1.0, kw_weight + 0.1)
            vec_weight = max(0.0, vec_weight - 0.1)

        # Contains numbers → boost keyword (likely factual)
        if re.search(r'\d+', query):
            kw_weight = min(1.0, kw_weight + 0.1)
            vec_weight = max(0.0, vec_weight - 0.1)

        # Question words at start → ensure minimum vector weight
        if re.match(r'^(what|why|how|when|where|who|which)\s', query_lower):
            vec_weight = max(0.4, vec_weight)

        # Normalize to ensure they sum to 1
        total = vec_weight + kw_weight
        if total > 0:
            vec_weight = vec_weight / total
            kw_weight = kw_weight / total

        return round(vec_weight, 2), round(kw_weight, 2)

    def _classify_by_heuristics(self, query: str) -> QueryClassification:
        """
        Classify query using heuristics when no patterns match.

        Args:
            query: The query string

        Returns:
            QueryClassification based on heuristics
        """
        query_lower = query.lower()
        word_count = len(query.split())
        use_kg = self._should_use_kg_enhancement(query)

        # Very short queries are often navigational or factual
        if word_count <= 2:
            config = INTENT_RETRIEVAL_CONFIG[QueryIntent.FACTUAL]
            return QueryClassification(
                query=query,
                intent=QueryIntent.FACTUAL,
                confidence=0.4,
                vector_weight=0.5,
                keyword_weight=0.5,
                reasoning="Short query, defaulting to balanced approach",
                use_mmr=config["use_mmr"],
                diversity_weight=config["diversity_weight"],
                use_cot=config["use_cot"],
                suggested_top_k=config["suggested_top_k"],
                similarity_threshold=config["similarity_threshold"],
                use_kg_enhancement=use_kg,
                prompt_template=config["prompt_template"],
                matched_patterns=[],
            )

        # Questions are often conceptual or procedural
        if query.endswith('?'):
            if query_lower.startswith('how'):
                intent = QueryIntent.PROCEDURAL
            elif query_lower.startswith(('why', 'what does', 'explain')):
                intent = QueryIntent.CONCEPTUAL
            else:
                intent = QueryIntent.FACTUAL

            vec_weight, kw_weight = INTENT_WEIGHTS[intent]
            config = INTENT_RETRIEVAL_CONFIG[intent]
            return QueryClassification(
                query=query,
                intent=intent,
                confidence=0.5,
                vector_weight=vec_weight,
                keyword_weight=kw_weight,
                reasoning="Question format detected",
                use_mmr=config["use_mmr"],
                diversity_weight=config["diversity_weight"],
                use_cot=config["use_cot"],
                suggested_top_k=config["suggested_top_k"],
                similarity_threshold=config["similarity_threshold"],
                use_kg_enhancement=use_kg,
                prompt_template=config["prompt_template"],
                matched_patterns=[],
            )

        # Default to exploratory for longer queries
        if word_count >= 5:
            config = INTENT_RETRIEVAL_CONFIG[QueryIntent.EXPLORATORY]
            return QueryClassification(
                query=query,
                intent=QueryIntent.EXPLORATORY,
                confidence=0.4,
                vector_weight=0.7,
                keyword_weight=0.3,
                reasoning="Longer query, defaulting to exploratory",
                use_mmr=config["use_mmr"],
                diversity_weight=config["diversity_weight"],
                use_cot=config["use_cot"],
                suggested_top_k=config["suggested_top_k"],
                similarity_threshold=config["similarity_threshold"],
                use_kg_enhancement=use_kg,
                prompt_template=config["prompt_template"],
                matched_patterns=[],
            )

        # Fallback
        config = INTENT_RETRIEVAL_CONFIG[QueryIntent.UNKNOWN]
        return QueryClassification(
            query=query,
            intent=QueryIntent.UNKNOWN,
            confidence=0.3,
            vector_weight=0.7,
            keyword_weight=0.3,
            reasoning="No clear intent detected, using defaults",
            use_mmr=config["use_mmr"],
            diversity_weight=config["diversity_weight"],
            use_cot=config["use_cot"],
            suggested_top_k=config["suggested_top_k"],
            similarity_threshold=config["similarity_threshold"],
            use_kg_enhancement=use_kg,
            prompt_template=config["prompt_template"],
            matched_patterns=[],
        )

    def get_weights_for_query(self, query: str) -> Tuple[float, float]:
        """
        Convenience method to get just the weights for a query.

        Args:
            query: The search query

        Returns:
            Tuple of (vector_weight, keyword_weight)
        """
        classification = self.classify(query)
        return classification.vector_weight, classification.keyword_weight


# Singleton instance
_query_classifier: Optional[QueryClassifier] = None


def get_query_classifier(
    use_llm_fallback: bool = False,
    confidence_threshold: float = 0.6,
) -> QueryClassifier:
    """
    Get or create the query classifier singleton.

    Args:
        use_llm_fallback: Whether to use LLM for ambiguous queries
        confidence_threshold: Minimum confidence for pattern classification

    Returns:
        QueryClassifier singleton instance
    """
    global _query_classifier
    if _query_classifier is None:
        _query_classifier = QueryClassifier(
            use_llm_fallback=use_llm_fallback,
            confidence_threshold=confidence_threshold,
        )
    return _query_classifier
