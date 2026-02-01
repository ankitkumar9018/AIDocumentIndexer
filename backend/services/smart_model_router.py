"""
AIDocumentIndexer - Smart Model Router
=======================================

Routes RAG queries to cost-optimal LLM models based on query complexity.

Tiers:
  - SIMPLE: Factual lookups, keyword queries → cheap/fast model (e.g., gpt-4o-mini, ollama/llama3.2)
  - MODERATE: Analytical queries, comparisons → default model
  - COMPLEX: Multi-hop reasoning, aggregation, creative → premium model (e.g., gpt-4o, claude-3-opus)

Expected impact: 40-70% LLM cost reduction on mixed workloads.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Any

import structlog

logger = structlog.get_logger(__name__)


class QueryTier(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ModelRoute:
    """Result of model routing decision."""
    tier: QueryTier
    provider: Optional[str]
    model: Optional[str]
    reason: str


# Default model tier mappings (overridable via settings)
DEFAULT_TIER_MODELS = {
    QueryTier.SIMPLE: {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
        "ollama": None,  # Use default ollama model
    },
    QueryTier.MODERATE: None,  # Use default/session model
    QueryTier.COMPLEX: {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "ollama": None,  # Use default ollama model
    },
}

# Query intents that map to simple tier
SIMPLE_INTENTS = {"factual", "keyword", "definition", "simple"}

# Query intents that map to complex tier
COMPLEX_INTENTS = {"analytical", "multi_hop", "creative", "aggregation", "reasoning", "comparative"}


def classify_query_tier(
    query_intent: Optional[str] = None,
    query_confidence: float = 0.0,
    context_length: int = 0,
    num_documents: int = 0,
) -> QueryTier:
    """
    Classify a query into a cost tier based on its characteristics.

    Args:
        query_intent: Intent from query classifier (factual, analytical, etc.)
        query_confidence: Classification confidence (0-1)
        context_length: Length of retrieved context
        num_documents: Number of retrieved documents

    Returns:
        QueryTier indicating which model tier to use
    """
    intent = (query_intent or "").lower()

    # High-confidence simple queries
    if intent in SIMPLE_INTENTS and query_confidence > 0.7:
        return QueryTier.SIMPLE

    # Complex queries
    if intent in COMPLEX_INTENTS and query_confidence > 0.6:
        return QueryTier.COMPLEX

    # Large context suggests complexity
    if context_length > 50000 or num_documents > 15:
        return QueryTier.COMPLEX

    # Very short context with factual intent → simple
    if context_length < 2000 and num_documents <= 3:
        return QueryTier.SIMPLE

    return QueryTier.MODERATE


def get_model_for_tier(
    tier: QueryTier,
    current_provider: Optional[str] = None,
    settings_getter=None,
) -> ModelRoute:
    """
    Get the recommended model for a query tier.

    Args:
        tier: The query complexity tier
        current_provider: The provider currently configured
        settings_getter: Optional settings accessor function

    Returns:
        ModelRoute with provider/model recommendation
    """
    # Check settings for custom tier mappings
    if settings_getter:
        custom_simple = settings_getter("rag.smart_routing_simple_model")
        custom_complex = settings_getter("rag.smart_routing_complex_model")

        if tier == QueryTier.SIMPLE and custom_simple:
            # Format: "provider/model" or just "model"
            if "/" in str(custom_simple):
                provider, model = str(custom_simple).split("/", 1)
                return ModelRoute(tier=tier, provider=provider, model=model, reason="settings_override")
            return ModelRoute(tier=tier, provider=current_provider, model=str(custom_simple), reason="settings_override")

        if tier == QueryTier.COMPLEX and custom_complex:
            if "/" in str(custom_complex):
                provider, model = str(custom_complex).split("/", 1)
                return ModelRoute(tier=tier, provider=provider, model=model, reason="settings_override")
            return ModelRoute(tier=tier, provider=current_provider, model=str(custom_complex), reason="settings_override")

    # Use defaults
    if tier == QueryTier.MODERATE:
        return ModelRoute(tier=tier, provider=None, model=None, reason="use_default")

    tier_models = DEFAULT_TIER_MODELS.get(tier)
    if tier_models and current_provider:
        model = tier_models.get(current_provider)
        if model:
            return ModelRoute(tier=tier, provider=current_provider, model=model, reason="tier_default")

    # No specific model for this tier/provider combo → use default
    return ModelRoute(tier=tier, provider=None, model=None, reason="no_tier_model")


async def route_query_to_model(
    question: str,
    query_classification=None,
    context_length: int = 0,
    num_documents: int = 0,
    current_provider: Optional[str] = None,
    settings_getter=None,
) -> ModelRoute:
    """
    Main entry point: route a query to the optimal model.

    Args:
        question: The user query
        query_classification: Result from query classifier
        context_length: Length of retrieved context
        num_documents: Number of retrieved documents
        current_provider: Currently configured provider
        settings_getter: Settings accessor function

    Returns:
        ModelRoute with routing decision
    """
    # Check if smart routing is enabled
    enabled = True
    if settings_getter:
        enabled = settings_getter("rag.smart_model_routing_enabled", False)

    if not enabled:
        return ModelRoute(tier=QueryTier.MODERATE, provider=None, model=None, reason="routing_disabled")

    # Extract classification info
    query_intent = None
    query_confidence = 0.0
    if query_classification:
        query_intent = query_classification.intent.value if hasattr(query_classification, 'intent') else None
        query_confidence = getattr(query_classification, 'confidence', 0.0)

    # Classify tier
    tier = classify_query_tier(
        query_intent=query_intent,
        query_confidence=query_confidence,
        context_length=context_length,
        num_documents=num_documents,
    )

    # Get model for tier
    route = get_model_for_tier(
        tier=tier,
        current_provider=current_provider,
        settings_getter=settings_getter,
    )

    logger.info(
        "Smart model routing decision",
        query_preview=question[:80],
        tier=tier.value,
        intent=query_intent,
        provider=route.provider,
        model=route.model,
        reason=route.reason,
    )

    return route
