"""
AIDocumentIndexer - LLM Pricing Service
========================================

Cost calculation for various LLM providers and models.
Prices are per million tokens (or per million characters for some providers).
"""

from dataclasses import dataclass
from typing import Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for a single model."""
    input_price_per_million: float  # USD per million tokens
    output_price_per_million: float  # USD per million tokens
    is_local: bool = False  # Local models are free


# =============================================================================
# Pricing Data (Updated December 2024)
# =============================================================================

PRICING_DATA: Dict[str, Dict[str, ModelPricing]] = {
    # OpenAI models
    "openai": {
        # GPT-4o family
        "gpt-4o": ModelPricing(5.00, 15.00),
        "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00),
        "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00),
        "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
        "gpt-4o-mini": ModelPricing(0.15, 0.60),
        "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60),
        # GPT-4 Turbo
        "gpt-4-turbo": ModelPricing(10.00, 30.00),
        "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
        "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
        # GPT-4 (original)
        "gpt-4": ModelPricing(30.00, 60.00),
        "gpt-4-0613": ModelPricing(30.00, 60.00),
        "gpt-4-32k": ModelPricing(60.00, 120.00),
        # GPT-3.5 Turbo
        "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
        "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
        "gpt-3.5-turbo-1106": ModelPricing(1.00, 2.00),
        # o1 reasoning models
        "o1": ModelPricing(15.00, 60.00),
        "o1-preview": ModelPricing(15.00, 60.00),
        "o1-mini": ModelPricing(3.00, 12.00),
        # Embeddings
        "text-embedding-3-large": ModelPricing(0.13, 0.0),
        "text-embedding-3-small": ModelPricing(0.02, 0.0),
        "text-embedding-ada-002": ModelPricing(0.10, 0.0),
    },

    # Anthropic models
    "anthropic": {
        # Claude 3.5
        "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
        "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00),
        "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
        "claude-3-5-haiku-latest": ModelPricing(0.80, 4.00),
        # Claude 3
        "claude-3-opus-20240229": ModelPricing(15.00, 75.00),
        "claude-3-opus-latest": ModelPricing(15.00, 75.00),
        "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00),
        "claude-3-haiku-20240307": ModelPricing(0.25, 1.25),
        # Legacy aliases
        "claude-3-5-sonnet": ModelPricing(3.00, 15.00),
        "claude-3-opus": ModelPricing(15.00, 75.00),
        "claude-3-sonnet": ModelPricing(3.00, 15.00),
        "claude-3-haiku": ModelPricing(0.25, 1.25),
    },

    # Google AI (Gemini)
    "google": {
        "gemini-1.5-pro": ModelPricing(1.25, 5.00),
        "gemini-1.5-pro-latest": ModelPricing(1.25, 5.00),
        "gemini-1.5-flash": ModelPricing(0.075, 0.30),
        "gemini-1.5-flash-latest": ModelPricing(0.075, 0.30),
        "gemini-pro": ModelPricing(0.50, 1.50),
        "gemini-2.0-flash-exp": ModelPricing(0.075, 0.30),
    },

    # Groq (fast inference)
    "groq": {
        "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
        "llama-3.1-70b-versatile": ModelPricing(0.59, 0.79),
        "llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
        "llama3-70b-8192": ModelPricing(0.59, 0.79),
        "llama3-8b-8192": ModelPricing(0.05, 0.08),
        "mixtral-8x7b-32768": ModelPricing(0.24, 0.24),
        "gemma-7b-it": ModelPricing(0.07, 0.07),
        "gemma2-9b-it": ModelPricing(0.20, 0.20),
    },

    # Together AI
    "together": {
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ModelPricing(3.50, 3.50),
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelPricing(0.18, 0.18),
        "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelPricing(1.20, 1.20),
        "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelPricing(0.60, 0.60),
        "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelPricing(1.20, 1.20),
        # Embeddings
        "togethercomputer/m2-bert-80M-8k-retrieval": ModelPricing(0.008, 0.0),
        "BAAI/bge-large-en-v1.5": ModelPricing(0.02, 0.0),
    },

    # Cohere
    "cohere": {
        "command-r-plus": ModelPricing(2.50, 10.00),
        "command-r": ModelPricing(0.15, 0.60),
        "command": ModelPricing(1.00, 2.00),
        "command-light": ModelPricing(0.30, 0.60),
        # Embeddings
        "embed-english-v3.0": ModelPricing(0.10, 0.0),
        "embed-multilingual-v3.0": ModelPricing(0.10, 0.0),
        "embed-english-light-v3.0": ModelPricing(0.10, 0.0),
    },

    # Azure OpenAI (same pricing as OpenAI)
    "azure": {
        "gpt-4o": ModelPricing(5.00, 15.00),
        "gpt-4o-mini": ModelPricing(0.15, 0.60),
        "gpt-4-turbo": ModelPricing(10.00, 30.00),
        "gpt-4": ModelPricing(30.00, 60.00),
        "gpt-35-turbo": ModelPricing(0.50, 1.50),
        "text-embedding-3-large": ModelPricing(0.13, 0.0),
        "text-embedding-3-small": ModelPricing(0.02, 0.0),
    },

    # Ollama (local - all free)
    "ollama": {
        "llama3.2": ModelPricing(0.0, 0.0, is_local=True),
        "llama3.2:1b": ModelPricing(0.0, 0.0, is_local=True),
        "llama3.2:3b": ModelPricing(0.0, 0.0, is_local=True),
        "llama3.1": ModelPricing(0.0, 0.0, is_local=True),
        "llama3.1:8b": ModelPricing(0.0, 0.0, is_local=True),
        "llama3.1:70b": ModelPricing(0.0, 0.0, is_local=True),
        "llama3": ModelPricing(0.0, 0.0, is_local=True),
        "mistral": ModelPricing(0.0, 0.0, is_local=True),
        "mixtral": ModelPricing(0.0, 0.0, is_local=True),
        "codellama": ModelPricing(0.0, 0.0, is_local=True),
        "phi3": ModelPricing(0.0, 0.0, is_local=True),
        "qwen2.5": ModelPricing(0.0, 0.0, is_local=True),
        "gemma2": ModelPricing(0.0, 0.0, is_local=True),
        "nomic-embed-text": ModelPricing(0.0, 0.0, is_local=True),
        "mxbai-embed-large": ModelPricing(0.0, 0.0, is_local=True),
        # Catch-all for any Ollama model
        "_default": ModelPricing(0.0, 0.0, is_local=True),
    },

    # Custom OpenAI-compatible (default to zero, can be overridden)
    "custom": {
        "_default": ModelPricing(0.0, 0.0),
    },
}


class LLMPricingService:
    """Service for calculating LLM usage costs."""

    @classmethod
    def get_model_pricing(
        cls,
        provider_type: str,
        model: str,
    ) -> Optional[ModelPricing]:
        """
        Get pricing information for a specific model.

        Args:
            provider_type: The provider type (openai, anthropic, etc.)
            model: The model name

        Returns:
            ModelPricing or None if not found
        """
        provider_type = provider_type.lower()

        if provider_type not in PRICING_DATA:
            logger.warning(
                "Unknown provider type for pricing",
                provider_type=provider_type,
                model=model,
            )
            return None

        provider_prices = PRICING_DATA[provider_type]

        # Try exact match first
        if model in provider_prices:
            return provider_prices[model]

        # Try without version suffix (e.g., "gpt-4o-2024-11-20" -> "gpt-4o")
        base_model = model.split("-")[0] if "-" in model else model
        for key in provider_prices:
            if key.startswith(base_model):
                return provider_prices[key]

        # Try prefix match for model families
        for key in provider_prices:
            if model.startswith(key) or key.startswith(model):
                return provider_prices[key]

        # Use default if available
        if "_default" in provider_prices:
            return provider_prices["_default"]

        logger.warning(
            "No pricing found for model",
            provider_type=provider_type,
            model=model,
        )
        return None

    @classmethod
    def calculate_cost(
        cls,
        provider_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict[str, Optional[float]]:
        """
        Calculate the cost for a single LLM call.

        Args:
            provider_type: The provider type
            model: The model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with input_cost_usd, output_cost_usd, total_cost_usd
        """
        pricing = cls.get_model_pricing(provider_type, model)

        if pricing is None:
            return {
                "input_cost_usd": None,
                "output_cost_usd": None,
                "total_cost_usd": None,
            }

        # Calculate costs (prices are per million tokens)
        input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_million

        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }

    @classmethod
    def estimate_embedding_cost(
        cls,
        provider_type: str,
        model: str,
        token_count: int,
    ) -> Optional[float]:
        """
        Estimate embedding cost.

        Args:
            provider_type: The provider type
            model: The model name
            token_count: Number of tokens to embed

        Returns:
            Estimated cost in USD
        """
        pricing = cls.get_model_pricing(provider_type, model)

        if pricing is None:
            return None

        # Embeddings only use input pricing
        return round((token_count / 1_000_000) * pricing.input_price_per_million, 6)

    @classmethod
    def is_local_model(cls, provider_type: str, model: str = "") -> bool:
        """Check if a model is local (free)."""
        provider_type = provider_type.lower()

        # Ollama is always local
        if provider_type == "ollama":
            return True

        # Check if the model has is_local flag
        pricing = cls.get_model_pricing(provider_type, model)
        return pricing.is_local if pricing else False

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of providers with pricing data."""
        return list(PRICING_DATA.keys())

    @classmethod
    def get_provider_models(cls, provider_type: str) -> list:
        """Get list of models with pricing for a provider."""
        provider_type = provider_type.lower()
        if provider_type not in PRICING_DATA:
            return []
        return [k for k in PRICING_DATA[provider_type].keys() if not k.startswith("_")]
