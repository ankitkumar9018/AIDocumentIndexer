"""
AIDocumentIndexer - LLM Service Tests
======================================

Unit tests for the LLM service.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from backend.services.llm import (
    LLMConfig,
    LLMFactory,
    get_chat_model,
    get_embeddings,
    generate_response,
    generate_embedding,
    generate_embeddings,
)


# =============================================================================
# LLMConfig Tests
# =============================================================================

class TestLLMConfig:
    """Tests for LLMConfig class."""

    def test_default_values(self):
        """Test LLMConfig has expected default values."""
        with patch.dict("os.environ", {}, clear=True):
            config = LLMConfig()

            assert config.default_provider == "openai"
            assert config.openai_chat_model == "gpt-4o"
            assert config.openai_embedding_model == "text-embedding-3-small"
            assert config.ollama_enabled is True
            assert config.ollama_host == "http://localhost:11434"
            assert config.default_temperature == 0.7
            assert config.default_max_tokens == 4096
            assert config.embedding_dimension == 1536

    def test_custom_values(self):
        """Test LLMConfig reads from environment variables."""
        env_vars = {
            "DEFAULT_LLM_PROVIDER": "anthropic",
            "OPENAI_CHAT_MODEL": "gpt-4-turbo",
            "OLLAMA_ENABLED": "false",
            "DEFAULT_TEMPERATURE": "0.5",
            "DEFAULT_MAX_TOKENS": "8192",
        }

        with patch.dict("os.environ", env_vars):
            config = LLMConfig()

            assert config.default_provider == "anthropic"
            assert config.openai_chat_model == "gpt-4-turbo"
            assert config.ollama_enabled is False
            assert config.default_temperature == 0.5
            assert config.default_max_tokens == 8192


# =============================================================================
# LLMFactory Tests
# =============================================================================

class TestLLMFactory:
    """Tests for LLMFactory class."""

    def setup_method(self):
        """Clear factory caches before each test."""
        LLMFactory._instances.clear()
        LLMFactory._embedding_instances.clear()

    @patch("backend.services.llm.HAS_LITELLM", False)
    @patch("backend.services.llm.ChatOpenAI")
    def test_get_chat_model_openai(self, mock_chat_openai):
        """Test getting OpenAI chat model."""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        model = LLMFactory.get_chat_model(provider="openai", model="gpt-4o")

        assert model == mock_model
        mock_chat_openai.assert_called_once()

    @patch("backend.services.llm.HAS_LITELLM", False)
    @patch("backend.services.llm.ChatOpenAI")
    def test_get_chat_model_caching(self, mock_chat_openai):
        """Test that chat models are cached."""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        model1 = LLMFactory.get_chat_model(provider="openai", model="gpt-4o")
        model2 = LLMFactory.get_chat_model(provider="openai", model="gpt-4o")

        assert model1 is model2
        assert mock_chat_openai.call_count == 1

    @patch("backend.services.llm.HAS_LITELLM", False)
    @patch("backend.services.llm.ChatOpenAI")
    def test_get_chat_model_different_params(self, mock_chat_openai):
        """Test different params create different model instances."""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        LLMFactory.get_chat_model(provider="openai", temperature=0.5)
        LLMFactory.get_chat_model(provider="openai", temperature=0.8)

        assert mock_chat_openai.call_count == 2

    @patch("backend.services.llm.OpenAIEmbeddings")
    def test_get_embeddings_openai(self, mock_embeddings):
        """Test getting OpenAI embeddings."""
        mock_embed = MagicMock()
        mock_embeddings.return_value = mock_embed

        embeddings = LLMFactory.get_embeddings(provider="openai")

        assert embeddings == mock_embed
        mock_embeddings.assert_called_once()

    @patch("backend.services.llm.OpenAIEmbeddings")
    def test_get_embeddings_caching(self, mock_embeddings):
        """Test that embeddings are cached."""
        mock_embed = MagicMock()
        mock_embeddings.return_value = mock_embed

        embed1 = LLMFactory.get_embeddings(provider="openai")
        embed2 = LLMFactory.get_embeddings(provider="openai")

        assert embed1 is embed2
        assert mock_embeddings.call_count == 1


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Clear factory caches before each test."""
        LLMFactory._instances.clear()
        LLMFactory._embedding_instances.clear()

    @patch.object(LLMFactory, "get_chat_model")
    def test_get_chat_model_function(self, mock_factory):
        """Test get_chat_model convenience function."""
        mock_model = MagicMock()
        mock_factory.return_value = mock_model

        result = get_chat_model(provider="openai", model="gpt-4o")

        assert result == mock_model
        mock_factory.assert_called_once_with(provider="openai", model="gpt-4o")

    @patch.object(LLMFactory, "get_embeddings")
    def test_get_embeddings_function(self, mock_factory):
        """Test get_embeddings convenience function."""
        mock_embed = MagicMock()
        mock_factory.return_value = mock_embed

        result = get_embeddings(provider="openai")

        assert result == mock_embed
        mock_factory.assert_called_once_with(provider="openai", model=None)

    @pytest.mark.asyncio
    @patch.object(LLMFactory, "get_chat_model")
    async def test_generate_response(self, mock_factory):
        """Test generate_response function."""
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = MagicMock(content="Hello!")
        mock_factory.return_value = mock_model

        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="Hi")]

        result = await generate_response(messages, provider="openai")

        assert result == "Hello!"
        mock_model.ainvoke.assert_called_once_with(messages)

    @pytest.mark.asyncio
    @patch.object(LLMFactory, "get_embeddings")
    async def test_generate_embedding(self, mock_factory):
        """Test generate_embedding function."""
        mock_embed = AsyncMock()
        mock_embed.aembed_query.return_value = [0.1] * 1536
        mock_factory.return_value = mock_embed

        result = await generate_embedding("test text")

        assert len(result) == 1536
        mock_embed.aembed_query.assert_called_once_with("test text")

    @pytest.mark.asyncio
    @patch.object(LLMFactory, "get_embeddings")
    async def test_generate_embeddings_batch(self, mock_factory):
        """Test generate_embeddings function for batch processing."""
        mock_embed = AsyncMock()
        mock_embed.aembed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        mock_factory.return_value = mock_embed

        result = await generate_embeddings(["text 1", "text 2"])

        assert len(result) == 2
        assert len(result[0]) == 1536
        mock_embed.aembed_documents.assert_called_once_with(["text 1", "text 2"])


# =============================================================================
# LiteLLM Integration Tests
# =============================================================================

class TestLiteLLMIntegration:
    """Tests for LiteLLM integration."""

    def setup_method(self):
        """Clear factory caches before each test."""
        LLMFactory._instances.clear()
        LLMFactory._embedding_instances.clear()

    @patch("backend.services.llm.HAS_LITELLM", True)
    @patch("backend.services.llm.ChatLiteLLM")
    def test_litellm_openai_model_string(self, mock_litellm):
        """Test LiteLLM uses correct model string for OpenAI."""
        mock_model = MagicMock()
        mock_litellm.return_value = mock_model

        LLMFactory._create_litellm_model(
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
        )

        call_args = mock_litellm.call_args
        assert call_args.kwargs["model"] == "gpt-4o"

    @patch("backend.services.llm.HAS_LITELLM", True)
    @patch("backend.services.llm.ChatLiteLLM")
    def test_litellm_ollama_model_string(self, mock_litellm):
        """Test LiteLLM uses correct model string for Ollama."""
        mock_model = MagicMock()
        mock_litellm.return_value = mock_model

        LLMFactory._create_litellm_model(
            provider="ollama",
            model="llama3.2",
            temperature=0.7,
            max_tokens=4096,
        )

        call_args = mock_litellm.call_args
        assert call_args.kwargs["model"] == "ollama/llama3.2"

    @patch("backend.services.llm.HAS_LITELLM", True)
    @patch("backend.services.llm.ChatLiteLLM")
    def test_litellm_anthropic_model_string(self, mock_litellm):
        """Test LiteLLM uses correct model string for Anthropic."""
        mock_model = MagicMock()
        mock_litellm.return_value = mock_model

        LLMFactory._create_litellm_model(
            provider="anthropic",
            model="claude-3-5-sonnet",
            temperature=0.7,
            max_tokens=4096,
        )

        call_args = mock_litellm.call_args
        assert call_args.kwargs["model"] == "anthropic/claude-3-5-sonnet"
