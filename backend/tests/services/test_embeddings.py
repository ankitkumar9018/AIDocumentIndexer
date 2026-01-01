"""
AIDocumentIndexer - Embeddings Service Tests
=============================================

Tests for the embedding service including caching, batching, and providers.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import hashlib

from backend.services.embeddings import (
    EmbeddingService,
    EmbeddingResult,
    get_optimal_batch_size,
    get_cache_stats,
    clear_embedding_cache,
    PROVIDER_BATCH_CONFIG,
    _embedding_cache,
)
from backend.processors.chunker import Chunk


# =============================================================================
# Batch Size Tests
# =============================================================================

class TestBatchSizeOptimization:
    """Tests for optimal batch size calculation."""

    def test_openai_batch_size(self):
        """Test OpenAI optimal batch size."""
        assert get_optimal_batch_size("openai", 1000) == 500
        assert get_optimal_batch_size("openai", 100) == 100
        assert get_optimal_batch_size("openai", 50) == 50

    def test_ollama_batch_size(self):
        """Test Ollama optimal batch size (smaller for local models)."""
        assert get_optimal_batch_size("ollama", 1000) == 50
        assert get_optimal_batch_size("ollama", 30) == 30

    def test_huggingface_batch_size(self):
        """Test HuggingFace optimal batch size."""
        assert get_optimal_batch_size("huggingface", 500) == 100
        assert get_optimal_batch_size("huggingface", 50) == 50

    def test_unknown_provider_defaults_to_openai(self):
        """Test unknown provider uses OpenAI defaults."""
        assert get_optimal_batch_size("unknown_provider", 1000) == 500

    def test_small_batch_uses_actual_size(self):
        """Test that small batches use actual size, not optimal."""
        assert get_optimal_batch_size("openai", 10) == 10
        assert get_optimal_batch_size("ollama", 10) == 10


# =============================================================================
# Cache Tests
# =============================================================================

class TestEmbeddingCache:
    """Tests for embedding cache functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_embedding_cache()

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        global _embedding_cache
        _embedding_cache["test_hash"] = [0.1] * 1536

        clear_embedding_cache()

        stats = get_cache_stats()
        assert stats["size"] == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        stats = get_cache_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "utilization_percent" in stats
        assert stats["size"] == 0
        assert stats["utilization_percent"] == 0.0


# =============================================================================
# Embedding Service Tests
# =============================================================================

class TestEmbeddingService:
    """Tests for the EmbeddingService class."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_embedding_cache()

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock LangChain embeddings."""
        mock = MagicMock()
        mock.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        mock.embed_query.return_value = [0.3] * 1536
        return mock

    @pytest.fixture
    def embedding_service(self, mock_embeddings):
        """Create embedding service with mocked embeddings."""
        service = EmbeddingService.__new__(EmbeddingService)
        service.provider = "openai"
        service.model = "text-embedding-3-small"
        service.config = None
        service._embeddings = mock_embeddings
        service._dimensions = 1536
        return service

    def test_embed_text(self, embedding_service, mock_embeddings):
        """Test embedding a single text."""
        result = embedding_service.embed_text("Hello world")

        assert len(result) == 1536
        mock_embeddings.embed_query.assert_called_once_with("Hello world")

    def test_embed_texts_basic(self, embedding_service, mock_embeddings):
        """Test embedding multiple texts."""
        texts = ["Hello", "World"]
        result = embedding_service.embed_texts(texts, use_cache=False)

        assert len(result) == 2
        assert len(result[0]) == 1536
        mock_embeddings.embed_documents.assert_called_once()

    def test_embed_texts_with_empty_strings(self, embedding_service, mock_embeddings):
        """Test embedding handles empty strings."""
        mock_embeddings.embed_documents.return_value = [[0.1] * 1536]
        texts = ["Hello", "", "  ", "World"]
        result = embedding_service.embed_texts(texts, use_cache=False)

        assert len(result) == 4
        # Empty strings should get zero vectors
        assert result[1] == [0.0] * 1536
        assert result[2] == [0.0] * 1536

    def test_embed_texts_caching(self, embedding_service, mock_embeddings):
        """Test that caching avoids re-embedding identical content."""
        mock_embeddings.embed_documents.return_value = [[0.5] * 1536]

        # First call - should embed
        result1 = embedding_service.embed_texts(["Hello"], use_cache=True)
        assert mock_embeddings.embed_documents.call_count == 1

        # Second call with same text - should use cache
        result2 = embedding_service.embed_texts(["Hello"], use_cache=True)
        # embed_documents should NOT be called again
        assert mock_embeddings.embed_documents.call_count == 1

        # Results should be the same
        assert result1 == result2

    def test_embed_texts_cache_disabled(self, embedding_service, mock_embeddings):
        """Test embedding without cache."""
        mock_embeddings.embed_documents.return_value = [[0.5] * 1536]

        embedding_service.embed_texts(["Hello"], use_cache=False)
        embedding_service.embed_texts(["Hello"], use_cache=False)

        # Should call embed_documents twice
        assert mock_embeddings.embed_documents.call_count == 2

    def test_content_hash_includes_model(self, embedding_service):
        """Test that content hash includes model name for cache isolation."""
        hash1 = embedding_service._get_content_hash("Hello")

        # Change model name
        embedding_service.model = "different-model"
        hash2 = embedding_service._get_content_hash("Hello")

        # Hashes should be different
        assert hash1 != hash2


# =============================================================================
# Chunk Embedding Tests
# =============================================================================

class TestChunkEmbedding:
    """Tests for embedding document chunks."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_embedding_cache()

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock LangChain embeddings."""
        mock = MagicMock()
        mock.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        return mock

    @pytest.fixture
    def embedding_service(self, mock_embeddings):
        """Create embedding service with mocked embeddings."""
        service = EmbeddingService.__new__(EmbeddingService)
        service.provider = "openai"
        service.model = "text-embedding-3-small"
        service.config = None
        service._embeddings = mock_embeddings
        service._dimensions = 1536
        return service

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                content="First chunk content",
                chunk_index=0,
                chunk_hash="hash1",
                document_id="doc-1",
                metadata={"page": 1},
            ),
            Chunk(
                content="Second chunk content",
                chunk_index=1,
                chunk_hash="hash2",
                document_id="doc-1",
                metadata={"page": 1},
            ),
            Chunk(
                content="Third chunk content",
                chunk_index=2,
                chunk_hash="hash3",
                document_id="doc-1",
                metadata={"page": 2},
            ),
        ]

    def test_embed_chunks(self, embedding_service, sample_chunks):
        """Test embedding document chunks."""
        results = embedding_service.embed_chunks(sample_chunks)

        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)
        assert results[0].chunk_id == "doc-1_0"
        assert results[1].chunk_id == "doc-1_1"
        assert results[2].chunk_id == "doc-1_2"

    def test_embed_chunks_preserves_metadata(self, embedding_service, sample_chunks):
        """Test that chunk metadata is preserved in results."""
        results = embedding_service.embed_chunks(sample_chunks)

        assert results[0].chunk_hash == "hash1"
        assert results[0].model == "text-embedding-3-small"
        assert results[0].dimensions == 1536

    def test_embed_chunks_with_cache(self, embedding_service, sample_chunks, mock_embeddings):
        """Test that chunk embedding uses cache."""
        # First embedding
        results1 = embedding_service.embed_chunks(sample_chunks, use_cache=True)
        call_count_1 = mock_embeddings.embed_documents.call_count

        # Second embedding of same chunks - should hit cache
        results2 = embedding_service.embed_chunks(sample_chunks, use_cache=True)
        call_count_2 = mock_embeddings.embed_documents.call_count

        # Should not make additional calls
        assert call_count_1 == call_count_2

    def test_embed_empty_chunks(self, embedding_service):
        """Test embedding empty chunk list."""
        results = embedding_service.embed_chunks([])
        assert results == []


# =============================================================================
# Provider Configuration Tests
# =============================================================================

class TestProviderConfiguration:
    """Tests for provider-specific configurations."""

    def test_openai_config_exists(self):
        """Test OpenAI configuration is defined."""
        assert "openai" in PROVIDER_BATCH_CONFIG
        config = PROVIDER_BATCH_CONFIG["openai"]
        assert config["max_batch_size"] == 2048
        assert config["optimal_batch_size"] == 500

    def test_ollama_config_exists(self):
        """Test Ollama configuration is defined."""
        assert "ollama" in PROVIDER_BATCH_CONFIG
        config = PROVIDER_BATCH_CONFIG["ollama"]
        assert config["max_batch_size"] == 100
        assert config["optimal_batch_size"] == 50

    def test_huggingface_config_exists(self):
        """Test HuggingFace configuration is defined."""
        assert "huggingface" in PROVIDER_BATCH_CONFIG
        config = PROVIDER_BATCH_CONFIG["huggingface"]
        assert config["max_batch_size"] == 256
        assert config["optimal_batch_size"] == 100
