"""
AIDocumentIndexer - Embedding Service
=====================================

Embedding generation with Ray-parallel processing.
Supports multiple embedding providers via LangChain.

Performance optimizations:
- Adaptive batching based on provider rate limits
- Embedding cache to avoid re-embedding identical content
- Concurrent processing with ThreadPoolExecutor fallback
- Ray-parallel processing for large batches
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
import hashlib
import os
import structlog
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# LangChain embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

# Phase 29: Voyage AI embeddings support
try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False

# Phase 68: Qwen3 Embedding support (70.58 MTEB score - top performer)
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

# Phase 69: Google Gemini Embedding (gemini-embedding-001) - #1 on MTEB Multilingual
# Phase 87: Migrated from deprecated google-generativeai to google-genai SDK
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Ray for parallel processing
import ray

from backend.processors.chunker import Chunk
from backend.services.llm import LLMConfig

logger = structlog.get_logger(__name__)


# =============================================================================
# Phase 29: Voyage AI Embeddings Wrapper
# =============================================================================

class VoyageAIEmbeddings(Embeddings):
    """
    Voyage AI embeddings wrapper implementing LangChain Embeddings interface.

    Voyage AI models lead MTEB benchmarks and are built by Stanford researchers
    specializing in RAG with training data that includes "tricky" negatives.

    Models:
    - voyage-3-large: Best overall (1024D)
    - voyage-3-lite: Cost-effective (512D)
    - voyage-code-3: Code-optimized
    - voyage-finance-2: Finance domain
    - voyage-law-2: Legal domain
    """

    def __init__(
        self,
        model: str = "voyage-3-large",
        api_key: Optional[str] = None,
        batch_size: int = 128,
    ):
        """
        Initialize Voyage AI embeddings.

        Args:
            model: Voyage model name
            api_key: Voyage API key (or from VOYAGE_API_KEY env var)
            batch_size: Max texts per batch (Voyage supports up to 128)
        """
        if not VOYAGEAI_AVAILABLE:
            raise ImportError(
                "voyageai package not installed. Install with: pip install voyageai"
            )

        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.batch_size = min(batch_size, 128)  # Voyage max is 128

        if not self.api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize Voyage client
        self._client = voyageai.Client(api_key=self.api_key)

        logger.info(
            "Initialized Voyage AI embeddings",
            model=model,
            batch_size=self.batch_size,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                result = self._client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="document",
                )
                all_embeddings.extend(result.embeddings)

            except Exception as e:
                logger.error(
                    "Voyage embedding failed",
                    error=str(e),
                    batch_size=len(batch),
                )
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector - get dimensions from model
            dims = EmbeddingService.MODEL_DIMENSIONS.get(self.model, 1024)
            return [0.0] * dims

        try:
            result = self._client.embed(
                texts=[text],
                model=self.model,
                input_type="query",  # Use query input type for search queries
            )
            return result.embeddings[0]

        except Exception as e:
            logger.error("Voyage query embedding failed", error=str(e))
            raise

# =============================================================================
# Phase 68: Qwen3 Embeddings (70.58 MTEB - Top Performer)
# =============================================================================

class Qwen3Embeddings(Embeddings):
    """
    Qwen3 Embedding wrapper implementing LangChain Embeddings interface.

    Qwen3-Embedding-8B achieves the highest MTEB score (70.58) as of Jan 2026,
    outperforming OpenAI, Voyage, BGE, and E5 models.

    Features:
    - 70.58 MTEB score (top performer)
    - 4096 embedding dimensions
    - 8192 token context length
    - 100+ language support (multilingual)
    - Instruction-based embedding for better retrieval

    Models:
    - Qwen3-Embedding-8B: Best quality (4096D)
    - Qwen3-Embedding-4B: Balanced (2048D)
    - Qwen3-Embedding-0.6B: Fast/lightweight (1024D)
    """

    # Task-specific instructions for better retrieval
    TASK_INSTRUCTIONS = {
        "document": "Instruct: Represent this document for retrieval\nQuery: ",
        "query": "Instruct: Represent this query for searching relevant documents\nQuery: ",
        "classification": "Instruct: Classify this text\nQuery: ",
        "clustering": "Instruct: Cluster this text\nQuery: ",
    }

    def __init__(
        self,
        model: str = "Alibaba-NLP/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 8192,
        normalize: bool = True,
        use_fp16: bool = True,
    ):
        """
        Initialize Qwen3 embeddings.

        Args:
            model: Qwen3 model name (HuggingFace model ID)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Max texts per batch
            max_length: Max token length (Qwen3 supports up to 8192)
            normalize: Whether to L2 normalize embeddings
            use_fp16: Use FP16 for faster inference (GPU only)
        """
        if not QWEN3_AVAILABLE:
            raise ImportError(
                "transformers and torch packages required for Qwen3 embeddings. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model
        self.batch_size = batch_size
        self.max_length = min(max_length, 8192)
        self.normalize = normalize
        self.use_fp16 = use_fp16

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        logger.info(
            "Loading Qwen3 embedding model",
            model=model,
            device=self.device,
            use_fp16=use_fp16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )

        # Load with appropriate dtype
        if self.use_fp16 and self.device == "cuda":
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
            ).to(self.device)

        self._model.eval()

        logger.info(
            "Qwen3 embedding model loaded",
            model=model,
            device=self.device,
            dimensions=self._get_dimensions(),
        )

    def _get_dimensions(self) -> int:
        """Get embedding dimensions for the loaded model."""
        # Qwen3 embedding dimensions by model size
        if "8B" in self.model_name:
            return 4096
        elif "4B" in self.model_name:
            return 2048
        elif "0.6B" in self.model_name or "600M" in self.model_name:
            return 1024
        else:
            # Default to 8B dimensions
            return 4096

    def _get_embeddings(
        self,
        texts: List[str],
        instruction_prefix: str = "",
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            instruction_prefix: Task-specific instruction prefix

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Add instruction prefix if provided
            if instruction_prefix:
                batch_texts = [instruction_prefix + t for t in batch_texts]

            try:
                # Tokenize
                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                # Generate embeddings
                with torch.no_grad():
                    outputs = self._model(**inputs)

                    # Use last hidden state with mean pooling
                    # Qwen3 uses the [CLS] token or mean pooling
                    attention_mask = inputs["attention_mask"]
                    hidden_states = outputs.last_hidden_state

                    # Mean pooling over non-padding tokens
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                    # Normalize if requested
                    if self.normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # Convert to list
                    batch_embeddings = embeddings.cpu().tolist()
                    all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(
                    "Qwen3 embedding failed",
                    error=str(e),
                    batch_size=len(batch_texts),
                )
                raise

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._get_embeddings(
            texts,
            instruction_prefix=self.TASK_INSTRUCTIONS["document"],
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector
            dims = self._get_dimensions()
            return [0.0] * dims

        embeddings = self._get_embeddings(
            [text],
            instruction_prefix=self.TASK_INSTRUCTIONS["query"],
        )
        return embeddings[0] if embeddings else [0.0] * self._get_dimensions()


# =============================================================================
# Phase 69: Google Gemini Embeddings (#1 on MTEB Multilingual)
# =============================================================================

class GeminiEmbeddings(Embeddings):
    """
    Google Gemini Embedding wrapper implementing LangChain Embeddings interface.

    gemini-embedding-001 achieves 71.5% accuracy and is #1 on MTEB Multilingual
    leaderboard as of Jan 2026.

    Features:
    - 71.5% MTEB accuracy (#1 multilingual)
    - 3072-dimensional embeddings (truncatable to 768/1536)
    - Matryoshka Representation Learning for flexible dimensions
    - 2048 token context
    - $0.15 per 1M tokens
    - 81-87% recall in production tests (Box, Everlaw)

    Task types:
    - RETRIEVAL_DOCUMENT: For indexing documents
    - RETRIEVAL_QUERY: For search queries
    - SEMANTIC_SIMILARITY: For comparing texts
    - CLASSIFICATION: For text classification
    - CLUSTERING: For text clustering
    """

    # Gemini task types for better retrieval
    TASK_TYPES = {
        "document": "RETRIEVAL_DOCUMENT",
        "query": "RETRIEVAL_QUERY",
        "similarity": "SEMANTIC_SIMILARITY",
        "classification": "CLASSIFICATION",
        "clustering": "CLUSTERING",
    }

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        dimensions: int = 3072,  # Full dimensions (or 768, 1536 via Matryoshka)
        batch_size: int = 100,
    ):
        """
        Initialize Gemini embeddings.

        Args:
            model: Gemini embedding model name
            api_key: Google API key (or from GOOGLE_API_KEY env var)
            dimensions: Output dimensions (768, 1536, or 3072)
            batch_size: Max texts per batch
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.dimensions = dimensions
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize Gemini client (Phase 87: new unified google-genai SDK)
        self._client = google_genai.Client(api_key=self.api_key)

        # Validate dimensions (Matryoshka supports 768, 1536, 3072)
        if dimensions not in (768, 1536, 3072):
            logger.warning(
                f"Gemini dimension {dimensions} not standard. "
                "Using 3072 (supported: 768, 1536, 3072)"
            )
            self.dimensions = 3072

        logger.info(
            "Initialized Google Gemini embeddings",
            model=model,
            dimensions=self.dimensions,
            batch_size=batch_size,
        )

    def _embed_content(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """
        Generate embeddings using Gemini API.

        Args:
            texts: List of texts to embed
            task_type: Gemini task type for optimization

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                # Gemini batch embedding (Phase 87: new google-genai SDK)
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimensions,
                    ),
                )

                # Extract embedding vectors from response
                all_embeddings.extend([e.values for e in result.embeddings])

            except Exception as e:
                logger.error(
                    "Gemini embedding failed",
                    error=str(e),
                    batch_size=len(batch),
                )
                raise

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._embed_content(
            texts,
            task_type=self.TASK_TYPES["document"],
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * self.dimensions

        try:
            embeddings = self._embed_content(
                [text],
                task_type=self.TASK_TYPES["query"],
            )
            return embeddings[0] if embeddings else [0.0] * self.dimensions

        except Exception as e:
            logger.error("Gemini query embedding failed", error=str(e))
            raise


# =============================================================================
# Phase 69: NVIDIA NV-Embed-v2 (69.32 MTEB - GPU Optimized)
# =============================================================================

class NVEmbedEmbeddings(Embeddings):
    """
    NVIDIA NV-Embed-v2 embeddings wrapper implementing LangChain Embeddings interface.

    NV-Embed-v2 achieves 69.32 on MTEB benchmark (comparable to top models).
    Optimized for NVIDIA GPUs with better batching and throughput.

    Features:
    - 69.32 MTEB score (top-tier performance)
    - 4096 embedding dimensions
    - 32K token context length
    - GPU-optimized for high throughput
    - Instruction-following for task-specific embeddings

    Model: nvidia/NV-Embed-v2 (HuggingFace)
    """

    TASK_INSTRUCTIONS = {
        "document": "Represent this document for retrieval: ",
        "query": "Represent this query for retrieving relevant documents: ",
    }

    def __init__(
        self,
        model: str = "nvidia/NV-Embed-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 32768,
        normalize: bool = True,
        use_fp16: bool = True,
    ):
        """
        Initialize NV-Embed-v2 embeddings.

        Args:
            model: NV-Embed model name (HuggingFace model ID)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Max texts per batch
            max_length: Max token length (NV-Embed-v2 supports up to 32K)
            normalize: Whether to L2 normalize embeddings
            use_fp16: Use FP16 for faster inference (GPU only)
        """
        if not QWEN3_AVAILABLE:  # Uses same transformers/torch deps
            raise ImportError(
                "transformers and torch packages required for NV-Embed embeddings. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model
        self.batch_size = batch_size
        self.max_length = min(max_length, 32768)
        self.normalize = normalize
        self.use_fp16 = use_fp16

        # Auto-detect device (prefer CUDA for NV-Embed)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Log warning if not using GPU
        if self.device != "cuda":
            logger.warning(
                "NV-Embed-v2 is optimized for NVIDIA GPUs. "
                "Performance may be significantly slower on CPU."
            )

        logger.info(
            "Loading NV-Embed-v2 embedding model",
            model=model,
            device=self.device,
            use_fp16=use_fp16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )

        # Load with appropriate dtype
        if self.use_fp16 and self.device == "cuda":
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
            ).to(self.device)

        self._model.eval()

        logger.info(
            "NV-Embed-v2 model loaded",
            model=model,
            device=self.device,
            dimensions=4096,
        )

    def _get_embeddings(
        self,
        texts: List[str],
        instruction_prefix: str = "",
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            instruction_prefix: Task-specific instruction prefix

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Add instruction prefix
            if instruction_prefix:
                batch = [instruction_prefix + text for text in batch]

            # Tokenize
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)

                # Mean pooling over token embeddings
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Masked mean
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # Normalize if requested
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to list
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._get_embeddings(
            texts,
            instruction_prefix=self.TASK_INSTRUCTIONS["document"],
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * 4096

        embeddings = self._get_embeddings(
            [text],
            instruction_prefix=self.TASK_INSTRUCTIONS["query"],
        )
        return embeddings[0] if embeddings else [0.0] * 4096


# =============================================================================
# Phase 69: BGE-M3 Multi-Retrieval Embeddings
# =============================================================================

class BGEM3Embeddings(Embeddings):
    """
    BGE-M3 embeddings wrapper implementing LangChain Embeddings interface.

    BGE-M3 (BAAI General Embedding M3) is a multi-retrieval model that supports:
    - Dense retrieval (standard embedding similarity)
    - Sparse retrieval (lexical matching)
    - ColBERT-style late interaction

    Features:
    - 64.3 MTEB score (strong multilingual performance)
    - 1024 embedding dimensions
    - 8192 token context length
    - 100+ language support
    - Multi-retrieval capabilities in single model

    Model: BAAI/bge-m3 (HuggingFace)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 8192,
        normalize: bool = True,
        use_fp16: bool = True,
    ):
        """
        Initialize BGE-M3 embeddings.

        Args:
            model: BGE-M3 model name (HuggingFace model ID)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Max texts per batch
            max_length: Max token length (BGE-M3 supports up to 8192)
            normalize: Whether to L2 normalize embeddings
            use_fp16: Use FP16 for faster inference (GPU only)
        """
        if not QWEN3_AVAILABLE:  # Uses same transformers/torch deps
            raise ImportError(
                "transformers and torch packages required for BGE-M3 embeddings. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model
        self.batch_size = batch_size
        self.max_length = min(max_length, 8192)
        self.normalize = normalize
        self.use_fp16 = use_fp16

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Loading BGE-M3 embedding model",
            model=model,
            device=self.device,
            use_fp16=use_fp16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )

        # Load with appropriate dtype
        if self.use_fp16 and self.device == "cuda":
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
            ).to(self.device)

        self._model.eval()

        logger.info(
            "BGE-M3 model loaded",
            model=model,
            device=self.device,
            dimensions=1024,
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)

                # Use CLS token embedding for BGE-M3
                embeddings = outputs.last_hidden_state[:, 0]

                # Normalize if requested
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to list
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        BGE-M3 recommends adding instruction prefix for queries.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * 1024

        # BGE-M3 uses instruction prefix for queries
        prefixed_text = f"Represent this sentence for searching relevant passages: {text}"
        embeddings = self._get_embeddings([prefixed_text])
        return embeddings[0] if embeddings else [0.0] * 1024


# =============================================================================
# Phase 73: Jina Embeddings v3 (Flexible dimensions, 89 languages)
# =============================================================================

# Check for Jina API availability
try:
    import httpx
    JINA_AVAILABLE = True
except ImportError:
    JINA_AVAILABLE = False


class JinaEmbeddings(Embeddings):
    """
    Jina Embeddings v3 wrapper implementing LangChain Embeddings interface.

    Jina v3 features:
    - Task-specific LoRA adapters for different use cases
    - Flexible output dimensions (64-1024)
    - 89 language support
    - 8192 token context length
    - Late interaction support (ColBERT-style)

    Tasks:
    - retrieval.query: For search queries
    - retrieval.passage: For documents
    - separation: For clustering
    - classification: For categorization
    - text-matching: For similarity

    API: https://api.jina.ai/v1/embeddings
    """

    SUPPORTED_TASKS = [
        "retrieval.query",
        "retrieval.passage",
        "separation",
        "classification",
        "text-matching",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v3",
        dimensions: int = 1024,  # Flexible: 64-1024
        task: str = "retrieval.passage",
        late_chunking: bool = False,
        batch_size: int = 32,
    ):
        """
        Initialize Jina Embeddings v3.

        Args:
            api_key: Jina API key (or JINA_API_KEY env var)
            model: Model name (jina-embeddings-v3)
            dimensions: Output dimensions (64-1024)
            task: Task type for LoRA adapter selection
            late_chunking: Enable late chunking for long documents
            batch_size: Max texts per batch
        """
        if not JINA_AVAILABLE:
            raise ImportError(
                "httpx package required for Jina embeddings. "
                "Install with: pip install httpx"
            )

        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key required. Set JINA_API_KEY env var.")

        self.model = model
        self.dimensions = min(max(dimensions, 64), 1024)
        self.task = task if task in self.SUPPORTED_TASKS else "retrieval.passage"
        self.late_chunking = late_chunking
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=60.0)

        logger.info(
            "Jina Embeddings v3 initialized",
            model=model,
            dimensions=self.dimensions,
            task=self.task,
        )

    def _call_api(self, texts: List[str], task: Optional[str] = None) -> List[List[float]]:
        """Call Jina API to get embeddings."""
        if not texts:
            return []

        all_embeddings = []
        task = task or self.task

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = self._client.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": batch,
                    "task": task,
                    "dimensions": self.dimensions,
                    "late_chunking": self.late_chunking,
                },
            )
            response.raise_for_status()
            data = response.json()

            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using retrieval.passage task."""
        return self._call_api(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> List[float]:
        """Embed query using retrieval.query task."""
        if not text or not text.strip():
            return [0.0] * self.dimensions

        embeddings = self._call_api([text], task="retrieval.query")
        return embeddings[0] if embeddings else [0.0] * self.dimensions


# =============================================================================
# Phase 73: GTE-Multilingual-Base (Efficient multilingual embeddings)
# =============================================================================

class GTEEmbeddings(Embeddings):
    """
    GTE (General Text Embeddings) multilingual wrapper.

    GTE-Multilingual-Base features:
    - 305M parameters (efficient)
    - 768 dimensions
    - 8192 token context
    - Strong multilingual performance (~66 MTEB)
    - Alibaba DAMO Academy

    Model: Alibaba-NLP/gte-multilingual-base (HuggingFace)
    """

    def __init__(
        self,
        model: str = "Alibaba-NLP/gte-multilingual-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 8192,
        normalize: bool = True,
        use_fp16: bool = True,
    ):
        """
        Initialize GTE-Multilingual embeddings.

        Args:
            model: GTE model name (HuggingFace model ID)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Max texts per batch
            max_length: Max token length
            normalize: Whether to L2 normalize embeddings
            use_fp16: Use FP16 for faster inference (GPU only)
        """
        if not QWEN3_AVAILABLE:  # Uses same transformers/torch deps
            raise ImportError(
                "transformers and torch packages required for GTE embeddings. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model
        self.batch_size = batch_size
        self.max_length = min(max_length, 8192)
        self.normalize = normalize
        self.use_fp16 = use_fp16
        self.dimensions = 768  # GTE-multilingual-base uses 768

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Loading GTE-Multilingual embedding model",
            model=model,
            device=self.device,
            use_fp16=use_fp16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )

        # Load with appropriate dtype
        if self.use_fp16 and self.device == "cuda":
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self._model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
            ).to(self.device)

        self._model.eval()

        logger.info(
            "GTE-Multilingual model loaded",
            model=model,
            device=self.device,
            dimensions=self.dimensions,
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Mean pooling for GTE
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9
                )

                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query."""
        if not text or not text.strip():
            return [0.0] * self.dimensions

        embeddings = self._get_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * self.dimensions


# =============================================================================
# Phase 73: Cohere Embed v3.5 (Self-improving embeddings)
# =============================================================================

# Check for Cohere availability
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class CohereEmbeddings(Embeddings):
    """
    Cohere Embed v3.5 wrapper implementing LangChain Embeddings interface.

    Cohere Embed v3.5 features:
    - Self-learning from search interactions
    - Input type optimization (search_document, search_query)
    - Flexible dimensions (256, 384, 512, 768, 1024)
    - Compression support (int8, uint8, ubinary)
    - Strong MTEB performance (~68)

    API: Cohere API
    """

    INPUT_TYPES = ["search_document", "search_query", "classification", "clustering"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",  # or embed-multilingual-v3.0
        dimensions: Optional[int] = None,  # None = full dimensions
        embedding_types: Optional[List[str]] = None,  # float, int8, uint8, ubinary
        batch_size: int = 96,  # Cohere supports up to 96
    ):
        """
        Initialize Cohere Embeddings v3.

        Args:
            api_key: Cohere API key (or COHERE_API_KEY env var)
            model: Model name
            dimensions: Output dimensions (None for full)
            embedding_types: Types of embeddings to return
            batch_size: Max texts per batch
        """
        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere package required. Install with: pip install cohere"
            )

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY env var.")

        self.model = model
        self.dimensions = dimensions
        self.embedding_types = embedding_types or ["float"]
        self.batch_size = min(batch_size, 96)
        self._client = cohere.Client(api_key=self.api_key)

        # Default dimension based on model
        self._default_dim = 1024 if "v3" in model else 768

        logger.info(
            "Cohere Embeddings v3 initialized",
            model=model,
            dimensions=dimensions or "full",
        )

    def _call_api(
        self,
        texts: List[str],
        input_type: str = "search_document",
    ) -> List[List[float]]:
        """Call Cohere API to get embeddings."""
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            kwargs = {
                "texts": batch,
                "model": self.model,
                "input_type": input_type,
                "embedding_types": self.embedding_types,
            }

            if self.dimensions:
                kwargs["output_dimension"] = self.dimensions

            response = self._client.embed(**kwargs)

            # Get float embeddings (primary type)
            if hasattr(response, 'embeddings') and response.embeddings:
                if hasattr(response.embeddings, 'float_'):
                    batch_embeddings = response.embeddings.float_
                else:
                    batch_embeddings = response.embeddings
            else:
                batch_embeddings = [[0.0] * (self.dimensions or self._default_dim)] * len(batch)

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        return self._call_api(texts, input_type="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Embed query."""
        if not text or not text.strip():
            return [0.0] * (self.dimensions or self._default_dim)

        embeddings = self._call_api([text], input_type="search_query")
        return embeddings[0] if embeddings else [0.0] * (self.dimensions or self._default_dim)


# =============================================================================
# Phase 77: EmbeddingGemma (Google's Specialized Embedding Model)
# =============================================================================

class EmbeddingGemma(Embeddings):
    """
    Phase 77: Google EmbeddingGemma wrapper.

    EmbeddingGemma is a specialized embedding model based on Gemma:
    - 768 dimensions (compact but effective)
    - Strong performance on retrieval tasks
    - Optimized for semantic similarity
    - Fast inference on CPU and GPU

    Model: google/embedding-gemma-2b (or similar)
    """

    def __init__(
        self,
        model: str = "google/embedding-gemma-2b",
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Initialize EmbeddingGemma.

        Args:
            model: Model name from HuggingFace
            device: Device (cuda, cpu, mps, or auto)
            use_fp16: Use half precision
            max_length: Maximum sequence length
            batch_size: Batch size for processing
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required. pip install transformers torch")

        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size
        self._use_fp16 = use_fp16

        # Determine device
        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._model = None
        self._tokenizer = None
        self._initialized = False

        logger.info(
            "EmbeddingGemma configured",
            model=model,
            device=str(self._device),
        )

    def _initialize(self) -> None:
        """Lazy initialization of model."""
        if self._initialized:
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = self._model.to(self._device)
            if self._use_fp16 and self._device.type in ("cuda", "mps"):
                self._model = self._model.half()
            self._model.eval()
            self._initialized = True
            logger.info("EmbeddingGemma initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGemma: {e}")
            raise

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        self._initialize()

        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

                # Use mean pooling over last hidden state
                attention_mask = inputs["attention_mask"]
                hidden_state = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query."""
        if not text or not text.strip():
            return [0.0] * 768
        embeddings = self._get_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * 768


# =============================================================================
# Phase 77: Stella v3 Embeddings (High-Performance)
# =============================================================================

class StellaEmbeddings(Embeddings):
    """
    Phase 77: Stella v3 Embeddings wrapper.

    Stella v3 features:
    - State-of-the-art MTEB performance (69+)
    - Multiple size variants (400M, 1.5B)
    - Excellent multilingual support
    - Optimized for both retrieval and similarity
    - 1024 or 2048 dimensions

    Model: infgrad/stella-base-en-v3 (or stella-large-en-v3)
    """

    MODELS = {
        "stella-base": "infgrad/stella-base-en-v3",
        "stella-large": "infgrad/stella-large-en-v3",
    }

    def __init__(
        self,
        model: str = "stella-base",
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize Stella v3 Embeddings.

        Args:
            model: Model variant (stella-base, stella-large) or full HF name
            device: Device (cuda, cpu, mps, or auto)
            use_fp16: Use half precision
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required. pip install transformers torch")

        # Map short names to full names
        self.model_name = self.MODELS.get(model, model)
        self.max_length = max_length
        self.batch_size = batch_size
        self._use_fp16 = use_fp16
        self._normalize = normalize

        # Determine device
        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._dimensions = 1024  # Default, will be updated

        logger.info(
            "Stella v3 Embeddings configured",
            model=self.model_name,
            device=str(self._device),
        )

    def _initialize(self) -> None:
        """Lazy initialization of model."""
        if self._initialized:
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = self._model.to(self._device)
            if self._use_fp16 and self._device.type in ("cuda", "mps"):
                self._model = self._model.half()
            self._model.eval()

            # Detect dimensions from model
            if hasattr(self._model.config, "hidden_size"):
                self._dimensions = self._model.config.hidden_size

            self._initialized = True
            logger.info(
                "Stella v3 initialized",
                dimensions=self._dimensions,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Stella v3: {e}")
            raise

    def _get_embeddings(
        self,
        texts: List[str],
        is_query: bool = False,
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        self._initialize()

        if not texts:
            return []

        # Stella uses instruction prefix for queries
        if is_query:
            texts = [f"Represent this query for retrieval: {t}" for t in texts]

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

                # Stella uses CLS token embedding
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    # Fallback to mean pooling
                    attention_mask = inputs["attention_mask"]
                    hidden_state = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                # Normalize if requested
                if self._normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        return self._get_embeddings(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        """Embed query."""
        if not text or not text.strip():
            return [0.0] * self._dimensions
        embeddings = self._get_embeddings([text], is_query=True)
        return embeddings[0] if embeddings else [0.0] * self._dimensions


# Embedding cache for deduplication (content hash -> embedding)
_embedding_cache: Dict[str, List[float]] = {}
# Increased default cache size from 10k to 100k for better hit rate at scale
# At 1536 dimensions * 4 bytes * 100k = ~600MB memory usage
# For production with millions of docs, use Redis-backed cache instead
_CACHE_MAX_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "100000"))

# Phase 71.5: TTL-based Qwen3 settings cache (auto-invalidates after 5 minutes)
import time as _time_module

_qwen3_settings_cache: Optional[Dict[str, Any]] = None
_qwen3_settings_cache_time: Optional[float] = None
_QWEN3_SETTINGS_CACHE_TTL = 300  # 5 minutes


def _get_qwen3_settings_sync() -> Dict[str, Any]:
    """
    Get Qwen3 settings from admin settings with env var fallback (sync version).

    Settings hierarchy:
    1. Admin settings (database) - takes priority
    2. Environment variables - fallback

    Returns:
        Dict with enabled, model, device, use_fp16
    """
    global _qwen3_settings_cache, _qwen3_settings_cache_time

    now = _time_module.time()

    # Return cached settings if available and not expired
    if (_qwen3_settings_cache is not None
        and _qwen3_settings_cache_time is not None
        and (now - _qwen3_settings_cache_time) < _QWEN3_SETTINGS_CACHE_TTL):
        return _qwen3_settings_cache

    # Start with env var defaults
    settings = {
        "enabled": os.getenv("QWEN3_EMBEDDING_ENABLED", "false").lower() == "true",
        "model": os.getenv("QWEN3_MODEL", "Alibaba-NLP/Qwen3-Embedding-8B"),
        "device": os.getenv("QWEN3_DEVICE"),  # None for auto-detect
        "use_fp16": os.getenv("QWEN3_USE_FP16", "true").lower() == "true",
    }

    try:
        # Try to load from admin settings (synchronous version for embedding creation)
        from backend.db.database import get_sync_session
        from backend.db.models import SystemSettings
        from sqlalchemy import select

        session = get_sync_session()
        try:
            # Query relevant settings
            result = session.execute(
                select(SystemSettings).where(
                    SystemSettings.key.in_([
                        "embedding.qwen3_enabled",
                        "embedding.qwen3_model",
                        "embedding.qwen3_device",
                        "embedding.qwen3_use_fp16",
                    ])
                )
            )
            db_settings = {row.key: row.value for row in result.scalars()}

            # Override with admin settings if present
            if "embedding.qwen3_enabled" in db_settings:
                settings["enabled"] = db_settings["embedding.qwen3_enabled"].lower() == "true"

            if db_settings.get("embedding.qwen3_model"):
                settings["model"] = db_settings["embedding.qwen3_model"]

            if db_settings.get("embedding.qwen3_device"):
                settings["device"] = db_settings["embedding.qwen3_device"]
                if settings["device"] == "auto":
                    settings["device"] = None  # Auto-detect

            if "embedding.qwen3_use_fp16" in db_settings:
                settings["use_fp16"] = db_settings["embedding.qwen3_use_fp16"].lower() == "true"

            logger.debug(
                "Loaded Qwen3 embedding settings",
                enabled=settings["enabled"],
                model=settings["model"],
                device=settings["device"],
            )
        finally:
            session.close()

    except Exception as e:
        logger.debug(
            "Using env var defaults for Qwen3 settings",
            reason=str(e),
        )

    # Cache the settings with timestamp
    _qwen3_settings_cache = settings
    _qwen3_settings_cache_time = _time_module.time()
    return settings


def reset_qwen3_settings_cache() -> None:
    """Reset the Qwen3 settings cache to reload from admin settings."""
    global _qwen3_settings_cache, _qwen3_settings_cache_time
    _qwen3_settings_cache = None
    _qwen3_settings_cache_time = None
    logger.info("Qwen3 settings cache reset, will reload on next use")


# =============================================================================
# Phase 76: Embedding Model Auto-Selection
# =============================================================================

class ContentType:
    """Content types for auto-selection."""
    GENERAL = "general"
    CODE = "code"
    MULTILINGUAL = "multilingual"
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    TECHNICAL = "technical"


class EmbeddingModelSelector:
    """
    Phase 76: Automatically select optimal embedding model based on content.

    Analyzes content characteristics and selects the most appropriate
    embedding model for best retrieval performance.
    """

    # Code detection patterns
    CODE_PATTERNS = [
        r'\bdef\s+\w+\s*\(',      # Python functions
        r'\bclass\s+\w+',          # Classes
        r'\bfunction\s+\w+\s*\(',  # JavaScript functions
        r'\bimport\s+[\w.]+',      # Import statements
        r'\bpublic\s+class\b',     # Java
        r'#include\s*<',           # C/C++
        r'\bconst\s+\w+\s*=',      # JavaScript/TypeScript const
        r'\blet\s+\w+\s*=',        # JavaScript/TypeScript let
        r'\breturn\s+\w+',         # Return statements
        r'[\{\};\[\]]',            # Code brackets
    ]

    # Multilingual detection (non-ASCII ratio)
    # Scientific/technical patterns
    SCIENTIFIC_PATTERNS = [
        r'\b\d+\.\d+\s*[×x]\s*10\^?[-−]?\d+',  # Scientific notation
        r'\b(?:μ|α|β|γ|δ|σ|π|θ|λ|Σ|Δ|Ω)\b',   # Greek letters
        r'\b(?:equation|theorem|hypothesis|coefficient|variable)\b',
        r'\b(?:p\s*[<>=]\s*0\.\d+)',            # P-values
        r'\b(?:et\s+al\.?|cf\.|ibid\.?)',       # Academic citations
    ]

    # Legal patterns
    LEGAL_PATTERNS = [
        r'\b(?:pursuant|herein|thereof|whereby|notwithstanding)\b',
        r'\b(?:plaintiff|defendant|jurisdiction|statute)\b',
        r'\b§\s*\d+',              # Section symbols
        r'\b(?:Article|Section|Clause)\s+\d+',
    ]

    # Model recommendations by content type
    MODEL_RECOMMENDATIONS = {
        ContentType.CODE: {
            "primary": ("voyage4-code", "voyage4-code-3"),
            "fallback": ("openai", "text-embedding-3-large"),
            "reason": "Code-specialized model for better code understanding"
        },
        ContentType.MULTILINGUAL: {
            "primary": ("gemini", "text-embedding-004"),
            "fallback": ("bge-m3", "BAAI/bge-m3"),
            "reason": "Best multilingual performance on MTEB"
        },
        ContentType.SCIENTIFIC: {
            "primary": ("voyage", "voyage-3"),
            "fallback": ("openai", "text-embedding-3-large"),
            "reason": "Scientific/academic content handling"
        },
        ContentType.LEGAL: {
            "primary": ("voyage", "voyage-law-2"),
            "fallback": ("openai", "text-embedding-3-large"),
            "reason": "Legal domain specialization"
        },
        ContentType.TECHNICAL: {
            "primary": ("qwen3", "Alibaba-NLP/gte-Qwen2-7B-instruct"),
            "fallback": ("openai", "text-embedding-3-large"),
            "reason": "High technical accuracy"
        },
        ContentType.GENERAL: {
            "primary": ("openai", "text-embedding-3-small"),
            "fallback": ("ollama", "nomic-embed-text"),
            "reason": "Balanced general-purpose embedding"
        },
    }

    def __init__(self):
        import re
        self._code_patterns = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
        self._scientific_patterns = [re.compile(p, re.IGNORECASE) for p in self.SCIENTIFIC_PATTERNS]
        self._legal_patterns = [re.compile(p, re.IGNORECASE) for p in self.LEGAL_PATTERNS]

    def detect_content_type(self, text: str, sample_size: int = 5000) -> str:
        """
        Detect content type from text sample.

        Args:
            text: Text to analyze
            sample_size: Maximum chars to analyze

        Returns:
            Content type string
        """
        if not text:
            return ContentType.GENERAL

        sample = text[:sample_size]

        # Check for code
        code_matches = sum(1 for p in self._code_patterns if p.search(sample))
        if code_matches >= 3:
            return ContentType.CODE

        # Check for non-ASCII (multilingual)
        non_ascii_ratio = sum(1 for c in sample if ord(c) > 127) / max(len(sample), 1)
        if non_ascii_ratio > 0.15:
            return ContentType.MULTILINGUAL

        # Check for scientific content
        scientific_matches = sum(1 for p in self._scientific_patterns if p.search(sample))
        if scientific_matches >= 2:
            return ContentType.SCIENTIFIC

        # Check for legal content
        legal_matches = sum(1 for p in self._legal_patterns if p.search(sample))
        if legal_matches >= 2:
            return ContentType.LEGAL

        # Check for technical content (high density of technical terms)
        technical_terms = ['API', 'SDK', 'framework', 'configuration', 'implementation',
                         'architecture', 'protocol', 'interface', 'module', 'component']
        tech_count = sum(1 for term in technical_terms if term.lower() in sample.lower())
        if tech_count >= 4:
            return ContentType.TECHNICAL

        return ContentType.GENERAL

    def select_model(
        self,
        text: str,
        available_providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Select optimal embedding model for content.

        Args:
            text: Text content to analyze
            available_providers: List of available providers (None = all)

        Returns:
            Dict with provider, model, and reason
        """
        content_type = self.detect_content_type(text)
        recommendation = self.MODEL_RECOMMENDATIONS.get(
            content_type,
            self.MODEL_RECOMMENDATIONS[ContentType.GENERAL]
        )

        primary_provider, primary_model = recommendation["primary"]
        fallback_provider, fallback_model = recommendation["fallback"]

        # Check if primary provider is available
        if available_providers is None or primary_provider in available_providers:
            selected_provider = primary_provider
            selected_model = primary_model
        elif fallback_provider in (available_providers or [fallback_provider]):
            selected_provider = fallback_provider
            selected_model = fallback_model
        else:
            # Default to OpenAI if nothing else available
            selected_provider = "openai"
            selected_model = "text-embedding-3-small"

        logger.info(
            "Auto-selected embedding model",
            content_type=content_type,
            provider=selected_provider,
            model=selected_model,
            reason=recommendation["reason"],
        )

        return {
            "provider": selected_provider,
            "model": selected_model,
            "content_type": content_type,
            "reason": recommendation["reason"],
        }

    def analyze_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze a batch of texts to determine overall content type.

        Uses majority voting from samples.

        Args:
            texts: List of texts to analyze

        Returns:
            Analysis result with dominant content type
        """
        if not texts:
            return {"content_type": ContentType.GENERAL, "confidence": 0.0}

        # Sample up to 10 texts
        sample_size = min(10, len(texts))
        import random
        samples = random.sample(texts, sample_size) if len(texts) > sample_size else texts

        # Detect type for each sample
        type_counts: Dict[str, int] = {}
        for text in samples:
            content_type = self.detect_content_type(text)
            type_counts[content_type] = type_counts.get(content_type, 0) + 1

        # Find dominant type
        dominant_type = max(type_counts.keys(), key=lambda k: type_counts[k])
        confidence = type_counts[dominant_type] / len(samples)

        return {
            "content_type": dominant_type,
            "confidence": confidence,
            "type_distribution": type_counts,
            "samples_analyzed": len(samples),
        }


# Global selector instance
_embedding_model_selector: Optional[EmbeddingModelSelector] = None


def get_embedding_model_selector() -> EmbeddingModelSelector:
    """Get or create embedding model selector singleton."""
    global _embedding_model_selector
    if _embedding_model_selector is None:
        _embedding_model_selector = EmbeddingModelSelector()
    return _embedding_model_selector


async def auto_select_embedding_model(
    text: str,
    available_providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Phase 76: Automatically select optimal embedding model for content.

    Convenience function for async contexts.

    Args:
        text: Text sample to analyze
        available_providers: Optional list of available providers

    Returns:
        Dict with provider, model, content_type, and reason
    """
    selector = get_embedding_model_selector()
    return selector.select_model(text, available_providers)


@dataclass
class EmbeddingResult:
    """Result of embedding a single chunk."""
    chunk_id: str
    chunk_hash: str
    embedding: List[float]
    model: str
    dimensions: int
    metadata: Dict[str, Any]


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding operation."""
    results: List[EmbeddingResult]
    total_chunks: int
    successful: int
    failed: int
    model: str


class EmbeddingService:
    """
    Embedding generation service with multiple provider support.

    Features:
    - Multiple embedding providers (OpenAI, Ollama, HuggingFace)
    - Batch processing for efficiency
    - Ray-parallel processing for large batches
    - Caching support
    - Automatic chunking integration
    """

    # Default embedding models by provider
    DEFAULT_MODELS = {
        "openai": "text-embedding-3-small",  # 1536 dimensions
        "ollama": "nomic-embed-text",        # 768 dimensions
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
        # Phase 76: Auto-selection (starts with openai, updates based on content)
        "auto": "text-embedding-3-small",
        # Phase 29: Voyage AI - Best on MTEB benchmarks
        "voyage": "voyage-3-large",          # 1024 dimensions - Best overall
        "voyage-lite": "voyage-3-lite",      # 512 dimensions - Cost-effective
        # Phase 67: Voyage 4 series - shared embedding space for better retrieval
        "voyage4": "voyage-4",               # 1024 dimensions - Latest generation
        "voyage4-lite": "voyage-4-lite",     # 512 dimensions - Cost-effective v4
        "voyage4-code": "voyage-code-4",     # 1024 dimensions - Code-optimized v4
        # Phase 68: Qwen3 Embedding - Highest MTEB score (70.58)
        "qwen3": "Alibaba-NLP/Qwen3-Embedding-8B",      # 4096 dimensions - Best quality
        "qwen3-4b": "Alibaba-NLP/Qwen3-Embedding-4B",   # 2048 dimensions - Balanced
        "qwen3-small": "Alibaba-NLP/Qwen3-Embedding-0.6B",  # 1024 dimensions - Fast
        # Phase 69: Google Gemini Embedding - #1 on MTEB Multilingual (71.5%)
        "gemini": "gemini-embedding-001",               # 3072 dimensions - Best multilingual
        "gemini-768": "gemini-embedding-001",           # 768 dimensions - Matryoshka reduced
        "gemini-1536": "gemini-embedding-001",          # 1536 dimensions - Matryoshka reduced
        # Phase 69: NVIDIA NV-Embed-v2 - 69.32 MTEB (GPU optimized)
        "nv-embed": "nvidia/NV-Embed-v2",               # 4096 dimensions - GPU optimized
        "nv-embed-v2": "nvidia/NV-Embed-v2",            # 4096 dimensions - Alias
        # Phase 69: BGE-M3 - Multi-retrieval (dense+sparse+colbert)
        "bge-m3": "BAAI/bge-m3",                        # 1024 dimensions - Multi-retrieval
        # Phase 77: EmbeddingGemma - Google's specialized embedding model
        "embedding-gemma": "google/embedding-gemma-2b", # 768 dimensions
        # Phase 77: Stella v3 - High-performance embeddings
        "stella": "infgrad/stella-base-en-v3",          # 1024 dimensions
        "stella-base": "infgrad/stella-base-en-v3",     # 1024 dimensions
        "stella-large": "infgrad/stella-large-en-v3",   # 2048 dimensions
    }

    # Embedding dimensions by model
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "mxbai-embed-large": 1024,
        # Phase 29: Voyage AI models
        "voyage-3-large": 1024,
        "voyage-3-lite": 512,
        "voyage-3": 1024,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-multilingual-2": 1024,
        # Phase 67: Voyage 4 series - shared embedding space
        # Voyage 4 provides unified embedding space across queries and documents
        "voyage-4": 1024,
        "voyage-4-lite": 512,
        "voyage-code-4": 1024,
        # Phase 68: Qwen3 Embedding - 70.58 MTEB score (top performer)
        "Qwen3-Embedding-8B": 4096,
        "Qwen3-Embedding-4B": 2048,
        "Qwen3-Embedding-0.6B": 1024,
        "Alibaba-NLP/Qwen3-Embedding-8B": 4096,
        "Alibaba-NLP/Qwen3-Embedding-4B": 2048,
        "Alibaba-NLP/Qwen3-Embedding-0.6B": 1024,
        # Phase 69: Google Gemini Embedding - #1 on MTEB Multilingual (71.5%)
        "gemini-embedding-001": 3072,
        "gemini-embedding-001-768": 768,   # Matryoshka reduced
        "gemini-embedding-001-1536": 1536,  # Matryoshka reduced
        # Phase 69: NVIDIA NV-Embed-v2 - 69.32 MTEB (GPU optimized)
        "nvidia/NV-Embed-v2": 4096,
        "NV-Embed-v2": 4096,
        # Phase 69: BGE-M3 - Multi-retrieval (dense+sparse+colbert)
        "BAAI/bge-m3": 1024,
        "bge-m3": 1024,
        # Phase 77: EmbeddingGemma
        "google/embedding-gemma-2b": 768,
        "embedding-gemma-2b": 768,
        # Phase 77: Stella v3
        "infgrad/stella-base-en-v3": 1024,
        "infgrad/stella-large-en-v3": 2048,
        "stella-base-en-v3": 1024,
        "stella-large-en-v3": 2048,
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        """
        Initialize embedding service.

        Args:
            provider: Embedding provider ("openai", "ollama", "huggingface", "auto")
                     Use "auto" for Phase 76 automatic model selection based on content
            model: Specific model to use (defaults to provider's default)
            config: LLM configuration with API keys
        """
        self._auto_select = provider == "auto"
        self._auto_selection_result: Optional[Dict[str, Any]] = None

        # For "auto" mode, start with openai as default until content is analyzed
        if self._auto_select:
            self.provider = "openai"  # Will be updated on first embed
            self.model = "text-embedding-3-small"
        else:
            self.provider = provider
            self.model = model or self.DEFAULT_MODELS.get(provider, "text-embedding-3-small")

        self.config = config or LLMConfig.from_env()
        self._embeddings: Optional[Embeddings] = None
        self._dimensions: Optional[int] = None

        logger.info(
            "Initializing embedding service",
            provider=provider if not self._auto_select else "auto",
            model=self.model,
            auto_select=self._auto_select,
        )

    def _auto_select_for_content(self, text: str) -> None:
        """
        Phase 76: Automatically select embedding model based on content.

        Updates provider and model based on content analysis.
        Only runs once per service instance.

        Args:
            text: Sample text to analyze
        """
        if not self._auto_select or self._auto_selection_result is not None:
            return

        selector = get_embedding_model_selector()

        # Determine available providers based on env vars/config
        available_providers = []
        if os.getenv("OPENAI_API_KEY") or self.config.openai_api_key:
            available_providers.extend(["openai"])
        if os.getenv("VOYAGE_API_KEY"):
            available_providers.extend(["voyage", "voyage4", "voyage4-code"])
        if os.getenv("GOOGLE_API_KEY") and GEMINI_AVAILABLE:
            available_providers.append("gemini")
        if QWEN3_AVAILABLE:
            available_providers.extend(["qwen3", "qwen3-4b", "qwen3-small"])
        if os.getenv("OLLAMA_BASE_URL") or self.config.ollama_base_url:
            available_providers.append("ollama")

        # Select optimal model
        self._auto_selection_result = selector.select_model(text, available_providers or None)

        # Update service configuration
        new_provider = self._auto_selection_result["provider"]
        new_model = self._auto_selection_result["model"]

        if new_provider != self.provider or new_model != self.model:
            logger.info(
                "Auto-selection updated embedding model",
                old_provider=self.provider,
                new_provider=new_provider,
                old_model=self.model,
                new_model=new_model,
                content_type=self._auto_selection_result["content_type"],
            )
            self.provider = new_provider
            self.model = new_model
            # Reset embeddings to use new provider
            self._embeddings = None
            self._dimensions = None

    @property
    def embeddings(self) -> Embeddings:
        """Get or create embeddings instance (lazy initialization)."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for current model."""
        if self._dimensions is None:
            self._dimensions = self.MODEL_DIMENSIONS.get(self.model, 1536)
        return self._dimensions

    def _create_embeddings(self) -> Embeddings:
        """Create embeddings instance based on provider."""
        if self.provider == "openai":
            # Check for explicit dimension override (OpenAI v3 models support flexible dimensions)
            import os
            explicit_dim = os.getenv("EMBEDDING_DIMENSION")

            # OpenAI text-embedding-3-* models support dimension parameter
            if explicit_dim and ("text-embedding-3" in self.model.lower()):
                try:
                    dim = int(explicit_dim)
                    logger.info(f"Using OpenAI with reduced dimension: {dim}D")
                    return OpenAIEmbeddings(
                        model=self.model,
                        openai_api_key=self.config.openai_api_key,
                        dimensions=dim,  # OpenAI v3 supports dimension reduction
                    )
                except ValueError:
                    pass

            return OpenAIEmbeddings(
                model=self.model,
                openai_api_key=self.config.openai_api_key,
            )
        elif self.provider == "ollama":
            return OllamaEmbeddings(
                model=self.model,
                base_url=self.config.ollama_base_url,
            )
        elif self.provider in ("voyage", "voyage-lite", "voyageai"):
            # Phase 29: Voyage AI embeddings - Best on MTEB benchmarks
            return VoyageAIEmbeddings(
                model=self.model,
                api_key=os.getenv("VOYAGE_API_KEY"),
            )
        elif self.provider in ("voyage4", "voyage4-lite", "voyage4-code"):
            # Phase 67: Voyage 4 series - shared embedding space
            return VoyageAIEmbeddings(
                model=self.model,
                api_key=os.getenv("VOYAGE_API_KEY"),
            )
        elif self.provider in ("qwen3", "qwen3-4b", "qwen3-small"):
            # Phase 68: Qwen3 Embedding - 70.58 MTEB score (top performer)
            # Load settings from admin settings with env var fallback
            qwen3_settings = _get_qwen3_settings_sync()
            return Qwen3Embeddings(
                model=qwen3_settings.get("model", self.model),
                device=qwen3_settings.get("device"),  # None for auto-detect
                use_fp16=qwen3_settings.get("use_fp16", True),
            )
        elif self.provider in ("gemini", "gemini-768", "gemini-1536"):
            # Phase 69: Google Gemini Embedding - #1 on MTEB Multilingual (71.5%)
            # Matryoshka dimensions based on provider variant
            if self.provider == "gemini-768":
                dimensions = 768
            elif self.provider == "gemini-1536":
                dimensions = 1536
            else:
                dimensions = 3072  # Full dimensions

            return GeminiEmbeddings(
                model=self.model,
                api_key=os.getenv("GOOGLE_API_KEY"),
                dimensions=dimensions,
            )
        elif self.provider in ("nv-embed", "nv-embed-v2", "nvidia"):
            # Phase 69: NVIDIA NV-Embed-v2 - 69.32 MTEB (GPU optimized)
            return NVEmbedEmbeddings(
                model=self.model,
                device=os.getenv("NVEMBED_DEVICE"),  # None for auto-detect
                use_fp16=os.getenv("NVEMBED_USE_FP16", "true").lower() == "true",
            )
        elif self.provider in ("bge-m3", "bgem3"):
            # Phase 69: BGE-M3 - Multi-retrieval (dense+sparse+colbert)
            # 64.3 MTEB score, best for hybrid retrieval
            return BGEM3Embeddings(
                model=self.model,
                device=os.getenv("BGEM3_DEVICE"),  # None for auto-detect
                use_fp16=os.getenv("BGEM3_USE_FP16", "true").lower() == "true",
            )
        elif self.provider in ("embedding-gemma", "gemma-embed"):
            # Phase 77: EmbeddingGemma - Google's specialized embedding model
            return EmbeddingGemma(
                model=self.model,
                device=os.getenv("EMBEDDING_GEMMA_DEVICE"),
                use_fp16=os.getenv("EMBEDDING_GEMMA_FP16", "true").lower() == "true",
            )
        elif self.provider in ("stella", "stella-base", "stella-large"):
            # Phase 77: Stella v3 - High-performance embeddings (69+ MTEB)
            return StellaEmbeddings(
                model=self.model,
                device=os.getenv("STELLA_DEVICE"),
                use_fp16=os.getenv("STELLA_FP16", "true").lower() == "true",
            )
        else:
            # Default to OpenAI
            logger.warning(f"Unknown provider {self.provider}, defaulting to OpenAI")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.config.openai_api_key,
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            return [0.0] * self.dimensions

        # Phase 76: Auto-select model based on content if enabled
        if self._auto_select:
            self._auto_select_for_content(text)

        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error("Embedding failed", error=str(e), text_length=len(text))
            raise

    def _get_content_hash(self, text: str) -> str:
        """Generate a hash for text content for caching.

        Phase 71.5: Use SHA-256 instead of MD5 for collision resistance.
        MD5 is deprecated by NIST and has known collision vulnerabilities.
        """
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def embed_texts(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use embedding cache (default True)

        Returns:
            List of embedding vectors
        """
        global _embedding_cache

        if not texts:
            return []

        # Phase 76: Auto-select model based on content if enabled
        if self._auto_select and texts:
            # Use first non-empty text or combined sample for analysis
            sample_texts = [t for t in texts[:5] if t and t.strip()]
            if sample_texts:
                sample = " ".join(sample_texts)[:3000]
                self._auto_select_for_content(sample)

        # Track which texts need embedding vs are cached
        result = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                result[i] = [0.0] * self.dimensions
                continue

            if use_cache:
                content_hash = self._get_content_hash(text)
                if content_hash in _embedding_cache:
                    result[i] = _embedding_cache[content_hash]
                    continue

            texts_to_embed.append(text)
            indices_to_embed.append(i)

        # Log cache statistics
        cache_hits = len(texts) - len(texts_to_embed) - sum(1 for r in result if r is not None and r == [0.0] * self.dimensions)
        if cache_hits > 0:
            logger.debug(
                "Embedding cache hits",
                cache_hits=cache_hits,
                cache_misses=len(texts_to_embed),
            )

        # Embed texts that weren't cached
        if texts_to_embed:
            try:
                embeddings = self.embeddings.embed_documents(texts_to_embed)

                # Store in results and cache
                for idx, text, embedding in zip(indices_to_embed, texts_to_embed, embeddings):
                    result[idx] = embedding

                    if use_cache and len(_embedding_cache) < _CACHE_MAX_SIZE:
                        content_hash = self._get_content_hash(text)
                        _embedding_cache[content_hash] = embedding

            except Exception as e:
                logger.error(
                    "Batch embedding failed",
                    error=str(e),
                    num_texts=len(texts_to_embed),
                )
                raise

        # Fill any remaining None values with zero vectors
        for i in range(len(result)):
            if result[i] is None:
                result[i] = [0.0] * self.dimensions

        return result

    async def embed_text_async(self, text: str) -> List[float]:
        """Async version of embed_text."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_text, text)

    async def embed_query(self, text: str) -> List[float]:
        """
        Async method to embed a query text.

        This is an alias for embed_text_async, provided for compatibility
        with LangChain's Embeddings interface naming convention.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        return await self.embed_text_async(text)

    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_texts."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_texts, texts)

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for document chunks with caching and optimal batching.

        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to process at once (auto-detected if None)
            use_cache: Whether to use the embedding cache

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []

        # Use optimal batch size for provider if not specified
        if batch_size is None:
            batch_size = get_optimal_batch_size(self.provider, len(chunks))

        start_time = time.time()
        logger.info(
            "Embedding chunks",
            num_chunks=len(chunks),
            batch_size=batch_size,
            model=self.model,
            provider=self.provider,
            cache_enabled=use_cache,
        )

        results = []
        texts = [chunk.content for chunk in chunks]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            try:
                embeddings = self.embed_texts(batch_texts, use_cache=use_cache)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    result = EmbeddingResult(
                        chunk_id=f"{chunk.document_id}_{chunk.chunk_index}" if chunk.document_id else str(chunk.chunk_index),
                        chunk_hash=chunk.chunk_hash,
                        embedding=embedding,
                        model=self.model,
                        dimensions=len(embedding),
                        metadata=chunk.metadata,
                    )
                    results.append(result)

                logger.debug(
                    "Batch embedded",
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch_texts),
                )

            except Exception as e:
                logger.error(
                    "Batch embedding failed",
                    batch_num=i // batch_size + 1,
                    error=str(e),
                )
                # Create placeholder results for failed batch
                for chunk in batch_chunks:
                    result = EmbeddingResult(
                        chunk_id=f"{chunk.document_id}_{chunk.chunk_index}" if chunk.document_id else str(chunk.chunk_index),
                        chunk_hash=chunk.chunk_hash,
                        embedding=[0.0] * self.dimensions,  # Zero vector for failures
                        model=self.model,
                        dimensions=self.dimensions,
                        metadata={**chunk.metadata, "embedding_failed": True},
                    )
                    results.append(result)

        elapsed = time.time() - start_time
        logger.info(
            "Embedding complete",
            num_chunks=len(chunks),
            num_results=len(results),
            elapsed_seconds=round(elapsed, 2),
            chunks_per_second=round(len(chunks) / elapsed, 1) if elapsed > 0 else 0,
        )

        return results

    async def embed_chunks_async(
        self,
        chunks: List[Chunk],
        batch_size: int = 100,
    ) -> List[EmbeddingResult]:
        """Async version of embed_chunks."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.embed_chunks(chunks, batch_size)
            )


# =============================================================================
# Ray-Parallel Embedding Functions
# =============================================================================

@ray.remote
def embed_batch_ray(
    texts: List[str],
    provider: str = "openai",
    model: Optional[str] = None,
    config_dict: Optional[Dict] = None,
) -> List[List[float]]:
    """
    Ray remote function for batch embedding.

    This runs on Ray workers for distributed processing.
    """
    # Reconstruct config from dict (Ray requires serializable objects)
    # LLMConfig reads from env vars, so we create instance and override from dict if provided
    config = LLMConfig.from_env()
    if config_dict:
        if config_dict.get("openai_api_key"):
            config.openai_api_key = config_dict["openai_api_key"]
        if config_dict.get("ollama_base_url"):
            config.ollama_base_url = config_dict["ollama_base_url"]
        if config_dict.get("anthropic_api_key"):
            config.anthropic_api_key = config_dict["anthropic_api_key"]

    service = EmbeddingService(provider=provider, model=model, config=config)
    return service.embed_texts(texts)


class RayEmbeddingService:
    """
    Distributed embedding service using Ray.

    Automatically distributes embedding work across Ray cluster.
    Falls back to local processing if Ray is not available.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        num_workers: int = 4,
        batch_size_per_worker: int = 50,
    ):
        """
        Initialize Ray embedding service.

        Args:
            provider: Embedding provider
            model: Embedding model
            config: LLM configuration
            num_workers: Number of Ray workers to use
            batch_size_per_worker: Texts per worker batch
        """
        self.provider = provider
        self.model = model or EmbeddingService.DEFAULT_MODELS.get(provider)
        self.config = config or LLMConfig.from_env()
        self.num_workers = num_workers
        self.batch_size_per_worker = batch_size_per_worker

        # Local fallback service
        self._local_service = EmbeddingService(
            provider=provider,
            model=model,
            config=config,
        )

        logger.info(
            "Initialized Ray embedding service",
            provider=provider,
            model=self.model,
            num_workers=num_workers,
        )

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._local_service.dimensions

    def _is_ray_available(self) -> bool:
        """Check if Ray is initialized and available."""
        try:
            return ray.is_initialized()
        except Exception:
            return False

    def _embed_texts_concurrent(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using concurrent.futures when Ray is not available.

        Provides 3-5x speedup over sequential processing for large batches.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Determine number of workers (default: 4, configurable via env)
        max_workers = int(os.getenv("EMBEDDING_CONCURRENT_WORKERS", "4"))

        # Split texts into batches
        batches = []
        for i in range(0, len(texts), self.batch_size_per_worker):
            batch = texts[i:i + self.batch_size_per_worker]
            batches.append((i, batch))

        logger.info(
            "Starting concurrent embedding (Ray unavailable)",
            num_texts=len(texts),
            num_batches=len(batches),
            max_workers=max_workers,
        )

        # Process batches concurrently
        results = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_idx = {
                executor.submit(self._local_service.embed_texts, batch): idx
                for idx, batch in batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx // self.batch_size_per_worker] = future.result()
                except Exception as e:
                    logger.error(
                        "Concurrent batch embedding failed",
                        batch_idx=idx,
                        error=str(e),
                    )
                    # Return zero vectors for failed batch
                    batch_size = len(batches[idx // self.batch_size_per_worker][1])
                    results[idx // self.batch_size_per_worker] = [
                        [0.0] * self._local_service.dimensions for _ in range(batch_size)
                    ]

        # Flatten results
        embeddings = []
        for batch_result in results:
            if batch_result:
                embeddings.extend(batch_result)

        logger.info(
            "Concurrent embedding complete",
            num_embeddings=len(embeddings),
        )

        return embeddings

    def embed_texts_parallel(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Embed texts using Ray parallel processing.

        Falls back to concurrent.futures for 3-5x faster local processing when Ray unavailable.

        Args:
            texts: Texts to embed
            show_progress: Whether to log progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Use local processing for small batches or when Ray unavailable
        if len(texts) <= self.batch_size_per_worker or not self._is_ray_available():
            if not self._is_ray_available() and len(texts) > self.batch_size_per_worker:
                # Use concurrent embedding fallback for large batches without Ray
                return self._embed_texts_concurrent(texts)
            return self._local_service.embed_texts(texts)

        logger.info(
            "Starting Ray parallel embedding",
            num_texts=len(texts),
            num_workers=self.num_workers,
        )

        # Prepare config dict for Ray workers
        config_dict = {
            "openai_api_key": self.config.openai_api_key,
            "ollama_base_url": self.config.ollama_base_url,
            "anthropic_api_key": self.config.anthropic_api_key,
        }

        # Split texts into batches for workers
        batches = []
        for i in range(0, len(texts), self.batch_size_per_worker):
            batch = texts[i:i + self.batch_size_per_worker]
            batches.append(batch)

        # Submit all batches to Ray
        futures = [
            embed_batch_ray.remote(
                batch,
                self.provider,
                self.model,
                config_dict,
            )
            for batch in batches
        ]

        # Collect results
        try:
            all_results = ray.get(futures)

            # Flatten results
            embeddings = []
            for batch_result in all_results:
                embeddings.extend(batch_result)

            logger.info(
                "Ray parallel embedding complete",
                num_embeddings=len(embeddings),
            )

            return embeddings

        except Exception as e:
            logger.error(
                "Ray embedding failed, falling back to local",
                error=str(e),
            )
            return self._local_service.embed_texts(texts)

    def embed_chunks_parallel(
        self,
        chunks: List[Chunk],
    ) -> List[EmbeddingResult]:
        """
        Embed chunks using Ray parallel processing.

        Args:
            chunks: Chunks to embed

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts_parallel(texts)

        results = []
        for chunk, embedding in zip(chunks, embeddings):
            result = EmbeddingResult(
                chunk_id=f"{chunk.document_id}_{chunk.chunk_index}" if chunk.document_id else str(chunk.chunk_index),
                chunk_hash=chunk.chunk_hash,
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                metadata=chunk.metadata,
            )
            results.append(result)

        return results

    async def embed_texts_parallel_async(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Async version of embed_texts_parallel."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.embed_texts_parallel(texts)
            )


# =============================================================================
# Utility Functions
# =============================================================================

def get_embedding_service(
    provider: str = "openai",
    use_ray: bool = True,
) -> Union[EmbeddingService, RayEmbeddingService]:
    """
    Get appropriate embedding service based on configuration.

    Args:
        provider: Embedding provider
        use_ray: Whether to use Ray for parallel processing

    Returns:
        Embedding service instance
    """
    if use_ray:
        return RayEmbeddingService(provider=provider)
    return EmbeddingService(provider=provider)


def compute_similarity(
    embedding1: List[float],
    embedding2: List[float],
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    if len(embedding1) != len(embedding2):
        raise ValueError("Embeddings must have same dimensions")

    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def clear_embedding_cache():
    """Clear the global embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the embedding cache."""
    return {
        "size": len(_embedding_cache),
        "max_size": _CACHE_MAX_SIZE,
        "utilization_percent": (len(_embedding_cache) / _CACHE_MAX_SIZE) * 100 if _CACHE_MAX_SIZE > 0 else 0,
    }


# =============================================================================
# Persistent Embedding Cache (Redis-backed)
# =============================================================================


class EmbeddingCachePersistence:
    """
    Persistent embedding cache backed by Redis.

    Features:
    - Persists embeddings to Redis for cross-session reuse
    - Reduces API calls by caching expensive embedding operations
    - Supports bulk save/load for efficient startup preloading
    - Graceful fallback to in-memory when Redis unavailable

    Usage:
        cache = EmbeddingCachePersistence()

        # Save embedding
        await cache.save("hash123", [0.1, 0.2, ...])

        # Load embedding
        embedding = await cache.load("hash123")

        # Bulk preload on startup
        await cache.preload_to_memory()
    """

    def __init__(
        self,
        prefix: str = "emb_cache",
        ttl_days: int = 30,
        max_preload_items: int = 50000,
    ):
        """
        Initialize the persistent embedding cache.

        Args:
            prefix: Redis key prefix for embeddings
            ttl_days: Time-to-live for cached embeddings in days
            max_preload_items: Maximum items to preload from Redis on startup
        """
        self.prefix = prefix
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self.max_preload_items = max_preload_items
        self._redis_available = None  # Lazy check

        logger.info(
            "EmbeddingCachePersistence initialized",
            prefix=prefix,
            ttl_days=ttl_days,
            max_preload_items=max_preload_items,
        )

    def _make_key(self, content_hash: str) -> str:
        """Create a Redis key for an embedding hash."""
        return f"{self.prefix}:{content_hash}"

    async def _get_redis(self):
        """Get Redis client (lazy initialization)."""
        try:
            from backend.services.redis_client import get_redis_client
            return await get_redis_client()
        except Exception as e:
            logger.debug(f"Redis not available for embedding cache: {e}")
            return None

    async def is_redis_available(self) -> bool:
        """Check if Redis is available for persistent caching."""
        if self._redis_available is not None:
            return self._redis_available

        client = await self._get_redis()
        self._redis_available = client is not None
        return self._redis_available

    async def save(self, content_hash: str, embedding: List[float]) -> bool:
        """
        Save an embedding to Redis.

        Args:
            content_hash: Hash of the content that was embedded
            embedding: The embedding vector

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            client = await self._get_redis()
            if client is None:
                return False

            import json
            key = self._make_key(content_hash)
            await client.setex(key, self.ttl_seconds, json.dumps(embedding))
            return True

        except Exception as e:
            logger.debug(f"Failed to save embedding to Redis: {e}")
            return False

    async def load(self, content_hash: str) -> Optional[List[float]]:
        """
        Load an embedding from Redis.

        Args:
            content_hash: Hash of the content

        Returns:
            The embedding vector if found, None otherwise
        """
        try:
            client = await self._get_redis()
            if client is None:
                return None

            import json
            key = self._make_key(content_hash)
            data = await client.get(key)

            if data:
                return json.loads(data)
            return None

        except Exception as e:
            logger.debug(f"Failed to load embedding from Redis: {e}")
            return None

    async def load_batch(self, content_hashes: List[str]) -> Dict[str, List[float]]:
        """
        Load multiple embeddings from Redis in a single operation.

        Args:
            content_hashes: List of content hashes to load

        Returns:
            Dictionary mapping hashes to embeddings (only found entries)
        """
        if not content_hashes:
            return {}

        try:
            client = await self._get_redis()
            if client is None:
                return {}

            import json
            keys = [self._make_key(h) for h in content_hashes]
            results = {}

            # Use pipeline for efficient batch loading
            pipe = client.pipeline()
            for key in keys:
                pipe.get(key)

            values = await pipe.execute()

            for hash_val, value in zip(content_hashes, values):
                if value:
                    try:
                        results[hash_val] = json.loads(value)
                    except json.JSONDecodeError:
                        pass

            logger.debug(
                "Batch loaded embeddings from Redis",
                requested=len(content_hashes),
                found=len(results),
            )
            return results

        except Exception as e:
            logger.debug(f"Failed to batch load embeddings from Redis: {e}")
            return {}

    async def save_batch(self, embeddings: Dict[str, List[float]]) -> int:
        """
        Save multiple embeddings to Redis in a single operation.

        Args:
            embeddings: Dictionary mapping content hashes to embeddings

        Returns:
            Number of embeddings successfully saved
        """
        if not embeddings:
            return 0

        try:
            client = await self._get_redis()
            if client is None:
                return 0

            import json
            pipe = client.pipeline()

            for content_hash, embedding in embeddings.items():
                key = self._make_key(content_hash)
                pipe.setex(key, self.ttl_seconds, json.dumps(embedding))

            await pipe.execute()

            logger.debug(f"Batch saved {len(embeddings)} embeddings to Redis")
            return len(embeddings)

        except Exception as e:
            logger.debug(f"Failed to batch save embeddings to Redis: {e}")
            return 0

    async def preload_to_memory(self) -> int:
        """
        Preload embeddings from Redis into the in-memory cache.

        Call this during application startup to warm the cache.

        Returns:
            Number of embeddings loaded into memory
        """
        global _embedding_cache

        try:
            client = await self._get_redis()
            if client is None:
                logger.info("Redis not available, skipping embedding cache preload")
                return 0

            import json
            # Scan for embedding keys
            pattern = f"{self.prefix}:*"
            loaded = 0

            async for key in client.scan_iter(match=pattern, count=1000):
                if loaded >= self.max_preload_items:
                    break

                if len(_embedding_cache) >= _CACHE_MAX_SIZE:
                    break

                try:
                    value = await client.get(key)
                    if value:
                        # Extract hash from key
                        content_hash = key.replace(f"{self.prefix}:", "")
                        embedding = json.loads(value)
                        _embedding_cache[content_hash] = embedding
                        loaded += 1
                except Exception:
                    continue

            logger.info(
                "Preloaded embeddings from Redis to memory",
                loaded=loaded,
                memory_cache_size=len(_embedding_cache),
            )
            return loaded

        except Exception as e:
            logger.warning(f"Failed to preload embeddings from Redis: {e}")
            return 0

    async def persist_memory_cache(self) -> int:
        """
        Persist the current in-memory cache to Redis.

        Call this during application shutdown or periodically.

        Returns:
            Number of embeddings persisted
        """
        global _embedding_cache

        if not _embedding_cache:
            return 0

        return await self.save_batch(_embedding_cache)

    async def delete(self, content_hash: str) -> bool:
        """
        Delete an embedding from Redis.

        Args:
            content_hash: Hash of the content

        Returns:
            True if deleted, False otherwise
        """
        try:
            client = await self._get_redis()
            if client is None:
                return False

            key = self._make_key(content_hash)
            await client.delete(key)
            return True

        except Exception as e:
            logger.debug(f"Failed to delete embedding from Redis: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the persistent cache."""
        try:
            client = await self._get_redis()
            if client is None:
                return {
                    "redis_available": False,
                    "persistent_count": 0,
                    "memory_count": len(_embedding_cache),
                }

            # Count keys with our prefix
            pattern = f"{self.prefix}:*"
            count = 0
            async for _ in client.scan_iter(match=pattern, count=1000):
                count += 1
                if count > 100000:  # Cap counting at 100k to avoid long scans
                    break

            return {
                "redis_available": True,
                "persistent_count": count,
                "memory_count": len(_embedding_cache),
                "prefix": self.prefix,
                "ttl_days": self.ttl_seconds // (24 * 60 * 60),
            }

        except Exception as e:
            return {
                "redis_available": False,
                "error": str(e),
                "memory_count": len(_embedding_cache),
            }


# Singleton instance for persistent cache
_embedding_cache_persistence: Optional[EmbeddingCachePersistence] = None


def get_embedding_cache_persistence() -> EmbeddingCachePersistence:
    """Get or create the persistent embedding cache singleton."""
    global _embedding_cache_persistence
    if _embedding_cache_persistence is None:
        _embedding_cache_persistence = EmbeddingCachePersistence()
    return _embedding_cache_persistence


async def preload_embedding_cache():
    """
    Preload embeddings from Redis into memory cache on startup.

    Call this from application lifespan to warm the cache.
    """
    if os.getenv("EMBEDDING_CACHE_PRELOAD", "false").lower() == "true":
        cache = get_embedding_cache_persistence()
        await cache.preload_to_memory()


async def persist_embedding_cache():
    """
    Persist in-memory embedding cache to Redis on shutdown.

    Call this from application lifespan to save the cache.
    """
    if os.getenv("EMBEDDING_CACHE_PERSIST", "true").lower() == "true":
        cache = get_embedding_cache_persistence()
        await cache.persist_memory_cache()


# Provider-specific rate limits and optimal batch sizes
PROVIDER_BATCH_CONFIG = {
    "openai": {
        "max_batch_size": 2048,  # OpenAI supports up to 2048 texts per batch
        "requests_per_minute": 3000,  # Rate limit
        "tokens_per_minute": 1000000,  # TPM limit for embeddings
        "optimal_batch_size": 500,  # Good balance of speed and reliability
    },
    "ollama": {
        "max_batch_size": 100,  # Ollama is local, smaller batches
        "requests_per_minute": None,  # No rate limit
        "optimal_batch_size": 50,  # Keep batches small for local models
    },
    "huggingface": {
        "max_batch_size": 256,
        "requests_per_minute": 300,
        "optimal_batch_size": 100,
    },
    # Phase 29: Voyage AI - Best on MTEB benchmarks
    "voyage": {
        "max_batch_size": 128,  # Voyage max batch size
        "requests_per_minute": 300,  # Rate limit
        "optimal_batch_size": 100,  # Good balance
    },
    "voyage-lite": {
        "max_batch_size": 128,
        "requests_per_minute": 300,
        "optimal_batch_size": 100,
    },
    # Phase 67: Voyage 4 series
    "voyage4": {
        "max_batch_size": 128,
        "requests_per_minute": 300,
        "optimal_batch_size": 100,
    },
    "voyage4-lite": {
        "max_batch_size": 128,
        "requests_per_minute": 300,
        "optimal_batch_size": 100,
    },
    "voyage4-code": {
        "max_batch_size": 128,
        "requests_per_minute": 300,
        "optimal_batch_size": 100,
    },
    # Phase 68: Qwen3 Embedding - local model, no API rate limits
    "qwen3": {
        "max_batch_size": 64,  # Depends on GPU memory (8B model)
        "requests_per_minute": None,  # No rate limit (local)
        "optimal_batch_size": 32,  # Conservative for GPU memory
    },
    "qwen3-4b": {
        "max_batch_size": 128,  # 4B model needs less memory
        "requests_per_minute": None,
        "optimal_batch_size": 64,
    },
    "qwen3-small": {
        "max_batch_size": 256,  # 0.6B model is lightweight
        "requests_per_minute": None,
        "optimal_batch_size": 128,
    },
    # Phase 69: Google Gemini Embedding - API-based
    "gemini": {
        "max_batch_size": 100,  # Gemini batch limit
        "requests_per_minute": 1500,  # Rate limit
        "tokens_per_minute": 4000000,  # TPM limit
        "optimal_batch_size": 100,  # Good balance
    },
    "gemini-768": {
        "max_batch_size": 100,
        "requests_per_minute": 1500,
        "optimal_batch_size": 100,
    },
    "gemini-1536": {
        "max_batch_size": 100,
        "requests_per_minute": 1500,
        "optimal_batch_size": 100,
    },
    # Phase 69: NVIDIA NV-Embed-v2 - GPU optimized
    "nv-embed": {
        "max_batch_size": 64,  # GPU memory dependent
        "requests_per_minute": None,  # Local model
        "optimal_batch_size": 32,  # Conservative for GPU memory
    },
    "nv-embed-v2": {
        "max_batch_size": 64,
        "requests_per_minute": None,
        "optimal_batch_size": 32,
    },
    # Phase 69: BGE-M3 - Multi-retrieval model
    "bge-m3": {
        "max_batch_size": 64,  # GPU memory dependent
        "requests_per_minute": None,  # Local model
        "optimal_batch_size": 32,  # Conservative for multi-vector output
    },
    "bgem3": {
        "max_batch_size": 64,
        "requests_per_minute": None,
        "optimal_batch_size": 32,
    },
}


def get_optimal_batch_size(provider: str, num_texts: int) -> int:
    """
    Get the optimal batch size for a provider.

    Args:
        provider: The embedding provider
        num_texts: Total number of texts to embed

    Returns:
        Optimal batch size for the provider
    """
    config = PROVIDER_BATCH_CONFIG.get(provider, PROVIDER_BATCH_CONFIG["openai"])
    optimal = config["optimal_batch_size"]

    # For small batches, use the smaller of optimal or num_texts
    if num_texts <= optimal:
        return num_texts

    return optimal


# =============================================================================
# Phase 59: Distributed Embedding Integration
# =============================================================================

async def generate_embeddings_distributed(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 100,
    use_distributed: bool = True,
) -> List[List[float]]:
    """
    Generate embeddings using the distributed processor when available.

    Phase 59: Provides a unified entry point that routes to:
    - Ray cluster (if available and configured)
    - RayEmbeddingService (if Ray available but no cluster)
    - ThreadPoolExecutor (fallback)
    - Local sequential processing (final fallback)

    Args:
        texts: List of texts to embed
        model: Embedding model name (optional)
        batch_size: Batch size for processing
        use_distributed: Whether to try distributed processing first

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    # Try distributed processor first if enabled
    if use_distributed:
        try:
            from backend.services.distributed_processor import get_distributed_processor

            processor = await get_distributed_processor()
            embeddings = await processor.process_embeddings(
                texts=texts,
                model=model,
                batch_size=batch_size,
            )

            logger.info(
                "Generated embeddings via distributed processor",
                count=len(embeddings),
                ray_available=processor.ray_available,
            )
            return embeddings

        except ImportError:
            logger.debug("Distributed processor not available, falling back")
        except Exception as e:
            logger.warning(f"Distributed processor failed: {e}, falling back")

    # Fallback to RayEmbeddingService (handles Ray or concurrent.futures)
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    service = RayEmbeddingService(provider=provider, model=model)
    embeddings = service.embed_texts_parallel(texts)

    logger.info(
        "Generated embeddings via fallback",
        count=len(embeddings),
        provider=provider,
    )
    return embeddings


async def embed_documents_distributed(
    texts: List[str],
    provider: str = "openai",
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Convenience function for embedding documents with automatic distribution.

    Phase 59: This is the recommended entry point for embedding generation
    throughout the codebase. It automatically uses Ray when available.

    Args:
        texts: List of texts to embed
        provider: Embedding provider
        model: Model name (optional)

    Returns:
        List of embedding vectors
    """
    # Check if distributed processing is enabled in settings
    try:
        from backend.core.config import settings as core_settings
        use_ray = getattr(core_settings, 'USE_RAY_FOR_EMBEDDINGS', True)
    except ImportError:
        use_ray = True

    return await generate_embeddings_distributed(
        texts=texts,
        model=model,
        use_distributed=use_ray,
    )
