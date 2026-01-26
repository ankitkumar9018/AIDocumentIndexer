"""
AIDocumentIndexer - Advanced Reranking Service
===============================================

Phase 17: Multi-stage reranking pipeline for improved retrieval precision.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │ Stage 1: Fast Filtering (100 → 50)                  │
    │ • BM25 keyword match                                │
    │ • Quick relevance check                             │
    │ • Sub-10ms latency                                  │
    └─────────────────────────────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────────┐
    │ Stage 2: Cross-Encoder (50 → 20)                    │
    │ • BGE-reranker-v2 or similar                        │
    │ • Deep semantic relevance                           │
    │ • ~50ms latency                                     │
    └─────────────────────────────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────────┐
    │ Stage 3: ColBERT MaxSim (20 → 10) [Optional]        │
    │ • Late interaction scoring                          │
    │ • Fine-grained token matching                       │
    │ • ~30ms latency                                     │
    └─────────────────────────────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────────┐
    │ Stage 4: LLM Verification (10 → 5) [Optional]       │
    │ • Relevance verification                            │
    │ • Hallucination filtering                           │
    │ • ~200ms latency                                    │
    └─────────────────────────────────────────────────────┘

Reranker Models (2024-2026):
| Model | BEIR NDCG@10 | Latency | Size | Context |
|-------|--------------|---------|------|---------|
| Voyage rerank-2.5 | 71.8* | 80ms | API | 16K |
| Zerank-2 | 71.5* | 60ms | Local | 8K |
| Cohere rerank-v4 | 71.5* | 100ms | API | 32K |
| Cohere rerank-v4-fast | 69.2* | 50ms | API | 32K |
| Qwen3-Reranker-8B | 69.02* | 80ms | 8B | 8K |
| BGE-reranker-v2-gemma | 67.2 | 50ms | 2.5B | 8K |
| BGE-reranker-v2-m3 | 65.8 | 30ms | 568M | 8K |
| Cohere rerank-v3.5 | 68.1 | 80ms | API | 8K |
| ColBERT v2 | 64.2 | 10ms | 110M | 512 |
| Jina reranker v2 | 65.1 | 25ms | 278M | 8K |
| mxbai-rerank-v2 | 66.3 | 20ms | 278M | 8K |

* Voyage rerank-2.5 (Phase 69): Top-tier production consistency, built by Stanford researchers.
* Zerank-2 (Phase 69): Highest accuracy, open-source (Apache 2.0), self-hostable.
* Cohere v4 shows +400 ELO improvement on business document retrieval tasks.
  Self-learning capability improves accuracy over time with usage.
* mxbai-rerank-v2 (Phase 67): Open-source, 100+ languages, strong multilingual.
* Qwen3-Reranker-8B (Phase 68): Best open-source multilingual reranker (69.02 BEIR).
  Instruction-following, 100+ languages, 8K context window.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class RerankerModel(str, Enum):
    """Available reranker models."""
    BGE_V2_GEMMA = "bge-reranker-v2-gemma"      # Best quality (2.5B)
    BGE_V2_M3 = "bge-reranker-v2-m3"            # Good balance (568M)
    BGE_LARGE = "bge-reranker-large"            # Smaller (560M)
    COHERE_V4 = "cohere-rerank-v4"              # API-based, 32K context, best quality
    COHERE_V4_FAST = "cohere-rerank-v4-fast"    # API-based, 32K context, low latency
    COHERE_V3_5 = "cohere-rerank-v3.5"          # API-based, 8K context
    COHERE_V3 = "cohere-rerank-v3"              # API-based (legacy)
    JINA_V2 = "jina-reranker-v2"                # Fast (278M)
    # Phase 67: mxbai-rerank-v2 - open-source, 100+ languages
    MXBAI_RERANK_V2 = "mxbai-rerank-v2"         # mixedbread.ai (278M, 100+ languages)
    MXBAI_RERANK_LARGE = "mxbai-rerank-large"   # mixedbread.ai large variant
    # Phase 68: Qwen3 Reranker - 69.02 multilingual ranking (top performer)
    QWEN3_RERANKER_8B = "qwen3-reranker-8b"     # Alibaba (8B, 8K context, best multilingual)
    QWEN3_RERANKER_4B = "qwen3-reranker-4b"     # Alibaba (4B, balanced)
    QWEN3_RERANKER_SMALL = "qwen3-reranker-small"  # Alibaba (0.6B, fast)
    # Phase 69: Voyage Rerank 2.5 & Zerank 2 (2026 Top Performers)
    VOYAGE_RERANK_2_5 = "voyage-rerank-2.5"     # Voyage AI (top-tier production consistency)
    VOYAGE_RERANK_2 = "voyage-rerank-2"         # Voyage AI (previous gen)
    ZERANK_2 = "zerank-2"                       # ZeroEntropy (highest accuracy, Apache 2.0)
    ZERANK_MINI = "zerank-mini"                 # ZeroEntropy (fast, lightweight)
    COLBERT_V2 = "colbert-v2"                   # Late interaction
    LLM = "llm"                                  # LLM-based verification


class RerankerBackend(str, Enum):
    """Reranker backends."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    FASTEMBED = "fastembed"
    COHERE = "cohere"
    OPENAI = "openai"
    COLBERT = "colbert"


@dataclass
class RerankerConfig:
    """Configuration for reranking pipeline."""

    # Model selection
    primary_model: RerankerModel = RerankerModel.BGE_V2_M3
    secondary_model: Optional[RerankerModel] = None  # For multi-stage
    use_colbert: bool = False
    use_llm_verification: bool = False

    # Stage configurations
    stage1_top_k: int = 50    # After fast filtering
    stage2_top_k: int = 20    # After cross-encoder
    stage3_top_k: int = 10    # After ColBERT (if enabled)
    final_top_k: int = 5      # Final results

    # Performance
    batch_size: int = 32
    max_length: int = 512     # For local cross-encoders
    use_fp16: bool = True

    # Cohere v4 specific
    cohere_max_chunks_per_doc: int = 10  # Max chunks for long docs in Cohere v4

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    # Thresholds
    min_relevance_score: float = 0.1  # Filter out low-relevance results
    llm_verification_threshold: float = 0.7  # Min score for LLM check

    @classmethod
    def cohere_v4_config(cls) -> "RerankerConfig":
        """Create config optimized for Cohere v4 (best quality, 32K context)."""
        return cls(
            primary_model=RerankerModel.COHERE_V4,
            stage1_top_k=100,  # Can handle more with 32K context
            stage2_top_k=50,
            cohere_max_chunks_per_doc=20,
        )

    @classmethod
    def cohere_v4_fast_config(cls) -> "RerankerConfig":
        """Create config optimized for Cohere v4-fast (low latency, 32K context)."""
        return cls(
            primary_model=RerankerModel.COHERE_V4_FAST,
            stage1_top_k=50,
            stage2_top_k=20,
            cohere_max_chunks_per_doc=10,
        )

    @classmethod
    def qwen3_reranker_config(cls, model_size: str = "8b") -> "RerankerConfig":
        """
        Create config optimized for Qwen3 Reranker (best multilingual, 8K context).

        Args:
            model_size: Model size - "8b" (best), "4b" (balanced), "small" (fast)

        Phase 68: Qwen3-Reranker features:
        - 69.02 BEIR NDCG@10 (best open-source multilingual)
        - 8K context window
        - Instruction-following capability
        - 100+ language support
        """
        model_map = {
            "8b": RerankerModel.QWEN3_RERANKER_8B,
            "4b": RerankerModel.QWEN3_RERANKER_4B,
            "small": RerankerModel.QWEN3_RERANKER_SMALL,
        }
        return cls(
            primary_model=model_map.get(model_size, RerankerModel.QWEN3_RERANKER_8B),
            max_length=8192,  # Qwen3 supports 8K context
            stage1_top_k=100,  # Can handle more with 8K context
            stage2_top_k=30,
        )

    @classmethod
    def voyage_rerank_config(cls, version: str = "2.5") -> "RerankerConfig":
        """
        Create config optimized for Voyage Rerank (top-tier production consistency).

        Args:
            version: "2.5" (latest) or "2" (previous gen)

        Phase 69: Voyage Rerank features:
        - Top-tier production consistency
        - Strong performance on business documents
        - Built by Stanford researchers
        """
        model_map = {
            "2.5": RerankerModel.VOYAGE_RERANK_2_5,
            "2": RerankerModel.VOYAGE_RERANK_2,
        }
        return cls(
            primary_model=model_map.get(version, RerankerModel.VOYAGE_RERANK_2_5),
            stage1_top_k=100,
            stage2_top_k=30,
        )

    @classmethod
    def zerank_config(cls, model_size: str = "full") -> "RerankerConfig":
        """
        Create config optimized for Zerank (highest accuracy, Apache 2.0).

        Args:
            model_size: "full" (zerank-2) or "mini" (fast)

        Phase 69: Zerank features:
        - Highest accuracy in benchmarks
        - Open-source (Apache 2.0)
        - Self-hostable
        """
        model_map = {
            "full": RerankerModel.ZERANK_2,
            "mini": RerankerModel.ZERANK_MINI,
        }
        return cls(
            primary_model=model_map.get(model_size, RerankerModel.ZERANK_2),
            max_length=8192,
            stage1_top_k=100,
            stage2_top_k=30,
        )


@dataclass
class RerankResult:
    """Result from reranking."""
    chunk_id: str
    content: str
    score: float
    original_rank: int
    new_rank: int
    stage_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResponse:
    """Full reranking response."""
    results: List[RerankResult]
    query: str
    total_candidates: int
    stages_used: List[str]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Reranker Interface
# =============================================================================

class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model."""
        pass

    @abstractmethod
    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Score documents against query.

        Args:
            query: Search query
            documents: List of document texts

        Returns:
            List of relevance scores (higher = more relevant)
        """
        pass

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents and return top-k.

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        scores = await self.score(query, documents)

        # Create (index, score) pairs and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]


# =============================================================================
# Cross-Encoder Reranker (BGE, Jina)
# =============================================================================

class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Supports:
    - BGE-reranker-v2-gemma (2.5B, best quality)
    - BGE-reranker-v2-m3 (568M, good balance)
    - Jina-reranker-v2 (278M, fast)
    """

    MODEL_MAP = {
        RerankerModel.BGE_V2_GEMMA: "BAAI/bge-reranker-v2-gemma",
        RerankerModel.BGE_V2_M3: "BAAI/bge-reranker-v2-m3",
        RerankerModel.BGE_LARGE: "BAAI/bge-reranker-large",
        RerankerModel.JINA_V2: "jinaai/jina-reranker-v2-base-multilingual",
        # Phase 67: mxbai-rerank-v2 from mixedbread.ai
        # 100+ languages, strong multilingual performance
        RerankerModel.MXBAI_RERANK_V2: "mixedbread-ai/mxbai-rerank-xsmall-v2",
        RerankerModel.MXBAI_RERANK_LARGE: "mixedbread-ai/mxbai-rerank-large-v1",
        # Phase 68: Qwen3 Reranker - 69.02 BEIR score (best multilingual)
        # 8K context, instruction-following, 100+ languages
        RerankerModel.QWEN3_RERANKER_8B: "Alibaba-NLP/Qwen3-Reranker-8B",
        RerankerModel.QWEN3_RERANKER_4B: "Alibaba-NLP/Qwen3-Reranker-4B",
        RerankerModel.QWEN3_RERANKER_SMALL: "Alibaba-NLP/Qwen3-Reranker-0.6B",
    }

    def __init__(
        self,
        model: RerankerModel = RerankerModel.BGE_V2_M3,
        max_length: int = 512,
        use_fp16: bool = True,
        batch_size: int = 32,
    ):
        model_path = self.MODEL_MAP.get(model, self.MODEL_MAP[RerankerModel.BGE_V2_M3])
        super().__init__(model_path)

        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the cross-encoder model."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                # Import in async context to avoid blocking
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    None, self._load_model
                )
                self._initialized = True
                logger.info(f"Cross-encoder loaded: {self.model_name}")

            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                raise

    def _load_model(self):
        """Load model (runs in executor)."""
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device="cuda" if self._has_cuda() else "cpu",
            )

            # Enable fp16 if supported
            if self.use_fp16 and self._has_cuda():
                model.model.half()

            return model

        except ImportError:
            logger.warning("sentence-transformers not available, using fastembed fallback")
            return self._load_fastembed_model()

    def _load_fastembed_model(self):
        """Load using fastembed as fallback."""
        try:
            from fastembed import TextEmbedding

            # FastEmbed doesn't have cross-encoders, so we use bi-encoder
            # with cosine similarity as approximation
            return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except ImportError:
            logger.error("Neither sentence-transformers nor fastembed available")
            return None

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score documents using cross-encoder."""
        if not self._initialized:
            await self.initialize()

        if self._model is None:
            # Fallback: return uniform scores
            return [0.5] * len(documents)

        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Score in batches
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self._model.predict(pairs, batch_size=self.batch_size),
            )

            return [float(s) for s in scores]

        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return [0.5] * len(documents)


# =============================================================================
# Cohere Reranker (API-based)
# =============================================================================

class CohereReranker(BaseReranker):
    """
    Cohere rerank API reranker.

    Supports Cohere's rerank models:
    - rerank-v4 (2025): Best quality, 32K context, +400 ELO on business tasks
    - rerank-v4-fast (2025): Low latency variant, 32K context
    - rerank-v3.5 (2024): Previous gen, 8K context
    - rerank-v3.0 (legacy): Original v3, 8K context
    """

    # Model name mappings
    MODEL_MAP = {
        RerankerModel.COHERE_V4: "rerank-v4",
        RerankerModel.COHERE_V4_FAST: "rerank-v4-fast",
        RerankerModel.COHERE_V3_5: "rerank-v3.5",
        RerankerModel.COHERE_V3: "rerank-english-v3.0",
    }

    # Context window limits per model
    CONTEXT_LIMITS = {
        "rerank-v4": 32000,
        "rerank-v4-fast": 32000,
        "rerank-v3.5": 8000,
        "rerank-english-v3.0": 4096,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Union[RerankerModel, str] = RerankerModel.COHERE_V4,
        max_chunks_per_doc: int = 10,
    ):
        # Resolve model name
        if isinstance(model, RerankerModel):
            model_name = self.MODEL_MAP.get(model, "rerank-v4")
        else:
            model_name = model

        super().__init__(model_name)
        self.api_key = api_key or getattr(settings, "COHERE_API_KEY", None)
        self.max_chunks_per_doc = max_chunks_per_doc
        self.context_limit = self.CONTEXT_LIMITS.get(model_name, 8000)
        self._client = None

    async def initialize(self) -> None:
        """Initialize Cohere client."""
        if self._initialized:
            return

        if not self.api_key:
            logger.warning("Cohere API key not configured")
            return

        try:
            import cohere
            self._client = cohere.AsyncClient(api_key=self.api_key)
            self._initialized = True
            logger.info(
                "Cohere reranker initialized",
                model=self.model_name,
                context_limit=self.context_limit,
            )

        except ImportError:
            logger.warning("cohere package not installed")

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score using Cohere rerank API."""
        if not self._initialized:
            await self.initialize()

        if self._client is None:
            return [0.5] * len(documents)

        try:
            # Truncate documents to context limit if needed
            truncated_docs = []
            for doc in documents:
                if len(doc) > self.context_limit:
                    # Truncate intelligently at word boundary
                    truncated = doc[:self.context_limit]
                    last_space = truncated.rfind(' ')
                    if last_space > self.context_limit * 0.8:
                        truncated = truncated[:last_space]
                    truncated_docs.append(truncated)
                else:
                    truncated_docs.append(doc)

            response = await self._client.rerank(
                query=query,
                documents=truncated_docs,
                model=self.model_name,
                top_n=len(documents),  # Get all scores
                max_chunks_per_doc=self.max_chunks_per_doc,
            )

            # Create score map
            scores = [0.0] * len(documents)
            for result in response.results:
                scores[result.index] = result.relevance_score

            return scores

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return [0.5] * len(documents)

    async def rerank_with_reasoning(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rerank with v4's self-explaining capability.

        Returns relevance scores with reasoning for top results.
        Only available with rerank-v4 model.
        """
        if not self._initialized:
            await self.initialize()

        if self._client is None:
            return []

        if "v4" not in self.model_name:
            logger.warning("Reasoning only available with rerank-v4 models")
            # Fall back to regular scoring
            scores = await self.score(query, documents)
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return [
                {"index": idx, "score": score, "reasoning": None}
                for idx, score in indexed[:top_k]
            ]

        try:
            response = await self._client.rerank(
                query=query,
                documents=documents,
                model=self.model_name,
                top_n=top_k,
                return_documents=True,
            )

            results = []
            for result in response.results:
                results.append({
                    "index": result.index,
                    "score": result.relevance_score,
                    "document": result.document.text if result.document else None,
                    "reasoning": getattr(result, 'reasoning', None),
                })

            return results

        except Exception as e:
            logger.error(f"Cohere rerank with reasoning failed: {e}")
            return []


# =============================================================================
# Phase 69: Voyage Reranker (API-based)
# =============================================================================

class VoyageReranker(BaseReranker):
    """
    Voyage AI rerank API reranker.

    Supports Voyage's rerank models:
    - rerank-2.5 (2026): Top-tier production consistency
    - rerank-2 (2025): Previous gen, strong accuracy

    Known for:
    - Production consistency and reliability
    - Strong performance on business documents
    - Built by Stanford researchers
    """

    # Model name mappings
    MODEL_MAP = {
        RerankerModel.VOYAGE_RERANK_2_5: "rerank-2.5",
        RerankerModel.VOYAGE_RERANK_2: "rerank-2",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Union[RerankerModel, str] = RerankerModel.VOYAGE_RERANK_2_5,
        top_k: int = 100,
    ):
        # Resolve model name
        if isinstance(model, RerankerModel):
            model_name = self.MODEL_MAP.get(model, "rerank-2.5")
        else:
            model_name = model

        super().__init__(model_name)
        import os
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.top_k = top_k
        self._client = None

    async def initialize(self) -> None:
        """Initialize Voyage client."""
        if self._initialized:
            return

        if not self.api_key:
            logger.warning("Voyage API key not configured")
            return

        try:
            import voyageai
            self._client = voyageai.Client(api_key=self.api_key)
            self._initialized = True
            logger.info(
                "Voyage reranker initialized",
                model=self.model_name,
            )

        except ImportError:
            logger.warning("voyageai package not installed")

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score using Voyage rerank API."""
        if not self._initialized:
            await self.initialize()

        if self._client is None:
            return [0.5] * len(documents)

        try:
            # Use sync API in thread pool
            import asyncio
            loop = asyncio.get_running_loop()

            reranking = await loop.run_in_executor(
                None,
                lambda: self._client.rerank(
                    query=query,
                    documents=documents,
                    model=self.model_name,
                    top_k=min(self.top_k, len(documents)),
                ),
            )

            # Create score map
            scores = [0.0] * len(documents)
            for result in reranking.results:
                scores[result.index] = result.relevance_score

            return scores

        except Exception as e:
            logger.error(f"Voyage reranking failed: {e}")
            return [0.5] * len(documents)


# =============================================================================
# Phase 69: Zerank Reranker (Open-source, Apache 2.0)
# =============================================================================

class ZerankReranker(BaseReranker):
    """
    Zerank reranker from ZeroEntropy (Apache 2.0 license).

    Supports Zerank models:
    - zerank-2: Highest accuracy scores
    - zerank-mini: Fast, lightweight variant

    Known for:
    - Highest accuracy in benchmarks
    - Open-source (Apache 2.0)
    - Can be self-hosted or API
    """

    # Model name mappings for HuggingFace
    MODEL_MAP = {
        RerankerModel.ZERANK_2: "zeroentropy/zerank-2",
        RerankerModel.ZERANK_MINI: "zeroentropy/zerank-mini",
    }

    def __init__(
        self,
        model: Union[RerankerModel, str] = RerankerModel.ZERANK_2,
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 8192,
    ):
        # Resolve model name
        if isinstance(model, RerankerModel):
            model_name = self.MODEL_MAP.get(model, "zeroentropy/zerank-2")
        else:
            model_name = model

        super().__init__(model_name)
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    async def initialize(self) -> None:
        """Initialize Zerank model."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                logger.info(
                    "Loading Zerank model",
                    model=self.model_name,
                    device=self.device,
                )

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )

                dtype = torch.float16 if self.use_fp16 and self.device == "cuda" else torch.float32
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).to(self.device)

                self._model.eval()
                self._initialized = True

                logger.info(
                    "Zerank reranker initialized",
                    model=self.model_name,
                    device=self.device,
                )

            except ImportError as e:
                logger.warning(f"transformers/torch not installed for Zerank: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Zerank: {e}")

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score using Zerank model."""
        if not self._initialized:
            await self.initialize()

        if self._model is None:
            return [0.5] * len(documents)

        try:
            import torch

            # Prepare pairs
            pairs = [[query, doc] for doc in documents]

            # Tokenize
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Score
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Sigmoid to get 0-1 scores
                scores = torch.sigmoid(outputs.logits.squeeze(-1))

            return scores.cpu().tolist()

        except Exception as e:
            logger.error(f"Zerank scoring failed: {e}")
            return [0.5] * len(documents)


# =============================================================================
# ColBERT Reranker (Late Interaction)
# =============================================================================

class ColBERTReranker(BaseReranker):
    """
    ColBERT-based reranker using late interaction.

    Provides fine-grained token-level matching for
    better precision on complex queries.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_length: int = 256,
    ):
        super().__init__(model_name)
        self.max_length = max_length
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize ColBERT model."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                # Try RAGatouille first
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    None, self._load_colbert
                )
                self._initialized = True
                logger.info("ColBERT reranker initialized")

            except Exception as e:
                logger.warning(f"ColBERT initialization failed: {e}")

    def _load_colbert(self):
        """Load ColBERT model."""
        try:
            from ragatouille import RAGPretrainedModel

            model = RAGPretrainedModel.from_pretrained(self.model_name)
            return model

        except ImportError:
            logger.warning("ragatouille not available for ColBERT")
            return None

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score using ColBERT MaxSim."""
        if not self._initialized:
            await self.initialize()

        if self._model is None:
            return [0.5] * len(documents)

        try:
            loop = asyncio.get_running_loop()

            # ColBERT rerank
            results = await loop.run_in_executor(
                None,
                lambda: self._model.rerank(
                    query=query,
                    documents=documents,
                    k=len(documents),
                ),
            )

            # Extract scores (results are sorted by score)
            scores = [0.0] * len(documents)
            for i, result in enumerate(results):
                # Find original index
                for j, doc in enumerate(documents):
                    if doc == result.get("content", ""):
                        scores[j] = result.get("score", 0.5)
                        break

            return scores

        except Exception as e:
            logger.error(f"ColBERT scoring failed: {e}")
            return [0.5] * len(documents)


# =============================================================================
# LLM Verification Reranker
# =============================================================================

class LLMVerificationReranker(BaseReranker):
    """
    LLM-based relevance verification.

    Uses an LLM to verify relevance of top candidates,
    helping filter out false positives and hallucination-prone content.
    """

    VERIFICATION_PROMPT = """You are evaluating document relevance for a search query.

Query: {query}

Document:
{document}

Rate the document's relevance to the query on a scale of 0-10:
- 0-2: Not relevant at all
- 3-4: Tangentially related
- 5-6: Somewhat relevant
- 7-8: Relevant
- 9-10: Highly relevant, directly answers the query

Respond with ONLY a number between 0 and 10."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 5,
    ):
        super().__init__(model)
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def initialize(self) -> None:
        """No initialization needed for LLM."""
        self._initialized = True

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score using LLM verification."""
        if not documents:
            return []

        # Score documents in parallel with concurrency limit
        tasks = [
            self._score_single(query, doc, i)
            for i, doc in enumerate(documents)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores = []
        for result in results:
            if isinstance(result, Exception):
                scores.append(0.5)  # Default on error
            else:
                scores.append(result)

        return scores

    async def _score_single(
        self,
        query: str,
        document: str,
        index: int,
    ) -> float:
        """Score a single document."""
        async with self._semaphore:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage

                llm = ChatOpenAI(model=self.model_name, temperature=0)

                prompt = self.VERIFICATION_PROMPT.format(
                    query=query,
                    document=document[:2000],  # Truncate long documents
                )

                response = await llm.ainvoke([HumanMessage(content=prompt)])
                text = response.content.strip()

                # Parse score
                try:
                    score = float(text) / 10.0  # Normalize to 0-1
                    return min(max(score, 0.0), 1.0)
                except ValueError:
                    return 0.5

            except Exception as e:
                logger.warning(f"LLM verification failed: {e}")
                return 0.5


# =============================================================================
# BM25 Fast Filter
# =============================================================================

class BM25FastFilter(BaseReranker):
    """
    Fast BM25 keyword matching for initial filtering.

    Removes obviously irrelevant documents before expensive reranking.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__("bm25")
        self.k1 = k1
        self.b = b
        self._initialized = True

    async def initialize(self) -> None:
        """No initialization needed."""
        pass

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score using BM25."""
        if not documents:
            return []

        try:
            from rank_bm25 import BM25Okapi

            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in documents]
            tokenized_query = query.lower().split()

            # Build BM25 index
            bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

            # Score
            scores = bm25.get_scores(tokenized_query)

            # Normalize to 0-1
            max_score = max(scores) if scores.max() > 0 else 1.0
            normalized = [s / max_score for s in scores]

            return normalized

        except ImportError:
            # Fallback: simple keyword matching
            return self._simple_keyword_score(query, documents)

    def _simple_keyword_score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Simple keyword overlap scoring as fallback."""
        query_words = set(query.lower().split())

        scores = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scores.append(score)

        return scores


# =============================================================================
# Multi-Stage Reranker Pipeline
# =============================================================================

class MultiStageReranker:
    """
    Multi-stage reranking pipeline.

    Combines multiple reranking strategies in stages for
    optimal precision/latency tradeoff.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()

        # Initialize rerankers based on config
        self._fast_filter = BM25FastFilter()
        self._primary_reranker: Optional[BaseReranker] = None
        self._secondary_reranker: Optional[BaseReranker] = None
        self._colbert_reranker: Optional[ColBERTReranker] = None
        self._llm_verifier: Optional[LLMVerificationReranker] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all rerankers."""
        if self._initialized:
            return

        # Primary cross-encoder
        if self.config.primary_model in [
            RerankerModel.BGE_V2_GEMMA,
            RerankerModel.BGE_V2_M3,
            RerankerModel.BGE_LARGE,
            RerankerModel.JINA_V2,
            # Phase 67: mxbai rerankers
            RerankerModel.MXBAI_RERANK_V2,
            RerankerModel.MXBAI_RERANK_LARGE,
            # Phase 68: Qwen3 rerankers
            RerankerModel.QWEN3_RERANKER_8B,
            RerankerModel.QWEN3_RERANKER_4B,
            RerankerModel.QWEN3_RERANKER_SMALL,
        ]:
            self._primary_reranker = CrossEncoderReranker(
                model=self.config.primary_model,
                max_length=self.config.max_length,
                use_fp16=self.config.use_fp16,
                batch_size=self.config.batch_size,
            )
        elif self.config.primary_model in [
            RerankerModel.COHERE_V4,
            RerankerModel.COHERE_V4_FAST,
            RerankerModel.COHERE_V3_5,
            RerankerModel.COHERE_V3,
        ]:
            self._primary_reranker = CohereReranker(model=self.config.primary_model)
        # Phase 69: Voyage Rerank 2.5
        elif self.config.primary_model in [
            RerankerModel.VOYAGE_RERANK_2_5,
            RerankerModel.VOYAGE_RERANK_2,
        ]:
            self._primary_reranker = VoyageReranker(model=self.config.primary_model)
        # Phase 69: Zerank (open-source)
        elif self.config.primary_model in [
            RerankerModel.ZERANK_2,
            RerankerModel.ZERANK_MINI,
        ]:
            self._primary_reranker = ZerankReranker(
                model=self.config.primary_model,
                use_fp16=self.config.use_fp16,
            )

        # Optional ColBERT
        if self.config.use_colbert:
            self._colbert_reranker = ColBERTReranker()

        # Optional LLM verification
        if self.config.use_llm_verification:
            self._llm_verifier = LLMVerificationReranker()

        # Initialize all
        init_tasks = [self._fast_filter.initialize()]

        if self._primary_reranker:
            init_tasks.append(self._primary_reranker.initialize())
        if self._colbert_reranker:
            init_tasks.append(self._colbert_reranker.initialize())
        if self._llm_verifier:
            init_tasks.append(self._llm_verifier.initialize())

        await asyncio.gather(*init_tasks, return_exceptions=True)

        self._initialized = True
        logger.info("Multi-stage reranker initialized")

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResponse:
        """
        Rerank documents through multi-stage pipeline.

        Args:
            query: Search query
            documents: List of document dicts with 'content' and 'chunk_id'
            top_k: Final number of results (defaults to config.final_top_k)

        Returns:
            RerankResponse with reranked results
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        top_k = top_k or self.config.final_top_k
        stages_used = []

        # Extract content for scoring
        contents = [doc.get("content", "") for doc in documents]
        total_candidates = len(documents)

        # Track scores through stages
        current_indices = list(range(len(documents)))
        stage_scores: Dict[int, Dict[str, float]] = {
            i: {} for i in range(len(documents))
        }

        # Stage 1: BM25 Fast Filter (100 → 50)
        if len(contents) > self.config.stage1_top_k:
            bm25_scores = await self._fast_filter.score(query, contents)

            for i, score in enumerate(bm25_scores):
                stage_scores[i]["bm25"] = score

            # Filter to top candidates
            indexed = [(i, bm25_scores[i]) for i in current_indices]
            indexed.sort(key=lambda x: x[1], reverse=True)
            current_indices = [i for i, _ in indexed[:self.config.stage1_top_k]]
            contents = [documents[i].get("content", "") for i in current_indices]
            stages_used.append("bm25")

        # Stage 2: Cross-Encoder (50 → 20)
        if self._primary_reranker and len(contents) > self.config.stage2_top_k:
            cross_scores = await self._primary_reranker.score(query, contents)

            for idx, score in zip(current_indices, cross_scores):
                stage_scores[idx]["cross_encoder"] = score

            # Filter to top candidates
            indexed = list(zip(current_indices, cross_scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            current_indices = [i for i, _ in indexed[:self.config.stage2_top_k]]
            contents = [documents[i].get("content", "") for i in current_indices]
            stages_used.append("cross_encoder")

        # Stage 3: ColBERT (20 → 10) [Optional]
        if self._colbert_reranker and len(contents) > self.config.stage3_top_k:
            colbert_scores = await self._colbert_reranker.score(query, contents)

            for idx, score in zip(current_indices, colbert_scores):
                stage_scores[idx]["colbert"] = score

            # Filter to top candidates
            indexed = list(zip(current_indices, colbert_scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            current_indices = [i for i, _ in indexed[:self.config.stage3_top_k]]
            contents = [documents[i].get("content", "") for i in current_indices]
            stages_used.append("colbert")

        # Stage 4: LLM Verification (10 → 5) [Optional]
        if self._llm_verifier and len(contents) > top_k:
            llm_scores = await self._llm_verifier.score(query, contents)

            for idx, score in zip(current_indices, llm_scores):
                stage_scores[idx]["llm"] = score

            # Filter by threshold and top-k
            indexed = list(zip(current_indices, llm_scores))
            indexed = [
                (i, s) for i, s in indexed
                if s >= self.config.llm_verification_threshold
            ]
            indexed.sort(key=lambda x: x[1], reverse=True)
            current_indices = [i for i, _ in indexed[:top_k]]
            stages_used.append("llm_verification")

        # Build final results
        results = []
        for new_rank, orig_idx in enumerate(current_indices[:top_k]):
            doc = documents[orig_idx]

            # Compute final score (weighted combination of stage scores)
            scores = stage_scores[orig_idx]
            final_score = self._compute_final_score(scores)

            results.append(RerankResult(
                chunk_id=doc.get("chunk_id", str(orig_idx)),
                content=doc.get("content", ""),
                score=final_score,
                original_rank=orig_idx,
                new_rank=new_rank,
                stage_scores=scores,
                metadata=doc.get("metadata", {}),
            ))

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "Reranking complete",
            query_length=len(query),
            candidates=total_candidates,
            results=len(results),
            stages=stages_used,
            latency_ms=round(latency_ms, 2),
        )

        return RerankResponse(
            results=results,
            query=query,
            total_candidates=total_candidates,
            stages_used=stages_used,
            latency_ms=latency_ms,
        )

    def _compute_final_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted final score from stage scores."""
        # Weights for each stage (tuned for quality)
        weights = {
            "bm25": 0.1,
            "cross_encoder": 0.5,
            "colbert": 0.3,
            "llm": 0.1,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for stage, score in scores.items():
            weight = weights.get(stage, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.5


# =============================================================================
# Admin Settings Integration (Phase 68)
# Phase 71.5: TTL-based settings cache (auto-invalidates after 5 minutes)
# =============================================================================

import time as _time_module

_qwen3_reranker_settings_cache: Optional[Dict[str, Any]] = None
_qwen3_reranker_settings_cache_time: Optional[float] = None
_QWEN3_RERANKER_CACHE_TTL = 300  # 5 minutes


def _get_qwen3_reranker_settings() -> Dict[str, Any]:
    """
    Get Qwen3 reranker settings from admin settings with env var fallback.

    Settings hierarchy: Admin settings (DB) → Environment variables → Defaults
    """
    global _qwen3_reranker_settings_cache, _qwen3_reranker_settings_cache_time

    now = _time_module.time()

    # Return cached settings if available and not expired
    if (_qwen3_reranker_settings_cache is not None
        and _qwen3_reranker_settings_cache_time is not None
        and (now - _qwen3_reranker_settings_cache_time) < _QWEN3_RERANKER_CACHE_TTL):
        return _qwen3_reranker_settings_cache

    import os

    # Default settings from environment variables
    settings = {
        "enabled": os.getenv("QWEN3_RERANKER_ENABLED", "false").lower() == "true",
        "model": os.getenv("QWEN3_RERANKER_MODEL", "qwen3-reranker-8b"),
        "max_length": int(os.getenv("QWEN3_RERANKER_MAX_LENGTH", "8192")),
        "use_fp16": os.getenv("QWEN3_RERANKER_USE_FP16", "true").lower() == "true",
        "batch_size": int(os.getenv("QWEN3_RERANKER_BATCH_SIZE", "32")),
    }

    try:
        # Try to load from database (sync version for non-async contexts)
        from backend.db.database import get_sync_session
        from backend.db.models import SystemSettings
        from sqlalchemy import select

        session = get_sync_session()
        try:
            result = session.execute(
                select(SystemSettings).where(
                    SystemSettings.key.in_([
                        "reranker.qwen3_enabled",
                        "reranker.qwen3_model",
                        "reranker.qwen3_max_length",
                        "reranker.qwen3_use_fp16",
                        "reranker.qwen3_batch_size",
                    ])
                )
            )
            db_settings = {row.key: row.value for row in result.scalars()}

            # Override with database settings if present
            if "reranker.qwen3_enabled" in db_settings:
                val = db_settings["reranker.qwen3_enabled"]
                settings["enabled"] = val if isinstance(val, bool) else str(val).lower() == "true"
            if "reranker.qwen3_model" in db_settings:
                settings["model"] = db_settings["reranker.qwen3_model"]
            if "reranker.qwen3_max_length" in db_settings:
                settings["max_length"] = int(db_settings["reranker.qwen3_max_length"])
            if "reranker.qwen3_use_fp16" in db_settings:
                val = db_settings["reranker.qwen3_use_fp16"]
                settings["use_fp16"] = val if isinstance(val, bool) else str(val).lower() == "true"
            if "reranker.qwen3_batch_size" in db_settings:
                settings["batch_size"] = int(db_settings["reranker.qwen3_batch_size"])
        finally:
            session.close()
    except Exception as e:
        logger.debug("Using env var defaults for Qwen3 reranker settings", reason=str(e))

    # Cache the settings with timestamp
    _qwen3_reranker_settings_cache = settings
    _qwen3_reranker_settings_cache_time = _time_module.time()
    return settings


def reset_qwen3_reranker_settings_cache() -> None:
    """Reset the Qwen3 reranker settings cache to reload from admin settings."""
    global _qwen3_reranker_settings_cache, _qwen3_reranker_settings_cache_time
    _qwen3_reranker_settings_cache = None
    _qwen3_reranker_settings_cache_time = None


def get_qwen3_reranker_config() -> Optional[RerankerConfig]:
    """
    Get RerankerConfig for Qwen3 if enabled in admin settings.

    Returns None if Qwen3 reranker is not enabled.
    """
    settings = _get_qwen3_reranker_settings()

    if not settings["enabled"]:
        return None

    # Map model string to enum
    model_map = {
        "qwen3-reranker-8b": RerankerModel.QWEN3_RERANKER_8B,
        "qwen3-reranker-4b": RerankerModel.QWEN3_RERANKER_4B,
        "qwen3-reranker-small": RerankerModel.QWEN3_RERANKER_SMALL,
    }

    model = model_map.get(settings["model"], RerankerModel.QWEN3_RERANKER_8B)

    return RerankerConfig(
        primary_model=model,
        max_length=settings["max_length"],
        use_fp16=settings["use_fp16"],
        batch_size=settings["batch_size"],
        stage1_top_k=100,  # Qwen3 can handle more with 8K context
        stage2_top_k=30,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

_reranker_instance: Optional[MultiStageReranker] = None


def get_reranker(config: Optional[RerankerConfig] = None) -> MultiStageReranker:
    """
    Get or create reranker singleton.

    If no config is provided, checks admin settings for Qwen3 reranker configuration.
    Falls back to default BGE-reranker-v2-m3 if Qwen3 is not enabled.
    """
    global _reranker_instance
    if _reranker_instance is None:
        # Check for Qwen3 config from admin settings if no explicit config
        if config is None:
            qwen3_config = get_qwen3_reranker_config()
            if qwen3_config is not None:
                logger.info(
                    "Using Qwen3 reranker from admin settings",
                    model=qwen3_config.primary_model.value,
                )
                config = qwen3_config
        _reranker_instance = MultiStageReranker(config)
    return _reranker_instance


def reset_reranker() -> None:
    """Reset the reranker singleton to reload settings."""
    global _reranker_instance
    _reranker_instance = None
    reset_qwen3_reranker_settings_cache()
    logger.info("Reranker singleton reset, will reload settings on next access")


async def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 10,
    config: Optional[RerankerConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query
        results: List of search results with 'content' field
        top_k: Number of results to return
        config: Optional reranker config

    Returns:
        Reranked results
    """
    reranker = get_reranker(config)
    response = await reranker.rerank(query, results, top_k)

    # Return in original format with scores
    return [
        {
            **results[r.original_rank],
            "rerank_score": r.score,
            "rerank_rank": r.new_rank,
            "stage_scores": r.stage_scores,
        }
        for r in response.results
    ]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "RerankerModel",
    "RerankerBackend",
    "RerankerConfig",
    "RerankResult",
    "RerankResponse",
    "BaseReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "ColBERTReranker",
    "LLMVerificationReranker",
    "BM25FastFilter",
    "MultiStageReranker",
    "get_reranker",
    "reset_reranker",
    "rerank_results",
    # Phase 68: Qwen3 Reranker
    "get_qwen3_reranker_config",
    "reset_qwen3_reranker_settings_cache",
]
