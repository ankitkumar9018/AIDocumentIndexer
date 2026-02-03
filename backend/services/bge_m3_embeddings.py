"""
AIDocumentIndexer - BGE-M3 Embeddings Service
===============================================

Multi-modal, multi-lingual embedding service using BGE-M3.

Features:
- Dense embeddings (1024 dimensions)
- Sparse embeddings (lexical/BM25-like)
- Multi-vector embeddings (ColBERT-style)
- Cross-lingual support (100+ languages)
- Unified retrieval pipeline
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import hashlib
import numpy as np

import structlog

logger = structlog.get_logger(__name__)

# Check if FlagEmbedding is available
try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    logger.info("FlagEmbedding not installed. Install with: pip install FlagEmbedding")


class EmbeddingType(str, Enum):
    """Types of embeddings from BGE-M3."""
    DENSE = "dense"  # Standard dense vector
    SPARSE = "sparse"  # Lexical sparse vector
    COLBERT = "colbert"  # Multi-vector (token-level)
    HYBRID = "hybrid"  # Combination of all


class RetrievalMode(str, Enum):
    """Retrieval modes for different use cases."""
    SEMANTIC = "semantic"  # Dense only - best for semantic similarity
    LEXICAL = "lexical"  # Sparse only - best for keyword matching
    MULTI_VECTOR = "multi_vector"  # ColBERT - best for fine-grained matching
    HYBRID_ALL = "hybrid_all"  # All three combined - best overall


@dataclass
class BGEM3Embedding:
    """Complete BGE-M3 embedding result."""
    text: str
    dense: np.ndarray  # Shape: (1024,)
    sparse: Dict[int, float]  # Token ID -> weight
    colbert: Optional[np.ndarray] = None  # Shape: (num_tokens, 1024)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_hash": hashlib.md5(self.text.encode()).hexdigest()[:16],
            "dense_dim": len(self.dense),
            "sparse_terms": len(self.sparse),
            "colbert_vectors": self.colbert.shape[0] if self.colbert is not None else 0,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Result of hybrid retrieval."""
    document_id: str
    text: str
    dense_score: float
    sparse_score: float
    colbert_score: float
    combined_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "scores": {
                "dense": round(self.dense_score, 4),
                "sparse": round(self.sparse_score, 4),
                "colbert": round(self.colbert_score, 4),
                "combined": round(self.combined_score, 4),
            },
            "metadata": self.metadata,
        }


class BGEM3EmbeddingsService:
    """
    BGE-M3 embedding service for multi-modal retrieval.

    BGE-M3 provides three types of embeddings in a single model:
    1. Dense: 1024-dim semantic embeddings
    2. Sparse: BM25-like lexical embeddings
    3. ColBERT: Token-level multi-vector embeddings

    This enables hybrid retrieval that combines the benefits of all approaches.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 8192,
        device: Optional[str] = None,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.2,
        colbert_weight: float = 0.4,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        # Retrieval weights (should sum to 1.0)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.colbert_weight = colbert_weight

        self._model = None
        # LRU cache with max 5000 entries to prevent unbounded memory growth
        self._embedding_cache: OrderedDict[str, BGEM3Embedding] = OrderedDict()
        self._cache_max_size = 5000

        if BGE_M3_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("BGE-M3 model not available - using fallback")

    def _initialize_model(self):
        """Initialize the BGE-M3 model."""
        try:
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            logger.info(f"BGE-M3 model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            self._model = None

    async def embed(
        self,
        texts: Union[str, List[str]],
        return_sparse: bool = True,
        return_colbert: bool = False,
        return_dense: bool = True,
    ) -> List[BGEM3Embedding]:
        """
        Generate BGE-M3 embeddings for texts.

        Args:
            texts: Single text or list of texts
            return_sparse: Include sparse embeddings
            return_colbert: Include ColBERT multi-vectors
            return_dense: Include dense embeddings

        Returns:
            List of BGEM3Embedding objects
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text, return_sparse, return_colbert)
            if cache_key in self._embedding_cache:
                # Move to end for LRU tracking
                self._embedding_cache.move_to_end(cache_key)
                results.append(self._embedding_cache[cache_key])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            if self._model is not None:
                new_embeddings = await self._generate_embeddings(
                    uncached_texts,
                    return_sparse=return_sparse,
                    return_colbert=return_colbert,
                    return_dense=return_dense,
                )
            else:
                # Fallback to random embeddings for testing
                new_embeddings = self._fallback_embeddings(
                    uncached_texts, return_sparse, return_colbert
                )

            # Update results and cache with LRU eviction
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                cache_key = self._cache_key(texts[idx], return_sparse, return_colbert)
                # Evict oldest if at capacity
                if len(self._embedding_cache) >= self._cache_max_size:
                    self._embedding_cache.popitem(last=False)
                self._embedding_cache[cache_key] = embedding

        return results

    async def _generate_embeddings(
        self,
        texts: List[str],
        return_sparse: bool,
        return_colbert: bool,
        return_dense: bool,
    ) -> List[BGEM3Embedding]:
        """Generate embeddings using the actual model."""
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                # Run model inference in thread pool to not block
                output = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._model.encode(
                        batch,
                        max_length=self.max_length,
                        return_dense=return_dense,
                        return_sparse=return_sparse,
                        return_colbert_vecs=return_colbert,
                    ),
                )

                # Parse output
                for j, text in enumerate(batch):
                    dense = output["dense_vecs"][j] if return_dense else np.zeros(1024)
                    sparse = (
                        dict(output["lexical_weights"][j])
                        if return_sparse and "lexical_weights" in output
                        else {}
                    )
                    colbert = (
                        output["colbert_vecs"][j]
                        if return_colbert and "colbert_vecs" in output
                        else None
                    )

                    embeddings.append(BGEM3Embedding(
                        text=text,
                        dense=dense,
                        sparse=sparse,
                        colbert=colbert,
                    ))

            except Exception as e:
                logger.error(f"BGE-M3 embedding failed: {e}")
                # Fallback for failed batch
                embeddings.extend(self._fallback_embeddings(batch, return_sparse, return_colbert))

        return embeddings

    def _fallback_embeddings(
        self,
        texts: List[str],
        return_sparse: bool,
        return_colbert: bool,
    ) -> List[BGEM3Embedding]:
        """Generate fallback embeddings when model is not available."""
        embeddings = []

        for text in texts:
            # Generate deterministic pseudo-random embedding based on text
            np.random.seed(hash(text) % (2**32))

            dense = np.random.randn(1024).astype(np.float32)
            dense = dense / np.linalg.norm(dense)  # Normalize

            sparse = {}
            if return_sparse:
                # Simple term frequency as sparse weights
                words = text.lower().split()
                for word in set(words):
                    word_hash = hash(word) % 100000
                    sparse[word_hash] = words.count(word) / len(words)

            colbert = None
            if return_colbert:
                num_tokens = min(len(text.split()), 512)
                colbert = np.random.randn(num_tokens, 1024).astype(np.float32)
                colbert = colbert / np.linalg.norm(colbert, axis=1, keepdims=True)

            embeddings.append(BGEM3Embedding(
                text=text,
                dense=dense,
                sparse=sparse,
                colbert=colbert,
                metadata={"fallback": True},
            ))

        return embeddings

    async def hybrid_search(
        self,
        query: str,
        document_embeddings: List[BGEM3Embedding],
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.HYBRID_ALL,
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search using multiple embedding types.

        Args:
            query: Query text
            document_embeddings: List of document embeddings
            top_k: Number of results to return
            mode: Retrieval mode

        Returns:
            List of RetrievalResult sorted by combined score
        """
        # Get query embedding
        query_embeddings = await self.embed(
            query,
            return_sparse=mode in [RetrievalMode.LEXICAL, RetrievalMode.HYBRID_ALL],
            return_colbert=mode in [RetrievalMode.MULTI_VECTOR, RetrievalMode.HYBRID_ALL],
        )
        query_emb = query_embeddings[0]

        n_docs = len(document_embeddings)

        # Batch compute dense similarities (10-50x faster than loop)
        dense_scores = np.zeros(n_docs, dtype=np.float32)
        if mode in [RetrievalMode.SEMANTIC, RetrievalMode.HYBRID_ALL]:
            # Stack all dense embeddings into matrix
            dense_matrix = np.array(
                [doc.dense for doc in document_embeddings],
                dtype=np.float32
            )
            # Normalize query
            query_dense = np.array(query_emb.dense, dtype=np.float32)
            query_norm = np.linalg.norm(query_dense)
            if query_norm > 0:
                query_dense = query_dense / query_norm
            # Normalize all docs at once
            doc_norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
            doc_norms = np.where(doc_norms == 0, 1, doc_norms)
            dense_matrix = dense_matrix / doc_norms
            # Single matrix-vector multiply
            dense_scores = np.dot(dense_matrix, query_dense)

        # Sparse and ColBERT still need per-doc computation (variable sizes)
        sparse_scores = np.zeros(n_docs, dtype=np.float32)
        colbert_scores = np.zeros(n_docs, dtype=np.float32)

        for i, doc_emb in enumerate(document_embeddings):
            if mode in [RetrievalMode.LEXICAL, RetrievalMode.HYBRID_ALL]:
                sparse_scores[i] = self._sparse_similarity(query_emb.sparse, doc_emb.sparse)

            if mode in [RetrievalMode.MULTI_VECTOR, RetrievalMode.HYBRID_ALL]:
                if query_emb.colbert is not None and doc_emb.colbert is not None:
                    colbert_scores[i] = self._colbert_similarity(query_emb.colbert, doc_emb.colbert)

        # Calculate combined scores (vectorized)
        if mode == RetrievalMode.SEMANTIC:
            combined_scores = dense_scores
        elif mode == RetrievalMode.LEXICAL:
            combined_scores = sparse_scores
        elif mode == RetrievalMode.MULTI_VECTOR:
            combined_scores = colbert_scores
        else:  # HYBRID_ALL
            combined_scores = (
                self.dense_weight * dense_scores +
                self.sparse_weight * sparse_scores +
                self.colbert_weight * colbert_scores
            )

        # Build results
        results = [
            RetrievalResult(
                document_id=str(i),
                text=document_embeddings[i].text,
                dense_score=float(dense_scores[i]),
                sparse_score=float(sparse_scores[i]),
                colbert_score=float(colbert_scores[i]),
                combined_score=float(combined_scores[i]),
                metadata=document_embeddings[i].metadata,
            )
            for i in range(n_docs)
        ]

        # Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)

        return results[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _sparse_similarity(
        self,
        sparse_a: Dict[int, float],
        sparse_b: Dict[int, float],
    ) -> float:
        """Calculate similarity between sparse vectors."""
        if not sparse_a or not sparse_b:
            return 0.0

        # Dot product of shared terms
        shared_keys = set(sparse_a.keys()) & set(sparse_b.keys())
        if not shared_keys:
            return 0.0

        dot_product = sum(sparse_a[k] * sparse_b[k] for k in shared_keys)

        # Normalize
        norm_a = np.sqrt(sum(v ** 2 for v in sparse_a.values()))
        norm_b = np.sqrt(sum(v ** 2 for v in sparse_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _colbert_similarity(
        self,
        colbert_a: np.ndarray,
        colbert_b: np.ndarray,
    ) -> float:
        """
        Calculate ColBERT-style MaxSim score.

        For each query token, find max similarity to any document token,
        then average across all query tokens.
        """
        if colbert_a is None or colbert_b is None:
            return 0.0
        if len(colbert_a) == 0 or len(colbert_b) == 0:
            return 0.0

        # Compute similarity matrix
        # Shape: (query_tokens, doc_tokens)
        sim_matrix = np.dot(colbert_a, colbert_b.T)

        # Normalize (assuming vectors are already normalized)
        norms_a = np.linalg.norm(colbert_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(colbert_b, axis=1, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            sim_matrix = sim_matrix / (norms_a @ norms_b.T)
            sim_matrix = np.nan_to_num(sim_matrix)

        # MaxSim: for each query token, max similarity to any doc token
        max_sims = np.max(sim_matrix, axis=1)

        # Average across query tokens
        return float(np.mean(max_sims))

    def _cache_key(self, text: str, sparse: bool, colbert: bool) -> str:
        """Generate cache key for embedding."""
        flags = f"s{int(sparse)}_c{int(colbert)}"
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{text_hash}_{flags}"

    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings."""
        return {
            "model": self.model_name,
            "model_loaded": self._model is not None,
            "cache_size": len(self._embedding_cache),
            "weights": {
                "dense": self.dense_weight,
                "sparse": self.sparse_weight,
                "colbert": self.colbert_weight,
            },
            "dimensions": {
                "dense": 1024,
                "sparse": "variable",
                "colbert": "1024 per token",
            },
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()


# Singleton instance
_bge_m3_service: Optional[BGEM3EmbeddingsService] = None


def get_bge_m3_service() -> BGEM3EmbeddingsService:
    """Get or create the BGE-M3 service singleton."""
    global _bge_m3_service
    if _bge_m3_service is None:
        _bge_m3_service = BGEM3EmbeddingsService()
    return _bge_m3_service
