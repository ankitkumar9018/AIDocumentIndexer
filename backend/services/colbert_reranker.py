"""
AIDocumentIndexer - ColBERT-Style Reranker
==========================================

Late interaction reranking for improved retrieval precision.

ColBERT (Contextualized Late Interaction over BERT) provides 10-20% better
precision than cross-encoders at similar speed by using MaxSim scoring
between query and document token embeddings.

Open-Source Models:
- colbert-ir/colbertv2.0 (MIT license)
- Can also use sentence-transformers models with token embeddings

The key insight: Instead of a single score from a cross-encoder, ColBERT
computes token-level similarities and aggregates them, preserving more
fine-grained semantic matching.

Settings-aware: Respects rag.colbert_reranking_enabled setting.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Check for model availability
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# Check for RAGatouille (dedicated ColBERT library)
try:
    from ragatouille import RAGPretrainedModel
    HAS_RAGATOUILLE = True
except ImportError:
    HAS_RAGATOUILLE = False
    RAGPretrainedModel = None

# Cached settings
_colbert_enabled: Optional[bool] = None
_colbert_model: Optional[str] = None


async def _get_colbert_settings() -> Tuple[bool, str]:
    """Get ColBERT reranking settings from database."""
    global _colbert_enabled, _colbert_model

    if _colbert_enabled is not None and _colbert_model is not None:
        return _colbert_enabled, _colbert_model

    try:
        from backend.services.settings import get_settings_service

        settings = get_settings_service()
        enabled = await settings.get_setting("rag.use_colbert_reranker")
        model = await settings.get_setting("rag.colbert_model")

        _colbert_enabled = enabled if enabled is not None else False
        _colbert_model = model if model else "colbert-ir/colbertv2.0"

        return _colbert_enabled, _colbert_model
    except Exception as e:
        logger.debug("Could not load ColBERT settings, using defaults", error=str(e))
        return False, "colbert-ir/colbertv2.0"


def invalidate_colbert_settings():
    """Invalidate cached settings (call after settings change)."""
    global _colbert_enabled, _colbert_model
    _colbert_enabled = None
    _colbert_model = None


@dataclass
class RerankedResult:
    """Result after reranking."""
    chunk_id: str
    document_id: str
    content: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any]


class ColBERTReranker:
    """
    ColBERT-style late interaction reranker.

    Provides more accurate relevance scoring than bi-encoders by computing
    token-level similarities between query and documents.

    Scoring: MaxSim(q, d) = Σ max_j sim(q_i, d_j) for each query token i
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        use_ragatouille: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize ColBERT reranker.

        Args:
            model_name: ColBERT model to use
            use_ragatouille: Whether to use RAGatouille library (recommended)
            device: Device for model (auto-detected if None)
        """
        self.model_name = model_name
        self.use_ragatouille = use_ragatouille and HAS_RAGATOUILLE
        self.device = device

        self._model = None
        self._initialized = False

        logger.info(
            "ColBERT reranker configured",
            model=model_name,
            use_ragatouille=self.use_ragatouille,
        )

    def _initialize_model(self) -> bool:
        """Lazy initialization of the model."""
        if self._initialized:
            return self._model is not None

        try:
            if self.use_ragatouille:
                # Use RAGatouille for native ColBERT support
                self._model = RAGPretrainedModel.from_pretrained(self.model_name)
                logger.info("Initialized ColBERT via RAGatouille", model=self.model_name)
            elif HAS_SENTENCE_TRANSFORMERS:
                # Fallback to sentence-transformers
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                logger.info("Initialized reranker via sentence-transformers", model=self.model_name)
            else:
                logger.warning("No reranking library available (install ragatouille or sentence-transformers)")
                self._model = None

            self._initialized = True
            return self._model is not None

        except Exception as e:
            logger.error("Failed to initialize ColBERT model", error=str(e))
            self._initialized = True
            self._model = None
            return False

    def _compute_maxsim_score(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> float:
        """
        Compute MaxSim score between query and document token embeddings.

        MaxSim: For each query token, find max similarity with any doc token,
        then sum all max similarities.

        Args:
            query_embeddings: [num_query_tokens, embedding_dim]
            doc_embeddings: [num_doc_tokens, embedding_dim]

        Returns:
            MaxSim score (higher is better)
        """
        # Compute similarity matrix [num_query_tokens, num_doc_tokens]
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)

        # For each query token, take max similarity with any doc token
        max_similarities = similarity_matrix.max(axis=1)

        # Sum all max similarities
        return float(max_similarities.sum())

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankedResult]:
        """
        Rerank documents using ColBERT MaxSim scoring.

        Args:
            query: Search query
            documents: List of documents with 'content', 'chunk_id', 'document_id', etc.
            top_k: Number of results to return after reranking

        Returns:
            List of RerankedResult sorted by rerank score
        """
        if not documents:
            return []

        if not self._initialize_model():
            # No model available, return documents with original ordering
            logger.warning("ColBERT model not available, skipping reranking")
            return [
                RerankedResult(
                    chunk_id=doc.get("chunk_id", ""),
                    document_id=doc.get("document_id", ""),
                    content=doc.get("content", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents[:top_k]
            ]

        try:
            if self.use_ragatouille:
                # Use RAGatouille's native reranking
                doc_contents = [doc.get("content", "") for doc in documents]
                scores = self._model.rerank(
                    query=query,
                    documents=doc_contents,
                    k=min(top_k, len(documents)),
                )

                # RAGatouille returns list of (content, score) tuples
                # Map back to original documents
                content_to_doc = {doc.get("content", ""): doc for doc in documents}
                results = []

                for content, score in scores:
                    if content in content_to_doc:
                        doc = content_to_doc[content]
                        results.append(RerankedResult(
                            chunk_id=doc.get("chunk_id", ""),
                            document_id=doc.get("document_id", ""),
                            content=content,
                            original_score=doc.get("score", 0.0),
                            rerank_score=float(score),
                            metadata={**doc.get("metadata", {}), "reranked": True},
                        ))

                return results

            else:
                # Sentence-transformers fallback with MaxSim
                # Encode query tokens
                query_embeddings = self._model.encode(
                    query,
                    output_value="token_embeddings",
                    convert_to_numpy=True,
                )

                scored_results = []
                for doc in documents:
                    content = doc.get("content", "")
                    if not content:
                        continue

                    # Encode document tokens
                    doc_embeddings = self._model.encode(
                        content,
                        output_value="token_embeddings",
                        convert_to_numpy=True,
                    )

                    # Compute MaxSim score
                    score = self._compute_maxsim_score(query_embeddings, doc_embeddings)

                    scored_results.append((doc, score))

                # Sort by score (descending)
                scored_results.sort(key=lambda x: x[1], reverse=True)

                return [
                    RerankedResult(
                        chunk_id=doc.get("chunk_id", ""),
                        document_id=doc.get("document_id", ""),
                        content=doc.get("content", ""),
                        original_score=doc.get("score", 0.0),
                        rerank_score=score,
                        metadata={**doc.get("metadata", {}), "reranked": True},
                    )
                    for doc, score in scored_results[:top_k]
                ]

        except Exception as e:
            logger.error("ColBERT reranking failed", error=str(e))
            # Return original documents on failure
            return [
                RerankedResult(
                    chunk_id=doc.get("chunk_id", ""),
                    document_id=doc.get("document_id", ""),
                    content=doc.get("content", ""),
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents[:top_k]
            ]


# Singleton instance
_colbert_reranker: Optional[ColBERTReranker] = None


def get_colbert_reranker() -> ColBERTReranker:
    """Get or create ColBERT reranker singleton."""
    global _colbert_reranker
    if _colbert_reranker is None:
        _colbert_reranker = ColBERTReranker()
    return _colbert_reranker


async def rerank_with_colbert(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[RerankedResult]:
    """
    Convenience function to rerank documents with ColBERT.

    Args:
        query: Search query
        documents: List of document dicts
        top_k: Number of results

    Returns:
        List of RerankedResult
    """
    # Check if enabled
    enabled, _ = await _get_colbert_settings()
    if not enabled:
        # Return as-is without reranking
        return [
            RerankedResult(
                chunk_id=doc.get("chunk_id", ""),
                document_id=doc.get("document_id", ""),
                content=doc.get("content", ""),
                original_score=doc.get("score", 0.0),
                rerank_score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
            )
            for doc in documents[:top_k]
        ]

    reranker = get_colbert_reranker()
    return await reranker.rerank(query, documents, top_k)
