"""
Pure Python Fallbacks for Cython Extensions
============================================

These functions provide identical functionality to the Cython versions
but run in pure Python. Used when:
1. Cython is not installed
2. Compilation fails at runtime
3. Running in environments without a C compiler

Performance: 10-50x slower than Cython, but still functional.
"""

import numpy as np
from typing import List, Tuple, Optional
import structlog

logger = structlog.get_logger(__name__)

# Flag to track if we're using fallbacks
_using_fallback = True


def cosine_similarity_batch(
    query: np.ndarray,
    corpus: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between query and all corpus vectors.

    Pure Python/NumPy implementation (fallback).

    Args:
        query: Query vector [D]
        corpus: Corpus matrix [N, D]

    Returns:
        Similarity scores [N]
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(corpus), dtype=np.float32)
    query_normalized = query / query_norm

    # Normalize corpus
    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus_norms = np.where(corpus_norms == 0, 1, corpus_norms)
    corpus_normalized = corpus / corpus_norms

    # Dot product
    return np.dot(corpus_normalized, query_normalized).astype(np.float32)


def cosine_similarity_matrix(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two matrices.

    Pure Python/NumPy implementation (fallback).

    Args:
        matrix_a: First matrix [M, D]
        matrix_b: Second matrix [N, D]

    Returns:
        Similarity matrix [M, N]
    """
    # Normalize rows
    norms_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    norms_a = np.where(norms_a == 0, 1, norms_a)
    a_normalized = matrix_a / norms_a

    norms_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
    norms_b = np.where(norms_b == 0, 1, norms_b)
    b_normalized = matrix_b / norms_b

    return np.dot(a_normalized, b_normalized.T).astype(np.float32)


def mmr_selection(
    relevance_scores: np.ndarray,
    similarity_matrix: np.ndarray,
    top_k: int,
    lambda_param: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance selection.

    Pure Python/NumPy implementation (fallback).

    Args:
        relevance_scores: Relevance scores [N]
        similarity_matrix: Pairwise similarity [N, N]
        top_k: Number of items to select
        lambda_param: Trade-off parameter (0=diversity, 1=relevance)

    Returns:
        List of selected indices in MMR order
    """
    n = len(relevance_scores)
    if n == 0:
        return []

    selected = []
    selected_mask = np.zeros(n, dtype=bool)

    for _ in range(min(top_k, n)):
        if len(selected) == 0:
            # First selection: pure relevance
            max_sim_to_selected = np.zeros(n)
        else:
            # Max similarity to any selected item
            selected_sims = similarity_matrix[:, selected]
            max_sim_to_selected = np.max(selected_sims, axis=1)

        # MMR score
        mmr_scores = lambda_param * relevance_scores - (1 - lambda_param) * max_sim_to_selected
        mmr_scores[selected_mask] = float('-inf')

        best_idx = int(np.argmax(mmr_scores))
        if mmr_scores[best_idx] == float('-inf'):
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True

    return selected


def hamming_distance_batch(
    query: np.ndarray,
    corpus: np.ndarray,
) -> np.ndarray:
    """
    Compute Hamming distance between binary query and corpus.

    Pure Python/NumPy implementation (fallback).

    Args:
        query: Binary query vector [D]
        corpus: Binary corpus matrix [N, D]

    Returns:
        Hamming distances [N]
    """
    # XOR and count differing bits
    xor_result = np.bitwise_xor(corpus, query)

    if xor_result.dtype == np.uint8:
        # Packed bits - unpack and sum
        unpacked = np.unpackbits(xor_result, axis=1)
        return np.sum(unpacked, axis=1, dtype=np.int32)
    else:
        # Unpacked - just sum differences
        return np.sum(xor_result != 0, axis=1, dtype=np.int32)


def jaccard_similarity_sets(
    set_a: set,
    set_b: set,
) -> float:
    """
    Compute Jaccard similarity between two sets.

    Pure Python implementation (fallback).

    Args:
        set_a: First set
        set_b: Second set

    Returns:
        Jaccard similarity (0-1)
    """
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def weighted_mean_pooling(
    token_embeddings: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted mean pooling of token embeddings.

    Pure Python/NumPy implementation (fallback).

    Args:
        token_embeddings: Token embeddings [B, T, D]
        attention_mask: Attention mask [B, T]

    Returns:
        Pooled embeddings [B, D]
    """
    # Expand mask to match embedding dimensions
    mask_expanded = np.expand_dims(attention_mask, -1)
    mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)

    # Masked sum
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(mask_expanded, axis=1), 1e-9, None)

    return (sum_embeddings / sum_mask).astype(np.float32)


def is_using_fallback() -> bool:
    """Check if we're using Python fallbacks instead of Cython."""
    return _using_fallback
