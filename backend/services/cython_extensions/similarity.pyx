# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
"""
Cython-optimized Similarity Functions
=====================================

High-performance implementations of similarity computations.
Provides 10-100x speedup over pure Python for hot loops.

Compile with: python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free

# Type definitions
ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
ctypedef np.int32_t INT32
ctypedef np.uint8_t UINT8

# Initialize NumPy
np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_similarity_batch(
    np.ndarray[FLOAT32, ndim=1] query,
    np.ndarray[FLOAT32, ndim=2] corpus,
):
    """
    Compute cosine similarity between query and all corpus vectors.

    Cython-optimized implementation (10-50x faster than pure Python).

    Args:
        query: Query vector [D] (float32)
        corpus: Corpus matrix [N, D] (float32)

    Returns:
        Similarity scores [N] (float32)
    """
    cdef int n = corpus.shape[0]
    cdef int d = corpus.shape[1]
    cdef int i, j
    cdef FLOAT32 dot_product, query_norm, corpus_norm
    cdef np.ndarray[FLOAT32, ndim=1] result = np.zeros(n, dtype=np.float32)

    # Compute query norm
    query_norm = 0.0
    for j in range(d):
        query_norm += query[j] * query[j]
    query_norm = sqrt(query_norm)

    if query_norm == 0:
        return result

    # Compute similarities
    for i in range(n):
        dot_product = 0.0
        corpus_norm = 0.0

        for j in range(d):
            dot_product += query[j] * corpus[i, j]
            corpus_norm += corpus[i, j] * corpus[i, j]

        corpus_norm = sqrt(corpus_norm)

        if corpus_norm > 0:
            result[i] = dot_product / (query_norm * corpus_norm)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_similarity_matrix(
    np.ndarray[FLOAT32, ndim=2] matrix_a,
    np.ndarray[FLOAT32, ndim=2] matrix_b,
):
    """
    Compute pairwise cosine similarity between two matrices.

    Cython-optimized implementation.

    Args:
        matrix_a: First matrix [M, D] (float32)
        matrix_b: Second matrix [N, D] (float32)

    Returns:
        Similarity matrix [M, N] (float32)
    """
    cdef int m = matrix_a.shape[0]
    cdef int n = matrix_b.shape[0]
    cdef int d = matrix_a.shape[1]
    cdef int i, j, k
    cdef FLOAT32 dot_product, norm_a, norm_b
    cdef np.ndarray[FLOAT32, ndim=2] result = np.zeros((m, n), dtype=np.float32)

    # Precompute norms for matrix_a
    cdef np.ndarray[FLOAT32, ndim=1] norms_a = np.zeros(m, dtype=np.float32)
    for i in range(m):
        norm_a = 0.0
        for k in range(d):
            norm_a += matrix_a[i, k] * matrix_a[i, k]
        norms_a[i] = sqrt(norm_a) if norm_a > 0 else 1.0

    # Precompute norms for matrix_b
    cdef np.ndarray[FLOAT32, ndim=1] norms_b = np.zeros(n, dtype=np.float32)
    for j in range(n):
        norm_b = 0.0
        for k in range(d):
            norm_b += matrix_b[j, k] * matrix_b[j, k]
        norms_b[j] = sqrt(norm_b) if norm_b > 0 else 1.0

    # Compute pairwise similarities
    for i in range(m):
        for j in range(n):
            dot_product = 0.0
            for k in range(d):
                dot_product += matrix_a[i, k] * matrix_b[j, k]
            result[i, j] = dot_product / (norms_a[i] * norms_b[j])

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def mmr_selection(
    np.ndarray[FLOAT32, ndim=1] relevance_scores,
    np.ndarray[FLOAT32, ndim=2] similarity_matrix,
    int top_k,
    float lambda_param = 0.5,
):
    """
    Maximal Marginal Relevance selection.

    Cython-optimized implementation (20-100x faster for large result sets).

    Args:
        relevance_scores: Relevance scores [N] (float32)
        similarity_matrix: Pairwise similarity [N, N] (float32)
        top_k: Number of items to select
        lambda_param: Trade-off parameter (0=diversity, 1=relevance)

    Returns:
        List of selected indices in MMR order
    """
    cdef int n = len(relevance_scores)
    cdef int i, j, best_idx, num_selected
    cdef FLOAT32 mmr_score, best_mmr, max_sim, sim
    cdef float one_minus_lambda = 1.0 - lambda_param

    if n == 0:
        return []

    # Track selected items
    cdef np.ndarray[UINT8, ndim=1] selected_mask = np.zeros(n, dtype=np.uint8)
    cdef list selected = []

    for num_selected in range(min(top_k, n)):
        best_idx = -1
        best_mmr = -1e30  # Very negative

        for i in range(n):
            if selected_mask[i]:
                continue

            # Compute max similarity to selected items
            max_sim = 0.0
            for j in range(num_selected):
                sim = similarity_matrix[i, selected[j]]
                if sim > max_sim:
                    max_sim = sim

            # MMR score
            mmr_score = lambda_param * relevance_scores[i] - one_minus_lambda * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = 1

    return selected


@cython.boundscheck(False)
@cython.wraparound(False)
def hamming_distance_batch(
    np.ndarray[UINT8, ndim=1] query,
    np.ndarray[UINT8, ndim=2] corpus,
):
    """
    Compute Hamming distance between binary query and corpus.

    Cython-optimized implementation with popcount.

    Args:
        query: Binary query vector [D] (uint8, packed bits)
        corpus: Binary corpus matrix [N, D] (uint8, packed bits)

    Returns:
        Hamming distances [N] (int32)
    """
    cdef int n = corpus.shape[0]
    cdef int d = corpus.shape[1]
    cdef int i, j, byte_val, count
    cdef np.ndarray[INT32, ndim=1] distances = np.zeros(n, dtype=np.int32)

    # Popcount lookup table for bytes
    cdef int[256] popcount_table
    for i in range(256):
        count = 0
        byte_val = i
        while byte_val:
            count += byte_val & 1
            byte_val >>= 1
        popcount_table[i] = count

    # Compute distances using XOR + popcount
    for i in range(n):
        count = 0
        for j in range(d):
            byte_val = query[j] ^ corpus[i, j]
            count += popcount_table[byte_val]
        distances[i] = count

    return distances


@cython.boundscheck(False)
@cython.wraparound(False)
def weighted_mean_pooling(
    np.ndarray[FLOAT32, ndim=3] token_embeddings,
    np.ndarray[FLOAT32, ndim=2] attention_mask,
):
    """
    Compute weighted mean pooling of token embeddings.

    Cython-optimized implementation.

    Args:
        token_embeddings: Token embeddings [B, T, D] (float32)
        attention_mask: Attention mask [B, T] (float32)

    Returns:
        Pooled embeddings [B, D] (float32)
    """
    cdef int b = token_embeddings.shape[0]
    cdef int t = token_embeddings.shape[1]
    cdef int d = token_embeddings.shape[2]
    cdef int bi, ti, di
    cdef FLOAT32 mask_sum
    cdef np.ndarray[FLOAT32, ndim=2] result = np.zeros((b, d), dtype=np.float32)

    for bi in range(b):
        # Sum mask for normalization
        mask_sum = 0.0
        for ti in range(t):
            mask_sum += attention_mask[bi, ti]

        if mask_sum < 1e-9:
            mask_sum = 1e-9

        # Weighted sum of embeddings
        for di in range(d):
            for ti in range(t):
                result[bi, di] += token_embeddings[bi, ti, di] * attention_mask[bi, ti]
            result[bi, di] /= mask_sum

    return result


# Flag to indicate Cython is active
_using_fallback = False

def is_using_fallback():
    """Check if we're using Python fallbacks instead of Cython."""
    return _using_fallback
