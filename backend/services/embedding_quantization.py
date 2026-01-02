"""
AIDocumentIndexer - Embedding Quantization
===========================================

Embedding quantization for faster search at scale.

Techniques:
- Binary Quantization: 32x compression with 92-96% accuracy retention
- Scalar Quantization (int8): 4x compression with 99%+ accuracy
- Matryoshka Embeddings: Variable dimensions for speed/quality tradeoff

Research:
- Binary quantization achieves 32x compression with minimal quality loss
- Matryoshka embeddings (from MRL paper) allow dimension reduction at inference
- Two-stage retrieval: Binary for fast recall, full precision for reranking

Open-source components:
- All techniques work with any embedding model
- No external dependencies beyond numpy

Usage:
    quantizer = EmbeddingQuantizer()

    # Binary quantization
    binary = quantizer.quantize_binary(embedding)
    similarity = quantizer.hamming_similarity(binary1, binary2)

    # Matryoshka dimension reduction
    reduced = quantizer.reduce_dimensions(embedding, target_dim=512)
"""

import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

import structlog

logger = structlog.get_logger(__name__)

# Try numpy for efficient operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available, using pure Python (slower)")


class QuantizationType(str, Enum):
    """Available quantization types."""
    NONE = "none"           # Full float32 precision
    INT8 = "int8"           # Scalar quantization (4x compression)
    BINARY = "binary"       # Binary quantization (32x compression)


@dataclass
class QuantizationConfig:
    """Configuration for embedding quantization."""
    quantization_type: QuantizationType = QuantizationType.NONE
    matryoshka_dim: Optional[int] = None  # Target dimension for Matryoshka
    binary_threshold: float = 0.0  # Threshold for binary quantization
    int8_min: float = -1.0  # Min value for int8 scaling
    int8_max: float = 1.0   # Max value for int8 scaling


@dataclass
class QuantizedEmbedding:
    """Container for quantized embedding data."""
    data: Union[bytes, List[float], List[int]]
    quantization_type: QuantizationType
    original_dimensions: int
    compressed_dimensions: int
    compression_ratio: float
    metadata: Dict[str, Any]


class EmbeddingQuantizer:
    """
    Embedding quantization for storage and search efficiency.

    Supports:
    - Binary quantization (32x compression)
    - Scalar int8 quantization (4x compression)
    - Matryoshka dimension reduction

    Usage:
        quantizer = EmbeddingQuantizer(QuantizationConfig(
            quantization_type=QuantizationType.BINARY
        ))

        # Quantize single embedding
        quantized = quantizer.quantize(embedding)

        # Compute similarity between quantized embeddings
        similarity = quantizer.similarity(quantized1, quantized2)
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantizer with configuration.

        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()

        # Load settings from environment if not configured
        if self.config.quantization_type == QuantizationType.NONE:
            env_type = os.getenv("EMBEDDING_QUANTIZATION", "none").lower()
            if env_type in ["binary", "int8"]:
                self.config.quantization_type = QuantizationType(env_type)

        if self.config.matryoshka_dim is None:
            env_dim = os.getenv("MATRYOSHKA_DIMENSIONS")
            if env_dim:
                self.config.matryoshka_dim = int(env_dim)

        logger.info(
            "Initialized embedding quantizer",
            quantization_type=self.config.quantization_type.value,
            matryoshka_dim=self.config.matryoshka_dim,
        )

    # =========================================================================
    # Binary Quantization
    # =========================================================================

    def quantize_binary(self, embedding: List[float]) -> bytes:
        """
        Convert float32 embedding to binary (1 bit per dimension).

        Binary quantization:
        - Each dimension becomes 1 bit (positive = 1, negative = 0)
        - 32x compression (float32 → 1 bit)
        - 92-96% accuracy retention for retrieval

        Args:
            embedding: Float embedding vector

        Returns:
            Binary packed bytes
        """
        if HAS_NUMPY:
            arr = np.array(embedding, dtype=np.float32)
            # Apply threshold (default 0.0)
            binary = (arr > self.config.binary_threshold).astype(np.uint8)
            return np.packbits(binary).tobytes()
        else:
            # Pure Python fallback
            threshold = self.config.binary_threshold
            bits = [1 if v > threshold else 0 for v in embedding]

            # Pad to multiple of 8
            while len(bits) % 8 != 0:
                bits.append(0)

            # Pack into bytes
            packed = []
            for i in range(0, len(bits), 8):
                byte = 0
                for j in range(8):
                    if bits[i + j]:
                        byte |= (1 << (7 - j))
                packed.append(byte)

            return bytes(packed)

    def dequantize_binary(
        self,
        binary_data: bytes,
        original_dim: int,
    ) -> List[float]:
        """
        Convert binary embedding back to float (lossy).

        Args:
            binary_data: Binary packed bytes
            original_dim: Original embedding dimensions

        Returns:
            Reconstructed float embedding (+1.0 or -1.0 values)
        """
        if HAS_NUMPY:
            bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
            # Convert to +1/-1
            return (bits[:original_dim].astype(np.float32) * 2 - 1).tolist()
        else:
            # Pure Python fallback
            bits = []
            for byte in binary_data:
                for i in range(7, -1, -1):
                    bits.append((byte >> i) & 1)

            # Convert to +1/-1
            return [1.0 if b else -1.0 for b in bits[:original_dim]]

    def hamming_distance(self, binary1: bytes, binary2: bytes) -> int:
        """
        Compute Hamming distance between binary embeddings.

        Fast bitwise comparison using XOR and popcount.

        Args:
            binary1: First binary embedding
            binary2: Second binary embedding

        Returns:
            Number of differing bits
        """
        if len(binary1) != len(binary2):
            raise ValueError("Binary embeddings must have same length")

        if HAS_NUMPY:
            a = np.frombuffer(binary1, dtype=np.uint8)
            b = np.frombuffer(binary2, dtype=np.uint8)
            xor = np.bitwise_xor(a, b)
            return int(np.sum(np.unpackbits(xor)))
        else:
            # Pure Python fallback
            distance = 0
            for b1, b2 in zip(binary1, binary2):
                xor = b1 ^ b2
                distance += bin(xor).count('1')
            return distance

    def hamming_similarity(
        self,
        binary1: bytes,
        binary2: bytes,
        num_bits: Optional[int] = None,
    ) -> float:
        """
        Compute normalized similarity from Hamming distance.

        Args:
            binary1: First binary embedding
            binary2: Second binary embedding
            num_bits: Total number of bits (for normalization)

        Returns:
            Similarity score (0-1)
        """
        distance = self.hamming_distance(binary1, binary2)
        total_bits = num_bits or (len(binary1) * 8)

        # Convert distance to similarity
        # similarity = 1 - (distance / total_bits)
        return 1.0 - (distance / total_bits)

    # =========================================================================
    # Scalar (int8) Quantization
    # =========================================================================

    def quantize_int8(
        self,
        embedding: List[float],
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Tuple[bytes, float, float]:
        """
        Convert float32 embedding to int8 (4x compression).

        Scalar quantization:
        - Maps float values to int8 (-128 to 127)
        - 4x compression (float32 → int8)
        - 99%+ accuracy retention

        Args:
            embedding: Float embedding vector
            min_val: Minimum value for scaling (auto-detected if None)
            max_val: Maximum value for scaling (auto-detected if None)

        Returns:
            Tuple of (int8 bytes, min_val, max_val) for dequantization
        """
        if HAS_NUMPY:
            arr = np.array(embedding, dtype=np.float32)

            # Auto-detect range if not provided
            if min_val is None:
                min_val = float(arr.min())
            if max_val is None:
                max_val = float(arr.max())

            # Scale to 0-255, then shift to -128 to 127
            scale = 255.0 / (max_val - min_val) if max_val != min_val else 1.0
            scaled = ((arr - min_val) * scale).astype(np.uint8)

            return scaled.tobytes(), min_val, max_val
        else:
            # Pure Python fallback
            if min_val is None:
                min_val = min(embedding)
            if max_val is None:
                max_val = max(embedding)

            scale = 255.0 / (max_val - min_val) if max_val != min_val else 1.0

            int8_values = []
            for v in embedding:
                scaled = int((v - min_val) * scale)
                scaled = max(0, min(255, scaled))
                int8_values.append(scaled)

            return bytes(int8_values), min_val, max_val

    def dequantize_int8(
        self,
        int8_data: bytes,
        min_val: float,
        max_val: float,
    ) -> List[float]:
        """
        Convert int8 embedding back to float32.

        Args:
            int8_data: Int8 packed bytes
            min_val: Minimum value used in quantization
            max_val: Maximum value used in quantization

        Returns:
            Reconstructed float embedding
        """
        if HAS_NUMPY:
            arr = np.frombuffer(int8_data, dtype=np.uint8).astype(np.float32)
            scale = (max_val - min_val) / 255.0 if max_val != min_val else 0.0
            return (arr * scale + min_val).tolist()
        else:
            # Pure Python fallback
            scale = (max_val - min_val) / 255.0 if max_val != min_val else 0.0
            return [byte * scale + min_val for byte in int8_data]

    def int8_dot_product(
        self,
        int8_1: bytes,
        int8_2: bytes,
        min1: float,
        max1: float,
        min2: float,
        max2: float,
    ) -> float:
        """
        Compute approximate dot product between int8 embeddings.

        Args:
            int8_1, int8_2: Int8 embeddings
            min1, max1: Range of first embedding
            min2, max2: Range of second embedding

        Returns:
            Approximate dot product
        """
        if HAS_NUMPY:
            a = np.frombuffer(int8_1, dtype=np.uint8).astype(np.float32)
            b = np.frombuffer(int8_2, dtype=np.uint8).astype(np.float32)

            scale1 = (max1 - min1) / 255.0
            scale2 = (max2 - min2) / 255.0

            # Dequantize on-the-fly for dot product
            a_float = a * scale1 + min1
            b_float = b * scale2 + min2

            return float(np.dot(a_float, b_float))
        else:
            # Dequantize first
            a = self.dequantize_int8(int8_1, min1, max1)
            b = self.dequantize_int8(int8_2, min2, max2)
            return sum(x * y for x, y in zip(a, b))

    # =========================================================================
    # Matryoshka Embeddings (Dimension Reduction)
    # =========================================================================

    def reduce_dimensions(
        self,
        embedding: List[float],
        target_dim: Optional[int] = None,
    ) -> List[float]:
        """
        Reduce embedding dimensions using Matryoshka-style truncation.

        Matryoshka embeddings are trained so that the first N dimensions
        contain most of the semantic information. Simply truncating works
        well for models that support it:
        - text-embedding-3-small/large (OpenAI)
        - nomic-embed-text-v1.5
        - jina-embeddings-v3

        Args:
            embedding: Full embedding vector
            target_dim: Target dimensions (uses config if None)

        Returns:
            Truncated embedding
        """
        target = target_dim or self.config.matryoshka_dim

        if target is None or target >= len(embedding):
            return embedding

        # Simply truncate to first N dimensions
        reduced = embedding[:target]

        # Optionally renormalize (for cosine similarity)
        if HAS_NUMPY:
            arr = np.array(reduced, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
                return arr.tolist()
        else:
            norm = sum(x * x for x in reduced) ** 0.5
            if norm > 0:
                reduced = [x / norm for x in reduced]

        return reduced

    def expand_dimensions(
        self,
        embedding: List[float],
        target_dim: int,
    ) -> List[float]:
        """
        Expand embedding back to original dimensions (zero-padding).

        Note: This is lossy - the original high-dimension values are lost.
        Use for compatibility when mixing different dimension embeddings.

        Args:
            embedding: Reduced embedding
            target_dim: Target (original) dimensions

        Returns:
            Zero-padded embedding
        """
        if len(embedding) >= target_dim:
            return embedding[:target_dim]

        # Zero-pad
        return embedding + [0.0] * (target_dim - len(embedding))

    # =========================================================================
    # Unified Interface
    # =========================================================================

    def quantize(
        self,
        embedding: List[float],
        quantization_type: Optional[QuantizationType] = None,
    ) -> QuantizedEmbedding:
        """
        Quantize embedding using configured or specified method.

        Args:
            embedding: Float embedding vector
            quantization_type: Override quantization type

        Returns:
            QuantizedEmbedding container
        """
        q_type = quantization_type or self.config.quantization_type
        original_dim = len(embedding)

        # Apply Matryoshka dimension reduction first if configured
        if self.config.matryoshka_dim and self.config.matryoshka_dim < original_dim:
            embedding = self.reduce_dimensions(embedding)
            working_dim = len(embedding)
        else:
            working_dim = original_dim

        if q_type == QuantizationType.BINARY:
            data = self.quantize_binary(embedding)
            compressed_dim = len(data) * 8  # bits
            compression_ratio = (original_dim * 32) / (len(data) * 8)  # float32 bits / binary bits
            metadata = {"threshold": self.config.binary_threshold}

        elif q_type == QuantizationType.INT8:
            data, min_val, max_val = self.quantize_int8(embedding)
            compressed_dim = len(data)
            compression_ratio = (original_dim * 4) / len(data)  # float32 bytes / int8 bytes
            metadata = {"min_val": min_val, "max_val": max_val}

        else:
            # No quantization
            data = embedding
            compressed_dim = original_dim
            compression_ratio = 1.0
            metadata = {}

        return QuantizedEmbedding(
            data=data,
            quantization_type=q_type,
            original_dimensions=original_dim,
            compressed_dimensions=compressed_dim,
            compression_ratio=compression_ratio,
            metadata=metadata,
        )

    def quantize_batch(
        self,
        embeddings: List[List[float]],
        quantization_type: Optional[QuantizationType] = None,
    ) -> List[QuantizedEmbedding]:
        """
        Quantize multiple embeddings.

        Args:
            embeddings: List of embedding vectors
            quantization_type: Override quantization type

        Returns:
            List of QuantizedEmbedding containers
        """
        return [
            self.quantize(emb, quantization_type)
            for emb in embeddings
        ]

    def similarity(
        self,
        quantized1: QuantizedEmbedding,
        quantized2: QuantizedEmbedding,
    ) -> float:
        """
        Compute similarity between quantized embeddings.

        Uses appropriate distance metric based on quantization type.

        Args:
            quantized1: First quantized embedding
            quantized2: Second quantized embedding

        Returns:
            Similarity score (0-1)
        """
        if quantized1.quantization_type != quantized2.quantization_type:
            raise ValueError("Cannot compare embeddings with different quantization types")

        q_type = quantized1.quantization_type

        if q_type == QuantizationType.BINARY:
            return self.hamming_similarity(
                quantized1.data,
                quantized2.data,
                quantized1.original_dimensions,
            )

        elif q_type == QuantizationType.INT8:
            # Compute cosine similarity from int8
            dot = self.int8_dot_product(
                quantized1.data,
                quantized2.data,
                quantized1.metadata["min_val"],
                quantized1.metadata["max_val"],
                quantized2.metadata["min_val"],
                quantized2.metadata["max_val"],
            )
            # Normalize (approximate)
            norm1 = len(quantized1.data) ** 0.5
            norm2 = len(quantized2.data) ** 0.5
            return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

        else:
            # Full precision cosine similarity
            if HAS_NUMPY:
                a = np.array(quantized1.data, dtype=np.float32)
                b = np.array(quantized2.data, dtype=np.float32)
                dot = np.dot(a, b)
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                return float(dot / norm) if norm > 0 else 0.0
            else:
                a, b = quantized1.data, quantized2.data
                dot = sum(x * y for x, y in zip(a, b))
                norm1 = sum(x * x for x in a) ** 0.5
                norm2 = sum(x * x for x in b) ** 0.5
                return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0


# =============================================================================
# Two-Stage Retrieval with Quantization
# =============================================================================

class TwoStageQuantizedRetriever:
    """
    Two-stage retrieval using quantization for speed.

    Architecture:
    1. Stage 1: Binary search for fast candidate recall
    2. Stage 2: Full-precision reranking for accuracy

    Benefits:
    - 10-50x faster initial retrieval
    - 92-96% recall compared to full precision
    - Significant storage savings (32x with binary)
    """

    def __init__(
        self,
        config: Optional[QuantizationConfig] = None,
        stage1_candidates: int = 200,
        use_binary_stage1: bool = True,
    ):
        """
        Initialize two-stage quantized retriever.

        Args:
            config: Quantization configuration
            stage1_candidates: Candidates to retrieve in stage 1
            use_binary_stage1: Use binary quantization for stage 1
        """
        self.quantizer = EmbeddingQuantizer(config)
        self.stage1_candidates = stage1_candidates
        self.use_binary_stage1 = use_binary_stage1

        logger.info(
            "Initialized TwoStageQuantizedRetriever",
            stage1_candidates=stage1_candidates,
            use_binary_stage1=use_binary_stage1,
        )

    def prepare_index(
        self,
        embeddings: List[List[float]],
        ids: List[str],
    ) -> Dict[str, Any]:
        """
        Prepare quantized index for fast retrieval.

        Args:
            embeddings: Document embeddings
            ids: Document IDs

        Returns:
            Index data structure
        """
        if self.use_binary_stage1:
            quantized = [
                self.quantizer.quantize_binary(emb)
                for emb in embeddings
            ]
        else:
            quantized = [
                self.quantizer.quantize_int8(emb)
                for emb in embeddings
            ]

        return {
            "ids": ids,
            "quantized": quantized,
            "full_precision": embeddings,  # Keep for stage 2
            "quantization_type": "binary" if self.use_binary_stage1 else "int8",
        }

    async def search(
        self,
        query_embedding: List[float],
        index: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Two-stage search with quantization.

        Args:
            query_embedding: Query embedding
            index: Prepared index
            top_k: Number of final results

        Returns:
            List of (id, score) tuples
        """
        import time
        start_time = time.time()

        # Stage 1: Fast quantized search
        if self.use_binary_stage1:
            query_binary = self.quantizer.quantize_binary(query_embedding)

            # Compute Hamming distances
            distances = []
            for i, doc_binary in enumerate(index["quantized"]):
                dist = self.quantizer.hamming_distance(query_binary, doc_binary)
                distances.append((i, dist))

            # Sort by distance (ascending)
            distances.sort(key=lambda x: x[1])
            candidates = [idx for idx, _ in distances[:self.stage1_candidates]]
        else:
            # Int8 dot product
            query_int8, q_min, q_max = self.quantizer.quantize_int8(query_embedding)

            scores = []
            for i, (doc_int8, d_min, d_max) in enumerate(index["quantized"]):
                score = self.quantizer.int8_dot_product(
                    query_int8, doc_int8, q_min, q_max, d_min, d_max
                )
                scores.append((i, score))

            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [idx for idx, _ in scores[:self.stage1_candidates]]

        stage1_time = (time.time() - start_time) * 1000

        # Stage 2: Full precision reranking
        stage2_start = time.time()

        results = []
        for idx in candidates:
            doc_embedding = index["full_precision"][idx]

            # Compute cosine similarity
            if HAS_NUMPY:
                q = np.array(query_embedding, dtype=np.float32)
                d = np.array(doc_embedding, dtype=np.float32)
                score = float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d)))
            else:
                dot = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                norm_q = sum(a * a for a in query_embedding) ** 0.5
                norm_d = sum(a * a for a in doc_embedding) ** 0.5
                score = dot / (norm_q * norm_d) if norm_q * norm_d > 0 else 0.0

            results.append((index["ids"][idx], score))

        # Sort by score and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        stage2_time = (time.time() - stage2_start) * 1000
        total_time = (time.time() - start_time) * 1000

        logger.debug(
            "Two-stage quantized search complete",
            stage1_time_ms=round(stage1_time, 2),
            stage2_time_ms=round(stage2_time, 2),
            total_time_ms=round(total_time, 2),
            candidates=len(candidates),
            results=len(results),
        )

        return results


# =============================================================================
# Factory and Utilities
# =============================================================================

_default_quantizer: Optional[EmbeddingQuantizer] = None


def get_quantizer(config: Optional[QuantizationConfig] = None) -> EmbeddingQuantizer:
    """
    Get or create default embedding quantizer.

    Args:
        config: Optional configuration override

    Returns:
        EmbeddingQuantizer instance
    """
    global _default_quantizer

    if config is not None:
        return EmbeddingQuantizer(config)

    if _default_quantizer is None:
        _default_quantizer = EmbeddingQuantizer()

    return _default_quantizer


async def quantize_embeddings_async(
    embeddings: List[List[float]],
    quantization_type: QuantizationType = QuantizationType.BINARY,
) -> List[QuantizedEmbedding]:
    """
    Async batch quantization.

    Args:
        embeddings: List of embedding vectors
        quantization_type: Type of quantization

    Returns:
        List of quantized embeddings
    """
    loop = asyncio.get_event_loop()
    quantizer = get_quantizer()

    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor,
            lambda: quantizer.quantize_batch(embeddings, quantization_type)
        )


def get_storage_savings(
    num_embeddings: int,
    embedding_dim: int = 1536,
    quantization_type: QuantizationType = QuantizationType.BINARY,
) -> Dict[str, Any]:
    """
    Calculate storage savings from quantization.

    Args:
        num_embeddings: Number of embeddings
        embedding_dim: Embedding dimensions
        quantization_type: Quantization type

    Returns:
        Dictionary with storage statistics
    """
    float32_bytes = num_embeddings * embedding_dim * 4  # 4 bytes per float32

    if quantization_type == QuantizationType.BINARY:
        quantized_bytes = num_embeddings * (embedding_dim // 8)
        compression_ratio = 32.0
    elif quantization_type == QuantizationType.INT8:
        quantized_bytes = num_embeddings * embedding_dim
        compression_ratio = 4.0
    else:
        quantized_bytes = float32_bytes
        compression_ratio = 1.0

    return {
        "original_bytes": float32_bytes,
        "quantized_bytes": quantized_bytes,
        "savings_bytes": float32_bytes - quantized_bytes,
        "compression_ratio": compression_ratio,
        "original_mb": round(float32_bytes / (1024 * 1024), 2),
        "quantized_mb": round(quantized_bytes / (1024 * 1024), 2),
        "savings_mb": round((float32_bytes - quantized_bytes) / (1024 * 1024), 2),
    }
