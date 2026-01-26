"""
AIDocumentIndexer - LLMLingua-2 Compression Service
====================================================

Token-level prompt compression using XLM-RoBERTa classifier.

LLMLingua-2 (Microsoft Research, 2024) achieves:
- 3-6x compression efficiency
- 95-98% accuracy retention
- 3-6x faster than original LLMLingua
- Works with any LLM (black-box compatible)

Key insight: Instead of LLM-based compression (slow, expensive), use a
small encoder model to classify which tokens are "important" and keep only those.

How it works:
1. Train a token classifier on (prompt, compressed_prompt) pairs
2. At inference, classify each token as keep/discard
3. Concatenate kept tokens to form compressed prompt
4. The compressed prompt retains essential information

Advantages over LLM summarization:
- 10-100x faster (no LLM inference)
- Preserves exact wording (no paraphrasing errors)
- Deterministic (same input = same output)
- Works with any downstream LLM

Research:
- LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic
  Prompt Compression (Microsoft Research, 2024)
- Original LLMLingua: Prompt Compression with Selective Context (2023)
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Check for required packages
try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoModelForTokenClassification = None
    AutoTokenizer = None
    torch = None


# =============================================================================
# Configuration
# =============================================================================

class CompressionMode(str, Enum):
    """Compression aggressiveness modes."""
    LIGHT = "light"          # Keep ~70% of tokens (1.4x compression)
    MODERATE = "moderate"    # Keep ~50% of tokens (2x compression)
    AGGRESSIVE = "aggressive"  # Keep ~30% of tokens (3x compression)
    EXTREME = "extreme"      # Keep ~20% of tokens (5x compression)
    ADAPTIVE = "adaptive"    # Adjust based on content importance


class ContentType(str, Enum):
    """Content type for optimized compression."""
    GENERAL = "general"
    CODE = "code"
    ACADEMIC = "academic"
    CONVERSATION = "conversation"
    TECHNICAL = "technical"


@dataclass
class LLMLingua2Config:
    """Configuration for LLMLingua-2 compression."""
    # Model settings
    model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

    # Compression settings
    mode: CompressionMode = CompressionMode.MODERATE
    target_ratio: Optional[float] = None  # Override mode with specific ratio

    # Token handling
    preserve_special_tokens: bool = True  # Keep [CLS], [SEP], etc.
    preserve_entities: bool = True        # Attempt to keep named entities
    min_kept_tokens: int = 10             # Minimum tokens to keep

    # Quality settings
    importance_threshold: float = 0.5     # Threshold for token importance
    context_window: int = 512             # Max tokens per segment

    # Performance
    batch_size: int = 8
    use_gpu: bool = True


@dataclass
class CompressionResult:
    """Result from LLMLingua-2 compression."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time_ms: float
    tokens_kept: List[Tuple[str, float]]  # (token, importance_score)
    mode_used: str


@dataclass
class BatchCompressionResult:
    """Result from batch compression."""
    results: List[CompressionResult]
    total_original_tokens: int
    total_compressed_tokens: int
    avg_compression_ratio: float
    total_processing_time_ms: float


# =============================================================================
# LLMLingua-2 Compression Engine
# =============================================================================

class LLMLingua2Engine:
    """
    Production-grade LLMLingua-2 compression engine.

    Usage:
        engine = LLMLingua2Engine()
        await engine.initialize()

        # Compress single text
        result = engine.compress(text, mode=CompressionMode.MODERATE)
        print(f"Compressed {result.compression_ratio:.1f}x: {result.compressed_text}")

        # Compress batch
        results = engine.compress_batch(texts)

        # Compress for RAG context
        compressed_chunks = engine.compress_rag_context(query, chunks, target_tokens=2000)
    """

    # Mode to target ratio mapping
    MODE_RATIOS = {
        CompressionMode.LIGHT: 0.7,
        CompressionMode.MODERATE: 0.5,
        CompressionMode.AGGRESSIVE: 0.3,
        CompressionMode.EXTREME: 0.2,
    }

    def __init__(self, config: Optional[LLMLingua2Config] = None):
        self.config = config or LLMLingua2Config()
        self._model = None
        self._tokenizer = None
        self._device = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the compression model."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            if not HAS_TRANSFORMERS:
                raise ImportError(
                    "transformers and torch required for LLMLingua-2. "
                    "Install with: pip install transformers torch"
                )

            logger.info("Initializing LLMLingua-2 engine", model=self.config.model_name)

            loop = asyncio.get_event_loop()

            def _load_model():
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                model = AutoModelForTokenClassification.from_pretrained(
                    self.config.model_name
                )

                # Determine device
                if self.config.use_gpu and torch.cuda.is_available():
                    device = torch.device("cuda")
                elif self.config.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")

                model = model.to(device)
                model.eval()

                return tokenizer, model, device

            self._tokenizer, self._model, self._device = await loop.run_in_executor(
                None, _load_model
            )

            self._initialized = True
            logger.info(
                "LLMLingua-2 engine initialized",
                model=self.config.model_name,
                device=str(self._device),
            )

    def compress(
        self,
        text: str,
        mode: Optional[CompressionMode] = None,
        target_ratio: Optional[float] = None,
    ) -> CompressionResult:
        """
        Compress text using LLMLingua-2.

        Args:
            text: Text to compress
            mode: Compression mode (overrides config)
            target_ratio: Specific target ratio (overrides mode)

        Returns:
            CompressionResult with compressed text and metrics
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        start_time = time.time()

        # Determine target ratio
        effective_mode = mode or self.config.mode
        effective_ratio = target_ratio or self.config.target_ratio

        if effective_ratio is None:
            if effective_mode == CompressionMode.ADAPTIVE:
                effective_ratio = self._compute_adaptive_ratio(text)
            else:
                effective_ratio = self.MODE_RATIOS.get(effective_mode, 0.5)

        # Tokenize
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_window,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)
        offset_mapping = encoding.get("offset_mapping", [[]])[0]

        # Get token importance scores
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            # Logits shape: [1, seq_len, num_labels]
            # For binary classification: label 1 = keep, label 0 = discard
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            # Get probability of "keep" label (usually index 1)
            importance_scores = probs[:, 1].cpu().numpy()

        # Get tokens
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        original_tokens = len(tokens)

        # Determine how many tokens to keep
        target_tokens = max(
            int(original_tokens * effective_ratio),
            self.config.min_kept_tokens,
        )

        # Select tokens to keep based on importance
        kept_indices = self._select_tokens(
            tokens,
            importance_scores,
            target_tokens,
            offset_mapping,
        )

        # Build compressed text
        compressed_text = self._reconstruct_text(
            text, tokens, kept_indices, offset_mapping
        )

        # Calculate metrics
        compressed_tokens = len(kept_indices)
        compression_ratio = original_tokens / max(compressed_tokens, 1)
        processing_time = (time.time() - start_time) * 1000

        # Build tokens kept list
        tokens_kept = [
            (tokens[i], float(importance_scores[i]))
            for i in kept_indices
        ]

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            processing_time_ms=processing_time,
            tokens_kept=tokens_kept,
            mode_used=effective_mode.value if isinstance(effective_mode, CompressionMode) else str(effective_mode),
        )

    def _select_tokens(
        self,
        tokens: List[str],
        importance_scores: np.ndarray,
        target_tokens: int,
        offset_mapping: List[Tuple[int, int]],
    ) -> List[int]:
        """Select which tokens to keep based on importance."""
        # Always keep special tokens
        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

        candidates = []
        forced_keep = []

        for i, (token, score) in enumerate(zip(tokens, importance_scores)):
            if token in special_tokens and self.config.preserve_special_tokens:
                forced_keep.append(i)
            else:
                candidates.append((i, score))

        # Sort by importance (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top tokens up to target
        remaining_slots = target_tokens - len(forced_keep)
        selected_indices = forced_keep + [
            idx for idx, _ in candidates[:remaining_slots]
        ]

        # Sort by position to maintain order
        selected_indices.sort()

        return selected_indices

    def _reconstruct_text(
        self,
        original_text: str,
        tokens: List[str],
        kept_indices: List[int],
        offset_mapping: List[Tuple[int, int]],
    ) -> str:
        """Reconstruct text from kept tokens."""
        if not kept_indices:
            return ""

        # Try to use offset mapping for accurate reconstruction
        if offset_mapping and len(offset_mapping) == len(tokens):
            text_parts = []
            for i in kept_indices:
                if i < len(offset_mapping):
                    start, end = offset_mapping[i]
                    if isinstance(start, torch.Tensor):
                        start = start.item()
                    if isinstance(end, torch.Tensor):
                        end = end.item()
                    if start < end:
                        text_parts.append(original_text[start:end])

            if text_parts:
                # Join with space, handling subword tokens
                result = ""
                for part in text_parts:
                    if result and not result.endswith(" ") and not part.startswith(" "):
                        result += " "
                    result += part
                return result.strip()

        # Fallback: use token strings directly
        kept_tokens = [tokens[i] for i in kept_indices]

        # Handle subword tokens (marked with ## or Ġ)
        result_parts = []
        for token in kept_tokens:
            # Skip special tokens
            if token in {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}:
                continue

            # Handle subword markers
            if token.startswith("##"):
                if result_parts:
                    result_parts[-1] += token[2:]
                else:
                    result_parts.append(token[2:])
            elif token.startswith("Ġ"):
                result_parts.append(token[1:])
            else:
                result_parts.append(token)

        return " ".join(result_parts)

    def _compute_adaptive_ratio(self, text: str) -> float:
        """Compute adaptive compression ratio based on content."""
        # Heuristics for adaptive compression
        text_lower = text.lower()

        # Code content: compress less (important syntax)
        if any(kw in text_lower for kw in ["def ", "function", "class ", "import ", "const ", "let ", "var "]):
            return 0.6

        # Academic/technical: compress moderately
        if any(kw in text_lower for kw in ["according to", "research shows", "study", "hypothesis"]):
            return 0.5

        # Conversational: can compress more
        if any(kw in text_lower for kw in ["i think", "maybe", "probably", "basically", "actually"]):
            return 0.4

        # Default moderate compression
        return 0.5

    def compress_batch(
        self,
        texts: List[str],
        mode: Optional[CompressionMode] = None,
        target_ratio: Optional[float] = None,
    ) -> BatchCompressionResult:
        """
        Compress multiple texts.

        Args:
            texts: List of texts to compress
            mode: Compression mode
            target_ratio: Target compression ratio

        Returns:
            BatchCompressionResult with all compressed texts
        """
        if not texts:
            return BatchCompressionResult(
                results=[],
                total_original_tokens=0,
                total_compressed_tokens=0,
                avg_compression_ratio=1.0,
                total_processing_time_ms=0.0,
            )

        start_time = time.time()
        results = []

        for text in texts:
            result = self.compress(text, mode, target_ratio)
            results.append(result)

        total_original = sum(r.original_tokens for r in results)
        total_compressed = sum(r.compressed_tokens for r in results)
        total_time = (time.time() - start_time) * 1000

        return BatchCompressionResult(
            results=results,
            total_original_tokens=total_original,
            total_compressed_tokens=total_compressed,
            avg_compression_ratio=total_original / max(total_compressed, 1),
            total_processing_time_ms=total_time,
        )

    def compress_rag_context(
        self,
        query: str,
        chunks: List[str],
        target_tokens: int = 2000,
        preserve_query_terms: bool = True,
    ) -> List[str]:
        """
        Compress RAG context chunks while preserving query relevance.

        Args:
            query: User query (for preserving relevant terms)
            chunks: Retrieved chunks to compress
            target_tokens: Target total tokens
            preserve_query_terms: Boost importance of query terms

        Returns:
            List of compressed chunks
        """
        if not chunks:
            return []

        # Estimate current token count
        total_tokens = sum(len(c.split()) * 1.3 for c in chunks)  # Rough estimate

        if total_tokens <= target_tokens:
            # No compression needed
            return chunks

        # Calculate per-chunk target
        compression_ratio = target_tokens / total_tokens
        effective_ratio = max(compression_ratio, 0.2)  # Don't compress below 20%

        compressed = []
        for chunk in chunks:
            # If preserve_query_terms, boost importance of query-related content
            if preserve_query_terms:
                # Simple approach: compress less aggressively for query-matching chunks
                query_terms = set(query.lower().split())
                chunk_terms = set(chunk.lower().split())
                overlap = len(query_terms & chunk_terms)

                if overlap > 2:
                    # High relevance: compress less
                    chunk_ratio = min(effective_ratio * 1.5, 0.8)
                else:
                    chunk_ratio = effective_ratio
            else:
                chunk_ratio = effective_ratio

            result = self.compress(chunk, target_ratio=chunk_ratio)
            compressed.append(result.compressed_text)

        return compressed


# =============================================================================
# Async Wrapper
# =============================================================================

class AsyncLLMLingua2Engine:
    """
    Async wrapper for LLMLingua-2 engine.

    Usage:
        engine = AsyncLLMLingua2Engine()
        await engine.initialize()

        result = await engine.compress(text)
    """

    def __init__(self, config: Optional[LLMLingua2Config] = None):
        self._sync_engine = LLMLingua2Engine(config)

    async def initialize(self) -> None:
        """Initialize the engine."""
        await self._sync_engine.initialize()

    async def compress(
        self,
        text: str,
        mode: Optional[CompressionMode] = None,
        target_ratio: Optional[float] = None,
    ) -> CompressionResult:
        """Compress text asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.compress(text, mode, target_ratio),
        )

    async def compress_batch(
        self,
        texts: List[str],
        mode: Optional[CompressionMode] = None,
        target_ratio: Optional[float] = None,
    ) -> BatchCompressionResult:
        """Compress batch asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.compress_batch(texts, mode, target_ratio),
        )

    async def compress_rag_context(
        self,
        query: str,
        chunks: List[str],
        target_tokens: int = 2000,
    ) -> List[str]:
        """Compress RAG context asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_engine.compress_rag_context(query, chunks, target_tokens),
        )


# =============================================================================
# Singleton Management
# =============================================================================

_llmlingua_engine: Optional[AsyncLLMLingua2Engine] = None
_engine_lock = asyncio.Lock()


async def get_llmlingua_engine(
    config: Optional[LLMLingua2Config] = None,
) -> AsyncLLMLingua2Engine:
    """Get or create LLMLingua-2 engine singleton."""
    global _llmlingua_engine

    async with _engine_lock:
        if _llmlingua_engine is None:
            _llmlingua_engine = AsyncLLMLingua2Engine(config)
            await _llmlingua_engine.initialize()

        return _llmlingua_engine


async def compress_text(
    text: str,
    mode: CompressionMode = CompressionMode.MODERATE,
) -> CompressionResult:
    """
    Convenience function to compress text.

    Usage:
        from backend.services.llmlingua_compression import compress_text

        result = await compress_text(long_prompt, mode=CompressionMode.AGGRESSIVE)
        short_prompt = result.compressed_text
    """
    engine = await get_llmlingua_engine()
    return await engine.compress(text, mode)


async def compress_rag_chunks(
    query: str,
    chunks: List[str],
    target_tokens: int = 2000,
) -> List[str]:
    """
    Convenience function to compress RAG context.

    Usage:
        from backend.services.llmlingua_compression import compress_rag_chunks

        compressed = await compress_rag_chunks(query, retrieved_chunks, target_tokens=3000)
    """
    engine = await get_llmlingua_engine()
    return await engine.compress_rag_context(query, chunks, target_tokens)
