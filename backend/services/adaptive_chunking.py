"""
AIDocumentIndexer - Adaptive Chunking Service
==============================================

Moltbot/OpenClaw-inspired adaptive chunking with progressive fallback.

Features:
- Target ~400 tokens per chunk with 80-token overlap (configurable)
- Progressive fallback when chunks are too large
- Split long lines to keep embeddings under token limits
- Sentence-aware splitting for natural breaks
- Token counting with tiktoken

Usage:
    from backend.services.adaptive_chunking import AdaptiveChunker

    chunker = AdaptiveChunker()
    chunks = await chunker.chunk_adaptive(text)

    # Or with custom settings
    chunker = AdaptiveChunker(
        target_tokens=400,
        overlap_tokens=80,
        max_retries=3,
    )
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import structlog

logger = structlog.get_logger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token counting")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AdaptiveChunkingConfig:
    """Configuration for adaptive chunking."""
    # Token targets (OpenClaw defaults)
    target_tokens: int = 400
    overlap_tokens: int = 80
    max_tokens: int = 512  # Hard limit per chunk

    # Character estimates (fallback when tiktoken unavailable)
    chars_per_token: float = 4.0

    # Progressive fallback
    max_retries: int = 3
    reduction_factor: float = 0.8  # Reduce by 20% each retry

    # Line splitting
    max_line_tokens: int = 200  # Split lines longer than this
    line_split_pattern: str = r"[.!?;,]\s+"

    # Sentence detection
    sentence_pattern: str = r"(?<=[.!?])\s+"
    paragraph_pattern: str = r"\n\n+"

    # Model for tiktoken (cl100k_base for GPT-4/Claude)
    tiktoken_model: str = "cl100k_base"


@dataclass
class AdaptiveChunk:
    """A chunk from adaptive chunking."""
    content: str
    index: int
    tokens: int
    chars: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Chunk quality metrics
    is_complete_sentence: bool = True
    split_reason: Optional[str] = None  # Why this chunk boundary was chosen


# =============================================================================
# Token Counter
# =============================================================================

class TokenCounter:
    """Token counter with tiktoken support."""

    def __init__(self, model: str = "cl100k_base", chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
        self._encoder = None

        if TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(model)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoder: {e}")

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: approximate by character count
        return int(len(text) / self.chars_per_token)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if self._encoder:
            tokens = self._encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._encoder.decode(tokens[:max_tokens])
        # Fallback: truncate by estimated characters
        max_chars = int(max_tokens * self.chars_per_token)
        return text[:max_chars]


# =============================================================================
# Adaptive Chunker
# =============================================================================

class AdaptiveChunker:
    """
    Adaptive chunker with progressive fallback (Moltbot-inspired).

    Process:
    1. Start with target chunk size
    2. If chunk too large, reduce size progressively
    3. Split long lines to keep under token limits
    4. Preserve sentence boundaries where possible

    Example:
        chunker = AdaptiveChunker()

        # Basic usage
        chunks = chunker.chunk(text)

        # With metadata
        chunks = chunker.chunk(text, metadata={"source": "doc.pdf"})

        # Get statistics
        stats = chunker.get_stats(chunks)
    """

    def __init__(self, config: Optional[AdaptiveChunkingConfig] = None):
        self.config = config or AdaptiveChunkingConfig()
        self._token_counter = TokenCounter(
            model=self.config.tiktoken_model,
            chars_per_token=self.config.chars_per_token,
        )

        # Compile patterns
        self._sentence_re = re.compile(self.config.sentence_pattern)
        self._paragraph_re = re.compile(self.config.paragraph_pattern)
        self._line_split_re = re.compile(self.config.line_split_pattern)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self._token_counter.count(text)

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[AdaptiveChunk]:
        """
        Chunk text with adaptive sizing and progressive fallback.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of AdaptiveChunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Pre-process: split long lines
        text = self._split_long_lines(text)

        # Try chunking with progressively smaller targets
        target = self.config.target_tokens
        chunks = []

        for attempt in range(self.config.max_retries):
            try:
                chunks = self._chunk_with_target(text, target, metadata)

                # Verify all chunks are within limits
                oversized = [c for c in chunks if c.tokens > self.config.max_tokens * 1.5]
                if not oversized:
                    break

                # Reduce target and retry
                target = int(target * self.config.reduction_factor)
                logger.debug(
                    "Reducing chunk target",
                    attempt=attempt + 1,
                    new_target=target,
                    oversized_count=len(oversized),
                )

            except Exception as e:
                logger.warning(f"Chunking attempt {attempt + 1} failed: {e}")
                target = int(target * self.config.reduction_factor)

        # Final fallback: split by sentences
        if not chunks or any(c.tokens > self.config.max_tokens * 2 for c in chunks):
            logger.info("Using sentence-based fallback chunking")
            chunks = self._chunk_by_sentences(text, metadata)

        logger.info(
            "Adaptive chunking complete",
            num_chunks=len(chunks),
            avg_tokens=sum(c.tokens for c in chunks) // max(len(chunks), 1),
            target_tokens=self.config.target_tokens,
        )

        return chunks

    def _split_long_lines(self, text: str) -> str:
        """Split lines that exceed max_line_tokens."""
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            if self.count_tokens(line) <= self.config.max_line_tokens:
                result_lines.append(line)
                continue

            # Split long line at natural break points
            parts = self._line_split_re.split(line)
            current = ""

            for part in parts:
                test = current + part if current else part
                if self.count_tokens(test) <= self.config.max_line_tokens:
                    current = test
                else:
                    if current:
                        result_lines.append(current.strip())
                    current = part

            if current:
                result_lines.append(current.strip())

        return "\n".join(result_lines)

    def _chunk_with_target(
        self,
        text: str,
        target_tokens: int,
        metadata: Dict[str, Any],
    ) -> List[AdaptiveChunk]:
        """Chunk text aiming for target token count."""
        # Split into paragraphs first
        paragraphs = self._paragraph_re.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_content = ""
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds target, split it
            if para_tokens > target_tokens:
                # Flush current chunk if any
                if current_content:
                    chunks.append(self._create_chunk(
                        current_content,
                        chunk_index,
                        current_tokens,
                        metadata,
                        split_reason="paragraph_boundary",
                    ))
                    chunk_index += 1
                    current_content = ""
                    current_tokens = 0

                # Split large paragraph by sentences
                para_chunks = self._split_paragraph(para, target_tokens, chunk_index, metadata)
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
                continue

            # Check if adding this paragraph exceeds target
            test_content = f"{current_content}\n\n{para}" if current_content else para
            test_tokens = self.count_tokens(test_content)

            if test_tokens <= target_tokens + self.config.overlap_tokens:
                current_content = test_content
                current_tokens = test_tokens
            else:
                # Save current chunk and start new one
                if current_content:
                    chunks.append(self._create_chunk(
                        current_content,
                        chunk_index,
                        current_tokens,
                        metadata,
                        split_reason="token_limit",
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_content)
                current_content = f"{overlap_text}\n\n{para}" if overlap_text else para
                current_tokens = self.count_tokens(current_content)

        # Don't forget the last chunk
        if current_content:
            chunks.append(self._create_chunk(
                current_content,
                chunk_index,
                current_tokens,
                metadata,
                split_reason="end_of_document",
            ))

        return chunks

    def _split_paragraph(
        self,
        paragraph: str,
        target_tokens: int,
        start_index: int,
        metadata: Dict[str, Any],
    ) -> List[AdaptiveChunk]:
        """Split a large paragraph into smaller chunks by sentences."""
        sentences = self._sentence_re.split(paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_content = ""
        current_tokens = 0
        chunk_index = start_index

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds target, we have to include it anyway
            if sentence_tokens > target_tokens and not current_content:
                chunks.append(self._create_chunk(
                    sentence,
                    chunk_index,
                    sentence_tokens,
                    metadata,
                    split_reason="oversized_sentence",
                    is_complete=True,
                ))
                chunk_index += 1
                continue

            test_content = f"{current_content} {sentence}" if current_content else sentence
            test_tokens = self.count_tokens(test_content)

            if test_tokens <= target_tokens:
                current_content = test_content
                current_tokens = test_tokens
            else:
                # Save current and start new
                if current_content:
                    chunks.append(self._create_chunk(
                        current_content,
                        chunk_index,
                        current_tokens,
                        metadata,
                        split_reason="sentence_boundary",
                        is_complete=True,
                    ))
                    chunk_index += 1

                current_content = sentence
                current_tokens = sentence_tokens

        if current_content:
            chunks.append(self._create_chunk(
                current_content,
                chunk_index,
                current_tokens,
                metadata,
                split_reason="paragraph_end",
                is_complete=True,
            ))

        return chunks

    def _chunk_by_sentences(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[AdaptiveChunk]:
        """Fallback: chunk strictly by sentences."""
        sentences = self._sentence_re.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_content = ""
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            test_tokens = current_tokens + sentence_tokens + 1  # +1 for space

            if test_tokens <= self.config.target_tokens:
                current_content = f"{current_content} {sentence}" if current_content else sentence
                current_tokens = test_tokens
            else:
                if current_content:
                    chunks.append(self._create_chunk(
                        current_content,
                        chunk_index,
                        current_tokens,
                        metadata,
                        split_reason="fallback_sentence",
                        is_complete=True,
                    ))
                    chunk_index += 1

                current_content = sentence
                current_tokens = sentence_tokens

        if current_content:
            chunks.append(self._create_chunk(
                current_content,
                chunk_index,
                current_tokens,
                metadata,
                split_reason="fallback_end",
                is_complete=True,
            ))

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of previous chunk."""
        if not text or self.config.overlap_tokens <= 0:
            return ""

        # Get last few sentences for overlap
        sentences = self._sentence_re.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        overlap = ""
        for sentence in reversed(sentences):
            test = f"{sentence} {overlap}" if overlap else sentence
            if self.count_tokens(test) <= self.config.overlap_tokens:
                overlap = test
            else:
                break

        return overlap

    def _create_chunk(
        self,
        content: str,
        index: int,
        tokens: int,
        metadata: Dict[str, Any],
        split_reason: Optional[str] = None,
        is_complete: bool = True,
    ) -> AdaptiveChunk:
        """Create an AdaptiveChunk object."""
        return AdaptiveChunk(
            content=content.strip(),
            index=index,
            tokens=tokens,
            chars=len(content),
            metadata=metadata.copy(),
            is_complete_sentence=is_complete,
            split_reason=split_reason,
        )

    def get_stats(self, chunks: List[AdaptiveChunk]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {
                "count": 0,
                "total_tokens": 0,
                "total_chars": 0,
            }

        tokens = [c.tokens for c in chunks]
        chars = [c.chars for c in chunks]

        return {
            "count": len(chunks),
            "total_tokens": sum(tokens),
            "total_chars": sum(chars),
            "avg_tokens": sum(tokens) // len(chunks),
            "avg_chars": sum(chars) // len(chunks),
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
            "complete_sentences": sum(1 for c in chunks if c.is_complete_sentence),
            "split_reasons": {
                reason: sum(1 for c in chunks if c.split_reason == reason)
                for reason in set(c.split_reason for c in chunks if c.split_reason)
            },
        }


# =============================================================================
# Integration with existing DocumentChunker
# =============================================================================

def chunk_with_adaptive_fallback(
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 80,
    max_retries: int = 3,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function for adaptive chunking with progressive fallback.

    Returns chunks in a format compatible with existing pipeline.

    Args:
        text: Text to chunk
        target_tokens: Target tokens per chunk
        overlap_tokens: Overlap between chunks
        max_retries: Max fallback attempts
        metadata: Optional metadata

    Returns:
        List of chunk dicts with 'content', 'tokens', 'index', 'metadata'
    """
    config = AdaptiveChunkingConfig(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        max_retries=max_retries,
    )

    chunker = AdaptiveChunker(config)
    chunks = chunker.chunk(text, metadata)

    return [
        {
            "content": c.content,
            "tokens": c.tokens,
            "chars": c.chars,
            "index": c.index,
            "metadata": c.metadata,
            "is_complete_sentence": c.is_complete_sentence,
            "split_reason": c.split_reason,
        }
        for c in chunks
    ]


# =============================================================================
# Singleton Instance
# =============================================================================

_chunker_instance: Optional[AdaptiveChunker] = None


def get_adaptive_chunker(
    config: Optional[AdaptiveChunkingConfig] = None,
) -> AdaptiveChunker:
    """Get or create adaptive chunker singleton."""
    global _chunker_instance

    if _chunker_instance is None or config is not None:
        _chunker_instance = AdaptiveChunker(config)

    return _chunker_instance
