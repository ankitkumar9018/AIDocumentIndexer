"""
AIDocumentIndexer - Text Preprocessor Service
==============================================

Preprocesses text before embedding to reduce token costs and improve quality.
All preprocessing steps are optional and configurable.

Features:
- Whitespace normalization
- Boilerplate removal (headers, footers, page numbers)
- Content deduplication (optional, more aggressive)
- Text cleaning and standardization
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    # Master toggle
    enabled: bool = True

    # Whitespace normalization (lightweight, recommended)
    normalize_whitespace: bool = True
    max_consecutive_newlines: int = 2

    # Boilerplate removal
    remove_boilerplate: bool = True
    boilerplate_patterns: List[str] = field(default_factory=lambda: [
        r"^Page\s+\d+\s*(of\s+\d+)?$",  # Page numbers
        r"^\d+\s*$",  # Standalone page numbers
        r"^(CONFIDENTIAL|DRAFT|INTERNAL USE ONLY)$",  # Common watermarks
        r"^(Copyright|©)\s+\d{4}.*$",  # Copyright notices
        r"^All rights reserved\.?$",
        r"^Printed in.*$",
        r"^www\.[^\s]+\s*$",  # URLs alone on a line
    ])

    # Content deduplication (more aggressive, off by default)
    deduplicate_content: bool = False
    dedup_similarity_threshold: float = 0.95  # Jaccard similarity
    min_dedup_length: int = 100  # Only dedup chunks >= this length

    # Text cleaning
    remove_excessive_punctuation: bool = True
    max_repeated_chars: int = 3  # Convert "!!!!" to "!!!"
    normalize_quotes: bool = True
    remove_control_chars: bool = True

    # Statistics tracking
    track_stats: bool = True


@dataclass
class PreprocessingResult:
    """Result of text preprocessing."""
    original_text: str
    processed_text: str
    original_length: int
    processed_length: int
    reduction_percent: float
    stats: Dict[str, Any] = field(default_factory=dict)


class TextPreprocessor:
    """
    Text preprocessing service for RAG optimization.

    Reduces token costs by cleaning and normalizing text before embedding.
    All operations preserve semantic meaning while removing noise.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or PreprocessingConfig()
        self._compiled_patterns: List[re.Pattern] = []

        if self.config.remove_boilerplate:
            self._compile_patterns()

        logger.info(
            "TextPreprocessor initialized",
            enabled=self.config.enabled,
            normalize_whitespace=self.config.normalize_whitespace,
            remove_boilerplate=self.config.remove_boilerplate,
            deduplicate_content=self.config.deduplicate_content,
        )

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.config.boilerplate_patterns
        ]

    def preprocess(self, text: str) -> PreprocessingResult:
        """
        Preprocess text with all configured operations.

        Args:
            text: Raw text to preprocess

        Returns:
            PreprocessingResult with processed text and statistics
        """
        if not self.config.enabled or not text:
            return PreprocessingResult(
                original_text=text,
                processed_text=text,
                original_length=len(text) if text else 0,
                processed_length=len(text) if text else 0,
                reduction_percent=0.0,
            )

        original_length = len(text)
        stats = {}
        processed = text

        # Step 1: Remove control characters
        if self.config.remove_control_chars:
            processed = self._remove_control_chars(processed)
            stats["control_chars_removed"] = original_length - len(processed)

        # Step 2: Normalize whitespace
        if self.config.normalize_whitespace:
            before = len(processed)
            processed = self._normalize_whitespace(processed)
            stats["whitespace_reduced"] = before - len(processed)

        # Step 3: Remove boilerplate
        if self.config.remove_boilerplate:
            before = len(processed)
            processed = self._remove_boilerplate(processed)
            stats["boilerplate_removed"] = before - len(processed)

        # Step 4: Clean excessive punctuation
        if self.config.remove_excessive_punctuation:
            processed = self._clean_punctuation(processed)

        # Step 5: Normalize quotes
        if self.config.normalize_quotes:
            processed = self._normalize_quotes(processed)

        # Final whitespace cleanup
        processed = processed.strip()

        processed_length = len(processed)
        reduction = ((original_length - processed_length) / original_length * 100) if original_length > 0 else 0

        if self.config.track_stats:
            stats["original_length"] = original_length
            stats["processed_length"] = processed_length
            stats["reduction_percent"] = round(reduction, 2)

        logger.debug(
            "Text preprocessed",
            original_length=original_length,
            processed_length=processed_length,
            reduction_percent=round(reduction, 2),
        )

        return PreprocessingResult(
            original_text=text,
            processed_text=processed,
            original_length=original_length,
            processed_length=processed_length,
            reduction_percent=reduction,
            stats=stats,
        )

    def preprocess_chunks(
        self,
        chunks: List[Dict[str, Any]],
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """
        Preprocess a list of chunks.

        Args:
            chunks: List of chunk dictionaries
            content_key: Key for text content in each chunk dict

        Returns:
            List of chunks with preprocessed content
        """
        if not self.config.enabled:
            return chunks

        processed_chunks = []
        total_reduction = 0

        for chunk in chunks:
            if content_key not in chunk:
                processed_chunks.append(chunk)
                continue

            result = self.preprocess(chunk[content_key])

            # Create new chunk with processed content
            new_chunk = chunk.copy()
            new_chunk[content_key] = result.processed_text

            # Add preprocessing metadata
            if self.config.track_stats:
                new_chunk["preprocessing_stats"] = {
                    "original_length": result.original_length,
                    "processed_length": result.processed_length,
                    "reduction_percent": result.reduction_percent,
                }

            processed_chunks.append(new_chunk)
            total_reduction += result.original_length - result.processed_length

        if self.config.deduplicate_content:
            before_dedup = len(processed_chunks)
            processed_chunks = self._deduplicate_chunks(processed_chunks, content_key)
            logger.debug(
                "Deduplicated chunks",
                before=before_dedup,
                after=len(processed_chunks),
            )

        logger.info(
            "Preprocessed chunks batch",
            chunk_count=len(chunks),
            total_chars_reduced=total_reduction,
        )

        return processed_chunks

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        # Keep \n, \t, \r but remove other control chars
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace tabs with spaces
        text = text.replace("\t", "    ")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace on each line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Collapse multiple spaces (but not newlines)
        text = re.sub(r" {2,}", " ", text)

        # Limit consecutive newlines
        max_newlines = self.config.max_consecutive_newlines
        pattern = r"\n{" + str(max_newlines + 1) + r",}"
        replacement = "\n" * max_newlines
        text = re.sub(pattern, replacement, text)

        return text

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate patterns."""
        lines = text.split("\n")
        filtered_lines = []

        for line in lines:
            stripped = line.strip()
            is_boilerplate = False

            for pattern in self._compiled_patterns:
                if pattern.match(stripped):
                    is_boilerplate = True
                    break

            if not is_boilerplate:
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _clean_punctuation(self, text: str) -> str:
        """Remove excessive repeated punctuation."""
        max_chars = self.config.max_repeated_chars

        # Handle common repeated punctuation
        for char in [".", "!", "?", "-", "*", "_", "="]:
            escaped = re.escape(char)
            pattern = f"({escaped}){{{max_chars + 1},}}"
            replacement = char * max_chars
            text = re.sub(pattern, replacement, text)

        return text

    def _normalize_quotes(self, text: str) -> str:
        """Normalize different quote styles to standard ASCII."""
        # Smart quotes to regular quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace("„", '"').replace("‟", '"')

        # Other unicode quotes
        text = text.replace("«", '"').replace("»", '"')
        text = text.replace("‹", "'").replace("›", "'")

        return text

    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        content_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate chunks using Jaccard similarity.

        Args:
            chunks: List of chunk dictionaries
            content_key: Key for text content

        Returns:
            Deduplicated list of chunks
        """
        if not chunks:
            return chunks

        seen_hashes: Set[str] = set()
        unique_chunks = []

        for chunk in chunks:
            content = chunk.get(content_key, "")

            # Skip small chunks
            if len(content) < self.config.min_dedup_length:
                unique_chunks.append(chunk)
                continue

            # Create content hash for exact dedup
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash in seen_hashes:
                continue

            # Check similarity with existing chunks
            is_duplicate = False
            content_words = set(content.lower().split())

            for existing in unique_chunks:
                existing_content = existing.get(content_key, "")
                if len(existing_content) < self.config.min_dedup_length:
                    continue

                existing_words = set(existing_content.lower().split())

                # Jaccard similarity
                intersection = len(content_words & existing_words)
                union = len(content_words | existing_words)

                if union > 0:
                    similarity = intersection / union
                    if similarity >= self.config.dedup_similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)

        return unique_chunks


# Singleton instance
_preprocessor: Optional[TextPreprocessor] = None


def get_text_preprocessor(config: Optional[PreprocessingConfig] = None) -> TextPreprocessor:
    """
    Get or create the text preprocessor singleton.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        TextPreprocessor singleton instance
    """
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor(config)
    return _preprocessor
