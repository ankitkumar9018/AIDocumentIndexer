"""
AIDocumentIndexer - Streaming with Citations (Phase 65)
======================================================

Real-time citation highlighting during LLM streaming responses.

Features:
- Token-by-token citation matching
- Source highlighting as content streams
- Confidence scores per citation
- Inline citation markers [1], [2], etc.
- Post-stream citation summary

Research:
- Attributed QA: Grounding LLM outputs in sources
- Citation Generation: Inline vs post-hoc citation

Usage:
    from backend.services.streaming_citations import (
        StreamingCitationMatcher,
        get_citation_matcher,
    )

    matcher = get_citation_matcher(sources)

    async for chunk in llm.stream(query, context):
        enriched = await matcher.process_token(chunk.token)
        yield enriched
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class CitationStyle(str, Enum):
    """Citation formatting styles."""
    NUMBERED = "numbered"  # [1], [2]
    SUPERSCRIPT = "superscript"  # ¹, ²
    INLINE = "inline"  # (Source: doc.pdf)
    NONE = "none"  # No inline markers, just tracking


@dataclass
class CitationConfig:
    """Configuration for streaming citations."""
    # Citation style
    style: CitationStyle = CitationStyle.NUMBERED

    # Matching settings
    min_match_length: int = 5  # Minimum chars to trigger citation
    similarity_threshold: float = 0.8  # For fuzzy matching
    exact_match_boost: float = 0.3  # Bonus for exact matches

    # Token buffering
    buffer_size: int = 50  # Tokens to buffer for phrase matching
    flush_on_sentence: bool = True  # Flush buffer at sentence boundaries

    # Confidence
    min_confidence: float = 0.5  # Minimum confidence to show citation
    show_confidence: bool = False  # Include confidence in output


@dataclass
class Citation:
    """A citation reference."""
    source_index: int
    source_id: str
    source_title: str
    matched_text: str
    start_position: int
    end_position: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedToken:
    """A token enriched with citation information."""
    token: str
    citations: List[Citation] = field(default_factory=list)
    is_cited: bool = False
    buffer_flushed: bool = False
    position: int = 0


@dataclass
class StreamingCitationResult:
    """Final result from streaming with citations."""
    text: str
    citations: List[Citation]
    citation_summary: Dict[int, Dict[str, Any]]  # source_index → details
    total_tokens: int
    cited_tokens: int
    coverage: float  # Percentage of text with citations


# =============================================================================
# Source Matcher
# =============================================================================

class SourceMatcher:
    """
    Matches streamed text against source documents.

    Uses multiple strategies:
    1. Exact phrase matching (highest confidence)
    2. N-gram overlap (medium confidence)
    3. Keyword matching (lower confidence)
    """

    def __init__(self, sources: List[Dict[str, Any]]):
        """
        Initialize with source documents.

        Args:
            sources: List of source dicts with 'content', 'id', 'title'
        """
        self._sources = sources
        self._source_texts: List[str] = []
        self._source_ngrams: List[set] = []
        self._source_keywords: List[set] = []

        self._preprocess_sources()

    def _preprocess_sources(self) -> None:
        """Preprocess sources for efficient matching."""
        for source in self._sources:
            content = source.get("content", "")
            self._source_texts.append(content.lower())

            # Extract n-grams (3-grams for phrase matching)
            words = content.lower().split()
            ngrams = set()
            for i in range(len(words) - 2):
                ngrams.add(" ".join(words[i:i + 3]))
            self._source_ngrams.append(ngrams)

            # Extract keywords (unique words > 3 chars)
            keywords = {w for w in words if len(w) > 3}
            self._source_keywords.append(keywords)

    def match(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """
        Find matching sources for text.

        Args:
            text: Text to match
            min_confidence: Minimum confidence threshold

        Returns:
            List of (source_index, confidence) tuples
        """
        text_lower = text.lower().strip()
        if len(text_lower) < 5:
            return []

        matches = []

        for i, source_text in enumerate(self._source_texts):
            confidence = 0.0

            # Exact phrase match
            if text_lower in source_text:
                confidence = 0.95
            else:
                # N-gram overlap
                text_words = text_lower.split()
                if len(text_words) >= 3:
                    text_ngrams = set()
                    for j in range(len(text_words) - 2):
                        text_ngrams.add(" ".join(text_words[j:j + 3]))

                    overlap = len(text_ngrams & self._source_ngrams[i])
                    if text_ngrams:
                        confidence = max(confidence, 0.7 * (overlap / len(text_ngrams)))

                # Keyword overlap
                text_keywords = {w for w in text_words if len(w) > 3}
                if text_keywords:
                    keyword_overlap = len(text_keywords & self._source_keywords[i])
                    keyword_confidence = 0.5 * (keyword_overlap / len(text_keywords))
                    confidence = max(confidence, keyword_confidence)

            if confidence >= min_confidence:
                matches.append((i, confidence))

        # Sort by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches


# =============================================================================
# Streaming Citation Matcher
# =============================================================================

class StreamingCitationMatcher:
    """
    Matches streamed LLM output against sources in real-time.

    Processes tokens as they arrive, buffers for phrase matching,
    and emits citations with confidence scores.
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        config: Optional[CitationConfig] = None,
    ):
        """
        Initialize citation matcher.

        Args:
            sources: List of source documents
            config: Citation configuration
        """
        self.config = config or CitationConfig()
        self._sources = sources
        self._matcher = SourceMatcher(sources)

        # State
        self._buffer: List[str] = []
        self._full_text: str = ""
        self._citations: List[Citation] = []
        self._token_count: int = 0
        self._cited_positions: set = set()

        # Build source index
        self._source_index = {
            i: {
                "id": s.get("chunk_id") or s.get("id", str(i)),
                "title": s.get("document_title") or s.get("title", f"Source {i + 1}"),
                "filename": s.get("document_filename"),
            }
            for i, s in enumerate(sources)
        }

    async def process_token(self, token: str) -> EnrichedToken:
        """
        Process a single token from the stream.

        Args:
            token: Incoming token

        Returns:
            EnrichedToken with citation information
        """
        self._token_count += 1
        position = len(self._full_text)
        self._full_text += token
        self._buffer.append(token)

        # Check if we should flush buffer
        flush = False
        if len(self._buffer) >= self.config.buffer_size:
            flush = True
        elif self.config.flush_on_sentence and self._is_sentence_end(token):
            flush = True

        citations = []
        if flush:
            citations = await self._process_buffer(position)

        return EnrichedToken(
            token=token,
            citations=citations,
            is_cited=len(citations) > 0,
            buffer_flushed=flush,
            position=position,
        )

    async def _process_buffer(self, end_position: int) -> List[Citation]:
        """Process buffered tokens for citations."""
        if not self._buffer:
            return []

        buffer_text = "".join(self._buffer)
        start_position = end_position - len(buffer_text)

        # Find matches
        matches = self._matcher.match(buffer_text, self.config.min_confidence)

        citations = []
        for source_idx, confidence in matches[:3]:  # Top 3 matches
            source_info = self._source_index.get(source_idx, {})

            citation = Citation(
                source_index=source_idx,
                source_id=source_info.get("id", str(source_idx)),
                source_title=source_info.get("title", f"Source {source_idx + 1}"),
                matched_text=buffer_text[:100],  # Truncate for storage
                start_position=start_position,
                end_position=end_position,
                confidence=confidence,
                metadata={"filename": source_info.get("filename")},
            )

            self._citations.append(citation)
            citations.append(citation)

            # Track cited positions
            for pos in range(start_position, end_position):
                self._cited_positions.add(pos)

        # Clear buffer
        self._buffer.clear()

        return citations

    def _is_sentence_end(self, token: str) -> bool:
        """Check if token ends a sentence."""
        return token.rstrip().endswith(('.', '!', '?', '\n'))

    async def finalize(self) -> StreamingCitationResult:
        """
        Finalize streaming and return citation summary.

        Call this after all tokens have been processed.

        Returns:
            StreamingCitationResult with all citations
        """
        # Process any remaining buffer
        if self._buffer:
            await self._process_buffer(len(self._full_text))

        # Build citation summary
        summary = {}
        for citation in self._citations:
            idx = citation.source_index
            if idx not in summary:
                summary[idx] = {
                    "source_id": citation.source_id,
                    "title": citation.source_title,
                    "count": 0,
                    "avg_confidence": 0.0,
                    "positions": [],
                }

            summary[idx]["count"] += 1
            summary[idx]["positions"].append({
                "start": citation.start_position,
                "end": citation.end_position,
            })

        # Calculate average confidence
        for idx in summary:
            related = [c for c in self._citations if c.source_index == idx]
            if related:
                summary[idx]["avg_confidence"] = sum(c.confidence for c in related) / len(related)

        # Calculate coverage
        coverage = len(self._cited_positions) / len(self._full_text) if self._full_text else 0.0

        return StreamingCitationResult(
            text=self._full_text,
            citations=self._citations,
            citation_summary=summary,
            total_tokens=self._token_count,
            cited_tokens=len([c for c in self._citations]),
            coverage=coverage,
        )

    def format_with_citations(self, style: Optional[CitationStyle] = None) -> str:
        """
        Format full text with inline citations.

        Args:
            style: Citation style (uses config default if None)

        Returns:
            Text with inline citation markers
        """
        style = style or self.config.style

        if style == CitationStyle.NONE:
            return self._full_text

        # Group citations by position
        position_citations: Dict[int, List[Citation]] = {}
        for citation in self._citations:
            pos = citation.end_position
            if pos not in position_citations:
                position_citations[pos] = []
            position_citations[pos].append(citation)

        # Insert markers (from end to preserve positions)
        text = self._full_text
        for pos in sorted(position_citations.keys(), reverse=True):
            citations = position_citations[pos]
            unique_indices = sorted(set(c.source_index for c in citations))

            if style == CitationStyle.NUMBERED:
                markers = "".join(f"[{idx + 1}]" for idx in unique_indices)
            elif style == CitationStyle.SUPERSCRIPT:
                superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
                markers = "".join(
                    "".join(superscripts[int(d)] for d in str(idx + 1))
                    for idx in unique_indices
                )
            else:  # INLINE
                titles = [self._source_index[idx]["title"][:20] for idx in unique_indices]
                markers = f" ({', '.join(titles)})"

            text = text[:pos] + markers + text[pos:]

        return text

    def get_citation_footer(self) -> str:
        """Generate citation footer with source list."""
        seen = set()
        lines = ["\n\n---\n**Sources:**"]

        for citation in self._citations:
            if citation.source_index in seen:
                continue
            seen.add(citation.source_index)

            info = self._source_index[citation.source_index]
            title = info.get("title", f"Source {citation.source_index + 1}")
            filename = info.get("filename", "")

            line = f"[{citation.source_index + 1}] {title}"
            if filename:
                line += f" ({filename})"

            lines.append(line)

        return "\n".join(lines)


# =============================================================================
# Stream Processing Helper
# =============================================================================

async def stream_with_citations(
    stream: AsyncGenerator[str, None],
    sources: List[Dict[str, Any]],
    config: Optional[CitationConfig] = None,
) -> AsyncGenerator[EnrichedToken, None]:
    """
    Wrap an LLM stream with citation matching.

    Args:
        stream: Async generator yielding tokens
        sources: Source documents
        config: Citation configuration

    Yields:
        EnrichedToken objects with citations
    """
    matcher = StreamingCitationMatcher(sources, config)

    async for token in stream:
        enriched = await matcher.process_token(token)
        yield enriched

    # Yield final summary as special token
    result = await matcher.finalize()
    yield EnrichedToken(
        token="",
        citations=[],
        is_cited=False,
        buffer_flushed=True,
        position=-1,  # Signals end of stream
    )


async def enrich_streaming_response(
    stream: AsyncGenerator[str, None],
    sources: List[Dict[str, Any]],
    include_footer: bool = True,
    citation_style: CitationStyle = CitationStyle.NUMBERED,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Enrich streaming response with citations.

    High-level helper that yields structured chunks.

    Args:
        stream: LLM token stream
        sources: Source documents
        include_footer: Add citation footer at end
        citation_style: How to format inline citations

    Yields:
        Dict with 'token', 'citations', 'is_complete'
    """
    config = CitationConfig(style=citation_style)
    matcher = StreamingCitationMatcher(sources, config)

    async for token in stream:
        enriched = await matcher.process_token(token)

        yield {
            "token": token,
            "citations": [
                {
                    "source_index": c.source_index,
                    "source_title": c.source_title,
                    "confidence": c.confidence,
                }
                for c in enriched.citations
            ],
            "is_cited": enriched.is_cited,
            "is_complete": False,
        }

    # Final chunk with summary
    result = await matcher.finalize()

    final_data = {
        "token": "",
        "citations": [],
        "is_cited": False,
        "is_complete": True,
        "summary": {
            "total_citations": len(result.citations),
            "coverage": result.coverage,
            "sources_used": list(result.citation_summary.keys()),
        },
    }

    if include_footer:
        final_data["footer"] = matcher.get_citation_footer()

    yield final_data


# =============================================================================
# Singleton
# =============================================================================

def get_citation_matcher(
    sources: List[Dict[str, Any]],
    config: Optional[CitationConfig] = None,
) -> StreamingCitationMatcher:
    """Create a new citation matcher for a set of sources."""
    return StreamingCitationMatcher(sources, config)
