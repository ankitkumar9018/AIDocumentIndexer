"""
AIDocumentIndexer - Semantic Chunker with Contextual Headers
=============================================================

Advanced semantic chunking that:
1. Detects natural document boundaries (sections, paragraphs)
2. Prepends contextual headers to each chunk for better retrieval
3. Maintains parent-child relationships between chunks

This improves RAG retrieval accuracy by providing context to each chunk.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import hashlib
import structlog

from backend.processors.chunker import Chunk, ChunkingConfig, ChunkingStrategy, DocumentChunker

logger = structlog.get_logger(__name__)


class ContextualChunkingMode(str, Enum):
    """Modes for contextual chunking."""
    NONE = "none"  # No contextual headers (original behavior)
    TITLE_ONLY = "title_only"  # Only prepend document title
    SECTION_HEADERS = "section_headers"  # Prepend section hierarchy
    FULL_CONTEXT = "full_context"  # Title + section + summary context


@dataclass
class DocumentSection:
    """A detected section in a document."""
    title: str
    content: str
    level: int  # 1 = top level, 2 = subsection, etc.
    start_pos: int
    end_pos: int
    parent_section: Optional[str] = None


@dataclass
class ContextualChunk(Chunk):
    """A chunk with contextual header prepended."""
    # Original content without context header
    original_content: str = ""
    # The contextual header that was prepended
    context_header: str = ""
    # Section path (e.g., "Introduction > Background")
    section_path: str = ""
    # Confidence that section detection was accurate
    section_confidence: float = 1.0


@dataclass
class SemanticChunkingConfig:
    """Configuration for semantic chunking with context."""
    # Base chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 50

    # Contextual header settings
    mode: ContextualChunkingMode = ContextualChunkingMode.SECTION_HEADERS
    max_context_length: int = 200  # Max chars for context header
    include_document_title: bool = True
    include_section_path: bool = True

    # Section detection settings
    detect_markdown_headers: bool = True
    detect_numbered_sections: bool = True
    detect_capitalized_headers: bool = True
    min_section_length: int = 100  # Min chars to be considered a section


class SemanticChunker:
    """
    Semantic chunker that adds contextual headers to chunks.

    This improves retrieval by ensuring each chunk contains enough
    context to understand its content without seeing the full document.

    Example output:
    - Original chunk: "The results show a 15% improvement..."
    - With context: "[Document: Q4 Report | Section: Results > Performance] The results show a 15% improvement..."
    """

    # Patterns for detecting section headers
    MARKDOWN_HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    NUMBERED_SECTION_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$', re.MULTILINE)
    CAPITALIZED_HEADER_PATTERN = re.compile(r'^([A-Z][A-Z\s]{3,50})$', re.MULTILINE)

    def __init__(self, config: Optional[SemanticChunkingConfig] = None):
        """Initialize semantic chunker."""
        self.config = config or SemanticChunkingConfig()

        # Create base chunker with matching settings
        base_config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            preserve_paragraphs=True,
            preserve_sentences=True,
        )
        self._base_chunker = DocumentChunker(base_config)

        logger.info(
            "Initialized semantic chunker",
            mode=self.config.mode.value,
            chunk_size=self.config.chunk_size,
        )

    def chunk_with_context(
        self,
        text: str,
        document_title: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ContextualChunk]:
        """
        Chunk text with contextual headers prepended.

        Args:
            text: Document text to chunk
            document_title: Title of the document (optional)
            document_id: Document identifier
            metadata: Additional metadata

        Returns:
            List of ContextualChunk objects with context headers
        """
        if not text or not text.strip():
            return []

        if self.config.mode == ContextualChunkingMode.NONE:
            # Use base chunker without context
            base_chunks = self._base_chunker.chunk(
                text,
                metadata=metadata,
                document_id=document_id
            )
            return [self._to_contextual_chunk(c, "", "") for c in base_chunks]

        logger.info(
            "Chunking with contextual headers",
            text_length=len(text),
            document_title=document_title,
            mode=self.config.mode.value,
        )

        # Detect document sections
        sections = self._detect_sections(text)

        # Create chunks with section context
        contextual_chunks = self._create_contextual_chunks(
            text=text,
            sections=sections,
            document_title=document_title,
            document_id=document_id,
            metadata=metadata,
        )

        logger.info(
            "Contextual chunking complete",
            num_chunks=len(contextual_chunks),
            sections_detected=len(sections),
        )

        return contextual_chunks

    def _detect_sections(self, text: str) -> List[DocumentSection]:
        """
        Detect section boundaries in the document.

        Returns a list of detected sections with their titles and content.
        """
        sections: List[DocumentSection] = []

        # Collect all potential section markers
        markers: List[Tuple[int, str, int]] = []  # (position, title, level)

        # Detect markdown headers
        if self.config.detect_markdown_headers:
            for match in self.MARKDOWN_HEADER_PATTERN.finditer(text):
                level = len(match.group(1))  # Number of # signs
                title = match.group(2).strip()
                markers.append((match.start(), title, level))

        # Detect numbered sections (1. Introduction, 2.1 Background)
        if self.config.detect_numbered_sections:
            for match in self.NUMBERED_SECTION_PATTERN.finditer(text):
                number = match.group(1)
                level = number.count('.') + 1
                title = f"{number} {match.group(2).strip()}"
                markers.append((match.start(), title, level))

        # Detect capitalized headers
        if self.config.detect_capitalized_headers:
            for match in self.CAPITALIZED_HEADER_PATTERN.finditer(text):
                title = match.group(1).strip()
                if len(title) > 3:  # Avoid short acronyms
                    markers.append((match.start(), title, 1))

        # Sort markers by position
        markers.sort(key=lambda x: x[0])

        # Remove duplicates (same position)
        seen_positions = set()
        unique_markers = []
        for pos, title, level in markers:
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_markers.append((pos, title, level))

        # Create sections from markers
        for i, (pos, title, level) in enumerate(unique_markers):
            # Determine section end
            if i < len(unique_markers) - 1:
                end_pos = unique_markers[i + 1][0]
            else:
                end_pos = len(text)

            content = text[pos:end_pos].strip()

            # Skip if section is too short
            if len(content) < self.config.min_section_length:
                continue

            # Find parent section
            parent = None
            for j in range(i - 1, -1, -1):
                if unique_markers[j][2] < level:
                    parent = unique_markers[j][1]
                    break

            section = DocumentSection(
                title=title,
                content=content,
                level=level,
                start_pos=pos,
                end_pos=end_pos,
                parent_section=parent,
            )
            sections.append(section)

        # If no sections detected, create a single section for entire document
        if not sections:
            sections.append(DocumentSection(
                title="",
                content=text,
                level=1,
                start_pos=0,
                end_pos=len(text),
            ))

        return sections

    def _create_contextual_chunks(
        self,
        text: str,
        sections: List[DocumentSection],
        document_title: Optional[str],
        document_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> List[ContextualChunk]:
        """Create chunks with contextual headers from detected sections."""
        all_chunks: List[ContextualChunk] = []
        chunk_index = 0

        for section in sections:
            # Build section path
            section_path = self._build_section_path(section)

            # Build context header
            context_header = self._build_context_header(
                document_title=document_title,
                section_path=section_path,
            )

            # Chunk this section
            section_metadata = {
                **(metadata or {}),
                "section_title": section.title,
                "section_level": section.level,
                "section_path": section_path,
            }

            base_chunks = self._base_chunker.chunk(
                section.content,
                metadata=section_metadata,
                document_id=document_id,
            )

            # Convert to contextual chunks
            for base_chunk in base_chunks:
                contextual_chunk = self._create_contextual_chunk(
                    base_chunk=base_chunk,
                    context_header=context_header,
                    section_path=section_path,
                    chunk_index=chunk_index,
                )
                all_chunks.append(contextual_chunk)
                chunk_index += 1

        return all_chunks

    def _build_section_path(self, section: DocumentSection) -> str:
        """Build hierarchical section path."""
        if section.parent_section and section.title:
            return f"{section.parent_section} > {section.title}"
        return section.title

    def _build_context_header(
        self,
        document_title: Optional[str],
        section_path: str,
    ) -> str:
        """Build the context header to prepend to chunks."""
        parts = []

        if self.config.include_document_title and document_title:
            parts.append(f"Document: {document_title}")

        if self.config.include_section_path and section_path:
            parts.append(f"Section: {section_path}")

        if not parts:
            return ""

        header = "[" + " | ".join(parts) + "] "

        # Truncate if too long
        if len(header) > self.config.max_context_length:
            header = header[:self.config.max_context_length - 3] + "...] "

        return header

    def _create_contextual_chunk(
        self,
        base_chunk: Chunk,
        context_header: str,
        section_path: str,
        chunk_index: int,
    ) -> ContextualChunk:
        """Create a ContextualChunk from a base chunk."""
        # Prepend context header to content
        content_with_context = context_header + base_chunk.content

        return ContextualChunk(
            content=content_with_context,
            chunk_index=chunk_index,
            metadata=base_chunk.metadata,
            document_id=base_chunk.document_id,
            page_number=base_chunk.page_number,
            slide_number=base_chunk.slide_number,
            section=base_chunk.section or section_path.split(" > ")[-1] if section_path else None,
            chunk_hash=hashlib.md5(content_with_context.encode()).hexdigest()[:12],
            char_count=len(content_with_context),
            word_count=len(content_with_context.split()),
            chunk_level=base_chunk.chunk_level,
            is_summary=base_chunk.is_summary,
            parent_chunk_id=base_chunk.parent_chunk_id,
            child_chunk_ids=base_chunk.child_chunk_ids,
            # Contextual-specific fields
            original_content=base_chunk.content,
            context_header=context_header,
            section_path=section_path,
            section_confidence=1.0,
        )

    def _to_contextual_chunk(
        self,
        chunk: Chunk,
        context_header: str,
        section_path: str
    ) -> ContextualChunk:
        """Convert a regular Chunk to ContextualChunk."""
        return ContextualChunk(
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            metadata=chunk.metadata,
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            slide_number=chunk.slide_number,
            section=chunk.section,
            chunk_hash=chunk.chunk_hash,
            char_count=chunk.char_count,
            word_count=chunk.word_count,
            chunk_level=chunk.chunk_level,
            is_summary=chunk.is_summary,
            parent_chunk_id=chunk.parent_chunk_id,
            child_chunk_ids=chunk.child_chunk_ids,
            original_content=chunk.content,
            context_header=context_header,
            section_path=section_path,
        )


# Convenience function
def chunk_with_context(
    text: str,
    document_title: Optional[str] = None,
    chunk_size: int = 1000,
    mode: ContextualChunkingMode = ContextualChunkingMode.SECTION_HEADERS,
) -> List[ContextualChunk]:
    """
    Quick function to chunk text with contextual headers.

    Args:
        text: Text to chunk
        document_title: Document title for context
        chunk_size: Target chunk size in characters
        mode: Contextual chunking mode

    Returns:
        List of ContextualChunk objects
    """
    config = SemanticChunkingConfig(
        chunk_size=chunk_size,
        mode=mode,
    )
    chunker = SemanticChunker(config)
    return chunker.chunk_with_context(text, document_title=document_title)
