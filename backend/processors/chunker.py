"""
AIDocumentIndexer - Document Chunker
====================================

Semantic text chunking for RAG using LangChain text splitters.
Supports multiple chunking strategies optimized for different document types.

Features:
- Multiple chunking strategies (recursive, semantic, token, markdown, code, etc.)
- Adaptive chunking based on document type detection
- Hierarchical chunking for large documents (document → section → detail)
- All advanced features are optional via configuration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple
from enum import Enum
import hashlib
import re
import structlog

# LangChain text splitters (modern imports)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter,
    Language,
    RecursiveCharacterTextSplitter as CodeSplitter,
)

logger = structlog.get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"  # Default, works well for most text
    SEMANTIC = "semantic"    # Uses sentence boundaries
    TOKEN = "token"          # Fixed token count
    MARKDOWN = "markdown"    # Preserves markdown structure
    HTML = "html"            # Preserves HTML structure
    CODE = "code"            # Language-aware code splitting
    SLIDE = "slide"          # Preserve slide boundaries for presentations
    HIERARCHICAL = "hierarchical"  # Multi-level chunking for large docs


class DocumentType(str, Enum):
    """Document types for adaptive chunking."""
    CODE = "code"            # Source code files
    LEGAL = "legal"          # Legal documents, contracts
    TECHNICAL = "technical"  # Technical documentation
    ACADEMIC = "academic"    # Research papers, academic texts
    NARRATIVE = "narrative"  # Stories, articles, prose
    TABULAR = "tabular"      # Tables, spreadsheet-like content
    GENERAL = "general"      # Default for unknown types


@dataclass
class Chunk:
    """A single document chunk."""
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source tracking
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    section: Optional[str] = None

    # Chunk identification
    chunk_hash: str = ""
    char_count: int = 0
    word_count: int = 0

    # Hierarchical chunking support
    # chunk_level: 0 = detail (default), 1 = section summary, 2 = document summary
    chunk_level: int = 0
    is_summary: bool = False
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate hash and counts after initialization."""
        if not self.chunk_hash:
            self.chunk_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        if not self.char_count:
            self.char_count = len(self.content)
        if not self.word_count:
            self.word_count = len(self.content.split())


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Size parameters (in characters unless using token strategy)
    chunk_size: int = 1000  # ~250 tokens
    chunk_overlap: int = 200  # 20% overlap

    # Token-based parameters
    token_chunk_size: int = 512
    token_chunk_overlap: int = 50

    # Minimum chunk size (skip smaller chunks)
    min_chunk_size: int = 50

    # Preserve structure
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True

    # Code-specific
    code_language: Optional[str] = None

    # Metadata to include
    include_position_metadata: bool = True
    include_statistics: bool = True

    # Adaptive chunking (optional - OFF by default)
    enable_adaptive_chunking: bool = False
    # Type-specific chunk sizes (characters)
    adaptive_settings: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "code": {"chunk_size": 256, "chunk_overlap": 50, "strategy": "code"},
        "legal": {"chunk_size": 1024, "chunk_overlap": 200, "preserve_paragraphs": True},
        "technical": {"chunk_size": 512, "chunk_overlap": 100, "preserve_sections": True},
        "academic": {"chunk_size": 800, "chunk_overlap": 150, "preserve_paragraphs": True},
        "narrative": {"chunk_size": 1000, "chunk_overlap": 200},
        "tabular": {"chunk_size": 500, "chunk_overlap": 50},
        "general": {"chunk_size": 1000, "chunk_overlap": 200},
    })

    # Hierarchical chunking (optional - OFF by default)
    enable_hierarchical_chunking: bool = False
    hierarchical_threshold_chars: int = 100000  # ~25k tokens - threshold to enable
    hierarchical_levels: int = 3  # 2=document summary, 1=section summary, 0=detail
    sections_per_document: int = 10  # Target number of section summaries


@dataclass
class QueryAdaptiveSettings:
    """
    Settings for query-adaptive chunk retrieval.

    Different query intents benefit from different chunk sizes:
    - Factual queries: Short, precise chunks for specific answers
    - Conceptual queries: Longer chunks for context and explanation
    - Comparative queries: Medium chunks for balanced comparison

    This doesn't affect chunking at index time, but rather guides
    retrieval-time chunk filtering and context building.
    """
    # Chunk size targets by query intent (in characters)
    factual_chunk_size: int = 500        # Short, precise chunks
    conceptual_chunk_size: int = 1500    # Longer context chunks
    comparative_chunk_size: int = 1000   # Medium balanced chunks
    procedural_chunk_size: int = 1200    # Steps need context
    navigational_chunk_size: int = 500   # Specific targets
    exploratory_chunk_size: int = 1200   # Broad exploration
    aggregation_chunk_size: int = 800    # Numerical data extraction
    default_chunk_size: int = 1000       # Fallback

    # Whether to enable query-adaptive chunk selection
    enable_adaptive_retrieval: bool = True

    # Tolerance for chunk size matching (chunks within this % of target are accepted)
    size_tolerance_percent: float = 30.0


def get_optimal_chunk_size_for_intent(
    query_intent: str,
    settings: Optional[QueryAdaptiveSettings] = None,
) -> int:
    """
    Get the optimal chunk size based on query intent.

    Args:
        query_intent: Query intent from QueryClassifier (e.g., "factual", "conceptual")
        settings: Optional adaptive settings, uses defaults if not provided

    Returns:
        Optimal chunk size in characters for this intent
    """
    settings = settings or QueryAdaptiveSettings()

    intent_sizes = {
        "factual": settings.factual_chunk_size,
        "conceptual": settings.conceptual_chunk_size,
        "comparative": settings.comparative_chunk_size,
        "navigational": settings.navigational_chunk_size,
        "procedural": settings.procedural_chunk_size,
        "exploratory": settings.exploratory_chunk_size,
        "aggregation": settings.aggregation_chunk_size,
        "unknown": settings.default_chunk_size,
    }

    return intent_sizes.get(query_intent.lower(), settings.default_chunk_size)


def filter_chunks_by_intent(
    chunks: List[Chunk],
    query_intent: str,
    target_count: int = 10,
    settings: Optional[QueryAdaptiveSettings] = None,
) -> List[Chunk]:
    """
    Filter and select chunks based on query intent.

    Selects chunks that best match the optimal size for the query intent,
    while maintaining relevance order (assumes chunks are pre-sorted by relevance).

    Args:
        chunks: List of chunks sorted by relevance
        query_intent: Query intent from classifier
        target_count: Number of chunks to return
        settings: Optional adaptive settings

    Returns:
        Filtered list of chunks optimized for the query intent
    """
    if not chunks:
        return []

    settings = settings or QueryAdaptiveSettings()

    if not settings.enable_adaptive_retrieval:
        return chunks[:target_count]

    optimal_size = get_optimal_chunk_size_for_intent(query_intent, settings)
    tolerance = settings.size_tolerance_percent / 100.0

    min_size = int(optimal_size * (1 - tolerance))
    max_size = int(optimal_size * (1 + tolerance))

    # Score chunks by how well they match the optimal size
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        relevance_weight = 1.0 - (i / len(chunks))  # Higher relevance = higher weight

        # Size score: 1.0 if within tolerance, decreasing outside
        if min_size <= chunk.char_count <= max_size:
            size_score = 1.0
        else:
            distance = min(abs(chunk.char_count - min_size), abs(chunk.char_count - max_size))
            size_score = max(0.0, 1.0 - (distance / optimal_size))

        # Combined score (relevance is more important)
        combined_score = 0.7 * relevance_weight + 0.3 * size_score
        scored_chunks.append((chunk, combined_score))

    # Sort by combined score and take top results
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in scored_chunks[:target_count]]


class DocumentChunker:
    """
    Document chunker using LangChain text splitters.

    Supports multiple strategies optimized for different content types:
    - RECURSIVE: General-purpose, hierarchical splitting
    - SEMANTIC: Sentence-aware splitting for natural text
    - TOKEN: Fixed token count (best for embedding consistency)
    - MARKDOWN: Preserves markdown structure
    - HTML: Preserves HTML sections
    - CODE: Language-aware code splitting
    - SLIDE: Preserves presentation slide boundaries
    """

    # Separators for recursive splitting (priority order)
    DEFAULT_SEPARATORS = [
        "\n\n\n",     # Multiple blank lines (section breaks)
        "\n\n",       # Paragraph breaks
        "\n",         # Line breaks
        ". ",         # Sentence boundaries
        "! ",
        "? ",
        "; ",
        ", ",         # Clause boundaries
        " ",          # Word boundaries
        "",           # Character level (last resort)
    ]

    # Code language mappings for LangChain
    CODE_LANGUAGES = {
        "python": Language.PYTHON,
        "py": Language.PYTHON,
        "javascript": Language.JS,
        "js": Language.JS,
        "typescript": Language.TS,
        "ts": Language.TS,
        "java": Language.JAVA,
        "go": Language.GO,
        "rust": Language.RUST,
        "cpp": Language.CPP,
        "c++": Language.CPP,
        "c": Language.C,
        "csharp": Language.CSHARP,
        "cs": Language.CSHARP,
        "php": Language.PHP,
        "ruby": Language.RUBY,
        "rb": Language.RUBY,
        "swift": Language.SWIFT,
        "kotlin": Language.KOTLIN,
        "scala": Language.SCALA,
        "html": Language.HTML,
        "markdown": Language.MARKDOWN,
        "md": Language.MARKDOWN,
        "latex": Language.LATEX,
        "tex": Language.LATEX,
    }

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker with configuration."""
        self.config = config or ChunkingConfig()
        self._splitters: Dict[ChunkingStrategy, Any] = {}
        self._initialize_splitters()

    def _initialize_splitters(self):
        """Initialize LangChain text splitters based on config."""
        config = self.config

        # Recursive splitter (default)
        self._splitters[ChunkingStrategy.RECURSIVE] = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=self.DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        # Token-based splitter
        self._splitters[ChunkingStrategy.TOKEN] = TokenTextSplitter(
            chunk_size=config.token_chunk_size,
            chunk_overlap=config.token_chunk_overlap,
        )

        # Markdown splitter
        self._splitters[ChunkingStrategy.MARKDOWN] = MarkdownTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        # Semantic splitter (sentence-aware)
        semantic_separators = [
            "\n\n",       # Paragraph
            "\n",         # Line
            ". ",         # Sentence
            "! ",
            "? ",
            ".\n",
            "!\n",
            "?\n",
        ]
        self._splitters[ChunkingStrategy.SEMANTIC] = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=semantic_separators,
        )

    def chunk(
        self,
        text: str,
        strategy: Optional[ChunkingStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk text into smaller pieces for embedding.

        Args:
            text: Text content to chunk
            strategy: Chunking strategy (defaults to config strategy)
            metadata: Additional metadata to attach to chunks
            document_id: Document identifier for source tracking

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        strategy = strategy or self.config.strategy
        metadata = metadata or {}

        logger.info(
            "Chunking document",
            strategy=strategy.value,
            text_length=len(text),
            document_id=document_id,
        )

        # Get appropriate splitter
        if strategy == ChunkingStrategy.CODE:
            chunks = self._chunk_code(text, metadata)
        elif strategy == ChunkingStrategy.HTML:
            chunks = self._chunk_html(text, metadata)
        elif strategy == ChunkingStrategy.SLIDE:
            chunks = self._chunk_slides(text, metadata)
        else:
            splitter = self._splitters.get(strategy, self._splitters[ChunkingStrategy.RECURSIVE])
            raw_chunks = splitter.split_text(text)
            chunks = self._create_chunks(raw_chunks, metadata, document_id)

        # Filter out chunks that are too small
        chunks = [c for c in chunks if c.char_count >= self.config.min_chunk_size]

        logger.info(
            "Chunking complete",
            num_chunks=len(chunks),
            avg_chunk_size=sum(c.char_count for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def chunk_with_pages(
        self,
        pages: List[Dict[str, Any]],
        strategy: Optional[ChunkingStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk document with page information preserved.

        Args:
            pages: List of dicts with 'content' and 'page_number' keys
            strategy: Chunking strategy
            metadata: Additional metadata
            document_id: Document identifier

        Returns:
            List of Chunk objects with page_number set
        """
        all_chunks = []
        chunk_index = 0

        for page in pages:
            content = page.get("content", "")
            page_number = page.get("page_number", page.get("page", 1))
            page_metadata = {**metadata} if metadata else {}
            page_metadata["page_number"] = page_number

            # Chunk this page
            page_chunks = self.chunk(
                text=content,
                strategy=strategy,
                metadata=page_metadata,
                document_id=document_id,
            )

            # Update chunk indices and page numbers
            for chunk in page_chunks:
                chunk.chunk_index = chunk_index
                chunk.page_number = page_number
                chunk_index += 1

            all_chunks.extend(page_chunks)

        return all_chunks

    def chunk_with_slides(
        self,
        slides: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk presentation with slide boundaries preserved.

        Each slide becomes at least one chunk, with larger slides
        split further if needed.

        Args:
            slides: List of dicts with 'content' and 'slide_number' keys
            metadata: Additional metadata
            document_id: Document identifier

        Returns:
            List of Chunk objects with slide_number set
        """
        all_chunks = []
        chunk_index = 0

        for slide in slides:
            content = slide.get("content", "")
            slide_number = slide.get("slide_number", slide.get("slide", 1))
            title = slide.get("title", "")

            slide_metadata = {**(metadata or {})}
            slide_metadata["slide_number"] = slide_number
            if title:
                slide_metadata["slide_title"] = title

            # If slide content is small enough, keep as single chunk
            if len(content) <= self.config.chunk_size:
                chunk = Chunk(
                    content=content,
                    chunk_index=chunk_index,
                    metadata=slide_metadata,
                    document_id=document_id,
                    slide_number=slide_number,
                    section=title,
                )
                if chunk.char_count >= self.config.min_chunk_size:
                    all_chunks.append(chunk)
                    chunk_index += 1
            else:
                # Split larger slides
                slide_chunks = self.chunk(
                    text=content,
                    strategy=ChunkingStrategy.RECURSIVE,
                    metadata=slide_metadata,
                    document_id=document_id,
                )
                for chunk in slide_chunks:
                    chunk.chunk_index = chunk_index
                    chunk.slide_number = slide_number
                    chunk.section = title
                    chunk_index += 1
                all_chunks.extend(slide_chunks)

        return all_chunks

    def _create_chunks(
        self,
        raw_chunks: List[str],
        metadata: Dict[str, Any],
        document_id: Optional[str],
    ) -> List[Chunk]:
        """Create Chunk objects from raw text chunks."""
        chunks = []

        for i, content in enumerate(raw_chunks):
            content = content.strip()
            if not content:
                continue

            chunk_metadata = {**metadata}

            if self.config.include_position_metadata:
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(raw_chunks)

            chunk = Chunk(
                content=content,
                chunk_index=i,
                metadata=chunk_metadata,
                document_id=document_id,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_code(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk code with language-aware splitting."""
        language = self.config.code_language

        if language and language.lower() in self.CODE_LANGUAGES:
            lang_enum = self.CODE_LANGUAGES[language.lower()]
            splitter = CodeSplitter.from_language(
                language=lang_enum,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        else:
            # Fall back to generic code splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=[
                    "\nclass ",
                    "\ndef ",
                    "\n\ndef ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ],
            )

        raw_chunks = splitter.split_text(text)
        return self._create_chunks(raw_chunks, metadata, None)

    def _chunk_html(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk HTML preserving header structure."""
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]

        try:
            splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            docs = splitter.split_text(text)

            chunks = []
            for i, doc in enumerate(docs):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                chunk_metadata = {**metadata}

                # Add header hierarchy to metadata
                if hasattr(doc, 'metadata'):
                    chunk_metadata.update(doc.metadata)

                chunk = Chunk(
                    content=content,
                    chunk_index=i,
                    metadata=chunk_metadata,
                )

                # Further split if chunk is too large
                if len(content) > self.config.chunk_size:
                    sub_chunks = self.chunk(
                        content,
                        strategy=ChunkingStrategy.RECURSIVE,
                        metadata=chunk_metadata,
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.warning("HTML chunking failed, falling back to recursive", error=str(e))
            return self.chunk(text, strategy=ChunkingStrategy.RECURSIVE, metadata=metadata)

    def _chunk_slides(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Chunk text that represents slide content.
        Looks for slide markers like "Slide 1:", "---", etc.
        """
        # Try to detect slide boundaries
        slide_patterns = [
            r'(?:^|\n)(?:Slide|SLIDE)\s*\d+[:\.]',  # "Slide 1:"
            r'(?:^|\n)---+(?:\n|$)',                  # "---" separator
            r'(?:^|\n)#{1,3}\s+',                     # Markdown headers
            r'(?:^|\n)\[Slide\s*\d+\]',               # "[Slide 1]"
        ]

        # Try each pattern
        for pattern in slide_patterns:
            if re.search(pattern, text):
                parts = re.split(pattern, text)
                parts = [p.strip() for p in parts if p.strip()]

                if len(parts) > 1:
                    chunks = []
                    for i, part in enumerate(parts):
                        chunk_metadata = {**metadata, "slide_number": i + 1}

                        if len(part) <= self.config.chunk_size:
                            chunk = Chunk(
                                content=part,
                                chunk_index=i,
                                metadata=chunk_metadata,
                                slide_number=i + 1,
                            )
                            chunks.append(chunk)
                        else:
                            # Split larger sections
                            sub_chunks = self.chunk(
                                part,
                                strategy=ChunkingStrategy.RECURSIVE,
                                metadata=chunk_metadata,
                            )
                            for sc in sub_chunks:
                                sc.slide_number = i + 1
                            chunks.extend(sub_chunks)

                    return chunks

        # No slide markers found, fall back to recursive
        return self.chunk(text, strategy=ChunkingStrategy.RECURSIVE, metadata=metadata)

    def estimate_chunks(self, text: str) -> int:
        """Estimate number of chunks without actually chunking."""
        if not text:
            return 0

        text_len = len(text)
        effective_chunk_size = self.config.chunk_size - self.config.chunk_overlap

        if effective_chunk_size <= 0:
            effective_chunk_size = self.config.chunk_size

        estimated = (text_len // effective_chunk_size) + 1
        return max(1, estimated)

    @staticmethod
    def merge_chunks(chunks: List[Chunk], separator: str = "\n\n") -> str:
        """Merge chunks back into a single string."""
        return separator.join(chunk.content for chunk in chunks)

    def rechunk(
        self,
        chunks: List[Chunk],
        new_config: ChunkingConfig,
    ) -> List[Chunk]:
        """
        Re-chunk existing chunks with new configuration.
        Useful when changing chunk size or strategy.
        """
        # Merge back to text
        text = self.merge_chunks(chunks)

        # Update config and rechunk
        original_config = self.config
        self.config = new_config
        self._initialize_splitters()

        # Preserve document_id from first chunk
        document_id = chunks[0].document_id if chunks else None

        new_chunks = self.chunk(text, document_id=document_id)

        # Restore original config
        self.config = original_config
        self._initialize_splitters()

        return new_chunks

    # =========================================================================
    # Adaptive Chunking Methods
    # =========================================================================

    def detect_document_type(
        self,
        text: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> DocumentType:
        """
        Detect document type for adaptive chunking.

        Uses heuristics based on content analysis and file metadata.

        Args:
            text: Document text content
            filename: Optional filename for extension-based detection
            mime_type: Optional MIME type

        Returns:
            Detected DocumentType
        """
        if not text:
            return DocumentType.GENERAL

        text_lower = text.lower()
        text_sample = text[:5000]  # Analyze first 5000 chars

        # Check filename extension first
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in ('py', 'js', 'ts', 'java', 'go', 'rs', 'cpp', 'c', 'cs', 'rb', 'php'):
                return DocumentType.CODE
            if ext in ('md', 'markdown'):
                # Could be code docs or general markdown
                if re.search(r'```[\w]+', text_sample):
                    return DocumentType.CODE
                return DocumentType.TECHNICAL

        # Check for code patterns
        code_indicators = [
            r'^\s*(def|class|function|import|from|const|let|var|public|private)\s+',
            r'^\s*#include\s*<',
            r'^\s*package\s+[\w.]+;',
            r'^\s*using\s+[\w.]+;',
            r'{\s*\n.*}\s*$',
            r'\(\s*\)\s*{',
            r'=>\s*{',
        ]
        code_matches = sum(1 for p in code_indicators if re.search(p, text_sample, re.MULTILINE))
        if code_matches >= 2:
            return DocumentType.CODE

        # Check for legal document patterns
        legal_indicators = [
            r'\b(hereby|whereas|pursuant|notwithstanding|herein|thereof)\b',
            r'\b(agreement|contract|party|parties|terms|conditions)\b',
            r'\b(section|article|clause)\s+\d+',
            r'\b(shall|must|may not)\b',
            r'\b(plaintiff|defendant|court|jurisdiction)\b',
        ]
        legal_matches = sum(1 for p in legal_indicators if re.search(p, text_lower))
        if legal_matches >= 3:
            return DocumentType.LEGAL

        # Check for academic/research patterns
        academic_indicators = [
            r'\babstract\b',
            r'\bintroduction\b',
            r'\bmethodology\b',
            r'\bconclusion\b',
            r'\breferences\b',
            r'\b(et al\.|ibid\.)\b',
            r'\[\d+\]',  # Citation numbers
            r'\(\d{4}\)',  # Year citations
        ]
        academic_matches = sum(1 for p in academic_indicators if re.search(p, text_lower))
        if academic_matches >= 3:
            return DocumentType.ACADEMIC

        # Check for technical documentation patterns
        tech_indicators = [
            r'\b(api|endpoint|parameter|request|response)\b',
            r'\b(configuration|installation|setup)\b',
            r'```',  # Code blocks
            r'\b(example|usage|note|warning|tip)\b:?',
            r'^#+\s+',  # Markdown headers
        ]
        tech_matches = sum(1 for p in tech_indicators if re.search(p, text_lower, re.MULTILINE))
        if tech_matches >= 3:
            return DocumentType.TECHNICAL

        # Check for tabular content
        if text.count('|') > 10 or text.count('\t') > 20:
            return DocumentType.TABULAR

        # Check for narrative content (long paragraphs, less structure)
        paragraphs = text.split('\n\n')
        avg_para_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        if avg_para_length > 500 and len(paragraphs) > 5:
            return DocumentType.NARRATIVE

        return DocumentType.GENERAL

    def chunk_adaptive(
        self,
        text: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk text with automatic document type detection and adaptive settings.

        Uses different chunking strategies and sizes based on detected document type.

        Args:
            text: Text content to chunk
            filename: Optional filename for type detection
            mime_type: Optional MIME type
            metadata: Additional metadata
            document_id: Document identifier

        Returns:
            List of Chunk objects with type-optimized chunking
        """
        if not self.config.enable_adaptive_chunking:
            # Fall back to standard chunking
            return self.chunk(text, metadata=metadata, document_id=document_id)

        # Detect document type
        doc_type = self.detect_document_type(text, filename, mime_type)

        logger.info(
            "Adaptive chunking detected document type",
            document_type=doc_type.value,
            filename=filename,
        )

        # Get type-specific settings
        type_settings = self.config.adaptive_settings.get(doc_type.value, {})

        # Create type-optimized config
        adaptive_config = ChunkingConfig(
            chunk_size=type_settings.get("chunk_size", self.config.chunk_size),
            chunk_overlap=type_settings.get("chunk_overlap", self.config.chunk_overlap),
            strategy=ChunkingStrategy(type_settings.get("strategy", "recursive")),
            preserve_paragraphs=type_settings.get("preserve_paragraphs", self.config.preserve_paragraphs),
            code_language=type_settings.get("language") if doc_type == DocumentType.CODE else None,
            min_chunk_size=self.config.min_chunk_size,
            include_position_metadata=self.config.include_position_metadata,
            include_statistics=self.config.include_statistics,
        )

        # Save current config and use adaptive config
        original_config = self.config
        self.config = adaptive_config
        self._initialize_splitters()

        # Add document type to metadata
        chunk_metadata = {**(metadata or {}), "document_type": doc_type.value}

        # Perform chunking
        chunks = self.chunk(
            text,
            strategy=adaptive_config.strategy,
            metadata=chunk_metadata,
            document_id=document_id,
        )

        # Restore original config
        self.config = original_config
        self._initialize_splitters()

        return chunks

    # =========================================================================
    # Hierarchical Chunking Methods
    # =========================================================================

    def should_use_hierarchical(self, text: str) -> bool:
        """
        Check if document is large enough for hierarchical chunking.

        Args:
            text: Document text

        Returns:
            True if document exceeds threshold and hierarchical is enabled
        """
        if not self.config.enable_hierarchical_chunking:
            return False

        return len(text) >= self.config.hierarchical_threshold_chars

    def chunk_hierarchical(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Create hierarchical chunks for large documents.

        Creates a multi-level structure:
        - Level 2: Document summary (1 chunk)
        - Level 1: Section summaries (~10 chunks)
        - Level 0: Detailed chunks (many chunks)

        Note: This creates the structure for summaries but does NOT generate
        the summary content itself. Use DocumentSummarizer for that.

        Args:
            text: Full document text
            metadata: Additional metadata
            document_id: Document identifier

        Returns:
            List of Chunk objects with hierarchy metadata
        """
        if not self.should_use_hierarchical(text):
            # Document not large enough, use standard chunking
            return self.chunk(text, metadata=metadata, document_id=document_id)

        logger.info(
            "Creating hierarchical chunks",
            text_length=len(text),
            target_sections=self.config.sections_per_document,
        )

        all_chunks = []
        chunk_metadata = metadata or {}

        # Split document into sections
        sections = self._split_into_sections(text)

        # Create placeholder for document-level summary (Level 2)
        doc_summary_id = f"doc_summary_{document_id or hashlib.md5(text[:1000].encode()).hexdigest()[:8]}"
        doc_summary_chunk = Chunk(
            content="[DOCUMENT SUMMARY PLACEHOLDER - Use DocumentSummarizer to generate]",
            chunk_index=-2000,  # Negative index for document summary
            metadata={**chunk_metadata, "type": "document_summary"},
            document_id=document_id,
            chunk_level=2,
            is_summary=True,
            chunk_hash=doc_summary_id,
        )
        all_chunks.append(doc_summary_chunk)

        section_chunk_ids = []
        detail_chunk_index = 0

        for section_idx, section_text in enumerate(sections):
            if len(section_text.strip()) < self.config.min_chunk_size:
                continue

            # Create section summary placeholder (Level 1)
            section_id = f"section_{section_idx}_{hashlib.md5(section_text[:500].encode()).hexdigest()[:8]}"
            section_chunk = Chunk(
                content=f"[SECTION {section_idx + 1} SUMMARY PLACEHOLDER]",
                chunk_index=-1000 - section_idx,  # Negative index for section summaries
                metadata={**chunk_metadata, "type": "section_summary", "section_index": section_idx},
                document_id=document_id,
                chunk_level=1,
                is_summary=True,
                parent_chunk_id=doc_summary_id,
                chunk_hash=section_id,
            )
            all_chunks.append(section_chunk)
            section_chunk_ids.append(section_id)

            # Create detailed chunks for this section (Level 0)
            section_detailed_chunks = self.chunk(
                section_text,
                strategy=self.config.strategy,
                metadata={**chunk_metadata, "type": "detail", "section_index": section_idx},
                document_id=document_id,
            )

            # Update chunk metadata with hierarchy info
            child_ids = []
            for chunk in section_detailed_chunks:
                chunk.chunk_index = detail_chunk_index
                chunk.chunk_level = 0
                chunk.is_summary = False
                chunk.parent_chunk_id = section_id
                child_ids.append(chunk.chunk_hash)
                detail_chunk_index += 1

            # Update section chunk with child references
            section_chunk.child_chunk_ids = child_ids

            all_chunks.extend(section_detailed_chunks)

        # Update document summary with section references
        doc_summary_chunk.child_chunk_ids = section_chunk_ids

        logger.info(
            "Hierarchical chunking complete",
            total_chunks=len(all_chunks),
            sections=len(sections),
            detail_chunks=detail_chunk_index,
        )

        return all_chunks

    def _split_into_sections(self, text: str) -> List[str]:
        """
        Split document into logical sections for hierarchical chunking.

        Uses various heuristics to identify section boundaries.

        Args:
            text: Full document text

        Returns:
            List of section texts
        """
        target_sections = self.config.sections_per_document

        # Try to split by common section patterns
        section_patterns = [
            r'\n(?=(?:Chapter|Section|Part)\s+\d)',  # Chapter 1, Section 2
            r'\n(?=\d+\.\s+[A-Z])',  # Numbered headings: 1. Introduction
            r'\n{3,}',  # Multiple blank lines
            r'\n(?=[A-Z][A-Z\s]{10,})\n',  # ALL CAPS HEADINGS
            r'\n(?=#{1,3}\s+)',  # Markdown headers
        ]

        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend(p for p in parts if p.strip())
            if len(new_sections) > len(sections):
                sections = new_sections

        # If we got too many sections, merge small ones
        if len(sections) > target_sections * 2:
            merged_sections = []
            current_section = ""
            target_length = len(text) // target_sections

            for section in sections:
                current_section += section
                if len(current_section) >= target_length:
                    merged_sections.append(current_section)
                    current_section = ""

            if current_section:
                merged_sections.append(current_section)

            sections = merged_sections

        # If we still have too few sections, split by size
        if len(sections) < 2:
            char_per_section = len(text) // target_sections
            sections = [
                text[i:i + char_per_section]
                for i in range(0, len(text), char_per_section)
            ]

        return sections[:target_sections]


# Convenience functions
def quick_chunk(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """Quick chunking with default settings, returns raw strings."""
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk(text)
    return [c.content for c in chunks]


def chunk_for_embedding(
    text: str,
    max_tokens: int = 512,
) -> List[str]:
    """Chunk text optimized for embedding models (token-based)."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.TOKEN,
        token_chunk_size=max_tokens,
        token_chunk_overlap=50,
    )
    chunker = DocumentChunker(config)
    chunks = chunker.chunk(text)
    return [c.content for c in chunks]
