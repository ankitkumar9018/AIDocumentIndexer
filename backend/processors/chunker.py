"""
AIDocumentIndexer - Document Chunker
====================================

Semantic text chunking for RAG using LangChain text splitters.
Supports multiple chunking strategies optimized for different document types.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
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
