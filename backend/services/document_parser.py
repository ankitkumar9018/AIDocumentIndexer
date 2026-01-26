"""
AIDocumentIndexer - High-Performance Document Parser
=====================================================

Enterprise-grade document parsing using Docling for superior table extraction.

Features:
- 97.9% table extraction accuracy (Docling)
- 1.27s/page processing speed
- Structured document output (markdown, JSON)
- Multi-format support (PDF, DOCX, PPTX, images)
- Vision model fallback for scanned documents

Research & Benchmarks (2024-2025):
| Parser | Speed (CPU) | Table Accuracy | License |
|--------|-------------|----------------|---------|
| Docling | 1.27s/page | 97.9% | MIT |
| MinerU 2.5 | 2.12p/s | 90.67% | AGPL |
| Marker | 4.2s/page | 85% | GPL |
| Unstructured | 2.7s/page | 75% | Apache |

Usage:
    parser = DocumentParser()
    result = await parser.parse(file_path_or_bytes, filename="doc.pdf")
    print(result.markdown)  # Formatted markdown
    print(result.tables)    # Extracted tables as dicts
"""

import asyncio
import hashlib
import io
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Check for Docling availability
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    DocumentConverter = None
    PdfFormatOption = None
    InputFormat = None
    PdfPipelineOptions = None
    TableFormerMode = None
    TableStructureOptions = None


# =============================================================================
# Configuration
# =============================================================================

class ParserBackend(str, Enum):
    """Available parsing backends."""
    DOCLING = "docling"     # Best for tables (97.9% accuracy)
    MARKER = "marker"       # Fast markdown conversion
    PYMUPDF = "pymupdf"     # Lightweight, good default
    VISION = "vision"       # For scanned documents (Claude/GPT-4V)
    AUTO = "auto"           # Auto-select best backend


@dataclass
class ParserConfig:
    """Configuration for document parsing."""
    # Backend selection
    backend: ParserBackend = ParserBackend.AUTO

    # Table extraction settings
    extract_tables: bool = True
    table_structure_mode: str = "accurate"  # "fast" or "accurate"
    table_cell_matching: bool = True

    # OCR settings
    enable_ocr: bool = True
    ocr_language: str = "eng"
    force_ocr: bool = False  # Force OCR even if text is extractable

    # Output format
    output_format: str = "markdown"  # "markdown", "json", "text"
    include_images: bool = True
    include_metadata: bool = True

    # Performance settings
    max_pages: Optional[int] = None  # Limit pages for large docs
    parallel_pages: int = 4  # Pages to process in parallel

    # Vision model settings (for scanned docs)
    vision_model: str = "claude-3-5-sonnet-20241022"
    vision_provider: str = "anthropic"


@dataclass
class ParsedTable:
    """Extracted table from document."""
    __slots__ = ('page_number', 'table_index', 'headers', 'rows', 'markdown', 'confidence')

    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    markdown: str
    confidence: float


@dataclass
class ParsedPage:
    """Parsed content from a single page."""
    __slots__ = ('page_number', 'text', 'tables', 'images', 'metadata')

    page_number: int
    text: str
    tables: List[ParsedTable]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class ParsedDocument:
    """Complete parsed document."""
    filename: str
    text: str
    markdown: str
    pages: List[ParsedPage]
    tables: List[ParsedTable]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    page_count: int
    word_count: int
    parser_backend: str
    parse_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "filename": self.filename,
            "text": self.text,
            "markdown": self.markdown,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "table_count": len(self.tables),
            "image_count": len(self.images),
            "parser_backend": self.parser_backend,
            "parse_time_ms": self.parse_time_ms,
            "metadata": self.metadata,
            "tables": [
                {
                    "page": t.page_number,
                    "headers": t.headers,
                    "rows": t.rows,
                    "markdown": t.markdown,
                }
                for t in self.tables
            ],
        }


# =============================================================================
# Document Parser
# =============================================================================

class DocumentParser:
    """
    High-performance document parser using Docling.

    Provides 97.9% table extraction accuracy at 1.27s/page.

    Usage:
        parser = DocumentParser()

        # Parse from file path
        result = await parser.parse("/path/to/document.pdf")

        # Parse from bytes
        result = await parser.parse(pdf_bytes, filename="document.pdf")

        # Access results
        print(result.markdown)
        for table in result.tables:
            print(table.markdown)
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()
        self._converter: Optional[Any] = None
        self._initialized = False
        self._lock = asyncio.Lock()

        if not HAS_DOCLING:
            logger.warning(
                "Docling not installed - using fallback parser. "
                "Install with: pip install docling"
            )

    async def initialize(self) -> bool:
        """Initialize the document converter (lazy loading)."""
        if self._initialized:
            return True

        if not HAS_DOCLING:
            return False

        async with self._lock:
            if self._initialized:
                return True

            try:
                # Initialize Docling in thread pool
                loop = asyncio.get_running_loop()
                self._converter = await loop.run_in_executor(
                    None,
                    self._create_converter
                )
                self._initialized = True
                logger.info("Document parser initialized", backend="docling")
                return True

            except Exception as e:
                logger.error("Failed to initialize document parser", error=str(e))
                return False

    def _create_converter(self) -> Any:
        """Create Docling converter with optimal settings."""
        # Configure table extraction
        table_options = TableStructureOptions(
            do_cell_matching=self.config.table_cell_matching,
            mode=TableFormerMode.ACCURATE if self.config.table_structure_mode == "accurate" else TableFormerMode.FAST,
        )

        # Configure PDF pipeline
        pipeline_options = PdfPipelineOptions(
            do_ocr=self.config.enable_ocr,
            do_table_structure=self.config.extract_tables,
            table_structure_options=table_options,
        )

        # Create converter
        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.IMAGE,
                InputFormat.HTML,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        return converter

    def _select_backend(self, filename: str, file_size: int) -> ParserBackend:
        """Auto-select best parsing backend based on file characteristics."""
        if self.config.backend != ParserBackend.AUTO:
            return self.config.backend

        ext = Path(filename).suffix.lower()

        # For images, use vision model if available
        if ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.tiff'}:
            if settings.ENABLE_VISION_PROCESSING:
                return ParserBackend.VISION
            return ParserBackend.PYMUPDF

        # For PDFs, prefer Docling for table-heavy documents
        if ext == '.pdf':
            if HAS_DOCLING:
                return ParserBackend.DOCLING
            return ParserBackend.PYMUPDF

        # For Office documents, Docling handles them well
        if ext in {'.docx', '.pptx', '.xlsx'}:
            if HAS_DOCLING:
                return ParserBackend.DOCLING
            return ParserBackend.PYMUPDF

        return ParserBackend.PYMUPDF

    async def parse(
        self,
        source: Union[str, bytes, Path],
        filename: Optional[str] = None,
    ) -> ParsedDocument:
        """
        Parse a document from file path or bytes.

        Args:
            source: File path, Path object, or file bytes
            filename: Required if source is bytes

        Returns:
            ParsedDocument with text, tables, and metadata
        """
        import time
        start_time = time.time()

        # Handle different input types
        if isinstance(source, bytes):
            if not filename:
                raise ValueError("filename required when parsing bytes")
            file_bytes = source
            file_size = len(source)
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {source}")
            filename = filename or path.name
            file_size = path.stat().st_size
            with open(path, 'rb') as f:
                file_bytes = f.read()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Select backend
        backend = self._select_backend(filename, file_size)

        logger.info(
            "Parsing document",
            filename=filename,
            size_bytes=file_size,
            backend=backend.value,
        )

        # Parse with selected backend
        if backend == ParserBackend.DOCLING:
            result = await self._parse_with_docling(file_bytes, filename)
        elif backend == ParserBackend.VISION:
            result = await self._parse_with_vision(file_bytes, filename)
        else:
            result = await self._parse_with_fallback(file_bytes, filename)

        # Add timing
        result.parse_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Document parsed",
            filename=filename,
            pages=result.page_count,
            tables=len(result.tables),
            time_ms=round(result.parse_time_ms, 2),
        )

        return result

    async def _parse_with_docling(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> ParsedDocument:
        """Parse document using Docling (best table accuracy)."""
        if not await self.initialize():
            return await self._parse_with_fallback(file_bytes, filename)

        try:
            # Write to temp file (Docling needs file path)
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix,
                delete=False
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                # Convert document in thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._converter.convert(tmp_path)
                )

                # Extract content
                doc = result.document

                # Get markdown output
                markdown = doc.export_to_markdown()

                # Extract text (plain)
                text = doc.export_to_text() if hasattr(doc, 'export_to_text') else markdown

                # Extract tables
                tables = []
                if hasattr(doc, 'tables'):
                    for i, table in enumerate(doc.tables):
                        page_num = table.prov[0].page_no if table.prov else 0
                        parsed_table = self._convert_docling_table(table, page_num, i)
                        if parsed_table:
                            tables.append(parsed_table)

                # Extract images
                images = []
                if hasattr(doc, 'pictures') and self.config.include_images:
                    for i, pic in enumerate(doc.pictures):
                        images.append({
                            "index": i,
                            "page": pic.prov[0].page_no if pic.prov else 0,
                            "caption": pic.caption if hasattr(pic, 'caption') else None,
                        })

                # Build metadata
                metadata = {}
                if self.config.include_metadata and hasattr(doc, 'metadata'):
                    metadata = {
                        "title": getattr(doc.metadata, 'title', None),
                        "author": getattr(doc.metadata, 'author', None),
                        "created": getattr(doc.metadata, 'created', None),
                    }

                # Build pages
                pages = []
                page_count = len(doc.pages) if hasattr(doc, 'pages') else 1

                return ParsedDocument(
                    filename=filename,
                    text=text,
                    markdown=markdown,
                    pages=pages,
                    tables=tables,
                    images=images,
                    metadata=metadata,
                    page_count=page_count,
                    word_count=len(text.split()),
                    parser_backend="docling",
                    parse_time_ms=0,
                )

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error("Docling parsing failed", error=str(e))
            return await self._parse_with_fallback(file_bytes, filename)

    def _convert_docling_table(
        self,
        table: Any,
        page_number: int,
        table_index: int,
    ) -> Optional[ParsedTable]:
        """Convert Docling table to ParsedTable."""
        try:
            # Get table data
            if hasattr(table, 'export_to_dataframe'):
                df = table.export_to_dataframe()
                headers = list(df.columns)
                rows = df.values.tolist()
            elif hasattr(table, 'data'):
                data = table.data
                headers = data.get('header', [])
                rows = data.get('body', [])
            else:
                return None

            # Generate markdown
            markdown = self._table_to_markdown(headers, rows)

            return ParsedTable(
                page_number=page_number,
                table_index=table_index,
                headers=headers,
                rows=[[str(cell) for cell in row] for row in rows],
                markdown=markdown,
                confidence=0.98,  # Docling's high accuracy
            )

        except Exception as e:
            logger.debug("Table conversion failed", error=str(e))
            return None

    def _table_to_markdown(self, headers: List[str], rows: List[List]) -> str:
        """Convert table to markdown format."""
        if not headers and not rows:
            return ""

        lines = []

        # Header row
        if headers:
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")

        # Data rows
        for row in rows:
            cells = [str(cell).replace("|", "\\|") for cell in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    async def _parse_with_vision(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> ParsedDocument:
        """
        Parse scanned/visual document using Vision Language Model.

        Phase 51: Uses VLMProcessor with automatic fallback chain:
        Claude → OpenAI → Qwen → Local
        """
        # Check if VLM is enabled
        from backend.core.config import settings
        vlm_enabled = getattr(settings, 'ENABLE_VLM', True)

        if vlm_enabled:
            try:
                from backend.services.vlm_processor import get_vlm_processor

                processor = await get_vlm_processor()

                # Extract text using VLM
                result = await processor.extract_text(file_bytes)

                if result.success:
                    text = result.ocr_text or result.content

                    # Try to extract structured data (tables) if available
                    tables = []
                    if result.structured_data:
                        # VLM may return table data in structured format
                        if 'tables' in result.structured_data:
                            for i, table_data in enumerate(result.structured_data['tables']):
                                tables.append({
                                    "index": i,
                                    "page": 0,
                                    "caption": table_data.get("caption"),
                                    "data": table_data.get("data", []),
                                    "headers": table_data.get("headers", []),
                                    "markdown": table_data.get("markdown", ""),
                                })

                    return ParsedDocument(
                        filename=filename,
                        text=text,
                        markdown=text,  # VLM extracts in markdown format
                        pages=[],
                        tables=tables,
                        images=[],
                        metadata={
                            "extracted_by": "vlm_processor",
                            "vlm_provider": result.provider.value if result.provider else "unknown",
                            "vlm_model": result.model,
                            "processing_time_ms": result.processing_time_ms,
                        },
                        page_count=1,
                        word_count=len(text.split()),
                        parser_backend="vlm",
                        parse_time_ms=result.processing_time_ms,
                    )
                else:
                    logger.warning(
                        "VLM parsing failed, falling back to basic vision",
                        error=result.error,
                    )
            except Exception as e:
                logger.warning(
                    "VLM processor unavailable, falling back to basic vision",
                    error=str(e),
                )

        # Fallback to basic vision using LLMFactory
        try:
            from backend.services.llm import LLMFactory
            import base64

            # Get vision model
            llm = LLMFactory.get_chat_model(
                provider=self.config.vision_provider,
                model=self.config.vision_model,
            )

            # Convert to base64
            b64_image = base64.b64encode(file_bytes).decode()

            # Determine MIME type
            ext = Path(filename).suffix.lower()
            mime_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }
            mime_type = mime_types.get(ext, 'image/png')

            # Create vision prompt
            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """Extract all text from this document image.
If there are tables, format them as markdown tables.
If there are headers or sections, preserve the structure.
Return the content in clean markdown format."""
                    }
                ]
            )

            response = await llm.ainvoke([message])
            text = response.content

            return ParsedDocument(
                filename=filename,
                text=text,
                markdown=text,
                pages=[],
                tables=[],  # Vision model extracts tables inline
                images=[],
                metadata={"extracted_by": "vision_model_fallback"},
                page_count=1,
                word_count=len(text.split()),
                parser_backend="vision",
                parse_time_ms=0,
            )

        except Exception as e:
            logger.error("Vision parsing failed", error=str(e))
            return await self._parse_with_fallback(file_bytes, filename)

    async def _parse_with_fallback(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> ParsedDocument:
        """Fallback parsing using PyMuPDF or basic extraction."""
        try:
            import fitz  # PyMuPDF

            # Open document
            doc = fitz.open(stream=file_bytes, filetype=Path(filename).suffix[1:])

            text_parts = []
            tables = []

            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                text_parts.append(text)

                # Extract tables (basic)
                page_tables = page.find_tables()
                for i, table in enumerate(page_tables):
                    if table.header.external:
                        continue
                    headers = [str(c) for c in table.header.names] if table.header else []
                    rows = [[str(cell) for cell in row] for row in table.extract()]

                    tables.append(ParsedTable(
                        page_number=page_num + 1,
                        table_index=i,
                        headers=headers,
                        rows=rows,
                        markdown=self._table_to_markdown(headers, rows),
                        confidence=0.75,  # Lower confidence for PyMuPDF
                    ))

            full_text = "\n\n".join(text_parts)
            doc.close()

            return ParsedDocument(
                filename=filename,
                text=full_text,
                markdown=full_text,
                pages=[],
                tables=tables,
                images=[],
                metadata={},
                page_count=len(text_parts),
                word_count=len(full_text.split()),
                parser_backend="pymupdf",
                parse_time_ms=0,
            )

        except Exception as e:
            logger.error("Fallback parsing failed", error=str(e))
            # Return empty document
            return ParsedDocument(
                filename=filename,
                text="",
                markdown="",
                pages=[],
                tables=[],
                images=[],
                metadata={"error": str(e)},
                page_count=0,
                word_count=0,
                parser_backend="error",
                parse_time_ms=0,
            )

    async def parse_batch(
        self,
        sources: List[Union[str, bytes, Path]],
        filenames: Optional[List[str]] = None,
        max_concurrent: int = 4,
    ) -> List[ParsedDocument]:
        """
        Parse multiple documents concurrently.

        Args:
            sources: List of file paths or bytes
            filenames: Required if any source is bytes
            max_concurrent: Maximum concurrent parsing operations

        Returns:
            List of ParsedDocument objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_one(source, filename):
            async with semaphore:
                return await self.parse(source, filename)

        if filenames is None:
            filenames = [None] * len(sources)

        tasks = [parse_one(s, f) for s, f in zip(sources, filenames)]
        return await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# Singleton Management
# =============================================================================

_document_parser: Optional[DocumentParser] = None
_parser_lock = asyncio.Lock()


async def get_document_parser(
    config: Optional[ParserConfig] = None,
) -> DocumentParser:
    """Get or create document parser singleton."""
    global _document_parser

    if _document_parser is not None:
        return _document_parser

    async with _parser_lock:
        if _document_parser is not None:
            return _document_parser

        _document_parser = DocumentParser(config)
        return _document_parser


# =============================================================================
# Convenience Functions
# =============================================================================

async def parse_document(
    source: Union[str, bytes, Path],
    filename: Optional[str] = None,
) -> ParsedDocument:
    """
    Convenience function to parse a document.

    Args:
        source: File path or bytes
        filename: Required if source is bytes

    Returns:
        ParsedDocument
    """
    parser = await get_document_parser()
    return await parser.parse(source, filename)


async def extract_tables(
    source: Union[str, bytes, Path],
    filename: Optional[str] = None,
) -> List[ParsedTable]:
    """
    Extract only tables from a document.

    Args:
        source: File path or bytes
        filename: Required if source is bytes

    Returns:
        List of ParsedTable objects
    """
    parser = await get_document_parser()
    result = await parser.parse(source, filename)
    return result.tables
