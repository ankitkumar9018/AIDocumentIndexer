"""
AIDocumentIndexer - PDF Tools Service
======================================

Comprehensive PDF manipulation tools for:
- Merge multiple PDFs
- Split PDF into pages or ranges
- Extract specific pages
- Rotate pages
- Compress PDF
- Convert PDF to images
- Convert images to PDF
- Edit PDF metadata
- Add watermarks
- Rearrange pages

These features can be used:
1. At upload time (pre-processing)
2. On existing documents
3. Via API endpoints
4. In the desktop app

Dependencies:
- PyMuPDF (fitz) for fast PDF manipulation
- Pillow for image processing
- reportlab for PDF creation
"""

import asyncio
import io
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)

# Optional imports with graceful fallback
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available - PDF tools will be limited")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not available - image conversion disabled")


class PageRange:
    """Represents a range of pages."""

    def __init__(self, start: int, end: Optional[int] = None):
        """
        Initialize page range.

        Args:
            start: Start page (1-indexed)
            end: End page (1-indexed), inclusive. If None, only start page.
        """
        self.start = start
        self.end = end if end is not None else start

    def to_indices(self, total_pages: int) -> List[int]:
        """Convert to 0-indexed page indices."""
        # Normalize to 0-indexed
        start = max(0, self.start - 1)
        end = min(total_pages - 1, self.end - 1)
        return list(range(start, end + 1))

    @classmethod
    def parse(cls, spec: str) -> "PageRange":
        """
        Parse a page range specification.

        Examples:
            "5" -> PageRange(5, 5)
            "1-10" -> PageRange(1, 10)
            "5-" -> PageRange(5, last_page)
        """
        spec = spec.strip()
        if "-" in spec:
            parts = spec.split("-")
            start = int(parts[0]) if parts[0] else 1
            end = int(parts[1]) if parts[1] else None
            return cls(start, end)
        return cls(int(spec))


@dataclass
class PDFMetadata:
    """PDF document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None


@dataclass
class PDFInfo:
    """Information about a PDF document."""
    page_count: int
    file_size: int
    metadata: PDFMetadata
    is_encrypted: bool
    has_text: bool
    has_images: bool
    page_sizes: List[Tuple[float, float]]  # [(width, height), ...]


@dataclass
class PDFToolResult:
    """Result of a PDF operation."""
    success: bool
    output_path: Optional[str] = None
    output_bytes: Optional[bytes] = None
    message: str = ""
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    error: Optional[str] = None


class PDFToolsService:
    """
    Service for PDF manipulation operations.

    All operations can work with either file paths or bytes.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or "/tmp/pdf_tools"
        os.makedirs(self.temp_dir, exist_ok=True)

    def _open_pdf(self, source: Union[str, bytes, BinaryIO]) -> "fitz.Document":
        """Open a PDF from various sources."""
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF not installed")

        if isinstance(source, str):
            return fitz.open(source)
        elif isinstance(source, bytes):
            return fitz.open(stream=source, filetype="pdf")
        else:
            return fitz.open(stream=source.read(), filetype="pdf")

    def _save_pdf(
        self,
        doc: "fitz.Document",
        output_path: Optional[str] = None,
    ) -> PDFToolResult:
        """Save PDF and return result."""
        try:
            if output_path:
                doc.save(output_path)
                file_size = os.path.getsize(output_path)
                return PDFToolResult(
                    success=True,
                    output_path=output_path,
                    page_count=doc.page_count,
                    file_size=file_size,
                    message=f"Saved to {output_path}",
                )
            else:
                output_bytes = doc.tobytes()
                return PDFToolResult(
                    success=True,
                    output_bytes=output_bytes,
                    page_count=doc.page_count,
                    file_size=len(output_bytes),
                    message="PDF created",
                )
        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Save failed: {e}",
            )
        finally:
            doc.close()

    # =========================================================================
    # Information
    # =========================================================================

    def get_info(self, source: Union[str, bytes]) -> PDFInfo:
        """Get detailed information about a PDF."""
        doc = self._open_pdf(source)
        try:
            metadata = PDFMetadata(
                title=doc.metadata.get("title"),
                author=doc.metadata.get("author"),
                subject=doc.metadata.get("subject"),
                keywords=doc.metadata.get("keywords"),
                creator=doc.metadata.get("creator"),
                producer=doc.metadata.get("producer"),
            )

            page_sizes = []
            has_text = False
            has_images = False

            for page in doc:
                page_sizes.append((page.rect.width, page.rect.height))
                if page.get_text().strip():
                    has_text = True
                if page.get_images():
                    has_images = True

            file_size = len(source) if isinstance(source, bytes) else os.path.getsize(source)

            return PDFInfo(
                page_count=doc.page_count,
                file_size=file_size,
                metadata=metadata,
                is_encrypted=doc.is_encrypted,
                has_text=has_text,
                has_images=has_images,
                page_sizes=page_sizes,
            )
        finally:
            doc.close()

    # =========================================================================
    # Merge
    # =========================================================================

    def merge(
        self,
        sources: List[Union[str, bytes]],
        output_path: Optional[str] = None,
    ) -> PDFToolResult:
        """
        Merge multiple PDFs into one.

        Args:
            sources: List of PDF file paths or bytes
            output_path: Output file path (if None, returns bytes)

        Returns:
            PDFToolResult with merged PDF
        """
        if not HAS_PYMUPDF:
            return PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
                message="PDF merge requires PyMuPDF",
            )

        if len(sources) < 2:
            return PDFToolResult(
                success=False,
                error="At least 2 PDFs required",
                message="Need at least 2 PDFs to merge",
            )

        try:
            merged = fitz.open()

            for source in sources:
                doc = self._open_pdf(source)
                merged.insert_pdf(doc)
                doc.close()

            return self._save_pdf(merged, output_path)

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Merge failed: {e}",
            )

    # =========================================================================
    # Split
    # =========================================================================

    def split(
        self,
        source: Union[str, bytes],
        ranges: List[Union[PageRange, str]],
        output_dir: Optional[str] = None,
    ) -> List[PDFToolResult]:
        """
        Split a PDF into multiple PDFs based on page ranges.

        Args:
            source: Source PDF
            ranges: List of page ranges (e.g., ["1-5", "6-10"])
            output_dir: Directory for output files

        Returns:
            List of PDFToolResult for each split
        """
        if not HAS_PYMUPDF:
            return [PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )]

        output_dir = output_dir or self.temp_dir
        os.makedirs(output_dir, exist_ok=True)

        results = []
        doc = self._open_pdf(source)

        try:
            total_pages = doc.page_count

            for i, range_spec in enumerate(ranges):
                if isinstance(range_spec, str):
                    page_range = PageRange.parse(range_spec)
                else:
                    page_range = range_spec

                # Handle open-ended ranges
                if page_range.end is None:
                    page_range.end = total_pages

                indices = page_range.to_indices(total_pages)

                if not indices:
                    results.append(PDFToolResult(
                        success=False,
                        error="Invalid page range",
                        message=f"Range {range_spec} produced no pages",
                    ))
                    continue

                # Create new PDF with selected pages
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=indices[0], to_page=indices[-1])

                output_path = os.path.join(output_dir, f"split_{i+1}.pdf")
                result = self._save_pdf(new_doc, output_path)
                results.append(result)

        except Exception as e:
            results.append(PDFToolResult(
                success=False,
                error=str(e),
                message=f"Split failed: {e}",
            ))
        finally:
            doc.close()

        return results

    def split_by_pages(
        self,
        source: Union[str, bytes],
        pages_per_split: int = 1,
        output_dir: Optional[str] = None,
    ) -> List[PDFToolResult]:
        """
        Split a PDF into multiple PDFs with N pages each.

        Args:
            source: Source PDF
            pages_per_split: Number of pages per split
            output_dir: Directory for output files

        Returns:
            List of PDFToolResult for each split
        """
        doc = self._open_pdf(source)
        try:
            total_pages = doc.page_count
            ranges = []
            for start in range(0, total_pages, pages_per_split):
                end = min(start + pages_per_split, total_pages)
                ranges.append(PageRange(start + 1, end))  # 1-indexed
            return self.split(source, ranges, output_dir)
        finally:
            doc.close()

    # =========================================================================
    # Extract Pages
    # =========================================================================

    def extract_pages(
        self,
        source: Union[str, bytes],
        pages: List[int],
        output_path: Optional[str] = None,
    ) -> PDFToolResult:
        """
        Extract specific pages from a PDF.

        Args:
            source: Source PDF
            pages: List of page numbers (1-indexed)
            output_path: Output file path

        Returns:
            PDFToolResult with extracted pages
        """
        if not HAS_PYMUPDF:
            return PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )

        try:
            doc = self._open_pdf(source)
            new_doc = fitz.open()

            for page_num in pages:
                if 1 <= page_num <= doc.page_count:
                    new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)

            doc.close()
            return self._save_pdf(new_doc, output_path)

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Extract failed: {e}",
            )

    # =========================================================================
    # Rotate
    # =========================================================================

    def rotate_pages(
        self,
        source: Union[str, bytes],
        rotation: int,
        pages: Optional[List[int]] = None,
        output_path: Optional[str] = None,
    ) -> PDFToolResult:
        """
        Rotate pages in a PDF.

        Args:
            source: Source PDF
            rotation: Rotation angle (90, 180, 270, or -90)
            pages: Pages to rotate (1-indexed). If None, rotate all.
            output_path: Output file path

        Returns:
            PDFToolResult with rotated PDF
        """
        if not HAS_PYMUPDF:
            return PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )

        if rotation not in [90, 180, 270, -90]:
            return PDFToolResult(
                success=False,
                error="Invalid rotation",
                message="Rotation must be 90, 180, 270, or -90",
            )

        try:
            doc = self._open_pdf(source)

            if pages is None:
                pages = list(range(1, doc.page_count + 1))

            for page_num in pages:
                if 1 <= page_num <= doc.page_count:
                    page = doc[page_num - 1]
                    page.set_rotation(page.rotation + rotation)

            return self._save_pdf(doc, output_path)

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Rotate failed: {e}",
            )

    # =========================================================================
    # Compress
    # =========================================================================

    def compress(
        self,
        source: Union[str, bytes],
        output_path: Optional[str] = None,
        image_quality: int = 75,
    ) -> PDFToolResult:
        """
        Compress a PDF to reduce file size.

        Args:
            source: Source PDF
            output_path: Output file path
            image_quality: Quality for image compression (1-100)

        Returns:
            PDFToolResult with compressed PDF
        """
        if not HAS_PYMUPDF:
            return PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )

        try:
            doc = self._open_pdf(source)
            original_size = len(source) if isinstance(source, bytes) else os.path.getsize(source)

            # Compress images in the PDF
            for page in doc:
                images = page.get_images()
                for img in images:
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            # Compress image
                            img_data = base_image["image"]
                            pil_img = Image.open(io.BytesIO(img_data))

                            # Convert to RGB if necessary
                            if pil_img.mode in ("RGBA", "P"):
                                pil_img = pil_img.convert("RGB")

                            # Compress
                            buffer = io.BytesIO()
                            pil_img.save(buffer, "JPEG", quality=image_quality, optimize=True)
                            compressed = buffer.getvalue()

                            # Only replace if smaller
                            if len(compressed) < len(img_data):
                                page.replace_image(xref, stream=compressed)
                    except Exception:
                        pass  # Skip problematic images

            # Save with compression options
            if output_path:
                doc.save(
                    output_path,
                    garbage=4,  # Maximum garbage collection
                    deflate=True,
                    clean=True,
                )
                new_size = os.path.getsize(output_path)
                doc.close()
                return PDFToolResult(
                    success=True,
                    output_path=output_path,
                    file_size=new_size,
                    message=f"Compressed from {original_size:,} to {new_size:,} bytes ({100 - (new_size/original_size)*100:.1f}% reduction)",
                )
            else:
                output_bytes = doc.tobytes(garbage=4, deflate=True, clean=True)
                doc.close()
                return PDFToolResult(
                    success=True,
                    output_bytes=output_bytes,
                    file_size=len(output_bytes),
                    message=f"Compressed from {original_size:,} to {len(output_bytes):,} bytes",
                )

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Compress failed: {e}",
            )

    # =========================================================================
    # Convert to Images
    # =========================================================================

    def to_images(
        self,
        source: Union[str, bytes],
        output_dir: Optional[str] = None,
        dpi: int = 150,
        image_format: str = "png",
        pages: Optional[List[int]] = None,
    ) -> List[PDFToolResult]:
        """
        Convert PDF pages to images.

        Args:
            source: Source PDF
            output_dir: Directory for output images
            dpi: Image resolution
            image_format: Output format (png, jpeg)
            pages: Pages to convert (1-indexed). If None, all pages.

        Returns:
            List of PDFToolResult for each page image
        """
        if not HAS_PYMUPDF:
            return [PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )]

        output_dir = output_dir or self.temp_dir
        os.makedirs(output_dir, exist_ok=True)

        results = []
        doc = self._open_pdf(source)

        try:
            if pages is None:
                pages = list(range(1, doc.page_count + 1))

            zoom = dpi / 72  # 72 is default DPI

            for page_num in pages:
                if 1 <= page_num <= doc.page_count:
                    page = doc[page_num - 1]
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)

                    output_path = os.path.join(
                        output_dir,
                        f"page_{page_num}.{image_format}"
                    )

                    pix.save(output_path)

                    results.append(PDFToolResult(
                        success=True,
                        output_path=output_path,
                        file_size=os.path.getsize(output_path),
                        message=f"Page {page_num} converted",
                    ))

        except Exception as e:
            results.append(PDFToolResult(
                success=False,
                error=str(e),
                message=f"Conversion failed: {e}",
            ))
        finally:
            doc.close()

        return results

    # =========================================================================
    # Convert from Images
    # =========================================================================

    def from_images(
        self,
        images: List[Union[str, bytes]],
        output_path: Optional[str] = None,
        page_size: str = "a4",
    ) -> PDFToolResult:
        """
        Create a PDF from images.

        Args:
            images: List of image file paths or bytes
            output_path: Output file path
            page_size: Page size (a4, letter, or WxH in points)

        Returns:
            PDFToolResult with created PDF
        """
        if not HAS_PYMUPDF or not HAS_PIL:
            return PDFToolResult(
                success=False,
                error="PyMuPDF and Pillow required",
            )

        try:
            doc = fitz.open()

            # Page dimensions
            if page_size.lower() == "a4":
                width, height = 595, 842  # A4 in points
            elif page_size.lower() == "letter":
                width, height = 612, 792
            else:
                # Parse WxH
                parts = page_size.lower().split("x")
                width, height = float(parts[0]), float(parts[1])

            for img_source in images:
                # Load image
                if isinstance(img_source, str):
                    img = Image.open(img_source)
                else:
                    img = Image.open(io.BytesIO(img_source))

                # Convert to RGB if necessary
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Create page
                page = doc.new_page(width=width, height=height)

                # Calculate fit
                img_width, img_height = img.size
                scale = min(width / img_width, height / img_height) * 0.9
                new_width = img_width * scale
                new_height = img_height * scale

                # Center image
                x = (width - new_width) / 2
                y = (height - new_height) / 2

                rect = fitz.Rect(x, y, x + new_width, y + new_height)

                # Insert image
                img_buffer = io.BytesIO()
                img.save(img_buffer, "JPEG", quality=95)
                page.insert_image(rect, stream=img_buffer.getvalue())

            return self._save_pdf(doc, output_path)

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Image to PDF failed: {e}",
            )

    # =========================================================================
    # Metadata
    # =========================================================================

    def set_metadata(
        self,
        source: Union[str, bytes],
        metadata: PDFMetadata,
        output_path: Optional[str] = None,
    ) -> PDFToolResult:
        """
        Set PDF metadata.

        Args:
            source: Source PDF
            metadata: New metadata
            output_path: Output file path

        Returns:
            PDFToolResult with updated PDF
        """
        if not HAS_PYMUPDF:
            return PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )

        try:
            doc = self._open_pdf(source)

            new_metadata = {}
            if metadata.title:
                new_metadata["title"] = metadata.title
            if metadata.author:
                new_metadata["author"] = metadata.author
            if metadata.subject:
                new_metadata["subject"] = metadata.subject
            if metadata.keywords:
                new_metadata["keywords"] = metadata.keywords
            if metadata.creator:
                new_metadata["creator"] = metadata.creator

            doc.set_metadata(new_metadata)

            return self._save_pdf(doc, output_path)

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Metadata update failed: {e}",
            )

    # =========================================================================
    # Watermark
    # =========================================================================

    def add_watermark(
        self,
        source: Union[str, bytes],
        watermark_text: str,
        output_path: Optional[str] = None,
        opacity: float = 0.3,
        angle: int = 45,
        pages: Optional[List[int]] = None,
    ) -> PDFToolResult:
        """
        Add a text watermark to PDF pages.

        Args:
            source: Source PDF
            watermark_text: Watermark text
            output_path: Output file path
            opacity: Watermark opacity (0-1)
            angle: Rotation angle
            pages: Pages to watermark (1-indexed). If None, all pages.

        Returns:
            PDFToolResult with watermarked PDF
        """
        if not HAS_PYMUPDF:
            return PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )

        try:
            doc = self._open_pdf(source)

            if pages is None:
                pages = list(range(1, doc.page_count + 1))

            for page_num in pages:
                if 1 <= page_num <= doc.page_count:
                    page = doc[page_num - 1]
                    rect = page.rect

                    # Calculate center
                    center = fitz.Point(rect.width / 2, rect.height / 2)

                    # Add watermark
                    page.insert_text(
                        center,
                        watermark_text,
                        fontsize=72,
                        rotate=angle,
                        color=(0.5, 0.5, 0.5),
                        opacity=opacity,
                    )

            return self._save_pdf(doc, output_path)

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Watermark failed: {e}",
            )

    # =========================================================================
    # Rearrange Pages
    # =========================================================================

    def rearrange_pages(
        self,
        source: Union[str, bytes],
        new_order: List[int],
        output_path: Optional[str] = None,
    ) -> PDFToolResult:
        """
        Rearrange pages in a PDF.

        Args:
            source: Source PDF
            new_order: New page order (1-indexed). Can include duplicates.
            output_path: Output file path

        Returns:
            PDFToolResult with rearranged PDF
        """
        if not HAS_PYMUPDF:
            return PDFToolResult(
                success=False,
                error="PyMuPDF not installed",
            )

        try:
            doc = self._open_pdf(source)
            new_doc = fitz.open()

            for page_num in new_order:
                if 1 <= page_num <= doc.page_count:
                    new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)

            doc.close()
            return self._save_pdf(new_doc, output_path)

        except Exception as e:
            return PDFToolResult(
                success=False,
                error=str(e),
                message=f"Rearrange failed: {e}",
            )


# =============================================================================
# Singleton Access
# =============================================================================

_pdf_tools: Optional[PDFToolsService] = None


def get_pdf_tools(temp_dir: Optional[str] = None) -> PDFToolsService:
    """Get or create the PDF tools service."""
    global _pdf_tools
    if _pdf_tools is None:
        _pdf_tools = PDFToolsService(temp_dir)
    return _pdf_tools


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PDFToolsService",
    "PDFToolResult",
    "PDFInfo",
    "PDFMetadata",
    "PageRange",
    "get_pdf_tools",
]
