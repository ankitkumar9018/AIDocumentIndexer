"""
AIDocumentIndexer - Universal Document Processor
================================================

Processes all supported file types using appropriate extractors.
Supports PDF, Office documents, images, and more.
"""

import os
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedContent:
    """Extracted content from a document."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    word_count: int = 0
    page_count: int = 0
    language: str = "en"


class UniversalProcessor:
    """
    Universal document processor supporting multiple file formats.

    Automatically selects the appropriate extractor based on file type.
    """

    # File type to processor mapping
    PROCESSORS = {
        # PDF
        ".pdf": "_process_pdf",
        # Office documents
        ".docx": "_process_docx",
        ".doc": "_process_doc",
        ".pptx": "_process_pptx",
        ".ppt": "_process_ppt",
        ".xlsx": "_process_xlsx",
        ".xls": "_process_xls",
        # Open formats
        ".odt": "_process_odt",
        ".odp": "_process_odp",
        ".ods": "_process_ods",
        # Text files
        ".txt": "_process_text",
        ".md": "_process_text",
        ".rtf": "_process_rtf",
        ".html": "_process_html",
        ".htm": "_process_html",
        ".xml": "_process_text",
        ".json": "_process_text",
        # Images (OCR)
        ".png": "_process_image",
        ".jpg": "_process_image",
        ".jpeg": "_process_image",
        ".gif": "_process_image",
        ".webp": "_process_image",
        ".bmp": "_process_image",
        ".tiff": "_process_image",
        # Email
        ".eml": "_process_email",
        ".msg": "_process_msg",
        # CSV
        ".csv": "_process_csv",
    }

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_image_analysis: bool = True,
        ocr_language: str = "en",
        smart_image_handling: bool = True,
    ):
        """
        Initialize the universal processor.

        Args:
            enable_ocr: Enable OCR for scanned documents
            enable_image_analysis: Enable image description generation
            ocr_language: Language for OCR (default: English)
            smart_image_handling: Optimize image quality based on content
        """
        self.enable_ocr = enable_ocr
        self.enable_image_analysis = enable_image_analysis
        self.ocr_language = ocr_language
        self.smart_image_handling = smart_image_handling

        # Lazy-load heavy dependencies
        self._ocr_engine = None
        self._pdf_processor = None

    def process(
        self,
        file_path: str,
        processing_mode: str = "smart",
    ) -> ExtractedContent:
        """
        Process a document and extract content.

        Args:
            file_path: Path to the document
            processing_mode: "full", "smart", or "text_only"

        Returns:
            ExtractedContent: Extracted text and metadata
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension not in self.PROCESSORS:
            raise ValueError(f"Unsupported file type: {extension}")

        processor_method = getattr(self, self.PROCESSORS[extension])

        logger.info(
            "Processing document",
            file_path=file_path,
            extension=extension,
            processing_mode=processing_mode,
        )

        try:
            content = processor_method(file_path, processing_mode)
            content.word_count = len(content.text.split())
            return content
        except Exception as e:
            logger.error(
                "Error processing document",
                file_path=file_path,
                error=str(e),
            )
            raise

    # =========================================================================
    # PDF Processing
    # =========================================================================

    def _process_pdf(self, file_path: str, mode: str) -> ExtractedContent:
        """Process PDF document using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed, using fallback")
            return self._process_pdf_fallback(file_path, mode)

        doc = fitz.open(file_path)
        pages = []
        all_text = []
        images = []

        for page_num, page in enumerate(doc, start=1):
            # Extract text
            text = page.get_text()
            all_text.append(text)

            # Check if page needs OCR (mostly images, little text)
            if self.enable_ocr and len(text.strip()) < 50:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    all_text[-1] = ocr_text
                    text = ocr_text

            pages.append({
                "page_number": page_num,
                "text": text,
                "has_images": len(page.get_images()) > 0,
            })

            # Extract images if not text_only mode
            if mode != "text_only":
                for img_index, img in enumerate(page.get_images()):
                    images.append({
                        "page_number": page_num,
                        "index": img_index,
                        "xref": img[0],
                    })

        doc.close()

        return ExtractedContent(
            text="\n\n".join(all_text),
            metadata={
                "file_type": "pdf",
                "file_path": file_path,
            },
            pages=pages,
            images=images,
            page_count=len(pages),
        )

    def _process_pdf_fallback(self, file_path: str, mode: str) -> ExtractedContent:
        """Fallback PDF processing without PyMuPDF."""
        # Try pypdf as fallback
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            return ExtractedContent(
                text=text,
                metadata={"file_type": "pdf", "file_path": file_path},
                page_count=len(reader.pages),
            )
        except ImportError:
            raise ImportError("Neither PyMuPDF nor pypdf is installed")

    # =========================================================================
    # Office Documents
    # =========================================================================

    def _process_docx(self, file_path: str, mode: str) -> ExtractedContent:
        """Process DOCX using python-docx."""
        try:
            from docx import Document
        except ImportError:
            return self._unstructured_fallback(file_path)

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # Extract tables
        tables = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            tables.append({"data": table_data})

        return ExtractedContent(
            text="\n\n".join(paragraphs),
            metadata={"file_type": "docx", "file_path": file_path},
            tables=tables,
            page_count=1,  # DOCX doesn't have page concept in API
        )

    def _process_doc(self, file_path: str, mode: str) -> ExtractedContent:
        """Process legacy DOC files."""
        return self._unstructured_fallback(file_path)

    def _process_pptx(self, file_path: str, mode: str) -> ExtractedContent:
        """Process PPTX using python-pptx."""
        try:
            from pptx import Presentation
        except ImportError:
            return self._unstructured_fallback(file_path)

        prs = Presentation(file_path)
        slides = []
        all_text = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_text = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)

                # Extract table text
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = [cell.text for cell in row.cells]
                        slide_text.append(" | ".join(row_text))

            text = "\n".join(slide_text)
            slides.append({
                "page_number": slide_num,
                "text": text,
            })
            all_text.append(f"[Slide {slide_num}]\n{text}")

        return ExtractedContent(
            text="\n\n".join(all_text),
            metadata={"file_type": "pptx", "file_path": file_path},
            pages=slides,
            page_count=len(slides),
        )

    def _process_ppt(self, file_path: str, mode: str) -> ExtractedContent:
        """Process legacy PPT files."""
        return self._unstructured_fallback(file_path)

    def _process_xlsx(self, file_path: str, mode: str) -> ExtractedContent:
        """Process XLSX using openpyxl."""
        try:
            from openpyxl import load_workbook
        except ImportError:
            return self._unstructured_fallback(file_path)

        wb = load_workbook(file_path, read_only=True, data_only=True)
        all_text = []
        tables = []

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            sheet_data = []

            for row in sheet.iter_rows(values_only=True):
                row_values = [str(cell) if cell is not None else "" for cell in row]
                if any(row_values):  # Skip empty rows
                    sheet_data.append(row_values)

            if sheet_data:
                all_text.append(f"[Sheet: {sheet_name}]")
                for row in sheet_data:
                    all_text.append(" | ".join(row))

                tables.append({
                    "sheet_name": sheet_name,
                    "data": sheet_data,
                })

        wb.close()

        return ExtractedContent(
            text="\n".join(all_text),
            metadata={"file_type": "xlsx", "file_path": file_path},
            tables=tables,
            page_count=len(wb.sheetnames),
        )

    def _process_xls(self, file_path: str, mode: str) -> ExtractedContent:
        """Process legacy XLS files."""
        return self._unstructured_fallback(file_path)

    # =========================================================================
    # Open Document Formats
    # =========================================================================

    def _process_odt(self, file_path: str, mode: str) -> ExtractedContent:
        """Process ODT files."""
        return self._unstructured_fallback(file_path)

    def _process_odp(self, file_path: str, mode: str) -> ExtractedContent:
        """Process ODP files."""
        return self._unstructured_fallback(file_path)

    def _process_ods(self, file_path: str, mode: str) -> ExtractedContent:
        """Process ODS files."""
        return self._unstructured_fallback(file_path)

    # =========================================================================
    # Text Files
    # =========================================================================

    def _process_text(self, file_path: str, mode: str) -> ExtractedContent:
        """Process plain text files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        return ExtractedContent(
            text=text,
            metadata={"file_type": "text", "file_path": file_path},
            page_count=1,
        )

    def _process_rtf(self, file_path: str, mode: str) -> ExtractedContent:
        """Process RTF files."""
        try:
            from striprtf.striprtf import rtf_to_text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                rtf_content = f.read()
            text = rtf_to_text(rtf_content)
            return ExtractedContent(
                text=text,
                metadata={"file_type": "rtf", "file_path": file_path},
                page_count=1,
            )
        except ImportError:
            return self._unstructured_fallback(file_path)

    def _process_html(self, file_path: str, mode: str) -> ExtractedContent:
        """Process HTML files."""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()

            text = soup.get_text(separator="\n", strip=True)
            return ExtractedContent(
                text=text,
                metadata={"file_type": "html", "file_path": file_path},
                page_count=1,
            )
        except ImportError:
            # Fallback: basic text extraction
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return ExtractedContent(
                text=text,
                metadata={"file_type": "html", "file_path": file_path},
                page_count=1,
            )

    # =========================================================================
    # Images (OCR)
    # =========================================================================

    def _process_image(self, file_path: str, mode: str) -> ExtractedContent:
        """Process image using OCR."""
        if mode == "text_only":
            return ExtractedContent(
                text="",
                metadata={"file_type": "image", "file_path": file_path, "skipped": True},
                page_count=1,
            )

        text = self._ocr_image(file_path)

        return ExtractedContent(
            text=text,
            metadata={"file_type": "image", "file_path": file_path},
            page_count=1,
        )

    def _ocr_image(self, image_path: str) -> str:
        """Perform OCR on an image file."""
        if not self.enable_ocr:
            return ""

        try:
            from paddleocr import PaddleOCR

            if self._ocr_engine is None:
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.ocr_language,
                    show_log=False,
                )

            result = self._ocr_engine.ocr(image_path, cls=True)

            if result and result[0]:
                text_lines = [line[1][0] for line in result[0]]
                return "\n".join(text_lines)

            return ""

        except ImportError:
            logger.warning("PaddleOCR not installed, trying Tesseract")
            return self._ocr_tesseract(image_path)
        except Exception as e:
            logger.error("OCR failed", error=str(e))
            return ""

    def _ocr_tesseract(self, image_path: str) -> str:
        """Fallback OCR using Tesseract."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=self.ocr_language)
            return text
        except ImportError:
            logger.warning("Neither PaddleOCR nor pytesseract installed")
            return ""

    def _ocr_page(self, page) -> str:
        """OCR a PDF page."""
        try:
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")

            # Save temp file for OCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(img_data)
                temp_path = f.name

            text = self._ocr_image(temp_path)

            # Clean up
            os.unlink(temp_path)

            return text
        except Exception as e:
            logger.error("Page OCR failed", error=str(e))
            return ""

    # =========================================================================
    # Email
    # =========================================================================

    def _process_email(self, file_path: str, mode: str) -> ExtractedContent:
        """Process EML email files."""
        import email
        from email import policy

        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        parts = []
        parts.append(f"From: {msg.get('From', '')}")
        parts.append(f"To: {msg.get('To', '')}")
        parts.append(f"Subject: {msg.get('Subject', '')}")
        parts.append(f"Date: {msg.get('Date', '')}")
        parts.append("")

        # Get body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    parts.append(part.get_content())
        else:
            parts.append(msg.get_content())

        return ExtractedContent(
            text="\n".join(parts),
            metadata={
                "file_type": "email",
                "file_path": file_path,
                "subject": msg.get("Subject", ""),
            },
            page_count=1,
        )

    def _process_msg(self, file_path: str, mode: str) -> ExtractedContent:
        """Process MSG Outlook files."""
        try:
            import extract_msg
            msg = extract_msg.Message(file_path)

            parts = []
            parts.append(f"From: {msg.sender}")
            parts.append(f"To: {msg.to}")
            parts.append(f"Subject: {msg.subject}")
            parts.append(f"Date: {msg.date}")
            parts.append("")
            parts.append(msg.body or "")

            return ExtractedContent(
                text="\n".join(parts),
                metadata={
                    "file_type": "msg",
                    "file_path": file_path,
                    "subject": msg.subject,
                },
                page_count=1,
            )
        except ImportError:
            return self._unstructured_fallback(file_path)

    # =========================================================================
    # CSV
    # =========================================================================

    def _process_csv(self, file_path: str, mode: str) -> ExtractedContent:
        """Process CSV files."""
        import csv

        rows = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))

        return ExtractedContent(
            text="\n".join(rows),
            metadata={"file_type": "csv", "file_path": file_path},
            tables=[{"data": rows}],
            page_count=1,
        )

    # =========================================================================
    # Fallback
    # =========================================================================

    def _unstructured_fallback(self, file_path: str) -> ExtractedContent:
        """
        Fallback to Unstructured.io for unsupported or complex formats.
        """
        try:
            from unstructured.partition.auto import partition

            elements = partition(filename=file_path)
            text = "\n\n".join(str(el) for el in elements)

            return ExtractedContent(
                text=text,
                metadata={"file_type": "unstructured", "file_path": file_path},
                page_count=1,
            )
        except ImportError:
            logger.error("Unstructured not installed for fallback")
            raise ImportError(
                f"Cannot process {file_path}. "
                "Install 'unstructured' package for this file type."
            )
