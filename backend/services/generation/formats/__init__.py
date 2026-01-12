"""
AIDocumentIndexer - Document Format Generators
===============================================

Output format generators for document generation.

Each format generator is responsible for converting a GenerationJob
with completed sections into the target output format.

Available formats:
- PPTX: PowerPoint presentations
- DOCX: Word documents
- PDF: PDF documents
- Markdown: Markdown files
- HTML: HTML files
- XLSX: Excel spreadsheets
- TXT: Plain text files
"""

from backend.services.generation.formats.base import BaseFormatGenerator
from backend.services.generation.formats.pptx import PPTXGenerator
from backend.services.generation.formats.docx import DOCXGenerator
from backend.services.generation.formats.pdf import PDFGenerator
from backend.services.generation.formats.markdown import MarkdownGenerator
from backend.services.generation.formats.html import HTMLGenerator
from backend.services.generation.formats.xlsx import XLSXGenerator
from backend.services.generation.formats.txt import TXTGenerator

__all__ = [
    "BaseFormatGenerator",
    "PPTXGenerator",
    "DOCXGenerator",
    "PDFGenerator",
    "MarkdownGenerator",
    "HTMLGenerator",
    "XLSXGenerator",
    "TXTGenerator",
]
