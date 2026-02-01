"""
Format-Specific Document Generators

This package provides generators for different output formats:
- PPTX: PowerPoint presentations
- DOCX: Word documents
- PDF: PDF documents
- XLSX: Excel spreadsheets
- Markdown: Markdown text
- HTML: HTML documents
- TXT: Plain text
- Mind Map: Interactive D3.js-based mind maps
"""

from .base import BaseFormatGenerator
from .factory import FormatGeneratorFactory, register_generator

# Import format-specific generators to register them with the factory
from .pptx import PPTXGenerator
from .docx import DOCXGenerator
from .pdf import PDFGenerator
from .xlsx import XLSXGenerator
from .simple import MarkdownGenerator, HTMLGenerator, TXTGenerator
from .mindmap import MindMapGenerator

__all__ = [
    # Base classes
    "BaseFormatGenerator",
    "FormatGeneratorFactory",
    "register_generator",
    # Format generators
    "PPTXGenerator",
    "DOCXGenerator",
    "PDFGenerator",
    "XLSXGenerator",
    "MarkdownGenerator",
    "HTMLGenerator",
    "TXTGenerator",
    "MindMapGenerator",
]
