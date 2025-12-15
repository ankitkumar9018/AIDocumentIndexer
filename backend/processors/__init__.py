"""
AIDocumentIndexer - Document Processors
=======================================

Document processing modules for text extraction, OCR, and chunking.
"""

from backend.processors.universal import UniversalProcessor
from backend.processors.chunker import DocumentChunker

__all__ = [
    "UniversalProcessor",
    "DocumentChunker",
]
