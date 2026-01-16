"""
AIDocumentIndexer - Base Format Generator
==========================================

Abstract base class for document format generators.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from backend.services.generation.models import GenerationJob, GenerationConfig

logger = structlog.get_logger(__name__)


class BaseFormatGenerator(ABC):
    """
    Abstract base class for document format generators.

    Each format generator converts a GenerationJob into a specific
    output format (PPTX, DOCX, PDF, etc.).
    """

    def __init__(self, config: "GenerationConfig"):
        """
        Initialize the format generator.

        Args:
            config: Generation configuration
        """
        self.config = config

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this format (e.g., 'pptx', 'docx')."""
        pass

    @property
    @abstractmethod
    def content_type(self) -> str:
        """MIME content type for this format."""
        pass

    @abstractmethod
    async def generate(self, job: "GenerationJob", filename: str) -> str:
        """
        Generate the output file.

        Args:
            job: Generation job with completed sections
            filename: Base filename (without extension)

        Returns:
            Full path to the generated file
        """
        pass

    def get_format_guidelines(self) -> str:
        """
        Get format-specific generation guidelines for the LLM.

        Override in subclasses to provide format-specific instructions.

        Returns:
            Guidelines string for the LLM prompt
        """
        return ""
