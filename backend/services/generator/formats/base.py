"""
Base Format Generator

Abstract base class for all document format generators.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GenerationJob, Section
    from ..template_analyzer import TemplateAnalysis


class BaseFormatGenerator(ABC):
    """Abstract base class for document format generators.

    All format-specific generators (PPTX, DOCX, PDF, etc.) should
    inherit from this class and implement the generate method.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name (e.g., 'pptx', 'docx')."""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension (e.g., '.pptx', '.docx')."""
        pass

    @abstractmethod
    async def generate(
        self,
        job: "GenerationJob",
        filename: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Generate a document in this format.

        Args:
            job: The generation job containing all metadata and sections
            filename: The output filename (without path)
            template_analysis: Optional template analysis for styling

        Returns:
            Path to the generated file
        """
        pass

    def get_output_path(self, output_dir: str, filename: str) -> str:
        """Get the full output path for the generated file."""
        import os
        return os.path.join(output_dir, filename)

    def ensure_output_dir(self, output_dir: str) -> None:
        """Ensure the output directory exists."""
        import os
        os.makedirs(output_dir, exist_ok=True)
