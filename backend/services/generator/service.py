"""
Document Generation Service - Modular Orchestrator

This module provides the new modular interface to document generation,
delegating to the appropriate format generators, outline generator,
and content generator.

Note: For backward compatibility, the main DocumentGenerationService
is still imported from the old generator.py file. This module provides
the new modular architecture that can be used alongside or as a replacement.
"""

from typing import Optional, List, TYPE_CHECKING

import structlog

from .outline import OutlineGenerator
from .content import ContentGenerator
from .formats import FormatGeneratorFactory
from .template_analyzer import TemplateAnalyzer

if TYPE_CHECKING:
    from .models import GenerationJob, Section, DocumentOutline, OutputFormat
    from .template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


class ModularGenerationService:
    """Modular document generation service.

    This service orchestrates document generation using the new modular
    architecture. It provides a cleaner interface than the monolithic
    DocumentGenerationService while maintaining backward compatibility.

    Usage:
        service = ModularGenerationService()

        # Analyze template
        analysis = service.analyze_template("template.pptx")

        # Generate outline with template context
        outline = await service.generate_outline(job, analysis)

        # Generate content
        sections = await service.generate_content(job, analysis)

        # Render to output format
        output_path = await service.render(job, analysis)
    """

    def __init__(self):
        self.template_analyzer = TemplateAnalyzer()
        self.outline_generator = OutlineGenerator()
        self.content_generator = ContentGenerator()

    def analyze_template(self, template_path: str) -> "TemplateAnalysis":
        """Analyze a template file and extract theme/constraints.

        Args:
            template_path: Path to the template file

        Returns:
            TemplateAnalysis with theme, layouts, and constraints
        """
        return self.template_analyzer.analyze(template_path)

    async def generate_outline(
        self,
        job: "GenerationJob",
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> "DocumentOutline":
        """Generate a document outline.

        Args:
            job: The generation job
            template_analysis: Optional template analysis for constraints

        Returns:
            DocumentOutline with sections
        """
        return await self.outline_generator.generate(job, template_analysis)

    async def generate_content(
        self,
        job: "GenerationJob",
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> List["Section"]:
        """Generate content for all sections.

        Args:
            job: The generation job with sections
            template_analysis: Optional template analysis for constraints

        Returns:
            List of sections with generated content
        """
        return await self.content_generator.generate_all(job, template_analysis)

    async def render(
        self,
        job: "GenerationJob",
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Render the document to the specified output format.

        Args:
            job: The generation job with content
            template_analysis: Optional template analysis for styling

        Returns:
            Path to the rendered document
        """
        generator = FormatGeneratorFactory.get(job.output_format)
        if not generator:
            raise ValueError(f"Unsupported output format: {job.output_format}")

        filename = self._generate_filename(job)
        return await generator.generate(job, filename, template_analysis)

    async def generate_full(
        self,
        job: "GenerationJob",
        template_path: Optional[str] = None,
    ) -> str:
        """Full generation pipeline: analyze, outline, content, render.

        This is a convenience method that runs the entire generation
        pipeline in one call.

        Args:
            job: The generation job
            template_path: Optional template path

        Returns:
            Path to the generated document
        """
        # Analyze template if provided
        template_analysis = None
        if template_path:
            template_analysis = self.analyze_template(template_path)

        # Generate outline if not already set
        if not job.outline:
            job.outline = await self.generate_outline(job, template_analysis)

        # Generate content
        await self.generate_content(job, template_analysis)

        # Render
        return await self.render(job, template_analysis)

    def _generate_filename(self, job: "GenerationJob") -> str:
        """Generate a filename for the output document."""
        import re
        from datetime import datetime

        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s-]', '', job.title)
        safe_title = re.sub(r'\s+', '_', safe_title)[:50]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get extension from format
        generator = FormatGeneratorFactory.get(job.output_format)
        ext = generator.file_extension if generator else f".{job.output_format.value}"

        return f"{safe_title}_{timestamp}{ext}"


# Singleton instance
_modular_service: Optional[ModularGenerationService] = None


def get_modular_generation_service() -> ModularGenerationService:
    """Get the singleton ModularGenerationService instance."""
    global _modular_service
    if _modular_service is None:
        _modular_service = ModularGenerationService()
    return _modular_service
