"""
Plain Text Format Generator

Generates plain text documents from generation jobs.
"""

import os
from typing import Optional, TYPE_CHECKING

import structlog

from ..base import BaseFormatGenerator
from ..factory import register_generator
from ...models import OutputFormat, GenerationConfig

if TYPE_CHECKING:
    from ...models import GenerationJob
    from ...template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


@register_generator(OutputFormat.TXT)
class TXTGenerator(BaseFormatGenerator):
    """Generator for plain text documents.

    Features:
    - Clean plain text output
    - ASCII-style formatting
    - Simple structure with headings
    """

    @property
    def format_name(self) -> str:
        return "txt"

    @property
    def file_extension(self) -> str:
        return ".txt"

    async def generate(
        self,
        job: "GenerationJob",
        filename: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Generate a plain text document.

        Args:
            job: The generation job containing metadata and sections
            filename: The output filename
            template_analysis: Optional template analysis (not used for plain text)

        Returns:
            Path to the generated text file
        """
        from ...utils import strip_markdown

        config = GenerationConfig()
        output_path = os.path.join(config.output_dir, filename)

        # Build plain text content
        lines = []
        lines.append("=" * 60)
        lines.append(job.title.upper())
        lines.append("=" * 60)
        lines.append("")

        # Use outline description (consistent with HTML/Markdown generators)
        if job.outline and job.outline.description:
            lines.append(strip_markdown(job.outline.description))
            lines.append("")

        lines.append("-" * 60)
        lines.append("TABLE OF CONTENTS")
        lines.append("-" * 60)
        for i, section in enumerate(job.sections, 1):
            lines.append(f"  {i}. {section.title}")
        lines.append("")

        for section_idx, section in enumerate(job.sections):
            # Validate title
            if not section.title or not section.title.strip():
                logger.warning(f"Section {section_idx + 1} has empty title, using default", section_idx=section_idx)
                section.title = f"Section {section_idx + 1}"

            lines.append("-" * 60)
            lines.append(section.title.upper())
            lines.append("-" * 60)
            lines.append("")

            # Get content, strip any markdown
            content = section.revised_content if section.revised_content and section.revised_content.strip() else section.content
            if not content or not content.strip():
                logger.warning("Section has empty content, adding placeholder",
                               title=section.title[:50] if section.title else "No title")
                content = "Content not available for this section."
            content = strip_markdown(content)
            lines.append(content)
            lines.append("")

        # Add references section
        include_sources = job.metadata.get("include_sources", True) if job.metadata else True
        if include_sources and job.sources_used:
            lines.append("=" * 60)
            lines.append("REFERENCES")
            lines.append("=" * 60)
            lines.append("")

            # Group sources by usage type
            from ...models import SourceUsageType
            content_sources = [s for s in job.sources_used if getattr(s, 'usage_type', SourceUsageType.CONTENT) == SourceUsageType.CONTENT]
            style_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) == SourceUsageType.STYLE]
            other_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) not in [SourceUsageType.CONTENT, SourceUsageType.STYLE, None]]

            def format_source(source):
                doc_name = source.document_name or source.document_id
                location_info = ""
                if source.page_number:
                    if doc_name.lower().endswith('.pptx'):
                        location_info = f" (Slide {source.page_number})"
                    else:
                        location_info = f" (Page {source.page_number})"
                usage_info = ""
                if hasattr(source, 'usage_description') and source.usage_description:
                    usage_info = f" - {source.usage_description}"
                return f"  * {doc_name}{location_info}{usage_info}"

            if content_sources:
                lines.append("Content References:")
                for source in content_sources:
                    lines.append(format_source(source))
                lines.append("")

            if style_sources:
                lines.append("Style References:")
                for source in style_sources:
                    lines.append(format_source(source))
                lines.append("")

            if other_sources:
                lines.append("Other References:")
                for source in other_sources:
                    lines.append(format_source(source))
                lines.append("")

            # Fallback if no categorized sources
            if not content_sources and not style_sources and not other_sources:
                for source in job.sources_used:
                    lines.append(format_source(source))
                lines.append("")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info("Plain text generated", path=output_path)
        return output_path
