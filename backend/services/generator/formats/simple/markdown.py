"""
Markdown Format Generator

Generates Markdown documents from generation jobs.
Full implementation - no delegation to old generator.py.
"""

import os
import re
from typing import Optional, TYPE_CHECKING

import structlog

from ..base import BaseFormatGenerator
from ..factory import register_generator
from ...models import OutputFormat, GenerationJob

if TYPE_CHECKING:
    from ...template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


def _normalize_markdown_content(content: str) -> str:
    """Normalize LLM output to proper Markdown syntax.

    Converts non-standard bullet characters (•, ◦) to standard
    Markdown list markers (- ) and ensures proper indentation.
    """
    if not content:
        return ""

    lines = content.split('\n')
    normalized = []
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Convert • bullets to - bullets
        if stripped.startswith('\u2022 '):
            normalized.append(' ' * indent + '- ' + stripped[2:])
        # Convert ◦ sub-bullets to indented - bullets
        elif stripped.startswith('\u25e6 '):
            normalized.append('  - ' + stripped[2:])
        else:
            normalized.append(line)

    return '\n'.join(normalized)


@register_generator(OutputFormat.MARKDOWN)
class MarkdownGenerator(BaseFormatGenerator):
    """Generator for Markdown documents.

    Features:
    - GitHub-flavored Markdown
    - Table of contents with anchors
    - Code blocks
    - Section sources
    - References section
    """

    @property
    def format_name(self) -> str:
        return "markdown"

    @property
    def file_extension(self) -> str:
        return ".md"

    async def generate(
        self,
        job: GenerationJob,
        filename: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Generate a Markdown document.

        Args:
            job: The generation job containing metadata and sections
            filename: The output filename
            template_analysis: Optional template analysis (not used for Markdown)

        Returns:
            Path to the generated Markdown file
        """
        from ...models import GenerationConfig
        config = GenerationConfig()

        # Determine include_sources
        include_sources = job.metadata.get("include_sources", config.include_sources)

        lines = []

        # Title
        lines.append(f"# {job.title}")
        lines.append("")

        # Description
        if job.outline and job.outline.description:
            lines.append(job.outline.description)
            lines.append("")

        # Table of contents
        if config.include_toc:
            lines.append("## Table of Contents")
            lines.append("")
            for section in job.sections:
                # GitHub-flavored Markdown anchor: lowercase, strip non-word chars, spaces to dashes
                anchor = re.sub(r'[^\w\s-]', '', section.title.lower()).replace(" ", "-")
                lines.append(f"- [{section.title}](#{anchor})")
            lines.append("")

        # Content sections
        for section_idx, section in enumerate(job.sections):
            # Validate title
            if not section.title or not section.title.strip():
                logger.warning(f"Section {section_idx + 1} has empty title, using default", section_idx=section_idx)
                section.title = f"Section {section_idx + 1}"

            lines.append(f"## {section.title}")
            lines.append("")
            content = section.revised_content if section.revised_content and section.revised_content.strip() else section.content
            if not content or not content.strip():
                logger.warning("Section has empty content, adding placeholder",
                               title=section.title[:50] if section.title else "No title")
                content = "Content not available for this section."
            content = _normalize_markdown_content(content)
            lines.append(content)
            lines.append("")

            # Section sources
            if include_sources and section.sources:
                lines.append("**Sources:**")
                for source in section.sources[:3]:
                    doc_name = source.document_name or source.document_id
                    location_info = ""
                    if source.page_number:
                        if doc_name.lower().endswith('.pptx'):
                            location_info = f" (Slide {source.page_number})"
                        else:
                            location_info = f" (Page {source.page_number})"
                    lines.append(f"- {doc_name}{location_info}")
                lines.append("")

        # References section
        if include_sources and job.sources_used:
            lines.append("## References")
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
                    usage_info = f" — *{source.usage_description}*"

                # Add hyperlink if available
                hyperlink = None
                if hasattr(source, 'document_url') and source.document_url:
                    hyperlink = source.document_url
                elif hasattr(source, 'document_path') and source.document_path:
                    import urllib.parse
                    hyperlink = f"file://{urllib.parse.quote(source.document_path)}"

                if hyperlink:
                    return f"- [{doc_name}]({hyperlink}){location_info}{usage_info}"
                return f"- {doc_name}{location_info}{usage_info}"

            if content_sources:
                lines.append("### Content References")
                for source in content_sources:
                    lines.append(format_source(source))
                lines.append("")

            if style_sources:
                lines.append("### Style References")
                for source in style_sources:
                    lines.append(format_source(source))
                lines.append("")

            if other_sources:
                lines.append("### Other References")
                for source in other_sources:
                    lines.append(format_source(source))
                lines.append("")

            # Fallback if no categorized sources
            if not content_sources and not style_sources and not other_sources:
                for source in job.sources_used:
                    lines.append(format_source(source))

        # Save file
        output_path = os.path.join(config.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Markdown generated", path=output_path)
        return output_path
