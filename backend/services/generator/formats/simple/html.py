"""
HTML Format Generator

Generates HTML documents from generation jobs.
Full implementation - no delegation to old generator.py.
"""

import os
from typing import Optional, TYPE_CHECKING

import structlog

# Use aiofiles for async file I/O if available
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from ..base import BaseFormatGenerator
from ..factory import register_generator
from ...models import OutputFormat, GenerationJob
from ...config import THEMES

if TYPE_CHECKING:
    from ...template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


def get_theme_colors(theme_key: str = "business", custom_colors: dict = None) -> dict:
    """Get theme colors, with fallback to business theme."""
    theme = THEMES.get(theme_key, THEMES["business"]).copy()
    if custom_colors:
        for key in ["primary", "secondary", "accent", "text", "background"]:
            if key in custom_colors:
                theme[key] = custom_colors[key]
    return theme


@register_generator(OutputFormat.HTML)
class HTMLGenerator(BaseFormatGenerator):
    """Generator for HTML documents.

    Features:
    - Standalone HTML with embedded CSS
    - Responsive design
    - Theme-based styling
    - References section
    """

    @property
    def format_name(self) -> str:
        return "html"

    @property
    def file_extension(self) -> str:
        return ".html"

    async def generate(
        self,
        job: GenerationJob,
        filename: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Generate an HTML document.

        Args:
            job: The generation job containing metadata and sections
            filename: The output filename
            template_analysis: Optional template analysis for styling

        Returns:
            Path to the generated HTML file
        """
        from ...models import GenerationConfig
        config = GenerationConfig()

        # Determine include_sources
        include_sources = job.metadata.get("include_sources", config.include_sources)

        # Get theme colors
        theme_key = job.metadata.get("theme", "business")
        custom_colors = job.metadata.get("custom_colors")
        theme = get_theme_colors(theme_key, custom_colors)

        # Font family mapping
        HTML_FONT_MAP = {
            "modern": "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
            "classic": "Georgia, 'Times New Roman', Times, serif",
            "professional": "Arial, Helvetica, sans-serif",
            "technical": "'Courier New', Consolas, monospace",
        }
        font_family_key = job.metadata.get("font_family", "modern")
        font_family = HTML_FONT_MAP.get(font_family_key, HTML_FONT_MAP["modern"])

        # Build HTML using list accumulation (10-50x faster than string += in loops)
        html_parts = [f"""<!DOCTYPE html>
<html>
<head>
    <title>{job.title}</title>
    <style>
        body {{ font-family: {font_family}; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: {theme["primary"]}; margin-bottom: 0.5em; }}
        h2 {{ color: {theme["secondary"]}; border-bottom: 2px solid {theme["secondary"]}; padding-bottom: 0.3em; }}
        .section {{ margin-bottom: 30px; }}
        .description {{ color: {theme["secondary"]}; font-style: italic; margin-bottom: 2em; }}
        .sources {{ font-size: 0.9em; color: {theme["light_gray"]}; margin-top: 1em; padding-top: 0.5em; border-top: 1px dashed {theme["light_gray"]}; }}
        .references {{ margin-top: 3em; padding-top: 1em; border-top: 2px solid {theme["primary"]}; }}
        .references h2 {{ border-bottom: none; }}
        .references ul {{ list-style-type: disc; padding-left: 1.5em; }}
        .references li {{ color: {theme["text"]}; margin-bottom: 0.5em; }}
    </style>
</head>
<body>
    <h1>{job.title}</h1>
"""]

        # Description
        if job.outline:
            html_parts.append(f'    <p class="description">{job.outline.description}</p>\n')

        # Content sections
        for section in job.sections:
            content = section.revised_content or section.content
            html_parts.append(f"""    <div class="section">
        <h2>{section.title}</h2>
        <p>{content.replace(chr(10), '</p><p>')}</p>
""")
            # Section sources
            if include_sources and section.sources:
                source_items = []
                for s in section.sources[:3]:
                    doc_name = s.document_name or s.document_id
                    if s.page_number:
                        if doc_name.lower().endswith('.pptx'):
                            source_items.append(f"{doc_name} (Slide {s.page_number})")
                        else:
                            source_items.append(f"{doc_name} (Page {s.page_number})")
                    else:
                        source_items.append(doc_name)
                html_parts.append(f'        <div class="sources">Sources: {", ".join(source_items)}</div>\n')

            html_parts.append("    </div>\n")

        # References section
        if include_sources and job.sources_used:
            html_parts.append('    <div class="references">\n')
            html_parts.append("        <h2>References</h2>\n")

            # Group sources by usage type
            from ...models import SourceUsageType
            content_sources = [s for s in job.sources_used if getattr(s, 'usage_type', SourceUsageType.CONTENT) == SourceUsageType.CONTENT]
            style_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) == SourceUsageType.STYLE]
            other_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) not in [SourceUsageType.CONTENT, SourceUsageType.STYLE, None]]

            def format_source_html(source):
                doc_name = source.document_name or source.document_id
                location_info = ""
                if source.page_number:
                    if doc_name.lower().endswith('.pptx'):
                        location_info = f" (Slide {source.page_number})"
                    else:
                        location_info = f" (Page {source.page_number})"
                usage_info = ""
                if hasattr(source, 'usage_description') and source.usage_description:
                    usage_info = f" &mdash; <em>{source.usage_description}</em>"

                # Add hyperlink if available
                hyperlink = None
                if hasattr(source, 'document_url') and source.document_url:
                    hyperlink = source.document_url
                elif hasattr(source, 'document_path') and source.document_path:
                    import urllib.parse
                    hyperlink = f"file://{urllib.parse.quote(source.document_path)}"

                if hyperlink:
                    return f'<a href="{hyperlink}">{doc_name}</a>{location_info}{usage_info}'
                return f"{doc_name}{location_info}{usage_info}"

            if content_sources:
                html_parts.append("        <h3>Content References</h3>\n        <ul>\n")
                html_parts.extend(f"            <li>{format_source_html(source)}</li>\n" for source in content_sources)
                html_parts.append("        </ul>\n")

            if style_sources:
                html_parts.append("        <h3>Style References</h3>\n        <ul>\n")
                html_parts.extend(f"            <li>{format_source_html(source)}</li>\n" for source in style_sources)
                html_parts.append("        </ul>\n")

            if other_sources:
                html_parts.append("        <h3>Other References</h3>\n        <ul>\n")
                html_parts.extend(f"            <li>{format_source_html(source)}</li>\n" for source in other_sources)
                html_parts.append("        </ul>\n")

            # Fallback if no categorized sources
            if not content_sources and not style_sources and not other_sources:
                html_parts.append("        <ul>\n")
                html_parts.extend(f"            <li>{format_source_html(source)}</li>\n" for source in job.sources_used)
                html_parts.append("        </ul>\n")

            html_parts.append("    </div>\n")

        html_parts.append("""</body>
</html>""")

        # Join all parts at once (much faster than repeated += )
        html = ''.join(html_parts)

        # Save file (use async I/O if available to avoid blocking)
        output_path = os.path.join(config.output_dir, f"{filename}.html")
        if HAS_AIOFILES:
            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                await f.write(html)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        logger.info("HTML generated", path=output_path)
        return output_path
