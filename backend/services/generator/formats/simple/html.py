"""
HTML Format Generator

Generates HTML documents from generation jobs.
Full implementation - no delegation to old generator.py.
"""

import html as html_module
import os
import re
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


def _convert_inline_markdown(text: str) -> str:
    """Convert **bold** and *italic* markdown to HTML tags.

    Must be called AFTER html_module.escape() since the asterisks
    will survive escaping (they are not HTML special chars).
    """
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    return text


def _process_content_to_html(content: str) -> str:
    """Convert LLM-generated content to structured HTML with proper escaping.

    Handles: headings (##/###), bullet lists (-/•/*), bold/italic markdown,
    and regular prose paragraphs. All text is HTML-escaped first.
    """
    if not content:
        return ""

    paragraphs = content.split('\n\n')
    html_parts = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect headings
        if para.startswith('###'):
            text = html_module.escape(para.lstrip('#').strip())
            text = _convert_inline_markdown(text)
            html_parts.append(f'<h4>{text}</h4>')
        elif para.startswith('##'):
            text = html_module.escape(para.lstrip('#').strip())
            text = _convert_inline_markdown(text)
            html_parts.append(f'<h3>{text}</h3>')
        # Detect bullet lists
        elif any(line.strip().startswith(('-', '•', '*')) and len(line.strip()) > 2
                 for line in para.split('\n') if line.strip()):
            items = para.split('\n')
            html_parts.append('<ul>')
            for item in items:
                item_text = item.strip()
                if not item_text:
                    continue
                # Strip bullet prefix
                if item_text[:2] in ('- ', '• ', '* '):
                    item_text = item_text[2:]
                elif item_text[:3] in ('  -', '  •', '  *'):
                    item_text = item_text[3:].strip()
                item_text = html_module.escape(item_text)
                item_text = _convert_inline_markdown(item_text)
                html_parts.append(f'  <li>{item_text}</li>')
            html_parts.append('</ul>')
        # Regular paragraph
        else:
            text = html_module.escape(para.replace('\n', ' '))
            text = _convert_inline_markdown(text)
            html_parts.append(f'<p>{text}</p>')

    return '\n        '.join(html_parts)


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

        # Escape title for safe HTML output
        safe_title = html_module.escape(job.title)

        # Build HTML using list accumulation (10-50x faster than string += in loops)
        html_parts = [f"""<!DOCTYPE html>
<html>
<head>
    <title>{safe_title}</title>
    <style>
        body {{ font-family: {font_family}; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: {theme["primary"]}; margin-bottom: 0.5em; }}
        h2 {{ color: {theme["secondary"]}; border-bottom: 2px solid {theme["secondary"]}; padding-bottom: 0.3em; }}
        h3 {{ color: {theme["secondary"]}; margin-top: 1em; }}
        h4 {{ color: {theme["text"]}; margin-top: 0.8em; }}
        .section {{ margin-bottom: 30px; }}
        .section ul {{ padding-left: 1.5em; margin: 0.5em 0; }}
        .section li {{ margin-bottom: 0.3em; }}
        .description {{ color: {theme["secondary"]}; font-style: italic; margin-bottom: 2em; }}
        .sources {{ font-size: 0.9em; color: {theme["light_gray"]}; margin-top: 1em; padding-top: 0.5em; border-top: 1px dashed {theme["light_gray"]}; }}
        .references {{ margin-top: 3em; padding-top: 1em; border-top: 2px solid {theme["primary"]}; }}
        .references h2 {{ border-bottom: none; }}
        .references ul {{ list-style-type: disc; padding-left: 1.5em; }}
        .references li {{ color: {theme["text"]}; margin-bottom: 0.5em; }}
    </style>
</head>
<body>
    <h1>{safe_title}</h1>
"""]

        # Description
        if job.outline and job.outline.description:
            safe_desc = html_module.escape(job.outline.description)
            html_parts.append(f'    <p class="description">{safe_desc}</p>\n')

        # Content sections
        for section_idx, section in enumerate(job.sections):
            # Validate title
            if not section.title or not section.title.strip():
                logger.warning(f"Section {section_idx + 1} has empty title, using default", section_idx=section_idx)
                section.title = f"Section {section_idx + 1}"

            content = section.revised_content if section.revised_content and section.revised_content.strip() else section.content
            if not content or not content.strip():
                logger.warning("Section has empty content, adding placeholder",
                               title=section.title[:50] if section.title else "No title")
                content = "Content not available for this section."
            safe_section_title = html_module.escape(section.title)
            processed_content = _process_content_to_html(content)
            html_parts.append(f"""    <div class="section">
        <h2>{safe_section_title}</h2>
        {processed_content}
""")
            # Section sources
            if include_sources and section.sources:
                source_items = []
                for s in section.sources[:3]:
                    doc_name = html_module.escape(s.document_name or s.document_id)
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
                doc_name = html_module.escape(source.document_name or source.document_id)
                location_info = ""
                if source.page_number:
                    if (source.document_name or "").lower().endswith('.pptx'):
                        location_info = f" (Slide {source.page_number})"
                    else:
                        location_info = f" (Page {source.page_number})"
                usage_info = ""
                if hasattr(source, 'usage_description') and source.usage_description:
                    usage_info = f" &mdash; <em>{html_module.escape(source.usage_description)}</em>"

                # Add hyperlink if available (validate URL scheme)
                hyperlink = None
                if hasattr(source, 'document_url') and source.document_url:
                    url = source.document_url
                    if url.startswith(('http://', 'https://', 'file://')):
                        hyperlink = html_module.escape(url)
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
        output_path = os.path.join(config.output_dir, filename)
        if HAS_AIOFILES:
            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                await f.write(html)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        logger.info("HTML generated", path=output_path)
        return output_path
