"""
PDF Format Generator

Full implementation of PDF document generation using ReportLab.
Migrated from generator.py for modularity.
"""

import os
import re
from typing import Optional, TYPE_CHECKING

import structlog

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


@register_generator(OutputFormat.PDF)
class PDFGenerator(BaseFormatGenerator):
    """Full implementation of PDF document generator.

    Features:
    - ReportLab-based generation
    - Professional styling with multiple themes
    - Cover page with title and description
    - Table of contents with hyperlinks
    - Page headers and footers with page numbers
    - Image and chart support
    - References section
    """

    @property
    def format_name(self) -> str:
        return "pdf"

    @property
    def file_extension(self) -> str:
        return ".pdf"

    async def generate(
        self,
        job: GenerationJob,
        filename: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Generate a PDF document.

        Args:
            job: The generation job containing metadata and sections
            filename: The output filename
            template_analysis: Optional template analysis for styling

        Returns:
            Path to the generated PDF file
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                Table, TableStyle, ListFlowable, ListItem, Image
            )
            from reportlab.lib.units import inch

            # Get output directory from config
            from ...models import GenerationConfig
            config = GenerationConfig()

            # Determine include_sources
            include_sources = job.metadata.get("include_sources", config.include_sources)

            # Image generation
            include_images = job.metadata.get("include_images", config.include_images)
            section_images = {}

            if include_images:
                try:
                    from backend.services.image_generator import ImageGeneratorConfig, ImageBackend, get_image_generator

                    image_config = ImageGeneratorConfig(
                        enabled=True,
                        backend=ImageBackend(config.image_backend),
                        default_width=600,
                        default_height=400,
                    )
                    image_service = get_image_generator(image_config)

                    sections_data = [
                        (section.title, section.revised_content if section.revised_content and section.revised_content.strip() else section.content)
                        for section in job.sections
                    ]
                    images = await image_service.generate_batch(
                        sections=sections_data,
                        document_title=job.title,
                    )

                    for idx, img in enumerate(images):
                        if img and img.success and img.path:
                            section_images[idx] = img.path

                except Exception as e:
                    logger.warning(f"Image generation failed: {e}")

            # Get theme colors
            theme_key = job.metadata.get("theme", "business")
            custom_colors = job.metadata.get("custom_colors")
            theme = get_theme_colors(theme_key, custom_colors)

            # PDF font mapping
            PDF_FONT_MAP = {
                "modern": {
                    "heading": "Helvetica-Bold",
                    "heading_oblique": "Helvetica-BoldOblique",
                    "body": "Helvetica",
                    "body_bold": "Helvetica-Bold",
                    "body_italic": "Helvetica-Oblique",
                },
                "classic": {
                    "heading": "Times-Bold",
                    "heading_oblique": "Times-BoldItalic",
                    "body": "Times-Roman",
                    "body_bold": "Times-Bold",
                    "body_italic": "Times-Italic",
                },
                "professional": {
                    "heading": "Helvetica-Bold",
                    "heading_oblique": "Helvetica-BoldOblique",
                    "body": "Helvetica",
                    "body_bold": "Helvetica-Bold",
                    "body_italic": "Helvetica-Oblique",
                },
                "technical": {
                    "heading": "Courier-Bold",
                    "heading_oblique": "Courier-BoldOblique",
                    "body": "Courier",
                    "body_bold": "Courier-Bold",
                    "body_italic": "Courier-Oblique",
                },
            }

            font_family_key = job.metadata.get("font_family", "modern")
            pdf_fonts = PDF_FONT_MAP.get(font_family_key, PDF_FONT_MAP["modern"])
            heading_font = pdf_fonts["heading"]
            body_font = pdf_fonts["body"]
            body_italic = pdf_fonts["body_italic"]

            # Apply theme colors
            PRIMARY_COLOR = HexColor(theme["primary"])
            SECONDARY_COLOR = HexColor(theme["secondary"])
            TEXT_COLOR = HexColor(theme["text"])
            LIGHT_GRAY = HexColor(theme["light_gray"])

            output_path = os.path.join(config.output_dir, filename)

            # Page number callback
            def add_page_number(canvas, doc, font=body_font):
                canvas.saveState()
                canvas.setFont(font, 9)
                canvas.setFillColor(LIGHT_GRAY)
                page_num = canvas.getPageNumber()
                text = f"Page {page_num}"
                canvas.drawRightString(letter[0] - 0.75*inch, 0.5*inch, text)
                canvas.restoreState()

            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch,
            )

            # Create styles
            styles = getSampleStyleSheet()

            cover_title_style = ParagraphStyle(
                'CoverTitle',
                parent=styles['Title'],
                fontSize=36,
                textColor=PRIMARY_COLOR,
                alignment=TA_CENTER,
                spaceAfter=20,
                fontName=heading_font,
            )

            cover_subtitle_style = ParagraphStyle(
                'CoverSubtitle',
                parent=styles['Normal'],
                fontSize=14,
                textColor=SECONDARY_COLOR,
                alignment=TA_CENTER,
                fontName=body_italic,
                spaceAfter=12,
            )

            heading_style = ParagraphStyle(
                'SectionHeading',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=PRIMARY_COLOR,
                spaceBefore=16,
                spaceAfter=12,
                fontName=heading_font,
            )

            subheading_style = ParagraphStyle(
                'SubHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=SECONDARY_COLOR,
                spaceBefore=12,
                spaceAfter=8,
                fontName=pdf_fonts["body_bold"],
            )

            body_style = ParagraphStyle(
                'BodyText',
                parent=styles['Normal'],
                fontSize=11,
                textColor=TEXT_COLOR,
                alignment=TA_JUSTIFY,
                spaceBefore=4,
                spaceAfter=8,
                leading=16,
                fontName=body_font,
            )

            toc_style = ParagraphStyle(
                'TOCEntry',
                parent=styles['Normal'],
                fontSize=12,
                textColor=TEXT_COLOR,
                spaceBefore=6,
                spaceAfter=6,
                leftIndent=20,
                fontName=body_font,
            )

            source_style = ParagraphStyle(
                'SourceStyle',
                parent=styles['Normal'],
                fontSize=9,
                textColor=LIGHT_GRAY,
                fontName=body_italic,
                spaceBefore=4,
                spaceAfter=2,
            )

            story = []

            # ========== COVER PAGE ==========
            story.append(Spacer(1, 2.5*inch))
            story.append(Paragraph(job.title, cover_title_style))
            story.append(Spacer(1, 0.3*inch))

            # Decorative line
            line_data = [['─' * 50]]
            line_table = Table(line_data, colWidths=[5*inch])
            line_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TEXTCOLOR', (0, 0), (-1, -1), SECONDARY_COLOR),
            ]))
            story.append(line_table)
            story.append(Spacer(1, 0.3*inch))

            # Description
            if job.outline and job.outline.description:
                story.append(Paragraph(job.outline.description, cover_subtitle_style))

            story.append(Spacer(1, 2*inch))

            # Date
            from datetime import datetime
            date_style = ParagraphStyle(
                'DateStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=LIGHT_GRAY,
                alignment=TA_CENTER,
                fontName=body_font,
            )
            story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), date_style))
            story.append(PageBreak())

            # ========== TABLE OF CONTENTS ==========
            if config.include_toc:
                toc_title_style = ParagraphStyle(
                    'TOCTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=PRIMARY_COLOR,
                    spaceAfter=20,
                    fontName=heading_font,
                )
                story.append(Paragraph("Table of Contents", toc_title_style))
                story.append(Spacer(1, 0.3*inch))

                toc_link_style = ParagraphStyle(
                    'TOCLink',
                    parent=toc_style,
                    textColor=HexColor('#3D5A80'),
                )

                for idx, section in enumerate(job.sections):
                    section_title = section.title
                    roman_pattern = r'^[IVXLCDM]+\.\s+'
                    if re.match(roman_pattern, section_title):
                        section_title = re.sub(roman_pattern, '', section_title)

                    toc_entry = f'<b>{idx + 1}.</b>  <a href="#section_{idx + 1}" color="#3D5A80">{section_title}</a>'
                    story.append(Paragraph(toc_entry, toc_link_style))

                story.append(PageBreak())

            # Auto charts
            auto_charts_enabled = job.metadata.get("auto_charts", config.auto_charts)

            # ========== CONTENT SECTIONS ==========
            for idx, section in enumerate(job.sections):
                # Validate title
                if not section.title or not section.title.strip():
                    logger.warning(f"Section {idx + 1} has empty title, using default", section_idx=idx)
                    section.title = f"Section {idx + 1}"

                # Section heading with bookmark anchor
                heading_text = f'<a name="section_{idx + 1}"/>{idx + 1}. {section.title}'
                story.append(Paragraph(heading_text, heading_style))

                content = section.revised_content if section.revised_content and section.revised_content.strip() else section.content
                if not content or not content.strip():
                    logger.warning("Section has empty content, adding placeholder",
                                   section_idx=idx, title=section.title[:50] if section.title else "No title")
                    content = "Content not available for this section."

                # Chart detection
                rendered_as_chart = False
                if auto_charts_enabled:
                    try:
                        from backend.services.pptx_chart_generator import PPTXNativeChartGenerator
                        from backend.services.chart_generator import ChartGenerator, ChartType as MatplotlibChartType, ChartData, ChartStyle

                        chart_data = PPTXNativeChartGenerator.detect_chartable_data(content)
                        if chart_data and len(chart_data.categories) >= 3:
                            suggested_type = PPTXNativeChartGenerator.suggest_chart_type(chart_data)

                            type_mapping = {
                                "column": MatplotlibChartType.BAR,
                                "bar": MatplotlibChartType.HORIZONTAL_BAR,
                                "line": MatplotlibChartType.LINE,
                                "line_markers": MatplotlibChartType.LINE,
                                "pie": MatplotlibChartType.PIE,
                                "doughnut": MatplotlibChartType.PIE,
                                "area": MatplotlibChartType.AREA,
                                "scatter": MatplotlibChartType.SCATTER,
                                "stacked_column": MatplotlibChartType.STACKED_BAR,
                                "stacked_bar": MatplotlibChartType.STACKED_BAR,
                                "radar": MatplotlibChartType.BAR,
                            }
                            mpl_chart_type = type_mapping.get(suggested_type.value, MatplotlibChartType.BAR)

                            chart_gen = ChartGenerator()
                            mpl_data = ChartData(
                                labels=chart_data.categories,
                                values=chart_data.series[0].values,
                                series_name=chart_data.series[0].name if chart_data.series else "Data",
                            )

                            for series in chart_data.series[1:]:
                                mpl_data.additional_series[series.name] = series.values

                            chart_style = ChartStyle(
                                title=section.title or "",
                                primary_color=theme["primary"],
                                secondary_color=theme["secondary"],
                                accent_color=theme.get("accent", "#E0E1DD"),
                                figsize=(7, 4),
                                dpi=150,
                            )

                            generated_chart = chart_gen.generate_chart(mpl_chart_type, mpl_data, chart_style)

                            if generated_chart and generated_chart.image_bytes:
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                    tmp.write(generated_chart.image_bytes)
                                    tmp_path = tmp.name

                                story.append(Spacer(1, 0.2*inch))
                                chart_img = Image(tmp_path, width=5.5*inch, height=3.5*inch)
                                story.append(chart_img)
                                story.append(Spacer(1, 0.2*inch))

                                rendered_as_chart = True

                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass

                    except Exception as e:
                        logger.warning(f"Chart detection failed: {e}")

                # Parse content
                paragraphs = content.split('\n')
                current_list = []

                def flush_list(lst):
                    if lst:
                        story.append(ListFlowable(
                            lst,
                            bulletType='bullet',
                            leftIndent=20,
                        ))
                    return []

                for para_text in paragraphs:
                    original_text = para_text
                    para_text = para_text.strip()
                    if not para_text:
                        current_list = flush_list(current_list)
                        continue

                    if para_text.startswith('###'):
                        current_list = flush_list(current_list)
                        story.append(Paragraph(para_text.lstrip('#').strip(), subheading_style))
                    elif para_text.startswith('##'):
                        current_list = flush_list(current_list)
                        story.append(Paragraph(para_text.lstrip('#').strip(), subheading_style))
                    elif para_text.startswith(('- ', '• ', '* ')):
                        bullet_text = para_text.lstrip('-•* ').strip()
                        indent = len(original_text) - len(original_text.lstrip())
                        level = min(indent // 2, 3)
                        left_indent = 20 + (level * 15)

                        indented_style = ParagraphStyle(
                            f'BulletLevel{level}',
                            parent=body_style,
                            leftIndent=left_indent,
                            bulletIndent=left_indent - 10,
                        )
                        current_list.append(ListItem(Paragraph(bullet_text, indented_style), leftIndent=left_indent))
                    elif para_text[:2].replace('.', '').isdigit():
                        current_list = flush_list(current_list)
                        num_match = re.match(r'^(\d+\.)\s*(.+)', para_text)
                        if num_match:
                            story.append(Paragraph(f"<b>{num_match.group(1)}</b> {num_match.group(2)}", body_style))
                        else:
                            story.append(Paragraph(para_text, body_style))
                    else:
                        current_list = flush_list(current_list)
                        formatted = para_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        formatted = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', formatted)
                        formatted = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', formatted)
                        story.append(Paragraph(formatted, body_style))

                current_list = flush_list(current_list)

                # Add image
                if idx in section_images:
                    try:
                        image_path = section_images[idx]
                        story.append(Spacer(1, 0.2*inch))
                        img = Image(image_path, width=5*inch, height=3.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        logger.warning(f"Failed to add image: {e}")

                # Section sources
                if include_sources and section.sources:
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph("Sources for this section:", source_style))
                    for source in section.sources[:3]:
                        doc_name = source.document_name or source.document_id
                        location_info = ""
                        if source.page_number:
                            if doc_name.lower().endswith('.pptx'):
                                location_info = f" (Slide {source.page_number})"
                            else:
                                location_info = f" (Page {source.page_number})"
                        story.append(Paragraph(f"• {doc_name}{location_info}", source_style))

                # Page break
                if idx < len(job.sections) - 1:
                    story.append(PageBreak())
                else:
                    story.append(Spacer(1, 0.3*inch))

            # ========== REFERENCES ==========
            if include_sources and job.sources_used:
                story.append(PageBreak())
                story.append(Paragraph("References", heading_style))
                story.append(Spacer(1, 0.2*inch))

                # Group sources by usage type
                from ...models import SourceUsageType
                content_sources = [s for s in job.sources_used if getattr(s, 'usage_type', SourceUsageType.CONTENT) == SourceUsageType.CONTENT]
                style_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) == SourceUsageType.STYLE]
                other_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) not in [SourceUsageType.CONTENT, SourceUsageType.STYLE, None]]

                def format_source_paragraph(source):
                    """Format a source reference paragraph."""
                    doc_name = source.document_name or source.document_id
                    location_info = ""
                    if source.page_number:
                        if doc_name.lower().endswith('.pptx'):
                            location_info = f" (Slide {source.page_number})"
                        else:
                            location_info = f" (Page {source.page_number})"

                    # Add usage description if available (escape for ReportLab markup)
                    usage_info = ""
                    if hasattr(source, 'usage_description') and source.usage_description:
                        safe_desc = source.usage_description.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        usage_info = f" — {safe_desc}"

                    # Check for hyperlink (validate URL scheme)
                    hyperlink_url = None
                    if hasattr(source, 'document_url') and source.document_url:
                        url = source.document_url
                        if url.startswith(('http://', 'https://', 'file://')):
                            hyperlink_url = url
                    elif hasattr(source, 'document_path') and source.document_path:
                        import urllib.parse
                        hyperlink_url = f"file://{urllib.parse.quote(source.document_path)}"

                    # Escape doc_name for ReportLab markup
                    safe_doc_name = doc_name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    ref_text = f"• {safe_doc_name}{location_info}{usage_info}"

                    # Add hyperlink if available (PDF supports hyperlinks via <link> tag)
                    if hyperlink_url:
                        ref_text = f'• <link href="{hyperlink_url}"><u>{safe_doc_name}</u></link>{location_info}{usage_info}'

                    ref_style = ParagraphStyle(
                        'RefStyle',
                        parent=styles['Normal'],
                        fontSize=10,
                        textColor=TEXT_COLOR,
                        spaceBefore=4,
                        spaceAfter=4,
                        leftIndent=15,
                        fontName=body_font,
                    )
                    return Paragraph(ref_text, ref_style)

                def add_category_heading(text):
                    """Add a category subheading."""
                    subheading_style = ParagraphStyle(
                        'SubHeading',
                        parent=styles['Normal'],
                        fontSize=12,
                        textColor=PRIMARY_COLOR,
                        spaceBefore=12,
                        spaceAfter=6,
                        fontName=heading_font,
                    )
                    story.append(Paragraph(f"<b>{text}</b>", subheading_style))

                # Add content sources
                if content_sources:
                    add_category_heading("Content References")
                    for source in content_sources:
                        story.append(format_source_paragraph(source))

                # Add style sources
                if style_sources:
                    add_category_heading("Style References")
                    for source in style_sources:
                        story.append(format_source_paragraph(source))

                # Add other sources
                if other_sources:
                    add_category_heading("Other References")
                    for source in other_sources:
                        story.append(format_source_paragraph(source))

                # Fallback if no categorized sources
                if not content_sources and not style_sources and not other_sources:
                    for source in job.sources_used:
                        story.append(format_source_paragraph(source))

            # Build PDF
            doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

            logger.info("PDF generated", path=output_path)
            return output_path

        except ImportError as e:
            logger.error(f"reportlab import error: {e}")
            raise
