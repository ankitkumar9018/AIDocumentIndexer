"""
DOCX Format Generator

Full implementation of Word document generation.
Migrated from generator.py for modularity.
"""

import os
import re
from typing import Optional, TYPE_CHECKING

import structlog

from ..base import BaseFormatGenerator
from ..factory import register_generator
from ...models import OutputFormat, GenerationJob
from ...config import THEMES, FONT_FAMILIES

if TYPE_CHECKING:
    from ...template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_theme_colors(theme_key: str = "business", custom_colors: dict = None) -> dict:
    """Get theme colors, with fallback to business theme."""
    theme = THEMES.get(theme_key, THEMES["business"]).copy()
    if custom_colors:
        for key in ["primary", "secondary", "accent", "text", "background"]:
            if key in custom_colors:
                theme[key] = custom_colors[key]
    return theme


def strip_markdown(text: str) -> str:
    """Remove markdown formatting, returning clean text."""
    if not text:
        return ""
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text


@register_generator(OutputFormat.DOCX)
class DOCXGenerator(BaseFormatGenerator):
    """Full implementation of Word document generator.

    Features:
    - Template support with style preservation
    - Cover page with title and description
    - Table of contents with hyperlinks
    - Professional styling with themes
    - Image support
    - Chart auto-detection
    - References section
    """

    @property
    def format_name(self) -> str:
        return "docx"

    @property
    def file_extension(self) -> str:
        return ".docx"

    async def generate(
        self,
        job: GenerationJob,
        filename: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Generate a Word document.

        Args:
            job: The generation job containing metadata and sections
            filename: The output filename
            template_analysis: Optional template analysis for styling

        Returns:
            Path to the generated DOCX file
        """
        try:
            from docx import Document
            from docx.shared import Pt, Inches, RGBColor
            from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
            from docx.oxml.ns import qn
            from docx.oxml import OxmlElement

            # Get output directory from config
            from ...models import GenerationConfig
            config = GenerationConfig()

            # Check if using a template DOCX
            template_path = job.metadata.get("template_docx_path")
            use_template_styling = False

            if template_path and os.path.exists(template_path):
                doc = Document(template_path)
                use_template_styling = True
                # Remove all existing content from template
                for element in doc.element.body:
                    doc.element.body.remove(element)
                logger.info("Using DOCX template", template_path=template_path)
            else:
                doc = Document()

            # Get theme colors
            theme_key = job.metadata.get("theme", "business")
            custom_colors = job.metadata.get("custom_colors")
            theme = get_theme_colors(theme_key, custom_colors)

            # Get font family
            font_family_key = job.metadata.get("font_family", "modern")
            font_config = FONT_FAMILIES.get(font_family_key, FONT_FAMILIES["modern"])
            heading_font = font_config["heading"]
            body_font = font_config["body"]

            # Apply theme color scheme
            primary_rgb = hex_to_rgb(theme["primary"])
            secondary_rgb = hex_to_rgb(theme["secondary"])
            text_rgb = hex_to_rgb(theme["text"])
            light_gray_rgb = hex_to_rgb(theme["light_gray"])

            PRIMARY_COLOR = RGBColor(*primary_rgb)
            SECONDARY_COLOR = RGBColor(*secondary_rgb)
            TEXT_COLOR = RGBColor(*text_rgb)
            LIGHT_GRAY = RGBColor(*light_gray_rgb)

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

            # Configure margins
            if not use_template_styling:
                for section in doc.sections:
                    section.top_margin = Inches(1)
                    section.bottom_margin = Inches(1)
                    section.left_margin = Inches(1.25)
                    section.right_margin = Inches(1.25)

            # ========== COVER PAGE ==========
            for _ in range(8):
                doc.add_paragraph()

            # Title
            title_para = doc.add_paragraph()
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title_para.add_run(job.title)
            if use_template_styling:
                try:
                    title_para.style = doc.styles['Title']
                except KeyError:
                    title_run.font.name = heading_font
                    title_run.font.size = Pt(36)
                    title_run.font.bold = True
                    title_run.font.color.rgb = PRIMARY_COLOR
            else:
                title_run.font.name = heading_font
                title_run.font.size = Pt(36)
                title_run.font.bold = True
                title_run.font.color.rgb = PRIMARY_COLOR

            # Subtitle
            doc.add_paragraph()
            if job.outline and job.outline.description:
                desc_para = doc.add_paragraph()
                desc_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                desc_run = desc_para.add_run(job.outline.description)
                if use_template_styling:
                    try:
                        desc_para.style = doc.styles['Subtitle']
                    except KeyError:
                        desc_run.font.name = body_font
                        desc_run.font.size = Pt(14)
                        desc_run.font.color.rgb = SECONDARY_COLOR
                        desc_run.font.italic = True
                else:
                    desc_run.font.name = body_font
                    desc_run.font.size = Pt(14)
                    desc_run.font.color.rgb = SECONDARY_COLOR
                    desc_run.font.italic = True

            # Spacing
            for _ in range(6):
                doc.add_paragraph()

            # Decorative line
            if not use_template_styling:
                line_para = doc.add_paragraph()
                line_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                line_run = line_para.add_run("─" * 40)
                line_run.font.color.rgb = SECONDARY_COLOR

            # Date
            from datetime import datetime
            doc.add_paragraph()
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_run = date_para.add_run(datetime.now().strftime("%B %d, %Y"))
            date_run.font.name = body_font
            date_run.font.size = Pt(12)
            date_run.font.color.rgb = LIGHT_GRAY

            doc.add_page_break()

            # ========== TABLE OF CONTENTS ==========
            def add_bookmark_hyperlink(paragraph, bookmark_name: str, text: str, font_name: str, font_size, font_color, is_bold: bool = False):
                """Add a hyperlink to a bookmark within the document."""
                hyperlink = OxmlElement('w:hyperlink')
                hyperlink.set(qn('w:anchor'), bookmark_name)

                new_run = OxmlElement('w:r')
                rPr = OxmlElement('w:rPr')

                rFonts = OxmlElement('w:rFonts')
                rFonts.set(qn('w:ascii'), font_name)
                rFonts.set(qn('w:hAnsi'), font_name)
                rPr.append(rFonts)

                sz = OxmlElement('w:sz')
                sz.set(qn('w:val'), str(int(font_size.pt * 2)))
                rPr.append(sz)

                color_elem = OxmlElement('w:color')
                color_hex = f"{font_color[0]:02X}{font_color[1]:02X}{font_color[2]:02X}"
                color_elem.set(qn('w:val'), color_hex)
                rPr.append(color_elem)

                if is_bold:
                    b = OxmlElement('w:b')
                    rPr.append(b)

                u = OxmlElement('w:u')
                u.set(qn('w:val'), 'single')
                rPr.append(u)

                new_run.append(rPr)

                text_elem = OxmlElement('w:t')
                text_elem.text = text
                new_run.append(text_elem)

                hyperlink.append(new_run)
                paragraph._p.append(hyperlink)

            def add_bookmark_to_paragraph(paragraph, bookmark_name: str):
                """Add a bookmark anchor to a paragraph."""
                bookmark_start = OxmlElement('w:bookmarkStart')
                bookmark_start.set(qn('w:id'), str(hash(bookmark_name) % 10000))
                bookmark_start.set(qn('w:name'), bookmark_name)

                bookmark_end = OxmlElement('w:bookmarkEnd')
                bookmark_end.set(qn('w:id'), str(hash(bookmark_name) % 10000))

                paragraph._p.insert(0, bookmark_start)
                paragraph._p.append(bookmark_end)

            if config.include_toc:
                toc_heading = doc.add_heading("Table of Contents", level=1)
                toc_heading.runs[0].font.color.rgb = PRIMARY_COLOR
                toc_heading.runs[0].font.size = Pt(24)

                doc.add_paragraph()

                for idx, section in enumerate(job.sections):
                    toc_entry = doc.add_paragraph()
                    toc_entry.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    toc_entry.paragraph_format.space_after = Pt(6)

                    num_run = toc_entry.add_run(f"{idx + 1}.  ")
                    num_run.font.name = body_font
                    num_run.font.size = Pt(12)
                    num_run.font.bold = True
                    num_run.font.color.rgb = SECONDARY_COLOR

                    section_title = section.title
                    roman_pattern = r'^[IVXLCDM]+\.\s+'
                    if re.match(roman_pattern, section_title):
                        section_title = re.sub(roman_pattern, '', section_title)

                    bookmark_name = f"section_{idx + 1}"

                    try:
                        add_bookmark_hyperlink(
                            toc_entry,
                            bookmark_name,
                            section_title,
                            body_font,
                            Pt(12),
                            (TEXT_COLOR[0], TEXT_COLOR[1], TEXT_COLOR[2]),
                        )
                    except Exception as e:
                        logger.debug(f"Could not create TOC hyperlink: {e}")
                        title_run = toc_entry.add_run(section_title)
                        title_run.font.name = body_font
                        title_run.font.size = Pt(12)
                        title_run.font.color.rgb = TEXT_COLOR

                doc.add_page_break()

            # Auto charts
            auto_charts_enabled = job.metadata.get("auto_charts", config.auto_charts)

            # Section review settings (optional LLM-based quality review)
            enable_quality_review = job.metadata.get("enable_slide_review", False) or job.metadata.get("enable_quality_review", False)
            section_reviewer = None
            if enable_quality_review:
                from .reviewer import DOCXSectionReviewer
                llm_generate_func = job.metadata.get("llm_generate_func")
                section_reviewer = DOCXSectionReviewer(llm_generate_func=llm_generate_func)
                logger.info("Section review enabled (will review and auto-fix sections before rendering)")

            # ========== CONTENT SECTIONS ==========
            for idx, section in enumerate(job.sections):
                # Validate title
                if not section.title or not section.title.strip():
                    logger.warning(f"Section {idx + 1} has empty title, using default", section_idx=idx)
                    section.title = f"Section {idx + 1}"

                # ========== PRE-RENDER REVIEW & FIX (Optional) ==========
                if section_reviewer:
                    try:
                        section_spec = {
                            "heading": section.title or f"Section {idx + 1}",
                            "heading_level": 1,
                            "paragraphs": [{"text": (section.revised_content if section.revised_content and section.revised_content.strip() else section.content) or ""}],
                            "bullet_points": [],
                            "font_heading": heading_font,
                            "font_body": body_font,
                        }

                        constraints = {
                            "heading_max_chars": 80,
                            "paragraph_max_chars": 2000,
                            "bullet_max_chars": 150,
                        }

                        review_result, fix_result = await section_reviewer.review_and_fix(
                            section_spec, idx, constraints
                        )

                        if fix_result.fixes_applied > 0:
                            fixed = fix_result.fixed_content
                            if fixed.get("heading") and fixed["heading"] != section.title:
                                section.title = fixed["heading"]
                                logger.info(f"Section {idx + 1}: Fixed heading")

                            logger.info(
                                "Applied pre-render fixes",
                                section=idx + 1,
                                fixes=fix_result.fixes_applied,
                                changes=fix_result.changes_made,
                            )
                        elif review_result.has_issues:
                            for issue in review_result.issues:
                                logger.warning(
                                    f"Section {idx + 1}: [{issue.severity}] {issue.issue_type}: {issue.description}"
                                )
                    except Exception as review_err:
                        logger.warning(f"Pre-render review failed for section {idx + 1}: {review_err}")
                heading = doc.add_heading(section.title, level=1)
                if not use_template_styling:
                    heading.runs[0].font.color.rgb = PRIMARY_COLOR
                    heading.runs[0].font.size = Pt(20)
                heading.paragraph_format.space_before = Pt(12)
                heading.paragraph_format.space_after = Pt(12)

                add_bookmark_to_paragraph(heading, f"section_{idx + 1}")

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
                                figsize=(8, 5),
                                dpi=150,
                            )

                            generated_chart = chart_gen.generate_chart(mpl_chart_type, mpl_data, chart_style)

                            if generated_chart and generated_chart.image_bytes:
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                    tmp.write(generated_chart.image_bytes)
                                    tmp_path = tmp.name

                                doc.add_paragraph()
                                chart_para = doc.add_paragraph()
                                chart_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                chart_run = chart_para.add_run()
                                chart_run.add_picture(tmp_path, width=Inches(6.0))
                                doc.add_paragraph()

                                rendered_as_chart = True

                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass

                    except Exception as e:
                        logger.warning(f"Chart detection failed: {e}")

                # Content paragraphs
                paragraphs = content.split('\n\n')
                for para_text in paragraphs:
                    para_text = para_text.strip()
                    if not para_text:
                        continue

                    if para_text.startswith('###'):
                        sub = doc.add_heading(para_text.lstrip('#').strip(), level=3)
                        sub.runs[0].font.color.rgb = SECONDARY_COLOR
                        sub.runs[0].font.size = Pt(13)
                    elif para_text.startswith('##'):
                        sub = doc.add_heading(para_text.lstrip('#').strip(), level=2)
                        sub.runs[0].font.color.rgb = SECONDARY_COLOR
                        sub.runs[0].font.size = Pt(15)
                    elif para_text.startswith(('- ', '• ', '* ')) or '  -' in para_text:
                        lines = para_text.split('\n')
                        for line in lines:
                            stripped_line = line.lstrip()
                            indent_spaces = len(line) - len(stripped_line)
                            indent_level = min(indent_spaces // 2, 2)

                            if stripped_line.startswith(('- ', '• ', '* ')):
                                bullet_style = 'List Bullet' if indent_level == 0 else f'List Bullet {indent_level + 1}'
                                try:
                                    bullet_para = doc.add_paragraph(style=bullet_style)
                                except KeyError:
                                    bullet_para = doc.add_paragraph(style='List Bullet')
                                    bullet_para.paragraph_format.left_indent = Inches(0.25 * indent_level)

                                bullet_run = bullet_para.add_run(stripped_line.lstrip('-•* ').strip())
                                bullet_run.font.name = body_font
                                bullet_run.font.size = Pt(11 - indent_level)
                                bullet_run.font.color.rgb = TEXT_COLOR
                    else:
                        para = doc.add_paragraph()
                        para.paragraph_format.line_spacing = 1.5
                        para.paragraph_format.space_after = Pt(8)

                        pattern = r'(\*\*[^*]+\*\*|\*[^*]+\*|[^*]+)'
                        parts = re.findall(pattern, para_text)

                        for part in parts:
                            if part.startswith('**') and part.endswith('**'):
                                run = para.add_run(part[2:-2])
                                run.bold = True
                            elif part.startswith('*') and part.endswith('*') and len(part) > 2:
                                run = para.add_run(part[1:-1])
                                run.italic = True
                            else:
                                run = para.add_run(part)

                            run.font.name = body_font
                            run.font.size = Pt(11)
                            run.font.color.rgb = TEXT_COLOR

                # Add image
                if idx in section_images:
                    try:
                        image_path = section_images[idx]
                        doc.add_paragraph()
                        img_para = doc.add_paragraph()
                        img_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        run = img_para.add_run()
                        run.add_picture(image_path, width=Inches(5.0))
                        doc.add_paragraph()
                    except Exception as e:
                        logger.warning(f"Failed to add image: {e}")

                # Section sources
                if include_sources and section.sources:
                    doc.add_paragraph()
                    source_label = doc.add_paragraph()
                    label_run = source_label.add_run("Sources for this section:")
                    label_run.font.name = body_font
                    label_run.font.size = Pt(9)
                    label_run.font.italic = True
                    label_run.font.color.rgb = LIGHT_GRAY

                    for source in section.sources[:3]:
                        src_para = doc.add_paragraph()
                        src_para.paragraph_format.left_indent = Inches(0.25)
                        doc_name = source.document_name or source.document_id
                        location_info = ""
                        if source.page_number:
                            if doc_name.lower().endswith('.pptx'):
                                location_info = f" (Slide {source.page_number})"
                            else:
                                location_info = f" (Page {source.page_number})"
                        src_run = src_para.add_run(f"• {doc_name}{location_info}")
                        src_run.font.name = body_font
                        src_run.font.size = Pt(9)
                        src_run.font.color.rgb = LIGHT_GRAY

                # Page break between sections
                if idx < len(job.sections) - 1:
                    doc.add_page_break()

            # ========== REFERENCES SECTION ==========
            if include_sources and job.sources_used:
                doc.add_page_break()

                ref_heading = doc.add_heading("References", level=1)
                ref_heading.runs[0].font.color.rgb = PRIMARY_COLOR
                ref_heading.runs[0].font.size = Pt(20)

                doc.add_paragraph()

                # Group sources by usage type
                from ...models import SourceUsageType
                content_sources = [s for s in job.sources_used if getattr(s, 'usage_type', SourceUsageType.CONTENT) == SourceUsageType.CONTENT]
                style_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) == SourceUsageType.STYLE]
                other_sources = [s for s in job.sources_used if getattr(s, 'usage_type', None) not in [SourceUsageType.CONTENT, SourceUsageType.STYLE, None]]

                def add_source_reference(source):
                    """Add a single source reference with optional hyperlink."""
                    ref_para = doc.add_paragraph()
                    ref_para.paragraph_format.space_after = Pt(4)
                    doc_name = source.document_name or source.document_id
                    location_info = ""
                    if source.page_number:
                        if doc_name.lower().endswith('.pptx'):
                            location_info = f" (Slide {source.page_number})"
                        else:
                            location_info = f" (Page {source.page_number})"

                    # Get hyperlink URL
                    hyperlink_url = None
                    if hasattr(source, 'document_url') and source.document_url:
                        hyperlink_url = source.document_url
                    elif hasattr(source, 'document_path') and source.document_path:
                        import urllib.parse
                        hyperlink_url = f"file://{urllib.parse.quote(source.document_path)}"

                    ref_text = f"• {doc_name}{location_info}"

                    # Add usage description if available
                    if hasattr(source, 'usage_description') and source.usage_description:
                        ref_text += f" — {source.usage_description}"

                    if hyperlink_url:
                        # Add hyperlink (python-docx doesn't have direct hyperlink support,
                        # so we use add_hyperlink helper if available, otherwise plain text)
                        try:
                            from docx.oxml.ns import qn
                            from docx.oxml import OxmlElement

                            # Create hyperlink element
                            hyperlink = OxmlElement('w:hyperlink')
                            hyperlink.set(qn('r:id'), ref_para.part.relate_to(
                                hyperlink_url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True
                            ))

                            new_run = OxmlElement('w:r')
                            rPr = OxmlElement('w:rPr')
                            rFonts = OxmlElement('w:rFonts')
                            rFonts.set(qn('w:ascii'), body_font)
                            rPr.append(rFonts)

                            u = OxmlElement('w:u')
                            u.set(qn('w:val'), 'single')
                            rPr.append(u)

                            new_run.append(rPr)
                            t = OxmlElement('w:t')
                            t.text = ref_text
                            new_run.append(t)
                            hyperlink.append(new_run)

                            ref_para._p.append(hyperlink)
                        except Exception:
                            # Fallback to plain text
                            ref_run = ref_para.add_run(ref_text)
                            ref_run.font.name = body_font
                            ref_run.font.size = Pt(10)
                            ref_run.font.color.rgb = TEXT_COLOR
                    else:
                        ref_run = ref_para.add_run(ref_text)
                        ref_run.font.name = body_font
                        ref_run.font.size = Pt(10)
                        ref_run.font.color.rgb = TEXT_COLOR

                def add_category_heading(text):
                    """Add a category subheading."""
                    heading_para = doc.add_paragraph()
                    heading_para.paragraph_format.space_before = Pt(12)
                    heading_para.paragraph_format.space_after = Pt(4)
                    heading_run = heading_para.add_run(text)
                    heading_run.font.name = heading_font
                    heading_run.font.size = Pt(12)
                    heading_run.font.bold = True
                    heading_run.font.color.rgb = PRIMARY_COLOR

                # Add content sources
                if content_sources:
                    add_category_heading("Content References")
                    for source in content_sources:
                        add_source_reference(source)

                # Add style sources
                if style_sources:
                    add_category_heading("Style References")
                    for source in style_sources:
                        add_source_reference(source)

                # Add other sources
                if other_sources:
                    add_category_heading("Other References")
                    for source in other_sources:
                        add_source_reference(source)

                # Fallback if no categorized sources
                if not content_sources and not style_sources and not other_sources:
                    for source in job.sources_used:
                        add_source_reference(source)

            # Save document
            output_path = os.path.join(config.output_dir, filename)
            doc.save(output_path)

            logger.info("DOCX generated", path=output_path)
            return output_path

        except ImportError as e:
            logger.error(f"python-docx import error: {e}")
            raise
