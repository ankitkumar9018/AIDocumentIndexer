"""
PPTX Slide Builder

Provides functions for creating and building different types of slides.
Extracted from generator.py for modularity.
"""

from typing import Optional, List, Dict, Any, Tuple

import structlog

from .styler import (
    apply_title_style,
    apply_body_style,
    apply_font_to_paragraph,
    apply_color_to_paragraph,
    sanitize_text,
    get_bullet_chars,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Layout Selection
# =============================================================================

def get_slide_layout(presentation, layout_type: str = "content"):
    """Get the appropriate slide layout from the presentation.

    Args:
        presentation: python-pptx Presentation object
        layout_type: Type of layout (title, content, section, blank, two_column)

    Returns:
        SlideLayout object or None if not found
    """
    # Layout index mapping (standard PowerPoint template)
    layout_indices = {
        "title": 0,           # Title Slide
        "section": 2,         # Section Header
        "content": 1,         # Title and Content
        "two_content": 3,     # Two Content
        "comparison": 4,      # Comparison
        "title_only": 5,      # Title Only
        "blank": 6,           # Blank
        "content_caption": 7, # Content with Caption
        "picture_caption": 8, # Picture with Caption
    }

    # Try to get the layout by index
    idx = layout_indices.get(layout_type, 1)

    try:
        # Access slide layouts from the first master
        if presentation.slide_masters and presentation.slide_layouts:
            if idx < len(presentation.slide_layouts):
                return presentation.slide_layouts[idx]
            # Fallback to first content layout
            return presentation.slide_layouts[1] if len(presentation.slide_layouts) > 1 else presentation.slide_layouts[0]
    except Exception as e:
        logger.warning(f"Could not get layout '{layout_type}': {e}")

    # Return first available layout
    return presentation.slide_layouts[0]


# =============================================================================
# Slide Notes
# =============================================================================

def add_slide_notes(slide, notes_text: str):
    """Add speaker notes to a slide.

    Args:
        slide: python-pptx slide object
        notes_text: Text content for speaker notes
    """
    if not notes_text:
        return

    try:
        notes_slide = slide.notes_slide
        notes_frame = notes_slide.notes_text_frame
        notes_frame.text = sanitize_text(notes_text)
    except Exception as e:
        logger.warning(f"Could not add slide notes: {e}")


# =============================================================================
# Placeholder Management
# =============================================================================

def remove_empty_placeholders(slide, preserve_branding: bool = True):
    """Remove empty placeholder shapes from a slide.

    Args:
        slide: python-pptx slide object
        preserve_branding: If True, keep footer/date/slide number placeholders
    """
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.enum.shapes import PP_PLACEHOLDER

    # Placeholder types to potentially preserve
    branding_types = {
        PP_PLACEHOLDER.FOOTER,
        PP_PLACEHOLDER.DATE,
        PP_PLACEHOLDER.SLIDE_NUMBER,
    }

    shapes_to_remove = []

    for shape in slide.shapes:
        # Check if it's a placeholder
        if not shape.is_placeholder:
            continue

        # Get placeholder type
        ph_type = shape.placeholder_format.type

        # Preserve branding placeholders if requested
        if preserve_branding and ph_type in branding_types:
            continue

        # Check if placeholder is empty
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if not text:
                shapes_to_remove.append(shape)

    # Remove empty placeholders
    for shape in shapes_to_remove:
        try:
            sp = shape._element
            sp.getparent().remove(sp)
        except Exception as e:
            logger.debug(f"Could not remove placeholder: {e}")


# =============================================================================
# Footer Management
# =============================================================================

def add_footer(slide, page_num: int, total_pages: int,
               font_name: str = "Arial", font_size: int = 10,
               font_color=None, include_date: bool = False):
    """Add footer with page number to a slide.

    Args:
        slide: python-pptx slide object
        page_num: Current page number
        total_pages: Total number of pages
        font_name: Font for footer text
        font_size: Font size in points
        font_color: RGBColor for text
        include_date: Whether to include current date
    """
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN

    try:
        # Get slide dimensions
        slide_width = slide.part.package.presentation.slide_width
        slide_height = slide.part.package.presentation.slide_height

        # Create footer text box
        left = Inches(0.5)
        top = slide_height - Inches(0.4)
        width = slide_width - Inches(1)
        height = Inches(0.3)

        footer_box = slide.shapes.add_textbox(left, top, width, height)
        tf = footer_box.text_frame
        tf.word_wrap = False

        # Add page number
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.RIGHT

        run = p.add_run()
        run.text = f"{page_num} / {total_pages}"
        run.font.name = font_name
        run.font.size = Pt(font_size)
        if font_color:
            run.font.color.rgb = font_color

        # Optionally add date
        if include_date:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            date_run = p.add_run()
            date_run.text = f"  |  {date_str}"
            date_run.font.name = font_name
            date_run.font.size = Pt(font_size)
            if font_color:
                date_run.font.color.rgb = font_color

    except Exception as e:
        logger.warning(f"Could not add footer: {e}")


# =============================================================================
# Hyperlinks
# =============================================================================

def add_hyperlink_to_run(run, url: str, is_internal: bool = False,
                         target_slide=None):
    """Add hyperlink to a text run.

    Args:
        run: python-pptx run object
        url: URL or internal reference
        is_internal: Whether this is an internal slide link
        target_slide: Target slide object for internal links
    """
    try:
        if is_internal and target_slide:
            # Internal slide link
            rId = run.part.relate_to(target_slide.part, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide')
            run.hyperlink._rPr.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'] = rId
        else:
            # External URL
            run.hyperlink.address = url
    except Exception as e:
        logger.warning(f"Could not add hyperlink: {e}")


def add_slide_link_to_paragraph(paragraph, target_slide, text: str,
                                font_name: str, font_size, font_color):
    """Add internal slide link to a paragraph.

    Args:
        paragraph: python-pptx paragraph object
        target_slide: Target slide to link to
        text: Link text
        font_name: Font for link text
        font_size: Font size (Pt object)
        font_color: RGBColor for link text
    """
    try:
        run = paragraph.add_run()
        run.text = text
        run.font.name = font_name
        run.font.size = font_size
        run.font.color.rgb = font_color
        run.font.underline = True

        # Add the hyperlink relationship
        add_hyperlink_to_run(run, "", is_internal=True, target_slide=target_slide)

    except Exception as e:
        logger.warning(f"Could not add slide link: {e}")


# =============================================================================
# Bullet Point Parsing
# =============================================================================

def parse_bullet_hierarchy(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse bullet points into a hierarchical structure.

    Detects indentation levels and nested bullets.

    Args:
        lines: List of text lines

    Returns:
        List of dicts with 'text' and 'level' keys
    """
    bullets = []

    for line in lines:
        if not line.strip():
            continue

        # Detect indentation
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Remove bullet markers
        text = stripped.lstrip('-*•▸◦▪○ ')

        # Calculate level based on indentation (4 spaces = 1 level)
        level = min(indent // 4, 3)  # Max 4 levels (0-3)

        bullets.append({
            'text': text.strip(),
            'level': level,
        })

    return bullets


# =============================================================================
# Title Slide Builder
# =============================================================================

def build_title_slide(presentation, title: str, subtitle: str = "",
                      heading_font: str = "Arial", body_font: str = "Arial",
                      title_color=None, subtitle_color=None):
    """Build a title slide.

    Args:
        presentation: python-pptx Presentation object
        title: Main title text
        subtitle: Subtitle text
        heading_font: Font for title
        body_font: Font for subtitle
        title_color: RGBColor for title
        subtitle_color: RGBColor for subtitle

    Returns:
        Created slide object
    """
    from pptx.util import Pt

    layout = get_slide_layout(presentation, "title")
    slide = presentation.slides.add_slide(layout)

    # Set title
    if slide.shapes.title:
        slide.shapes.title.text = sanitize_text(title)
        apply_title_style(
            slide.shapes.title,
            font_name=heading_font,
            font_size=44,
            bold=True,
            color=title_color,
        )

    # Set subtitle
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 1:  # Subtitle placeholder
            shape.text = sanitize_text(subtitle)
            apply_body_style(
                shape,
                font_name=body_font,
                font_size=24,
                color=subtitle_color,
            )
            break

    return slide


# =============================================================================
# Content Slide Builder
# =============================================================================

def build_content_slide(presentation, title: str, content: List[str],
                        heading_font: str = "Arial", body_font: str = "Arial",
                        title_color=None, text_color=None,
                        bullet_style: str = "circle"):
    """Build a content slide with bullet points.

    Args:
        presentation: python-pptx Presentation object
        title: Slide title
        content: List of bullet point strings
        heading_font: Font for title
        body_font: Font for bullets
        title_color: RGBColor for title
        text_color: RGBColor for content
        bullet_style: Style of bullets (circle, square, arrow, etc.)

    Returns:
        Created slide object
    """
    from pptx.util import Pt
    from pptx.enum.text import PP_ALIGN

    layout = get_slide_layout(presentation, "content")
    slide = presentation.slides.add_slide(layout)

    # Set title
    if slide.shapes.title:
        slide.shapes.title.text = sanitize_text(title)
        apply_title_style(
            slide.shapes.title,
            font_name=heading_font,
            font_size=36,
            bold=True,
            color=title_color,
        )

    # Find content placeholder
    content_shape = None
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 1:  # Content placeholder
            content_shape = shape
            break

    if content_shape and content_shape.has_text_frame:
        tf = content_shape.text_frame
        tf.clear()

        bullet_char = get_bullet_chars(bullet_style)

        for i, bullet_text in enumerate(content):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()

            p.text = sanitize_text(bullet_text)
            p.level = 0
            p.alignment = PP_ALIGN.LEFT

            # Apply styling
            for run in p.runs:
                run.font.name = body_font
                run.font.size = Pt(18)
                if text_color:
                    run.font.color.rgb = text_color

    return slide


# =============================================================================
# Section Header Slide
# =============================================================================

def build_section_slide(presentation, title: str, subtitle: str = "",
                        heading_font: str = "Arial", body_font: str = "Arial",
                        title_color=None, subtitle_color=None):
    """Build a section header slide.

    Args:
        presentation: python-pptx Presentation object
        title: Section title
        subtitle: Section subtitle/description
        heading_font: Font for title
        body_font: Font for subtitle
        title_color: RGBColor for title
        subtitle_color: RGBColor for subtitle

    Returns:
        Created slide object
    """
    layout = get_slide_layout(presentation, "section")
    slide = presentation.slides.add_slide(layout)

    # Set title
    if slide.shapes.title:
        slide.shapes.title.text = sanitize_text(title)
        apply_title_style(
            slide.shapes.title,
            font_name=heading_font,
            font_size=40,
            bold=True,
            color=title_color,
        )

    # Set subtitle if there's a subtitle placeholder
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 1:
            shape.text = sanitize_text(subtitle)
            apply_body_style(
                shape,
                font_name=body_font,
                font_size=20,
                color=subtitle_color,
            )
            break

    return slide


# =============================================================================
# Table of Contents Slide
# =============================================================================

def build_toc_slide(presentation, title: str, sections: List[Dict[str, Any]],
                    heading_font: str = "Arial", body_font: str = "Arial",
                    title_color=None, text_color=None,
                    link_to_slides: bool = False):
    """Build a table of contents slide.

    Args:
        presentation: python-pptx Presentation object
        title: TOC title (e.g., "Agenda", "Contents")
        sections: List of dicts with 'title' and optionally 'slide' keys
        heading_font: Font for title
        body_font: Font for section items
        title_color: RGBColor for title
        text_color: RGBColor for items
        link_to_slides: Whether to add hyperlinks to section slides

    Returns:
        Created slide object
    """
    from pptx.util import Pt
    from pptx.enum.text import PP_ALIGN

    layout = get_slide_layout(presentation, "content")
    slide = presentation.slides.add_slide(layout)

    # Set title
    if slide.shapes.title:
        slide.shapes.title.text = sanitize_text(title)
        apply_title_style(
            slide.shapes.title,
            font_name=heading_font,
            font_size=36,
            bold=True,
            color=title_color,
        )

    # Find content placeholder
    content_shape = None
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 1:
            content_shape = shape
            break

    if content_shape and content_shape.has_text_frame:
        tf = content_shape.text_frame
        tf.clear()

        for i, section in enumerate(sections):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()

            section_title = section.get('title', f'Section {i+1}')
            p.text = f"{i+1}. {sanitize_text(section_title)}"
            p.alignment = PP_ALIGN.LEFT

            for run in p.runs:
                run.font.name = body_font
                run.font.size = Pt(20)
                if text_color:
                    run.font.color.rgb = text_color

    return slide
