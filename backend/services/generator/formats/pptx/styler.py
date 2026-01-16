"""
PPTX Styling Utilities

Provides functions for applying styles, colors, and fonts to PowerPoint elements.
Extracted from generator.py for modularity.
"""

from typing import Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Color Utilities
# =============================================================================

def calculate_luminance(r: int, g: int, b: int) -> float:
    """Calculate relative luminance of an RGB color.

    Uses the sRGB luminance formula (ITU-R BT.709).

    Args:
        r, g, b: RGB values (0-255)

    Returns:
        Luminance value between 0 and 1
    """
    # Normalize to 0-1
    rs = r / 255.0
    gs = g / 255.0
    bs = b / 255.0

    # Apply gamma correction
    def gamma(c):
        if c <= 0.03928:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * gamma(rs) + 0.7152 * gamma(gs) + 0.0722 * gamma(bs)


def get_contrast_ratio(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """Calculate contrast ratio between two colors.

    Args:
        color1: First RGB tuple (r, g, b)
        color2: Second RGB tuple (r, g, b)

    Returns:
        Contrast ratio (1:1 to 21:1)
    """
    l1 = calculate_luminance(*color1)
    l2 = calculate_luminance(*color2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def get_best_text_color(bg_r: int, bg_g: int, bg_b: int) -> Tuple[int, int, int]:
    """Determine the best text color (black or white) for a given background.

    Args:
        bg_r, bg_g, bg_b: Background RGB values

    Returns:
        Tuple of (r, g, b) for either white or black
    """
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    white_ratio = get_contrast_ratio((bg_r, bg_g, bg_b), WHITE)
    black_ratio = get_contrast_ratio((bg_r, bg_g, bg_b), BLACK)

    return WHITE if white_ratio > black_ratio else BLACK


def hex_to_rgb_tuple(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (with or without #)

    Returns:
        Tuple of (r, g, b) integers
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# =============================================================================
# Font Application
# =============================================================================

def apply_font_to_paragraph(paragraph, font_name: str, is_heading: bool = False):
    """Apply font to all runs in a paragraph.

    Args:
        paragraph: python-pptx paragraph object
        font_name: Name of font to apply
        is_heading: Whether this is heading text (affects any future styling)
    """
    for run in paragraph.runs:
        run.font.name = font_name


def apply_font_to_run(run, font_name: str):
    """Apply font to a single run.

    Args:
        run: python-pptx run object
        font_name: Name of font to apply
    """
    run.font.name = font_name


def apply_color_to_paragraph(paragraph, color):
    """Apply color to all runs in a paragraph.

    Args:
        paragraph: python-pptx paragraph object
        color: RGBColor object to apply
    """
    for run in paragraph.runs:
        run.font.color.rgb = color


def apply_color_to_run(run, color):
    """Apply color to a single run.

    Args:
        run: python-pptx run object
        color: RGBColor object to apply
    """
    run.font.color.rgb = color


# =============================================================================
# Shape Styling
# =============================================================================

def apply_title_style(shape, font_name: str, font_size: int = 44,
                      bold: bool = True, color=None):
    """Apply title styling to a shape.

    Args:
        shape: python-pptx shape with text_frame
        font_name: Font to use for title
        font_size: Font size in points (default 44)
        bold: Whether text should be bold (default True)
        color: RGBColor for text (optional)
    """
    from pptx.util import Pt

    for para in shape.text_frame.paragraphs:
        for run in para.runs:
            run.font.name = font_name
            run.font.size = Pt(font_size)
            run.font.bold = bold
            if color:
                run.font.color.rgb = color


def apply_body_style(shape, font_name: str, font_size: int = 18, color=None):
    """Apply body text styling to a shape.

    Args:
        shape: python-pptx shape with text_frame
        font_name: Font to use for body text
        font_size: Font size in points (default 18)
        color: RGBColor for text (optional)
    """
    from pptx.util import Pt

    for para in shape.text_frame.paragraphs:
        for run in para.runs:
            run.font.name = font_name
            run.font.size = Pt(font_size)
            if color:
                run.font.color.rgb = color


# =============================================================================
# Text Sanitization
# =============================================================================

def sanitize_text(text: str) -> str:
    """Sanitize text for use in PowerPoint.

    Removes/replaces characters that cause issues in PPTX.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text safe for PowerPoint
    """
    if not text:
        return ""

    # Replace problematic characters
    replacements = {
        '\x00': '',  # Null character
        '\x0b': '',  # Vertical tab
        '\x0c': '',  # Form feed
        '\r\n': '\n',  # Windows newlines
        '\r': '\n',  # Mac newlines
        '—': '-',  # Em dash
        '–': '-',  # En dash
        '"': '"',  # Smart quotes
        '"': '"',
        ''': "'",
        ''': "'",
        '…': '...',  # Ellipsis
    }

    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)

    return result


# =============================================================================
# Bullet Styles
# =============================================================================

def get_bullet_chars(style: str = "circle") -> str:
    """Get bullet character for a given style.

    Args:
        style: Bullet style name (circle, square, arrow, dash, diamond)

    Returns:
        Unicode bullet character
    """
    bullets = {
        "circle": "•",
        "square": "■",
        "arrow": "▸",
        "dash": "—",
        "diamond": "◆",
        "hollow_circle": "○",
        "checkmark": "✓",
    }
    return bullets.get(style, "•")


# =============================================================================
# Slide Background
# =============================================================================

def apply_slide_background(slide, color=None, gradient: bool = False,
                           is_title_slide: bool = False):
    """Apply background to a slide.

    Args:
        slide: python-pptx slide object
        color: RGBColor for solid background (optional)
        gradient: Whether to apply gradient (not implemented yet)
        is_title_slide: Whether this is a title slide (may affect styling)
    """
    if color is None:
        return

    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color
