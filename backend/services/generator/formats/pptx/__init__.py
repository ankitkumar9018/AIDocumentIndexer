"""
PPTX Format Generator Package

Provides modular PPTX generation with separate components for:
- slide_builder: Creating different slide types
- styler: Applying fonts, colors, and styles
- animations: Transitions and content animations
- generator: Main PPTXGenerator class
"""

from .generator import PPTXGenerator

# Slide building
from .slide_builder import (
    get_slide_layout,
    add_slide_notes,
    remove_empty_placeholders,
    add_footer,
    add_hyperlink_to_run,
    add_slide_link_to_paragraph,
    parse_bullet_hierarchy,
    build_title_slide,
    build_content_slide,
    build_section_slide,
    build_toc_slide,
)

# Styling
from .styler import (
    calculate_luminance,
    get_contrast_ratio,
    get_best_text_color,
    hex_to_rgb_tuple,
    apply_font_to_paragraph,
    apply_font_to_run,
    apply_color_to_paragraph,
    apply_color_to_run,
    apply_title_style,
    apply_body_style,
    sanitize_text,
    get_bullet_chars,
    apply_slide_background,
)

# Animations
from .animations import (
    add_slide_transition,
    add_bullet_animations,
    get_animation_preset,
    apply_presentation_animations,
)

__all__ = [
    # Main generator
    "PPTXGenerator",
    # Slide building
    "get_slide_layout",
    "add_slide_notes",
    "remove_empty_placeholders",
    "add_footer",
    "add_hyperlink_to_run",
    "add_slide_link_to_paragraph",
    "parse_bullet_hierarchy",
    "build_title_slide",
    "build_content_slide",
    "build_section_slide",
    "build_toc_slide",
    # Styling
    "calculate_luminance",
    "get_contrast_ratio",
    "get_best_text_color",
    "hex_to_rgb_tuple",
    "apply_font_to_paragraph",
    "apply_font_to_run",
    "apply_color_to_paragraph",
    "apply_color_to_run",
    "apply_title_style",
    "apply_body_style",
    "sanitize_text",
    "get_bullet_chars",
    "apply_slide_background",
    # Animations
    "add_slide_transition",
    "add_bullet_animations",
    "get_animation_preset",
    "apply_presentation_animations",
]
