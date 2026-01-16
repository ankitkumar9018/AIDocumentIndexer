"""
Document Generation Configuration

Theme, font, and layout configuration for document generation.
Extracted from generator.py for modularity.
"""

# =============================================================================
# Language Configuration
# =============================================================================

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}


# =============================================================================
# Theme Configuration
# =============================================================================

THEMES = {
    # === EXISTING THEMES (Enhanced with distinctive visual properties) ===
    "business": {
        "name": "Business Professional",
        "primary": "#1E3A5F",
        "secondary": "#3D5A80",
        "accent": "#E0E1DD",
        "text": "#2D3A45",
        "light_gray": "#888888",
        "description": "Clean, corporate look ideal for business presentations",
        "slide_background": "solid",
        "header_style": "underline",
        "bullet_style": "circle",
        "accent_position": "top",
    },
    "creative": {
        "name": "Creative & Bold",
        "primary": "#6B4C9A",
        "secondary": "#9B6B9E",
        "accent": "#F4E4BA",
        "text": "#333333",
        "light_gray": "#666666",
        "description": "Vibrant colors for marketing and creative content",
        "slide_background": "gradient",
        "header_style": "bar",
        "bullet_style": "arrow",
        "accent_position": "side",
    },
    "modern": {
        "name": "Modern Minimal",
        "primary": "#212529",
        "secondary": "#495057",
        "accent": "#00B4D8",
        "text": "#212529",
        "light_gray": "#6C757D",
        "description": "Sleek, contemporary design with bold accents",
        "slide_background": "solid",
        "header_style": "none",
        "bullet_style": "dash",
        "accent_position": "bottom",
    },
    "nature": {
        "name": "Nature & Organic",
        "primary": "#2D5016",
        "secondary": "#5A7D3A",
        "accent": "#F5F0E1",
        "text": "#2D3A2E",
        "light_gray": "#7A8B6E",
        "description": "Earthy tones for sustainability and wellness topics",
        "slide_background": "textured",
        "header_style": "leaf",
        "bullet_style": "leaf",
        "accent_position": "corner",
    },
    "elegant": {
        "name": "Elegant & Refined",
        "primary": "#2C3E50",
        "secondary": "#7F8C8D",
        "accent": "#BDC3C7",
        "text": "#2C3E50",
        "light_gray": "#95A5A6",
        "description": "Sophisticated look for executive presentations",
        "slide_background": "solid",
        "header_style": "serif",
        "bullet_style": "square",
        "accent_position": "top",
    },
    "vibrant": {
        "name": "Vibrant & Energetic",
        "primary": "#E74C3C",
        "secondary": "#F39C12",
        "accent": "#FDF2E9",
        "text": "#2D3436",
        "light_gray": "#BDC3C7",
        "description": "Bold colors for high-energy content",
        "slide_background": "gradient",
        "header_style": "colorblock",
        "bullet_style": "circle-filled",
        "accent_position": "diagonal",
    },
    "tech": {
        "name": "Tech & Digital",
        "primary": "#0984E3",
        "secondary": "#6C5CE7",
        "accent": "#DFE6E9",
        "text": "#2D3436",
        "light_gray": "#B2BEC3",
        "description": "Modern tech aesthetic for digital topics",
        "slide_background": "gradient",
        "header_style": "glow",
        "bullet_style": "chevron",
        "accent_position": "side",
    },
    "warm": {
        "name": "Warm & Inviting",
        "primary": "#D35400",
        "secondary": "#E67E22",
        "accent": "#FDEBD0",
        "text": "#2C3E50",
        "light_gray": "#A6ACAF",
        "description": "Cozy colors for community and wellness",
        "slide_background": "warm-gradient",
        "header_style": "rounded",
        "bullet_style": "circle",
        "accent_position": "corner",
    },
    # === NEW THEMES ===
    "minimalist": {
        "name": "Ultra Minimalist",
        "primary": "#333333",
        "secondary": "#666666",
        "accent": "#F5F5F5",
        "text": "#222222",
        "light_gray": "#AAAAAA",
        "description": "Ultra-clean design with maximum whitespace and focus on content",
        "slide_background": "white",
        "header_style": "none",
        "bullet_style": "dash",
        "accent_position": "none",
    },
    "dark": {
        "name": "Dark Mode",
        "primary": "#1A1A2E",
        "secondary": "#16213E",
        "accent": "#0F3460",
        "text": "#E4E4E4",
        "light_gray": "#7A7A8C",
        "description": "Elegant dark theme for low-light viewing and modern aesthetics",
        "slide_background": "dark",
        "header_style": "glow",
        "bullet_style": "square",
        "accent_position": "border",
    },
    "colorful": {
        "name": "Colorful & Fun",
        "primary": "#FF6B6B",
        "secondary": "#4ECDC4",
        "accent": "#FFE66D",
        "text": "#2C3E50",
        "light_gray": "#95A5A6",
        "description": "Bold, multi-color theme for engaging and memorable presentations",
        "slide_background": "gradient-multi",
        "header_style": "colorblock",
        "bullet_style": "circle-filled",
        "accent_position": "corners",
    },
    "academic": {
        "name": "Academic & Scholarly",
        "primary": "#2C3E50",
        "secondary": "#8E44AD",
        "accent": "#ECF0F1",
        "text": "#2C3E50",
        "light_gray": "#7F8C8D",
        "description": "Traditional academic style ideal for research and educational presentations",
        "slide_background": "solid",
        "header_style": "underline",
        "bullet_style": "number",
        "accent_position": "footer",
    },
}


# =============================================================================
# Font Configuration
# =============================================================================

FONT_FAMILIES = {
    "modern": {
        "name": "Modern",
        "heading": "Calibri",
        "body": "Calibri",
        "description": "Clean, contemporary sans-serif"
    },
    "classic": {
        "name": "Classic",
        "heading": "Georgia",
        "body": "Times New Roman",
        "description": "Traditional serif fonts for formal documents"
    },
    "professional": {
        "name": "Professional",
        "heading": "Arial",
        "body": "Arial",
        "description": "Universal business-standard fonts"
    },
    "technical": {
        "name": "Technical",
        "heading": "Consolas",
        "body": "Courier New",
        "description": "Monospace fonts for technical content"
    },
}


# =============================================================================
# Layout Configuration
# =============================================================================

LAYOUT_TEMPLATES = {
    "standard": {
        "name": "Standard",
        "description": "Title with bullet points",
        "content_width": 0.85,
        "image_position": "right"
    },
    "two_column": {
        "name": "Two Column",
        "description": "Split content into two columns",
        "content_width": 0.45,
        "image_position": "side"
    },
    "image_focused": {
        "name": "Image Focused",
        "description": "Large images with minimal text",
        "content_width": 0.4,
        "image_position": "center"
    },
    "minimal": {
        "name": "Minimal",
        "description": "Clean layout with lots of whitespace",
        "content_width": 0.7,
        "image_position": "bottom"
    },
}


# =============================================================================
# Default Values
# =============================================================================

DEFAULT_THEME = "business"
DEFAULT_FONT_FAMILY = "modern"
DEFAULT_LAYOUT = "standard"
