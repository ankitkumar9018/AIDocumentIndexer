"""
Theme extraction and management for document templates.

Supports extracting themes from PPTX, DOCX, XLSX, and PDF/HTML templates.

Components:
- models: Data models for themes (ThemeProfile, PPTXTheme, etc.)
- extractor: Extract themes from template files
- manager: CRUD operations for theme storage
- applier: Apply themes to documents during generation
"""

from .models import (
    BaseTheme,
    ThemeProfile,
    PPTXTheme,
    DOCXTheme,
    XLSXTheme,
    PDFTheme,
    LayoutInfo,
    PlaceholderSpec,
    Margins,
    CellStyle,
)
from .extractor import (
    ThemeExtractor,
    PPTXThemeExtractor,
    DOCXThemeExtractor,
    XLSXThemeExtractor,
    PDFThemeExtractor,
    ThemeExtractorFactory,
)
from .manager import (
    ThemeManager,
    get_theme_manager,
)
from .applier import (
    ThemeApplier,
    create_applier_from_dict,
    create_applier_from_profile,
)

__all__ = [
    # Models
    "BaseTheme",
    "ThemeProfile",
    "PPTXTheme",
    "DOCXTheme",
    "XLSXTheme",
    "PDFTheme",
    "LayoutInfo",
    "PlaceholderSpec",
    "Margins",
    "CellStyle",
    # Extractors
    "ThemeExtractor",
    "PPTXThemeExtractor",
    "DOCXThemeExtractor",
    "XLSXThemeExtractor",
    "PDFThemeExtractor",
    "ThemeExtractorFactory",
    # Manager
    "ThemeManager",
    "get_theme_manager",
    # Applier
    "ThemeApplier",
    "create_applier_from_dict",
    "create_applier_from_profile",
]
