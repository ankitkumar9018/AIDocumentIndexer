"""
Theme data models for all supported document formats.

These models represent the theme/styling information extracted from templates
and are used to provide context to the LLM for content generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class Margins:
    """Document margins in inches."""
    top: float = 1.0
    bottom: float = 1.0
    left: float = 1.0
    right: float = 1.0


@dataclass
class PlaceholderSpec:
    """Specification for a placeholder in a template."""
    type: str  # title, body, picture, chart, table, subtitle, footer, etc.
    idx: int = 0  # Placeholder index
    max_chars: int = 500
    max_lines: int = 10
    max_bullets: int = 7
    font_size_pt: int = 12
    width_inches: float = 8.0
    height_inches: float = 5.0
    x_inches: float = 0.0
    y_inches: float = 0.0


@dataclass
class LayoutInfo:
    """Information about a slide/page layout."""
    name: str
    layout_type: str  # title_only, title_content, two_column, image_text, blank
    placeholders: List[PlaceholderSpec] = field(default_factory=list)
    recommended_use: str = "general content"


@dataclass
class CellStyle:
    """Style for a spreadsheet cell."""
    font_name: str = "Calibri"
    font_size: int = 11
    bold: bool = False
    italic: bool = False
    fill_color: Optional[str] = None
    text_color: str = "000000"
    border: bool = False
    alignment: str = "left"  # left, center, right
    number_format: Optional[str] = None


class BaseTheme(ABC):
    """Base interface for all theme types."""

    @abstractmethod
    def get_primary_color(self) -> str:
        """Get the primary accent color."""
        ...

    @abstractmethod
    def get_heading_font(self) -> str:
        """Get the heading/title font name."""
        ...

    @abstractmethod
    def get_body_font(self) -> str:
        """Get the body text font name."""
        ...

    @abstractmethod
    def get_constraints(self) -> Dict[str, int]:
        """Get content constraints for this theme."""
        ...

    @abstractmethod
    def to_llm_context(self) -> str:
        """Generate context string for LLM prompt."""
        ...


@dataclass
class ThemeProfile:
    """
    Generic theme profile used across different document types.
    This is the simplified version for backward compatibility.
    """
    name: str = "Default"
    primary: str = "#2563EB"  # Blue
    secondary: str = "#1E40AF"
    accent: str = "#F59E0B"
    background: str = "#FFFFFF"
    text: str = "#1F2937"
    font_heading: str = "Calibri"
    font_body: str = "Calibri"
    description: str = "Professional theme"

    # Extended colors from OOXML
    dk1: str = "#000000"  # Dark 1 (usually text)
    lt1: str = "#FFFFFF"  # Light 1 (usually background)
    dk2: str = "#1F497D"  # Dark 2
    lt2: str = "#EEECE1"  # Light 2
    accent1: str = "#4F81BD"  # Accent 1 (primary)
    accent2: str = "#C0504D"  # Accent 2
    accent3: str = "#9BBB59"  # Accent 3
    accent4: str = "#8064A2"  # Accent 4
    accent5: str = "#4BACC6"  # Accent 5
    accent6: str = "#F79646"  # Accent 6
    hlink: str = "#0000FF"  # Hyperlink
    folHlink: str = "#800080"  # Followed hyperlink

    # Object defaults
    default_shape_fill: str = "#44484C"
    bullet_color: str = "#595959"


@dataclass
class PPTXTheme(BaseTheme):
    """Complete PPTX theme from theme1.xml."""

    # Color scheme (12 standard OOXML colors)
    colors: Dict[str, str] = field(default_factory=lambda: {
        "dk1": "000000",
        "lt1": "FFFFFF",
        "dk2": "1F497D",
        "lt2": "EEECE1",
        "accent1": "4F81BD",
        "accent2": "C0504D",
        "accent3": "9BBB59",
        "accent4": "8064A2",
        "accent5": "4BACC6",
        "accent6": "F79646",
        "hlink": "0000FF",
        "folHlink": "800080",
    })

    # Font scheme
    font_heading: str = "Calibri"
    font_body: str = "Calibri"
    font_scripts: Dict[str, str] = field(default_factory=dict)

    # Format scheme
    fill_styles: List[dict] = field(default_factory=list)
    line_styles: List[dict] = field(default_factory=list)
    effect_styles: List[dict] = field(default_factory=list)

    # Object defaults
    default_shape_fill: str = "44484C"
    default_shape_line: Optional[str] = None
    default_text_style: Dict = field(default_factory=dict)
    bullet_color: str = "595959"

    # Layout information
    layouts: List[LayoutInfo] = field(default_factory=list)

    def get_primary_color(self) -> str:
        return f"#{self.colors.get('accent1', '4F81BD')}"

    def get_heading_font(self) -> str:
        return self.font_heading

    def get_body_font(self) -> str:
        return self.font_body

    def get_constraints(self) -> Dict[str, int]:
        return {
            "title_max_chars": 60,
            "subtitle_max_chars": 100,
            "bullet_max_chars": 70,
            "bullets_per_slide": 7,
            "body_max_chars": 500,
            "speaker_notes_max_chars": 300,
        }

    def to_llm_context(self) -> str:
        constraints = self.get_constraints()
        layouts_str = "\n".join([
            f"  - {l.name}: {l.recommended_use} (placeholders: {len(l.placeholders)})"
            for l in self.layouts
        ]) if self.layouts else "  - Standard layouts available"

        return f"""PPTX THEME CONTEXT:
Colors:
  - Primary (accent1): #{self.colors.get('accent1', '4F81BD')}
  - Background (lt1): #{self.colors.get('lt1', 'FFFFFF')}
  - Text (dk1): #{self.colors.get('dk1', '000000')}
  - Accent colors: {', '.join([f"#{self.colors.get(f'accent{i}', '')}" for i in range(1, 7)])}

Typography:
  - Headings: {self.font_heading}
  - Body: {self.font_body}

Constraints:
  - Title: max {constraints['title_max_chars']} characters
  - Bullets: max {constraints['bullet_max_chars']} characters each
  - Bullets per slide: max {constraints['bullets_per_slide']}
  - Speaker notes: max {constraints['speaker_notes_max_chars']} characters

Available Layouts:
{layouts_str}
"""

    def to_theme_profile(self) -> ThemeProfile:
        """Convert to simpler ThemeProfile for backward compatibility."""
        return ThemeProfile(
            name="Extracted Theme",
            primary=f"#{self.colors.get('accent1', '4F81BD')}",
            secondary=f"#{self.colors.get('accent2', 'C0504D')}",
            accent=f"#{self.colors.get('accent3', '9BBB59')}",
            background=f"#{self.colors.get('lt1', 'FFFFFF')}",
            text=f"#{self.colors.get('dk1', '000000')}",
            font_heading=self.font_heading,
            font_body=self.font_body,
            dk1=self.colors.get('dk1', '000000'),
            lt1=self.colors.get('lt1', 'FFFFFF'),
            dk2=self.colors.get('dk2', '1F497D'),
            lt2=self.colors.get('lt2', 'EEECE1'),
            accent1=self.colors.get('accent1', '4F81BD'),
            accent2=self.colors.get('accent2', 'C0504D'),
            accent3=self.colors.get('accent3', '9BBB59'),
            accent4=self.colors.get('accent4', '8064A2'),
            accent5=self.colors.get('accent5', '4BACC6'),
            accent6=self.colors.get('accent6', 'F79646'),
            hlink=self.colors.get('hlink', '0000FF'),
            folHlink=self.colors.get('folHlink', '800080'),
            default_shape_fill=f"#{self.default_shape_fill}",
            bullet_color=f"#{self.bullet_color}",
        )


@dataclass
class DOCXTheme(BaseTheme):
    """Word document theme."""

    # From styles.xml - named styles
    styles: Dict[str, dict] = field(default_factory=dict)

    # From theme/theme1.xml (same structure as PPTX)
    colors: Dict[str, str] = field(default_factory=lambda: {
        "dk1": "000000",
        "lt1": "FFFFFF",
        "accent1": "4F81BD",
    })
    font_heading: str = "Calibri Light"
    font_body: str = "Calibri"

    # Document settings
    page_margins: Margins = field(default_factory=Margins)
    default_font_size: int = 11
    line_spacing: float = 1.15

    # Section constraints
    max_heading_chars: int = 80
    max_paragraph_chars: int = 1000

    def get_primary_color(self) -> str:
        return f"#{self.colors.get('accent1', '4F81BD')}"

    def get_heading_font(self) -> str:
        return self.font_heading

    def get_body_font(self) -> str:
        return self.font_body

    def get_constraints(self) -> Dict[str, int]:
        return {
            "heading_max_chars": self.max_heading_chars,
            "paragraph_max_chars": self.max_paragraph_chars,
            "default_font_size": self.default_font_size,
        }

    def to_llm_context(self) -> str:
        return f"""DOCX THEME CONTEXT:
Typography:
  - Headings: {self.font_heading}
  - Body: {self.font_body} ({self.default_font_size}pt)
  - Line spacing: {self.line_spacing}

Page Layout:
  - Margins: {self.page_margins.top}" top, {self.page_margins.bottom}" bottom

Constraints:
  - Headings: max {self.max_heading_chars} characters
  - Paragraphs: max {self.max_paragraph_chars} characters

Available Styles: {', '.join(self.styles.keys()) if self.styles else 'Standard Word styles'}
"""


@dataclass
class XLSXTheme(BaseTheme):
    """Excel workbook theme."""

    # From xl/theme/theme1.xml
    colors: Dict[str, str] = field(default_factory=lambda: {
        "dk1": "000000",
        "lt1": "FFFFFF",
        "accent1": "4F81BD",
    })
    fonts: Dict[str, str] = field(default_factory=lambda: {
        "heading": "Calibri",
        "body": "Calibri",
    })

    # Named styles from styles.xml
    named_styles: Dict[str, CellStyle] = field(default_factory=dict)

    # Cell formatting defaults
    header_style: CellStyle = field(default_factory=lambda: CellStyle(bold=True, fill_color="4F81BD", text_color="FFFFFF"))
    data_style: CellStyle = field(default_factory=CellStyle)
    currency_format: str = "$#,##0.00"
    date_format: str = "yyyy-mm-dd"

    # Constraints
    max_column_width: int = 50
    max_cell_chars: int = 256

    def get_primary_color(self) -> str:
        return f"#{self.colors.get('accent1', '4F81BD')}"

    def get_heading_font(self) -> str:
        return self.fonts.get("heading", "Calibri")

    def get_body_font(self) -> str:
        return self.fonts.get("body", "Calibri")

    def get_constraints(self) -> Dict[str, int]:
        return {
            "max_column_width": self.max_column_width,
            "max_cell_chars": self.max_cell_chars,
        }

    def to_llm_context(self) -> str:
        return f"""XLSX THEME CONTEXT:
Typography:
  - Headers: {self.get_heading_font()}
  - Data: {self.get_body_font()}

Formatting:
  - Currency: {self.currency_format}
  - Date: {self.date_format}

Constraints:
  - Max column width: {self.max_column_width} characters
  - Max cell content: {self.max_cell_chars} characters

Available Styles: {', '.join(self.named_styles.keys()) if self.named_styles else 'Standard Excel styles'}
"""


@dataclass
class PDFTheme(BaseTheme):
    """PDF theme from CSS/HTML template."""

    # Colors from CSS variables
    primary_color: str = "#2563EB"
    secondary_color: str = "#1E40AF"
    text_color: str = "#1F2937"
    background_color: str = "#FFFFFF"
    accent_color: str = "#F59E0B"

    # Typography
    font_heading: str = "Helvetica"
    font_body: str = "Helvetica"
    base_font_size: str = "12pt"
    line_height: str = "1.5"

    # Layout
    page_size: str = "A4"  # A4, Letter, etc.
    margins: Margins = field(default_factory=lambda: Margins(1.0, 1.0, 1.0, 1.0))
    header_height: str = "0.5in"
    footer_height: str = "0.5in"

    # Constraints
    max_heading_chars: int = 80
    max_paragraph_chars: int = 1500

    def get_primary_color(self) -> str:
        return self.primary_color

    def get_heading_font(self) -> str:
        return self.font_heading

    def get_body_font(self) -> str:
        return self.font_body

    def get_constraints(self) -> Dict[str, int]:
        return {
            "heading_max_chars": self.max_heading_chars,
            "paragraph_max_chars": self.max_paragraph_chars,
        }

    def to_llm_context(self) -> str:
        return f"""PDF THEME CONTEXT:
Colors:
  - Primary: {self.primary_color}
  - Text: {self.text_color}
  - Background: {self.background_color}

Typography:
  - Headings: {self.font_heading}
  - Body: {self.font_body} ({self.base_font_size})
  - Line height: {self.line_height}

Page Layout:
  - Size: {self.page_size}
  - Margins: {self.margins.top}" {self.margins.right}" {self.margins.bottom}" {self.margins.left}"

Constraints:
  - Headings: max {self.max_heading_chars} characters
  - Paragraphs: max {self.max_paragraph_chars} characters
"""
