"""
Template Analyzer

Analyzes templates to extract theme, layout, and constraint information
that is provided to the LLM before content generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from .theme import (
    ThemeProfile,
    PPTXTheme,
    DOCXTheme,
    XLSXTheme,
    PDFTheme,
    LayoutInfo,
    PlaceholderSpec,
    ThemeExtractorFactory,
    BaseTheme,
)
from .content_models import ContentConstraints


@dataclass
class TemplateAnalysis:
    """Complete analysis of a template for LLM context."""

    # Template info
    template_path: str
    file_type: str  # pptx, docx, xlsx, pdf

    # Theme information
    theme: BaseTheme = None
    theme_profile: ThemeProfile = None  # Simplified version for backward compat

    # Layout information (for PPTX)
    layouts: List[LayoutInfo] = field(default_factory=list)

    # Constraints for content generation
    constraints: ContentConstraints = field(default_factory=ContentConstraints)

    # Placeholder definitions
    placeholders: Dict[str, PlaceholderSpec] = field(default_factory=dict)

    # Raw analysis data
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_llm_context(self) -> str:
        """Generate complete context string for LLM prompt."""
        parts = []

        # Add theme context
        if self.theme:
            parts.append(self.theme.to_llm_context())
        elif self.theme_profile:
            parts.append(self._theme_profile_to_context())

        # Add constraints
        parts.append(self.constraints.to_llm_context())

        # Add layout info (for PPTX)
        if self.layouts:
            parts.append(self._layouts_to_context())

        return "\n\n".join(parts)

    def _theme_profile_to_context(self) -> str:
        """Convert ThemeProfile to context string."""
        if not self.theme_profile:
            return ""

        return f"""THEME:
- Primary color: {self.theme_profile.primary}
- Secondary color: {self.theme_profile.secondary}
- Heading font: {self.theme_profile.font_heading}
- Body font: {self.theme_profile.font_body}
- Style: {self.theme_profile.description}"""

    def _layouts_to_context(self) -> str:
        """Convert layouts to context string."""
        if not self.layouts:
            return ""

        layout_lines = []
        for layout in self.layouts:
            ph_types = [p.type for p in layout.placeholders]
            layout_lines.append(
                f"  - {layout.name} ({layout.layout_type}): {layout.recommended_use}"
                f" [placeholders: {', '.join(set(ph_types))}]"
            )

        return f"""AVAILABLE LAYOUTS:
{chr(10).join(layout_lines)}

When generating content, specify which layout to use for each slide."""


class TemplateAnalyzer:
    """
    Analyzes document templates to extract information for LLM context.

    Supports PPTX, DOCX, XLSX, and PDF/HTML templates.
    """

    def __init__(self):
        self.extractor_factory = ThemeExtractorFactory

    def analyze(self, template_path: str) -> TemplateAnalysis:
        """
        Analyze a template and extract all relevant information.

        Args:
            template_path: Path to the template file

        Returns:
            TemplateAnalysis with theme, layouts, and constraints
        """
        path = Path(template_path)

        if not path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Determine file type
        suffix = path.suffix.lower()
        file_type_map = {
            '.pptx': 'pptx',
            '.docx': 'docx',
            '.xlsx': 'xlsx',
            '.pdf': 'pdf',
            '.html': 'html',
            '.css': 'css',
        }
        file_type = file_type_map.get(suffix, 'unknown')

        # Extract theme
        theme = self.extractor_factory.extract(template_path)

        # Create analysis
        analysis = TemplateAnalysis(
            template_path=template_path,
            file_type=file_type,
            theme=theme,
        )

        # File-type specific analysis
        if file_type == 'pptx':
            self._analyze_pptx(analysis, theme)
        elif file_type == 'docx':
            self._analyze_docx(analysis, theme)
        elif file_type == 'xlsx':
            self._analyze_xlsx(analysis, theme)
        elif file_type in ['pdf', 'html', 'css']:
            self._analyze_pdf(analysis, theme)

        return analysis

    def _analyze_pptx(self, analysis: TemplateAnalysis, theme: BaseTheme) -> None:
        """PPTX-specific analysis."""
        if isinstance(theme, PPTXTheme):
            analysis.layouts = theme.layouts
            analysis.theme_profile = theme.to_theme_profile()

            # Set constraints based on layout analysis
            constraints = ContentConstraints()

            # Analyze placeholder sizes to determine constraints
            for layout in theme.layouts:
                for ph in layout.placeholders:
                    if ph.type in ['title', 'ctrTitle']:
                        # Calculate max chars based on width
                        # Rough estimate: ~10 chars per inch at 44pt
                        max_chars = min(int(ph.width_inches * 10), 80)
                        if max_chars < constraints.title_max_chars:
                            constraints.title_max_chars = max_chars
                    elif ph.type == 'body':
                        # Estimate bullets that fit
                        lines_that_fit = int(ph.height_inches / 0.4)  # ~0.4 inches per line
                        if lines_that_fit < constraints.bullets_per_slide:
                            constraints.bullets_per_slide = lines_that_fit

            analysis.constraints = constraints

            # Build placeholder dict
            for layout in theme.layouts:
                for ph in layout.placeholders:
                    key = f"{layout.name}_{ph.type}_{ph.idx}"
                    analysis.placeholders[key] = ph

    def _analyze_docx(self, analysis: TemplateAnalysis, theme: BaseTheme) -> None:
        """DOCX-specific analysis."""
        if isinstance(theme, DOCXTheme):
            analysis.theme_profile = ThemeProfile(
                font_heading=theme.font_heading,
                font_body=theme.font_body,
                primary=f"#{theme.colors.get('accent1', '4F81BD')}",
            )

            analysis.constraints = ContentConstraints(
                heading_max_chars=theme.max_heading_chars,
                paragraph_max_chars=theme.max_paragraph_chars,
            )

    def _analyze_xlsx(self, analysis: TemplateAnalysis, theme: BaseTheme) -> None:
        """XLSX-specific analysis."""
        if isinstance(theme, XLSXTheme):
            analysis.theme_profile = ThemeProfile(
                font_heading=theme.get_heading_font(),
                font_body=theme.get_body_font(),
                primary=theme.get_primary_color(),
            )

            analysis.constraints = ContentConstraints(
                cell_max_chars=theme.max_cell_chars,
                sheet_name_max_chars=31,
            )

    def _analyze_pdf(self, analysis: TemplateAnalysis, theme: BaseTheme) -> None:
        """PDF/HTML-specific analysis."""
        if isinstance(theme, PDFTheme):
            analysis.theme_profile = ThemeProfile(
                font_heading=theme.font_heading,
                font_body=theme.font_body,
                primary=theme.primary_color,
                secondary=theme.secondary_color,
                text=theme.text_color,
                background=theme.background_color,
            )

            analysis.constraints = ContentConstraints(
                heading_max_chars=theme.max_heading_chars,
                paragraph_max_chars=theme.max_paragraph_chars,
            )

    def get_layout_recommendations(self, analysis: TemplateAnalysis, content_type: str) -> List[str]:
        """
        Get layout recommendations based on content type.

        Args:
            analysis: Template analysis
            content_type: Type of content (title, content, comparison, image, etc.)

        Returns:
            List of recommended layout names
        """
        recommendations = []

        content_to_layout = {
            'title': ['title_slide', 'title_only'],
            'content': ['title_content', 'content'],
            'comparison': ['two_column'],
            'image': ['image_text'],
            'data': ['title_content', 'blank'],
            'summary': ['title_only', 'title_content'],
        }

        preferred_types = content_to_layout.get(content_type, ['title_content'])

        for layout in analysis.layouts:
            if layout.layout_type in preferred_types:
                recommendations.append(layout.name)

        return recommendations or [l.name for l in analysis.layouts[:1]]

    def calculate_slide_capacity(self, analysis: TemplateAnalysis, layout_name: str) -> dict:
        """
        Calculate how much content can fit on a slide with the given layout.

        Returns estimated capacities for different content types.
        """
        capacity = {
            'title_chars': 60,
            'bullet_count': 7,
            'chars_per_bullet': 120,  # PHASE 11: Increased from 70 to allow complete sentences
            'has_image_placeholder': False,
            'has_chart_placeholder': False,
            'body_chars': 500,
        }

        # Find the layout
        for layout in analysis.layouts:
            if layout.name == layout_name or layout.layout_type == layout_name:
                for ph in layout.placeholders:
                    if ph.type in ['title', 'ctrTitle']:
                        capacity['title_chars'] = ph.max_chars
                    elif ph.type == 'body':
                        capacity['bullet_count'] = ph.max_bullets
                        capacity['chars_per_bullet'] = min(
                            int(ph.width_inches * 12),  # ~12 chars per inch at 18pt
                            70
                        )
                        capacity['body_chars'] = ph.max_chars
                    elif ph.type == 'pic':
                        capacity['has_image_placeholder'] = True
                    elif ph.type in ['chart', 'dgm']:
                        capacity['has_chart_placeholder'] = True
                break

        return capacity


def analyze_template(template_path: str) -> TemplateAnalysis:
    """
    Convenience function to analyze a template.

    Args:
        template_path: Path to the template file

    Returns:
        TemplateAnalysis with all extracted information
    """
    analyzer = TemplateAnalyzer()
    return analyzer.analyze(template_path)
