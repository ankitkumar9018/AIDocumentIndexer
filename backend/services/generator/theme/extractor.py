"""
Theme extractors for different document formats.

These extractors parse template files and extract theme/styling information
that is used to provide context to the LLM for content generation.
"""

import zipfile
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from pathlib import Path

try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree

from .models import (
    PPTXTheme,
    DOCXTheme,
    XLSXTheme,
    PDFTheme,
    ThemeProfile,
    LayoutInfo,
    PlaceholderSpec,
    Margins,
    CellStyle,
    BaseTheme,
)


class ThemeExtractor(ABC):
    """Base class for theme extractors."""

    @abstractmethod
    def extract(self, template_path: str) -> BaseTheme:
        """Extract theme from a template file."""
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if this extractor supports the given file type."""
        ...


class PPTXThemeExtractor(ThemeExtractor):
    """Extract theme from PPTX templates."""

    # OOXML namespaces
    NAMESPACES = {
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pptx')

    def extract(self, template_path: str) -> PPTXTheme:
        """Extract complete theme from PPTX template."""
        theme = PPTXTheme()

        try:
            with zipfile.ZipFile(template_path, 'r') as zf:
                # Extract theme colors and fonts
                if 'ppt/theme/theme1.xml' in zf.namelist():
                    theme_xml = zf.read('ppt/theme/theme1.xml')
                    self._parse_theme_xml(theme_xml, theme)

                # Extract layout information
                self._extract_layouts(zf, theme)

        except Exception as e:
            print(f"Warning: Could not fully extract theme from {template_path}: {e}")

        return theme

    def _parse_theme_xml(self, xml_content: bytes, theme: PPTXTheme) -> None:
        """Parse theme1.xml to extract colors, fonts, and styles."""
        try:
            root = etree.fromstring(xml_content)
        except Exception:
            return

        # Extract color scheme
        self._extract_color_scheme(root, theme)

        # Extract font scheme
        self._extract_font_scheme(root, theme)

        # Extract format scheme (fill, line, effect styles)
        self._extract_format_scheme(root, theme)

        # Extract object defaults
        self._extract_object_defaults(root, theme)

    def _extract_color_scheme(self, root, theme: PPTXTheme) -> None:
        """Extract the 12 theme colors from clrScheme."""
        color_names = ['dk1', 'lt1', 'dk2', 'lt2', 'accent1', 'accent2',
                       'accent3', 'accent4', 'accent5', 'accent6', 'hlink', 'folHlink']

        for color_name in color_names:
            # Try to find the color element
            xpath = f".//a:clrScheme/a:{color_name}"
            try:
                elem = root.find(xpath, self.NAMESPACES)
                if elem is not None:
                    color = self._extract_color_value(elem)
                    if color:
                        theme.colors[color_name] = color
            except Exception:
                pass

    def _extract_color_value(self, elem) -> Optional[str]:
        """Extract color value from an element (srgbClr or sysClr)."""
        # Check for srgbClr (explicit RGB)
        srgb = elem.find('.//a:srgbClr', self.NAMESPACES)
        if srgb is not None:
            return srgb.get('val', '')

        # Check for sysClr (system color with lastClr fallback)
        sysclr = elem.find('.//a:sysClr', self.NAMESPACES)
        if sysclr is not None:
            return sysclr.get('lastClr', sysclr.get('val', ''))

        return None

    def _extract_font_scheme(self, root, theme: PPTXTheme) -> None:
        """Extract major and minor fonts from fontScheme."""
        try:
            # Major font (headings)
            major_latin = root.find('.//a:fontScheme/a:majorFont/a:latin', self.NAMESPACES)
            if major_latin is not None:
                theme.font_heading = major_latin.get('typeface', 'Calibri')

            # Minor font (body)
            minor_latin = root.find('.//a:fontScheme/a:minorFont/a:latin', self.NAMESPACES)
            if minor_latin is not None:
                theme.font_body = minor_latin.get('typeface', 'Calibri')

            # Script-specific fonts
            for font_elem in root.findall('.//a:fontScheme/a:majorFont/a:font', self.NAMESPACES):
                script = font_elem.get('script', '')
                typeface = font_elem.get('typeface', '')
                if script and typeface:
                    theme.font_scripts[script] = typeface

        except Exception:
            pass

    def _extract_format_scheme(self, root, theme: PPTXTheme) -> None:
        """Extract fill, line, and effect styles from fmtScheme."""
        try:
            # Fill styles
            fill_styles = root.findall('.//a:fmtScheme/a:fillStyleLst/*', self.NAMESPACES)
            for fill in fill_styles:
                theme.fill_styles.append({'tag': fill.tag.split('}')[-1]})

            # Line styles
            line_styles = root.findall('.//a:fmtScheme/a:lnStyleLst/*', self.NAMESPACES)
            for line in line_styles:
                w = line.get('w', '')
                theme.line_styles.append({'width': w})

            # Effect styles
            effect_styles = root.findall('.//a:fmtScheme/a:effectStyleLst/*', self.NAMESPACES)
            theme.effect_styles = [{'index': i} for i in range(len(effect_styles))]

        except Exception:
            pass

    def _extract_object_defaults(self, root, theme: PPTXTheme) -> None:
        """Extract object defaults (default shape fill, text styles)."""
        try:
            # Default shape fill
            sp_def = root.find('.//a:objectDefaults/a:spDef', self.NAMESPACES)
            if sp_def is not None:
                # Shape fill
                solid_fill = sp_def.find('.//a:spPr/a:solidFill/a:srgbClr', self.NAMESPACES)
                if solid_fill is not None:
                    theme.default_shape_fill = solid_fill.get('val', '44484C')

                # Bullet color
                bu_clr = sp_def.find('.//a:lstStyle//a:buClr/a:srgbClr', self.NAMESPACES)
                if bu_clr is not None:
                    theme.bullet_color = bu_clr.get('val', '595959')

                # Default text style
                def_rpr = sp_def.find('.//a:lstStyle//a:defRPr', self.NAMESPACES)
                if def_rpr is not None:
                    theme.default_text_style = {
                        'size': def_rpr.get('sz', '1200'),
                        'bold': def_rpr.get('b', '0'),
                    }

        except Exception:
            pass

    def _extract_layouts(self, zf: zipfile.ZipFile, theme: PPTXTheme) -> None:
        """Extract slide layout information."""
        try:
            # Find all slide layout files
            layout_files = [f for f in zf.namelist() if f.startswith('ppt/slideLayouts/slideLayout')]

            for layout_file in sorted(layout_files):
                try:
                    layout_xml = zf.read(layout_file)
                    root = etree.fromstring(layout_xml)

                    # Get layout name from cSld element
                    csld = root.find('.//p:cSld', self.NAMESPACES)
                    layout_name = csld.get('name', f'Layout {len(theme.layouts) + 1}') if csld is not None else f'Layout {len(theme.layouts) + 1}'

                    # Determine layout type based on placeholders
                    placeholders = self._extract_placeholders(root)
                    layout_type = self._determine_layout_type(placeholders)

                    layout_info = LayoutInfo(
                        name=layout_name,
                        layout_type=layout_type,
                        placeholders=placeholders,
                        recommended_use=self._get_recommended_use(layout_type),
                    )
                    theme.layouts.append(layout_info)

                except Exception:
                    pass

        except Exception:
            pass

    def _extract_placeholders(self, root) -> List[PlaceholderSpec]:
        """Extract placeholder specifications from a layout."""
        placeholders = []

        try:
            for sp in root.findall('.//p:sp', self.NAMESPACES):
                ph = sp.find('.//p:nvSpPr/p:nvPr/p:ph', self.NAMESPACES)
                if ph is not None:
                    ph_type = ph.get('type', 'body')
                    ph_idx = int(ph.get('idx', '0'))

                    # Get size from spPr/xfrm
                    xfrm = sp.find('.//p:spPr/a:xfrm', self.NAMESPACES)
                    width_emu = 0
                    height_emu = 0
                    x_emu = 0
                    y_emu = 0

                    if xfrm is not None:
                        ext = xfrm.find('a:ext', self.NAMESPACES)
                        off = xfrm.find('a:off', self.NAMESPACES)
                        if ext is not None:
                            width_emu = int(ext.get('cx', '0'))
                            height_emu = int(ext.get('cy', '0'))
                        if off is not None:
                            x_emu = int(off.get('x', '0'))
                            y_emu = int(off.get('y', '0'))

                    # Convert EMUs to inches (914400 EMUs per inch)
                    emu_per_inch = 914400
                    placeholder = PlaceholderSpec(
                        type=ph_type,
                        idx=ph_idx,
                        width_inches=width_emu / emu_per_inch if width_emu else 8.0,
                        height_inches=height_emu / emu_per_inch if height_emu else 5.0,
                        x_inches=x_emu / emu_per_inch if x_emu else 0.0,
                        y_inches=y_emu / emu_per_inch if y_emu else 0.0,
                    )

                    # Set constraints based on type
                    if ph_type in ['title', 'ctrTitle']:
                        placeholder.max_chars = 60
                        placeholder.max_lines = 2
                        placeholder.font_size_pt = 44
                    elif ph_type == 'subTitle':
                        placeholder.max_chars = 100
                        placeholder.max_lines = 3
                        placeholder.font_size_pt = 24
                    elif ph_type == 'body':
                        placeholder.max_chars = 500
                        placeholder.max_lines = 10
                        placeholder.max_bullets = 7
                        placeholder.font_size_pt = 18
                    elif ph_type == 'pic':
                        placeholder.max_chars = 0
                        placeholder.max_lines = 0

                    placeholders.append(placeholder)

        except Exception:
            pass

        return placeholders

    def _determine_layout_type(self, placeholders: List[PlaceholderSpec]) -> str:
        """Determine layout type based on placeholders present."""
        ph_types = {p.type for p in placeholders}

        if 'ctrTitle' in ph_types:
            return 'title_slide'
        elif 'pic' in ph_types and 'body' in ph_types:
            return 'image_text'
        elif len([p for p in placeholders if p.type == 'body']) >= 2:
            return 'two_column'
        elif 'title' in ph_types and 'body' in ph_types:
            return 'title_content'
        elif 'title' in ph_types and 'body' not in ph_types:
            return 'title_only'
        elif not ph_types or ph_types == {'body'}:
            return 'blank'
        else:
            return 'content'

    def _get_recommended_use(self, layout_type: str) -> str:
        """Get recommended use description for a layout type."""
        uses = {
            'title_slide': 'Opening slide, section dividers',
            'title_only': 'Section headers, emphasis slides',
            'title_content': 'Main content with bullet points',
            'two_column': 'Comparisons, side-by-side content',
            'image_text': 'Content with supporting image',
            'blank': 'Custom layouts, full-bleed images',
            'content': 'General content',
        }
        return uses.get(layout_type, 'General content')


class DOCXThemeExtractor(ThemeExtractor):
    """Extract theme from DOCX templates."""

    NAMESPACES = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    }

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.docx')

    def extract(self, template_path: str) -> DOCXTheme:
        """Extract theme from DOCX template."""
        theme = DOCXTheme()

        try:
            with zipfile.ZipFile(template_path, 'r') as zf:
                # Extract theme (if present)
                if 'word/theme/theme1.xml' in zf.namelist():
                    theme_xml = zf.read('word/theme/theme1.xml')
                    self._parse_theme_xml(theme_xml, theme)

                # Extract styles
                if 'word/styles.xml' in zf.namelist():
                    styles_xml = zf.read('word/styles.xml')
                    self._parse_styles_xml(styles_xml, theme)

                # Extract document settings
                if 'word/document.xml' in zf.namelist():
                    doc_xml = zf.read('word/document.xml')
                    self._parse_document_xml(doc_xml, theme)

        except Exception as e:
            print(f"Warning: Could not fully extract theme from {template_path}: {e}")

        return theme

    def _parse_theme_xml(self, xml_content: bytes, theme: DOCXTheme) -> None:
        """Parse theme XML for colors and fonts."""
        try:
            root = etree.fromstring(xml_content)

            # Extract colors (same as PPTX)
            for color_name in ['dk1', 'lt1', 'accent1']:
                xpath = f".//a:clrScheme/a:{color_name}"
                elem = root.find(xpath, {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                if elem is not None:
                    srgb = elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr')
                    if srgb is not None:
                        theme.colors[color_name] = srgb.get('val', '')

            # Extract fonts
            major = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}majorFont/{http://schemas.openxmlformats.org/drawingml/2006/main}latin')
            if major is not None:
                theme.font_heading = major.get('typeface', 'Calibri Light')

            minor = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}minorFont/{http://schemas.openxmlformats.org/drawingml/2006/main}latin')
            if minor is not None:
                theme.font_body = minor.get('typeface', 'Calibri')

        except Exception:
            pass

    def _parse_styles_xml(self, xml_content: bytes, theme: DOCXTheme) -> None:
        """Parse styles.xml for document styles."""
        try:
            root = etree.fromstring(xml_content)

            for style in root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}style'):
                style_id = style.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}styleId', '')
                if style_id:
                    theme.styles[style_id] = {'id': style_id}

        except Exception:
            pass

    def _parse_document_xml(self, xml_content: bytes, theme: DOCXTheme) -> None:
        """Parse document.xml for page settings."""
        # Page settings would be in sectPr
        pass


class XLSXThemeExtractor(ThemeExtractor):
    """Extract theme from XLSX templates."""

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.xlsx')

    def extract(self, template_path: str) -> XLSXTheme:
        """Extract theme from XLSX template."""
        theme = XLSXTheme()

        try:
            with zipfile.ZipFile(template_path, 'r') as zf:
                # Extract theme
                if 'xl/theme/theme1.xml' in zf.namelist():
                    theme_xml = zf.read('xl/theme/theme1.xml')
                    self._parse_theme_xml(theme_xml, theme)

                # Extract styles
                if 'xl/styles.xml' in zf.namelist():
                    styles_xml = zf.read('xl/styles.xml')
                    self._parse_styles_xml(styles_xml, theme)

        except Exception as e:
            print(f"Warning: Could not fully extract theme from {template_path}: {e}")

        return theme

    def _parse_theme_xml(self, xml_content: bytes, theme: XLSXTheme) -> None:
        """Parse theme XML."""
        try:
            root = etree.fromstring(xml_content)
            ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}

            for color_name in ['dk1', 'lt1', 'accent1']:
                elem = root.find(f".//a:clrScheme/a:{color_name}", ns)
                if elem is not None:
                    srgb = elem.find('.//a:srgbClr', ns)
                    if srgb is not None:
                        theme.colors[color_name] = srgb.get('val', '')

        except Exception:
            pass

    def _parse_styles_xml(self, xml_content: bytes, theme: XLSXTheme) -> None:
        """Parse styles.xml for cell styles."""
        # Would parse numFmts, fonts, fills, cellStyles
        pass


class PDFThemeExtractor(ThemeExtractor):
    """Extract theme from HTML/CSS templates for PDF generation."""

    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.html', '.css'))

    def extract(self, template_path: str) -> PDFTheme:
        """Extract theme from HTML/CSS template."""
        theme = PDFTheme()

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if template_path.endswith('.css'):
                self._parse_css(content, theme)
            else:
                # Extract embedded CSS from HTML
                css_matches = re.findall(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
                for css in css_matches:
                    self._parse_css(css, theme)

        except Exception as e:
            print(f"Warning: Could not extract theme from {template_path}: {e}")

        return theme

    def _parse_css(self, css_content: str, theme: PDFTheme) -> None:
        """Parse CSS for colors and fonts."""
        # Extract CSS variables
        var_pattern = r'--([a-zA-Z-]+):\s*([^;]+);'
        for match in re.finditer(var_pattern, css_content):
            var_name, value = match.groups()
            value = value.strip()

            if 'primary' in var_name:
                theme.primary_color = value
            elif 'secondary' in var_name:
                theme.secondary_color = value
            elif 'text' in var_name and 'color' in var_name:
                theme.text_color = value
            elif 'background' in var_name:
                theme.background_color = value

        # Extract font-family
        font_pattern = r'font-family:\s*([^;]+);'
        fonts = re.findall(font_pattern, css_content)
        if fonts:
            theme.font_body = fonts[0].split(',')[0].strip().strip('"\'')


class ThemeExtractorFactory:
    """Factory to get appropriate theme extractor for a file type."""

    _extractors: List[ThemeExtractor] = [
        PPTXThemeExtractor(),
        DOCXThemeExtractor(),
        XLSXThemeExtractor(),
        PDFThemeExtractor(),
    ]

    @classmethod
    def get_extractor(cls, file_path: str) -> Optional[ThemeExtractor]:
        """Get the appropriate extractor for a file."""
        for extractor in cls._extractors:
            if extractor.supports(file_path):
                return extractor
        return None

    @classmethod
    def extract(cls, file_path: str) -> Optional[BaseTheme]:
        """Extract theme from a file using the appropriate extractor."""
        extractor = cls.get_extractor(file_path)
        if extractor:
            return extractor.extract(file_path)
        return None

    @classmethod
    def extract_to_profile(cls, file_path: str) -> Optional[ThemeProfile]:
        """Extract theme and convert to ThemeProfile for backward compatibility."""
        theme = cls.extract(file_path)
        if isinstance(theme, PPTXTheme):
            return theme.to_theme_profile()
        elif theme:
            # Generic conversion for other types
            return ThemeProfile(
                primary=theme.get_primary_color(),
                font_heading=theme.get_heading_font(),
                font_body=theme.get_body_font(),
            )
        return None
