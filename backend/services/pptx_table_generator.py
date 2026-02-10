"""
PPTX Native Table Generator
============================

Generates native PowerPoint tables with theme-aware styling.
Uses python-pptx to create real editable tables, not images.
"""

import re
from typing import List, Optional, Tuple, Dict, Any

import structlog
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.table import Table, _Cell

logger = structlog.get_logger(__name__)


def hex_to_rgb(hex_color: str) -> RGBColor:
    """Convert hex color to RGBColor."""
    hex_color = hex_color.lstrip('#')
    return RGBColor(
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    )


class PPTXTableGenerator:
    """
    Generator for native PowerPoint tables.

    Creates professional, theme-colored tables from structured data.
    """

    def __init__(self, theme_colors: Dict[str, str]):
        """
        Initialize table generator with theme colors.

        Args:
            theme_colors: Dict with 'primary', 'secondary', 'accent', 'text' colors
        """
        self.theme_colors = theme_colors
        self.primary_color = hex_to_rgb(theme_colors.get('primary', '#3D5A80'))
        self.secondary_color = hex_to_rgb(theme_colors.get('secondary', '#98C1D9'))
        self.accent_color = hex_to_rgb(theme_colors.get('accent', '#EE6C4D'))
        self.text_color = hex_to_rgb(theme_colors.get('text', '#2D3748'))

    def add_table_to_slide(
        self,
        slide,
        table_data: List[List[str]],
        left: Optional[Emu] = None,
        top: Optional[Emu] = None,
        width: Optional[Emu] = None,
        height: Optional[Emu] = None,
        has_header: bool = True,
        font_size: int = 11,
        cell_padding: Tuple[Inches, Inches] = (Inches(0.1), Inches(0.05)),
    ):
        """
        Add a formatted table to a slide.

        Args:
            slide: pptx.slide.Slide object
            table_data: 2D list of cell values (first row = headers if has_header)
            left: Left position (defaults to centered)
            top: Top position (defaults to 2 inches from top)
            width: Table width (defaults to slide width - margins)
            height: Table height (auto-calculated if not provided)
            has_header: Whether first row is a header row
            font_size: Base font size in points
            cell_padding: (horizontal, vertical) padding

        Returns:
            The created Table shape
        """
        if not table_data or not table_data[0]:
            logger.warning("Empty table data provided")
            return None

        rows = len(table_data)
        cols = max(len(row) for row in table_data)

        # Default positioning
        if left is None:
            left = Inches(0.5)
        if top is None:
            top = Inches(2.0)
        if width is None:
            width = Inches(9.0)  # Standard slide width - margins
        if height is None:
            # Auto-calculate based on row count
            row_height = Inches(0.4)
            height = row_height * rows

        # Create the table
        table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
        table = table_shape.table

        # Set column widths (equal distribution)
        col_width = width // cols
        for i in range(cols):
            table.columns[i].width = col_width

        # Style each cell
        for row_idx, row_data in enumerate(table_data):
            is_header = has_header and row_idx == 0
            is_even_row = row_idx % 2 == 0

            for col_idx, cell_value in enumerate(row_data):
                if col_idx >= cols:
                    break

                cell = table.cell(row_idx, col_idx)
                self._style_cell(
                    cell,
                    str(cell_value) if cell_value else "",
                    is_header=is_header,
                    is_even_row=is_even_row,
                    font_size=font_size,
                )

            # Fill remaining columns if row is shorter
            for col_idx in range(len(row_data), cols):
                cell = table.cell(row_idx, col_idx)
                self._style_cell(
                    cell,
                    "",
                    is_header=is_header,
                    is_even_row=is_even_row,
                    font_size=font_size,
                )

        logger.info("Created table", rows=rows, cols=cols)
        return table_shape

    def _style_cell(
        self,
        cell: _Cell,
        text: str,
        is_header: bool = False,
        is_even_row: bool = False,
        font_size: int = 11,
    ):
        """
        Style a single table cell.

        Args:
            cell: pptx table cell
            text: Cell text content
            is_header: Whether this is a header cell
            is_even_row: Whether this is an even row (for alternating colors)
            font_size: Font size in points
        """
        # Set text
        cell.text = text
        paragraph = cell.text_frame.paragraphs[0]
        run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()

        # Font styling
        run.font.size = Pt(font_size + (2 if is_header else 0))
        run.font.bold = is_header

        # Cell background
        if is_header:
            cell.fill.solid()
            cell.fill.fore_color.rgb = self.primary_color
            run.font.color.rgb = RGBColor(255, 255, 255)  # White text on header
            paragraph.alignment = PP_ALIGN.CENTER
        else:
            if is_even_row:
                cell.fill.solid()
                # Use a light tint of secondary color
                cell.fill.fore_color.rgb = RGBColor(248, 249, 250)  # Very light gray
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255)  # White
            run.font.color.rgb = self.text_color
            paragraph.alignment = PP_ALIGN.LEFT

        # Vertical alignment
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    @staticmethod
    def detect_tabular_content(text: str) -> Optional[List[List[str]]]:
        """
        Detect and parse tabular content from text.

        Supports:
        - Markdown tables (|col1|col2|)
        - Key-value pairs (key: value)
        - Tab-separated values
        - Comma-separated lists with patterns

        Args:
            text: Text content to analyze

        Returns:
            2D list of table data if tabular content found, None otherwise
        """
        lines = text.strip().split('\n')

        # Try markdown table detection
        md_table = PPTXTableGenerator._parse_markdown_table(lines)
        if md_table:
            return md_table

        # Try key-value detection
        kv_table = PPTXTableGenerator._parse_key_value_pairs(lines)
        if kv_table:
            return kv_table

        # Try structured list detection
        list_table = PPTXTableGenerator._parse_structured_list(lines)
        if list_table:
            return list_table

        return None

    @staticmethod
    def _parse_markdown_table(lines: List[str]) -> Optional[List[List[str]]]:
        """Parse markdown table format."""
        table_data = []
        in_table = False

        for line in lines:
            line = line.strip()
            if '|' in line:
                # Skip separator rows
                if re.match(r'^[\|\s\-:]+$', line):
                    continue
                # Parse row
                cells = [c.strip() for c in line.split('|')]
                # Remove empty first/last cells from |col1|col2|
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                if cells:
                    table_data.append(cells)
                    in_table = True
            elif in_table:
                # End of table
                break

        if len(table_data) >= 2:  # At least header + 1 row
            return table_data
        return None

    @staticmethod
    def _parse_key_value_pairs(lines: List[str]) -> Optional[List[List[str]]]:
        """Parse key: value pairs into a 2-column table.

        Guards against matching bullet prose that incidentally contains colons
        by validating key/value lengths and requiring a high ratio of KV lines.
        """
        pairs = []
        kv_pattern = re.compile(r'^([^:]+):\s*(.+)$')

        total_non_empty = sum(1 for line in lines if line.strip())

        for line in lines:
            line = line.strip().lstrip('-*•● ').strip()
            match = kv_pattern.match(line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                # Reject keys that are too long (real keys are short labels)
                if len(key.split()) > 4:
                    continue
                # Reject values that are too long (sentence continuations)
                if len(value.split()) > 15:
                    continue
                pairs.append([key, value])

        # Require at least 3 KV pairs AND >50% of lines must be KV pairs
        if len(pairs) >= 3:
            if total_non_empty > 0 and len(pairs) / total_non_empty < 0.5:
                return None  # Less than half are KV pairs → not a real table
            return [["Attribute", "Value"]] + pairs
        return None

    @staticmethod
    def _parse_structured_list(lines: List[str]) -> Optional[List[List[str]]]:
        """
        Parse structured lists (numbered items with consistent structure).
        E.g., "1. Item - Description - Value"
        """
        data = []
        list_pattern = re.compile(r'^[\d]+[\.\)]\s*(.+)$')

        for line in lines:
            line = line.strip()
            match = list_pattern.match(line)
            if match:
                content = match.group(1)
                # Try to split by common delimiters
                if ' - ' in content:
                    parts = [p.strip() for p in content.split(' - ')]
                    data.append(parts)
                elif '\t' in content:
                    parts = [p.strip() for p in content.split('\t')]
                    data.append(parts)

        if len(data) >= 2 and all(len(row) == len(data[0]) for row in data):
            # Create header based on number of columns
            num_cols = len(data[0])
            headers = [f"Column {i+1}" for i in range(num_cols)]
            return [headers] + data

        return None

    @staticmethod
    def should_render_as_table(text: str) -> bool:
        """
        Determine if text content should be rendered as a table.

        Args:
            text: Text content to analyze

        Returns:
            True if content appears to be tabular
        """
        # Quick checks
        if not text or len(text) < 20:
            return False

        # Check for markdown table indicators
        if '|' in text and text.count('|') >= 4:
            lines = text.split('\n')
            pipe_lines = sum(1 for line in lines if '|' in line)
            if pipe_lines >= 2:
                return True

        # Check for key-value patterns
        lines = text.strip().split('\n')
        kv_pattern = re.compile(r'^[^:]+:\s+.+$')
        kv_count = sum(1 for line in lines if kv_pattern.match(line.strip()))
        if kv_count >= 3:
            return True

        return False

    def extract_and_remove_table(self, text: str) -> Tuple[str, Optional[List[List[str]]]]:
        """
        Extract table data from text and return the remaining text.

        Args:
            text: Text that may contain tabular content

        Returns:
            Tuple of (remaining_text, table_data)
        """
        table_data = self.detect_tabular_content(text)
        if not table_data:
            return text, None

        # Remove the table portion from text
        # This is a simplified version - could be improved
        lines = text.split('\n')
        non_table_lines = []
        in_table = False

        for line in lines:
            if '|' in line or re.match(r'^[^:]+:\s+.+$', line.strip()):
                in_table = True
                continue
            if in_table and not line.strip():
                in_table = False
            if not in_table:
                non_table_lines.append(line)

        remaining_text = '\n'.join(non_table_lines).strip()
        return remaining_text, table_data


# Helper function for easy integration
def create_table_generator(theme_colors: Dict[str, str]) -> PPTXTableGenerator:
    """Create a new table generator with the given theme."""
    return PPTXTableGenerator(theme_colors)
