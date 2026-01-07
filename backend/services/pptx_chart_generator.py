"""
PPTX Native Chart Generator
============================

Generates native PowerPoint charts using python-pptx.
Supports column, bar, line, pie, and area charts with theme colors.
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import structlog
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData, ChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_THEME_COLOR

logger = structlog.get_logger(__name__)


class ChartType(str, Enum):
    """Supported chart types."""
    COLUMN = "column"
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    AREA = "area"
    CLUSTERED_BAR = "clustered_bar"
    STACKED_COLUMN = "stacked_column"


def hex_to_rgb(hex_color: str) -> RGBColor:
    """Convert hex color to RGBColor."""
    hex_color = hex_color.lstrip('#')
    return RGBColor(
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    )


class PPTXNativeChartGenerator:
    """
    Generator for native PowerPoint charts.

    Creates professional, theme-colored charts from extracted data.
    """

    # Mapping of chart type enum to python-pptx chart type
    CHART_TYPE_MAP = {
        ChartType.COLUMN: XL_CHART_TYPE.COLUMN_CLUSTERED,
        ChartType.BAR: XL_CHART_TYPE.BAR_CLUSTERED,
        ChartType.LINE: XL_CHART_TYPE.LINE,
        ChartType.PIE: XL_CHART_TYPE.PIE,
        ChartType.AREA: XL_CHART_TYPE.AREA,
        ChartType.CLUSTERED_BAR: XL_CHART_TYPE.BAR_CLUSTERED,
        ChartType.STACKED_COLUMN: XL_CHART_TYPE.COLUMN_STACKED,
    }

    def __init__(self, theme_colors: Dict[str, str]):
        """
        Initialize chart generator with theme colors.

        Args:
            theme_colors: Dict with 'primary', 'secondary', 'accent', 'text' colors
        """
        self.theme_colors = theme_colors
        self.primary_color = hex_to_rgb(theme_colors.get('primary', '#3D5A80'))
        self.secondary_color = hex_to_rgb(theme_colors.get('secondary', '#98C1D9'))
        self.accent_color = hex_to_rgb(theme_colors.get('accent', '#EE6C4D'))
        self.text_color = hex_to_rgb(theme_colors.get('text', '#2D3748'))

        # Chart color palette derived from theme
        self.chart_colors = [
            self.primary_color,
            self.secondary_color,
            self.accent_color,
            hex_to_rgb('#66C2A5'),  # Teal
            hex_to_rgb('#FC8D62'),  # Orange
            hex_to_rgb('#8DA0CB'),  # Blue-gray
        ]

    def add_chart_to_slide(
        self,
        slide,
        chart_data: "ChartDataStructure",
        chart_type: ChartType = ChartType.COLUMN,
        left: Optional[int] = None,
        top: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: Optional[str] = None,
        show_legend: bool = True,
    ):
        """
        Add a formatted chart to a slide.

        Args:
            slide: pptx.slide.Slide object
            chart_data: ChartDataStructure with categories and series
            chart_type: Type of chart to create
            left: Left position (defaults to centered)
            top: Top position
            width: Chart width
            height: Chart height
            title: Chart title
            show_legend: Whether to show legend

        Returns:
            The created chart shape
        """
        # Default positioning
        if left is None:
            left = Inches(0.75)
        if top is None:
            top = Inches(2.0)
        if width is None:
            width = Inches(8.5)
        if height is None:
            height = Inches(4.5)

        # Create chart data
        pptx_chart_data = CategoryChartData()
        pptx_chart_data.categories = chart_data.categories

        for series in chart_data.series:
            pptx_chart_data.add_series(series.name, series.values)

        # Get chart type
        xl_chart_type = self.CHART_TYPE_MAP.get(
            chart_type, XL_CHART_TYPE.COLUMN_CLUSTERED
        )

        # Add chart to slide
        chart_shape = slide.shapes.add_chart(
            xl_chart_type, left, top, width, height, pptx_chart_data
        )
        chart = chart_shape.chart

        # Style the chart
        self._style_chart(chart, title, show_legend, chart_type)

        logger.info(
            "Created chart",
            type=chart_type.value,
            categories=len(chart_data.categories),
            series=len(chart_data.series),
        )
        return chart_shape

    def _style_chart(
        self,
        chart,
        title: Optional[str],
        show_legend: bool,
        chart_type: ChartType,
    ):
        """Apply theme styling to the chart."""
        # Title
        if title:
            chart.has_title = True
            chart.chart_title.text_frame.text = title
            chart.chart_title.text_frame.paragraphs[0].font.size = Pt(14)
            chart.chart_title.text_frame.paragraphs[0].font.bold = True
        else:
            chart.has_title = False

        # Legend
        if show_legend and chart_type != ChartType.PIE:
            chart.has_legend = True
            chart.legend.position = XL_LEGEND_POSITION.BOTTOM
            chart.legend.include_in_layout = False
        elif chart_type == ChartType.PIE:
            chart.has_legend = True
            chart.legend.position = XL_LEGEND_POSITION.RIGHT

        # Apply colors to series
        try:
            for i, series in enumerate(chart.series):
                color = self.chart_colors[i % len(self.chart_colors)]
                if chart_type == ChartType.LINE:
                    series.format.line.color.rgb = color
                    series.format.line.width = Pt(2.5)
                else:
                    fill = series.format.fill
                    fill.solid()
                    fill.fore_color.rgb = color
        except Exception as e:
            logger.warning("Could not apply chart colors", error=str(e))

    @staticmethod
    def detect_chartable_data(text: str) -> Optional["ChartDataStructure"]:
        """
        Detect and extract chartable data from text.

        Supports:
        - Numeric lists with labels
        - Percentage data
        - Time series data
        - Comparison data

        Args:
            text: Text content to analyze

        Returns:
            ChartDataStructure if chartable data found, None otherwise
        """
        lines = text.strip().split('\n')

        # Try pattern matching for numeric data
        numeric_data = PPTXNativeChartGenerator._extract_numeric_patterns(lines)
        if numeric_data:
            return numeric_data

        # Try percentage data
        percentage_data = PPTXNativeChartGenerator._extract_percentage_patterns(lines)
        if percentage_data:
            return percentage_data

        return None

    @staticmethod
    def _extract_numeric_patterns(lines: List[str]) -> Optional["ChartDataStructure"]:
        """
        Extract numeric patterns like "Category: 100" or "Category - 100".
        """
        data_points = []

        # Pattern: "Label: number" or "Label - number" or "Label (number)"
        patterns = [
            re.compile(r'^([^:\d]+):\s*([\d,\.]+)\s*$'),
            re.compile(r'^([^-\d]+)\s*-\s*([\d,\.]+)\s*$'),
            re.compile(r'^([^\(\d]+)\s*\(([\d,\.]+)\)\s*$'),
            re.compile(r'^(.+?)[\s:]+\$?([\d,\.]+)%?$'),
        ]

        for line in lines:
            line = line.strip()
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    label = match.group(1).strip()
                    value_str = match.group(2).replace(',', '')
                    try:
                        value = float(value_str)
                        data_points.append((label, value))
                        break
                    except ValueError:
                        continue

        if len(data_points) >= 3:
            categories = [dp[0] for dp in data_points]
            values = [dp[1] for dp in data_points]
            return ChartDataStructure(
                categories=categories,
                series=[SeriesData(name="Values", values=values)],
            )
        return None

    @staticmethod
    def _extract_percentage_patterns(lines: List[str]) -> Optional["ChartDataStructure"]:
        """Extract percentage data for pie charts."""
        data_points = []

        # Pattern: "Label: XX%" or "Label - XX%"
        pattern = re.compile(r'^([^:\d%]+)[:\-\s]+([\d\.]+)\s*%')

        for line in lines:
            match = pattern.match(line.strip())
            if match:
                label = match.group(1).strip()
                try:
                    value = float(match.group(2))
                    data_points.append((label, value))
                except ValueError:
                    continue

        if len(data_points) >= 2:
            categories = [dp[0] for dp in data_points]
            values = [dp[1] for dp in data_points]
            return ChartDataStructure(
                categories=categories,
                series=[SeriesData(name="Percentage", values=values)],
                suggested_type=ChartType.PIE,
            )
        return None

    @staticmethod
    def suggest_chart_type(data: "ChartDataStructure") -> ChartType:
        """
        Suggest the best chart type based on data characteristics.

        Args:
            data: Chart data structure

        Returns:
            Recommended chart type
        """
        if data.suggested_type:
            return data.suggested_type

        num_categories = len(data.categories)
        num_series = len(data.series)

        # Pie chart for small category sets with single series
        if num_categories <= 8 and num_series == 1:
            total = sum(data.series[0].values)
            if all(0 <= v <= 100 for v in data.series[0].values):
                return ChartType.PIE

        # Bar chart for many categories
        if num_categories > 6:
            return ChartType.BAR

        # Line chart for time-series-like data
        if num_series > 1 or num_categories > 8:
            return ChartType.LINE

        # Default to column chart
        return ChartType.COLUMN

    @staticmethod
    def should_render_as_chart(text: str) -> bool:
        """
        Determine if text content should be rendered as a chart.

        Args:
            text: Text content to analyze

        Returns:
            True if content appears to be chartable numeric data
        """
        if not text or len(text) < 20:
            return False

        lines = text.strip().split('\n')

        # Count lines with numeric data
        numeric_pattern = re.compile(r'[\d,\.]+\s*%?')
        numeric_lines = sum(
            1 for line in lines
            if numeric_pattern.search(line) and len(line) > 5
        )

        # Should have at least 3 data points
        return numeric_lines >= 3


class SeriesData:
    """Data for a single chart series."""

    def __init__(self, name: str, values: List[float]):
        self.name = name
        self.values = values


class ChartDataStructure:
    """Structured data for creating a chart."""

    def __init__(
        self,
        categories: List[str],
        series: List[SeriesData],
        suggested_type: Optional[ChartType] = None,
    ):
        self.categories = categories
        self.series = series
        self.suggested_type = suggested_type


# Helper function for easy integration
def create_chart_generator(theme_colors: Dict[str, str]) -> PPTXNativeChartGenerator:
    """Create a new chart generator with the given theme."""
    return PPTXNativeChartGenerator(theme_colors)
