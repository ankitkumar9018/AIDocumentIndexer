"""
AIDocumentIndexer - Chart Generation Service
=============================================

Generates charts and visualizations from data extracted from documents.
Uses matplotlib for chart generation with support for:
- Bar charts
- Line charts
- Pie charts
- Scatter plots
- Area charts

Charts can be embedded in generated documents (PPTX, DOCX, PDF).
"""

import io
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HORIZONTAL_BAR = "horizontal_bar"
    STACKED_BAR = "stacked_bar"


@dataclass
class ChartData:
    """Data for chart generation."""
    labels: List[str]
    values: List[float]
    series_name: str = "Data"
    # For multi-series charts
    additional_series: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class ChartStyle:
    """Styling options for charts."""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    colors: Optional[List[str]] = None
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 150
    show_legend: bool = True
    show_grid: bool = True
    font_size: int = 12
    title_font_size: int = 14
    # Corporate color scheme
    primary_color: str = "#1E3A5F"
    secondary_color: str = "#3D5A80"
    accent_color: str = "#98C1D9"


@dataclass
class GeneratedChart:
    """Result of chart generation."""
    chart_type: ChartType
    image_bytes: bytes
    format: str = "png"
    width: int = 0
    height: int = 0
    title: str = ""
    data_summary: str = ""


class ChartGenerator:
    """
    Generates charts from structured data.

    Uses matplotlib for rendering with a consistent style
    suitable for business documents.
    """

    # Default color palette for charts
    DEFAULT_COLORS = [
        "#1E3A5F",  # Dark blue
        "#3D5A80",  # Medium blue
        "#98C1D9",  # Light blue
        "#E0FBFC",  # Very light blue
        "#EE6C4D",  # Coral accent
        "#293241",  # Dark gray
        "#5C677D",  # Medium gray
        "#7D8597",  # Light gray
    ]

    def __init__(self):
        self._plt = None
        self._np = None

    def _get_matplotlib(self):
        """Lazy load matplotlib."""
        if self._plt is None:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            self._plt = plt
        return self._plt

    def _get_numpy(self):
        """Lazy load numpy."""
        if self._np is None:
            import numpy as np
            self._np = np
        return self._np

    def _apply_style(self, ax, style: ChartStyle):
        """Apply consistent styling to chart axes."""
        plt = self._get_matplotlib()

        if style.title:
            ax.set_title(style.title, fontsize=style.title_font_size, fontweight='bold')
        if style.xlabel:
            ax.set_xlabel(style.xlabel, fontsize=style.font_size)
        if style.ylabel:
            ax.set_ylabel(style.ylabel, fontsize=style.font_size)
        if style.show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=style.font_size - 2)

    def _get_colors(self, count: int, style: ChartStyle) -> List[str]:
        """Get color palette for chart elements."""
        if style.colors and len(style.colors) >= count:
            return style.colors[:count]

        # Use default colors, cycling if needed
        colors = []
        for i in range(count):
            colors.append(self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
        return colors

    def generate_bar_chart(
        self,
        data: ChartData,
        style: Optional[ChartStyle] = None,
    ) -> GeneratedChart:
        """Generate a vertical bar chart."""
        plt = self._get_matplotlib()
        np = self._get_numpy()
        style = style or ChartStyle()

        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

        colors = self._get_colors(len(data.labels), style)

        if data.additional_series:
            # Grouped bar chart
            x = np.arange(len(data.labels))
            width = 0.8 / (len(data.additional_series) + 1)

            # Plot main series
            bars = ax.bar(x - width * len(data.additional_series) / 2,
                         data.values, width, label=data.series_name, color=colors[0])

            # Plot additional series
            for i, (name, values) in enumerate(data.additional_series.items()):
                offset = width * (i + 1 - len(data.additional_series) / 2)
                ax.bar(x + offset, values, width, label=name,
                      color=colors[(i + 1) % len(colors)])

            ax.set_xticks(x)
            ax.set_xticklabels(data.labels, rotation=45, ha='right')
        else:
            # Simple bar chart
            ax.bar(data.labels, data.values, color=colors)
            plt.xticks(rotation=45, ha='right')

        self._apply_style(ax, style)

        if style.show_legend and data.additional_series:
            ax.legend()

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return GeneratedChart(
            chart_type=ChartType.BAR,
            image_bytes=image_bytes,
            format="png",
            width=int(style.figsize[0] * style.dpi),
            height=int(style.figsize[1] * style.dpi),
            title=style.title,
            data_summary=f"Bar chart with {len(data.labels)} categories",
        )

    def generate_horizontal_bar_chart(
        self,
        data: ChartData,
        style: Optional[ChartStyle] = None,
    ) -> GeneratedChart:
        """Generate a horizontal bar chart."""
        plt = self._get_matplotlib()
        style = style or ChartStyle()

        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

        colors = self._get_colors(len(data.labels), style)

        ax.barh(data.labels, data.values, color=colors)

        self._apply_style(ax, style)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return GeneratedChart(
            chart_type=ChartType.HORIZONTAL_BAR,
            image_bytes=image_bytes,
            format="png",
            width=int(style.figsize[0] * style.dpi),
            height=int(style.figsize[1] * style.dpi),
            title=style.title,
            data_summary=f"Horizontal bar chart with {len(data.labels)} categories",
        )

    def generate_line_chart(
        self,
        data: ChartData,
        style: Optional[ChartStyle] = None,
    ) -> GeneratedChart:
        """Generate a line chart."""
        plt = self._get_matplotlib()
        style = style or ChartStyle()

        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

        colors = self._get_colors(len(data.additional_series) + 1, style)

        # Plot main series
        ax.plot(data.labels, data.values, marker='o', color=colors[0],
                label=data.series_name, linewidth=2)

        # Plot additional series
        for i, (name, values) in enumerate(data.additional_series.items()):
            ax.plot(data.labels, values, marker='s',
                   color=colors[(i + 1) % len(colors)], label=name, linewidth=2)

        plt.xticks(rotation=45, ha='right')
        self._apply_style(ax, style)

        if style.show_legend:
            ax.legend()

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return GeneratedChart(
            chart_type=ChartType.LINE,
            image_bytes=image_bytes,
            format="png",
            width=int(style.figsize[0] * style.dpi),
            height=int(style.figsize[1] * style.dpi),
            title=style.title,
            data_summary=f"Line chart with {len(data.labels)} data points",
        )

    def generate_pie_chart(
        self,
        data: ChartData,
        style: Optional[ChartStyle] = None,
    ) -> GeneratedChart:
        """Generate a pie chart."""
        plt = self._get_matplotlib()
        style = style or ChartStyle()

        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

        colors = self._get_colors(len(data.labels), style)

        # Calculate percentages for display
        total = sum(data.values)
        percentages = [v / total * 100 for v in data.values]

        wedges, texts, autotexts = ax.pie(
            data.values,
            labels=data.labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(data.values),  # Slight separation
            shadow=True,
            startangle=90,
        )

        # Style the text
        for text in texts:
            text.set_fontsize(style.font_size - 2)
        for autotext in autotexts:
            autotext.set_fontsize(style.font_size - 2)
            autotext.set_color('white')
            autotext.set_weight('bold')

        if style.title:
            ax.set_title(style.title, fontsize=style.title_font_size, fontweight='bold')

        ax.axis('equal')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return GeneratedChart(
            chart_type=ChartType.PIE,
            image_bytes=image_bytes,
            format="png",
            width=int(style.figsize[0] * style.dpi),
            height=int(style.figsize[1] * style.dpi),
            title=style.title,
            data_summary=f"Pie chart with {len(data.labels)} segments",
        )

    def generate_scatter_chart(
        self,
        data: ChartData,
        style: Optional[ChartStyle] = None,
    ) -> GeneratedChart:
        """Generate a scatter plot."""
        plt = self._get_matplotlib()
        np = self._get_numpy()
        style = style or ChartStyle()

        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

        colors = self._get_colors(1, style)

        # Convert labels to numeric if possible, otherwise use indices
        try:
            x_values = [float(label) for label in data.labels]
        except ValueError:
            x_values = list(range(len(data.labels)))

        ax.scatter(x_values, data.values, c=colors[0], s=100, alpha=0.7, edgecolors='white')

        # Add trend line if enough points
        if len(x_values) >= 3:
            z = np.polyfit(x_values, data.values, 1)
            p = np.poly1d(z)
            ax.plot(x_values, p(x_values), "--", color=style.secondary_color, alpha=0.8)

        self._apply_style(ax, style)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return GeneratedChart(
            chart_type=ChartType.SCATTER,
            image_bytes=image_bytes,
            format="png",
            width=int(style.figsize[0] * style.dpi),
            height=int(style.figsize[1] * style.dpi),
            title=style.title,
            data_summary=f"Scatter plot with {len(data.labels)} points",
        )

    def generate_area_chart(
        self,
        data: ChartData,
        style: Optional[ChartStyle] = None,
    ) -> GeneratedChart:
        """Generate an area chart."""
        plt = self._get_matplotlib()
        style = style or ChartStyle()

        fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

        colors = self._get_colors(len(data.additional_series) + 1, style)

        # Plot main series
        ax.fill_between(range(len(data.labels)), data.values,
                       alpha=0.5, color=colors[0], label=data.series_name)
        ax.plot(range(len(data.labels)), data.values, color=colors[0], linewidth=2)

        # Plot additional series
        for i, (name, values) in enumerate(data.additional_series.items()):
            color = colors[(i + 1) % len(colors)]
            ax.fill_between(range(len(data.labels)), values,
                           alpha=0.3, color=color, label=name)
            ax.plot(range(len(data.labels)), values, color=color, linewidth=2)

        ax.set_xticks(range(len(data.labels)))
        ax.set_xticklabels(data.labels, rotation=45, ha='right')

        self._apply_style(ax, style)

        if style.show_legend:
            ax.legend()

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return GeneratedChart(
            chart_type=ChartType.AREA,
            image_bytes=image_bytes,
            format="png",
            width=int(style.figsize[0] * style.dpi),
            height=int(style.figsize[1] * style.dpi),
            title=style.title,
            data_summary=f"Area chart with {len(data.labels)} data points",
        )

    def generate_chart(
        self,
        chart_type: ChartType,
        data: ChartData,
        style: Optional[ChartStyle] = None,
    ) -> GeneratedChart:
        """
        Generate a chart of the specified type.

        Args:
            chart_type: Type of chart to generate
            data: Chart data (labels and values)
            style: Optional styling options

        Returns:
            GeneratedChart with image bytes
        """
        generators = {
            ChartType.BAR: self.generate_bar_chart,
            ChartType.HORIZONTAL_BAR: self.generate_horizontal_bar_chart,
            ChartType.LINE: self.generate_line_chart,
            ChartType.PIE: self.generate_pie_chart,
            ChartType.SCATTER: self.generate_scatter_chart,
            ChartType.AREA: self.generate_area_chart,
            ChartType.STACKED_BAR: self.generate_bar_chart,  # Handled by additional_series
        }

        generator = generators.get(chart_type)
        if not generator:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        return generator(data, style)


class DataExtractor:
    """
    Extracts chartable data from text content.

    Uses pattern matching and LLM assistance to find
    tabular data, percentages, and numeric comparisons.
    """

    # Patterns for extracting data
    PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')
    NUMBER_PATTERN = re.compile(r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[KkMmBb]illion)?')
    TABLE_ROW_PATTERN = re.compile(r'^\|(.+)\|$', re.MULTILINE)

    def __init__(self, llm_service=None):
        self._llm = llm_service

    def extract_percentages(self, text: str) -> List[Tuple[str, float]]:
        """Extract percentage values with context."""
        results = []

        # Find sentences with percentages
        sentences = text.split('.')
        for sentence in sentences:
            matches = self.PERCENTAGE_PATTERN.findall(sentence)
            if matches:
                # Try to extract context (words before the percentage)
                for match in matches:
                    value = float(match)
                    # Get context from the sentence
                    context = sentence.strip()[:50]
                    results.append((context, value))

        return results

    def extract_table_data(self, text: str) -> Optional[ChartData]:
        """Extract data from markdown tables."""
        rows = self.TABLE_ROW_PATTERN.findall(text)
        if len(rows) < 2:
            return None

        # Parse header and data rows
        header = [cell.strip() for cell in rows[0].split('|') if cell.strip()]

        # Skip separator row (contains dashes)
        data_rows = []
        for row in rows[1:]:
            if '---' in row:
                continue
            cells = [cell.strip() for cell in row.split('|') if cell.strip()]
            if cells:
                data_rows.append(cells)

        if not data_rows:
            return None

        # First column as labels, try to find numeric column for values
        labels = []
        values = []

        for row in data_rows:
            if len(row) >= 2:
                labels.append(row[0])
                # Try to parse the second column as numeric
                try:
                    # Remove currency symbols, commas, etc.
                    num_str = re.sub(r'[^\d.\-]', '', row[1])
                    values.append(float(num_str))
                except ValueError:
                    values.append(0.0)

        if labels and values:
            return ChartData(
                labels=labels,
                values=values,
                series_name=header[1] if len(header) > 1 else "Value",
            )

        return None

    async def extract_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Use LLM to extract chartable data from text."""
        if not self._llm:
            return None

        prompt = """Analyze the following text and extract any data that could be visualized in a chart.

Text:
{text}

If chartable data exists, respond with JSON in this format:
{{
    "chart_type": "bar|line|pie|scatter",
    "title": "Chart title",
    "labels": ["label1", "label2", ...],
    "values": [value1, value2, ...],
    "xlabel": "X axis label",
    "ylabel": "Y axis label"
}}

If no chartable data exists, respond with: {{"no_data": true}}

Respond only with JSON."""

        try:
            response = await self._llm.agenerate(
                prompt.format(text=text[:5000])  # Limit text length
            )

            # Parse JSON response
            content = response.content if hasattr(response, 'content') else str(response)
            # Clean up markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            data = json.loads(content.strip())

            if data.get("no_data"):
                return None

            return data
        except Exception as e:
            logger.warning("Failed to extract data with LLM", error=str(e))
            return None


# Singleton instance
_chart_generator: Optional[ChartGenerator] = None


def get_chart_generator() -> ChartGenerator:
    """Get the chart generator singleton."""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator()
    return _chart_generator
