"""
AIDocumentIndexer - Chart Generation API Routes
================================================

API endpoints for generating charts and visualizations.
"""

from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
import math
from pydantic import BaseModel, Field, field_validator
import structlog

from backend.api.deps import get_current_user

from backend.services.chart_generator import (
    ChartGenerator,
    ChartData,
    ChartStyle,
    ChartType,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/charts", tags=["charts"])

# Global chart generator instance
_chart_generator: Optional[ChartGenerator] = None


def get_chart_generator() -> ChartGenerator:
    """Get or create chart generator instance."""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator()
    return _chart_generator


# =============================================================================
# Request/Response Models
# =============================================================================

class ChartTypeEnum(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HORIZONTAL_BAR = "horizontal_bar"
    STACKED_BAR = "stacked_bar"


class ChartDataRequest(BaseModel):
    """Data for chart generation."""
    labels: List[str] = Field(..., max_length=10000, description="Labels for chart data points")
    values: List[float] = Field(..., max_length=10000, description="Numeric values for chart")
    series_name: str = Field("Data", max_length=200, description="Name of the data series")
    additional_series: Optional[Dict[str, List[float]]] = Field(
        None, description="Additional data series for multi-series charts (max 10 series)"
    )

    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v):
        for label in v:
            if len(label) > 1000:
                raise ValueError('Individual label must be <= 1000 characters')
        return v

    @field_validator('values')
    @classmethod
    def validate_values(cls, v):
        for val in v:
            if math.isinf(val) or math.isnan(val):
                raise ValueError('Infinity and NaN values are not allowed')
        return v


class ChartStyleRequest(BaseModel):
    """Styling options for charts."""
    title: str = Field("", description="Chart title")
    xlabel: str = Field("", description="X-axis label")
    ylabel: str = Field("", description="Y-axis label")
    colors: Optional[List[str]] = Field(None, description="Custom colors (hex codes)")
    width: int = Field(10, ge=4, le=20, description="Chart width in inches")
    height: int = Field(6, ge=3, le=15, description="Chart height in inches")
    dpi: int = Field(150, ge=72, le=300, description="Resolution (DPI)")
    show_legend: bool = Field(True, description="Show legend")
    show_grid: bool = Field(True, description="Show grid lines")
    font_size: int = Field(12, ge=8, le=24, description="Base font size")


class GenerateChartRequest(BaseModel):
    """Request to generate a chart."""
    chart_type: ChartTypeEnum = Field(..., description="Type of chart to generate")
    data: ChartDataRequest = Field(..., description="Chart data")
    style: Optional[ChartStyleRequest] = Field(None, description="Chart styling options")
    format: str = Field("png", description="Output format (png, svg, pdf)")


class ChartInfoResponse(BaseModel):
    """Response with chart metadata."""
    chart_type: str
    title: str
    data_points: int
    format: str
    width: int
    height: int


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/generate", response_class=Response)
async def generate_chart(request: GenerateChartRequest, user: dict = Depends(get_current_user)):
    """
    Generate a chart from provided data.

    Returns the chart image in the specified format (PNG, SVG, or PDF).
    """
    if len(request.data.labels) != len(request.data.values):
        raise HTTPException(status_code=400, detail="Labels and values must have the same length")

    if request.data.additional_series:
        if len(request.data.additional_series) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 additional series allowed")
        for name, series_values in request.data.additional_series.items():
            if len(series_values) > 10000:
                raise HTTPException(status_code=400, detail=f"Series '{name}' exceeds maximum of 10000 data points")

    logger.info(
        "Generating chart",
        chart_type=request.chart_type.value,
        data_points=len(request.data.labels),
    )

    try:
        generator = get_chart_generator()

        # Convert request to service models
        chart_data = ChartData(
            labels=request.data.labels,
            values=request.data.values,
            series_name=request.data.series_name,
            additional_series=request.data.additional_series or {},
        )

        chart_style = None
        if request.style:
            chart_style = ChartStyle(
                title=request.style.title,
                xlabel=request.style.xlabel,
                ylabel=request.style.ylabel,
                colors=request.style.colors,
                figsize=(request.style.width, request.style.height),
                dpi=request.style.dpi,
                show_legend=request.style.show_legend,
                show_grid=request.style.show_grid,
                font_size=request.style.font_size,
            )

        # Map enum to ChartType
        chart_type_map = {
            ChartTypeEnum.BAR: ChartType.BAR,
            ChartTypeEnum.LINE: ChartType.LINE,
            ChartTypeEnum.PIE: ChartType.PIE,
            ChartTypeEnum.SCATTER: ChartType.SCATTER,
            ChartTypeEnum.AREA: ChartType.AREA,
            ChartTypeEnum.HORIZONTAL_BAR: ChartType.HORIZONTAL_BAR,
            ChartTypeEnum.STACKED_BAR: ChartType.STACKED_BAR,
        }

        # Generate chart using the unified method
        result = generator.generate_chart(
            chart_type=chart_type_map[request.chart_type],
            data=chart_data,
            style=chart_style,
        )

        # Determine content type
        content_type_map = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
        }
        content_type = content_type_map.get(request.format, "image/png")

        logger.info(
            "Chart generated successfully",
            chart_type=request.chart_type.value,
            size_bytes=len(result.image_bytes),
        )

        return Response(
            content=result.image_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename=chart.{request.format}",
            },
        )

    except ValueError as e:
        logger.warning("Invalid chart request", error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chart request")
    except Exception as e:
        logger.error("Chart generation failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Chart generation failed")


@router.get("/types")
async def list_chart_types(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    List available chart types with descriptions.
    """
    return {
        "chart_types": [
            {
                "type": "bar",
                "name": "Bar Chart",
                "description": "Vertical bar chart for comparing categories",
                "best_for": "Comparing discrete categories",
            },
            {
                "type": "horizontal_bar",
                "name": "Horizontal Bar Chart",
                "description": "Horizontal bar chart for long category names",
                "best_for": "Categories with long labels",
            },
            {
                "type": "line",
                "name": "Line Chart",
                "description": "Line chart for trends over time",
                "best_for": "Time series and trends",
            },
            {
                "type": "pie",
                "name": "Pie Chart",
                "description": "Pie chart for showing proportions",
                "best_for": "Part-to-whole relationships (5-7 categories max)",
            },
            {
                "type": "scatter",
                "name": "Scatter Plot",
                "description": "Scatter plot for showing correlations",
                "best_for": "Correlation between two variables",
            },
            {
                "type": "area",
                "name": "Area Chart",
                "description": "Filled area chart for cumulative data",
                "best_for": "Volume changes over time",
            },
            {
                "type": "stacked_bar",
                "name": "Stacked Bar Chart",
                "description": "Stacked bars for multi-series comparison",
                "best_for": "Comparing compositions across categories",
            },
        ],
        "supported_formats": ["png", "svg", "pdf"],
    }


@router.get("/health")
async def chart_service_health() -> Dict[str, Any]:
    """Check chart generation service health."""
    try:
        generator = get_chart_generator()

        # Test matplotlib availability
        generator._get_matplotlib()
        generator._get_numpy()

        return {
            "status": "healthy",
            "matplotlib_available": True,
            "numpy_available": True,
        }
    except ImportError as e:
        return {
            "status": "degraded",
            "error": f"Missing dependency: {str(e)}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
