"""
AIDocumentIndexer - Document Generation Data Models
=====================================================

Data models for document generation: formats, sections, jobs, and configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.services.generation.config import get_generation_setting, DEFAULT_OUTPUT_DIR


class OutputFormat(str, Enum):
    """Supported output formats."""
    PPTX = "pptx"
    DOCX = "docx"
    PDF = "pdf"
    XLSX = "xlsx"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"


class GenerationStatus(str, Enum):
    """Document generation workflow status."""
    DRAFT = "draft"
    OUTLINE_PENDING = "outline_pending"
    OUTLINE_APPROVED = "outline_approved"
    GENERATING = "generating"
    SECTION_REVIEW = "section_review"
    REVISION = "revision"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class SourceReference:
    """Reference to a source document used in generation."""
    document_id: str
    document_name: str
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    snippet: str = ""


@dataclass
class Section:
    """A section of the generated document."""
    id: str
    title: str
    content: str
    order: int
    sources: List[SourceReference] = field(default_factory=list)
    approved: bool = False
    feedback: Optional[str] = None
    revised_content: Optional[str] = None
    rendered_content: Optional[str] = None  # Final content after processing (what appears in output)
    metadata: Optional[Dict[str, Any]] = None  # Quality scores, etc.


@dataclass
class DocumentOutline:
    """Outline for document generation."""
    title: str
    description: str
    sections: List[Dict[str, str]]  # List of {title, description}
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    word_count_target: Optional[int] = None


@dataclass
class GenerationJob:
    """A document generation job."""
    id: str
    user_id: str
    title: str
    description: str
    output_format: OutputFormat
    status: GenerationStatus
    outline: Optional[DocumentOutline] = None
    sections: List[Section] = field(default_factory=list)
    sources_used: List[SourceReference] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for document generation."""
    # LLM settings
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000

    # RAG settings
    use_rag: bool = True
    max_sources: int = 10
    min_relevance_score: float = 0.01  # RRF scores are typically 0-0.1, not 0-1

    # Output settings - loaded from database settings or environment
    # Use persistent storage instead of /tmp
    output_dir: str = DEFAULT_OUTPUT_DIR
    include_sources: bool = field(
        default_factory=lambda: get_generation_setting(
            "generation.include_sources", "GENERATION_INCLUDE_SOURCES", True
        )
    )
    include_toc: bool = True

    # Image generation settings - loaded from database settings
    # Configure via Admin UI: Settings > Document Generation
    include_images: bool = field(
        default_factory=lambda: get_generation_setting(
            "generation.include_images", "GENERATION_INCLUDE_IMAGES", True
        )
    )
    # Image backend: "picsum" (free, no API key), "unsplash" (requires API key),
    # "pexels" (requires API key), "openai" (DALL-E, requires API key),
    # "stability" (Stable Diffusion API), "automatic1111" (local SD), or "disabled"
    image_backend: str = field(
        default_factory=lambda: get_generation_setting(
            "generation.image_backend", "GENERATION_IMAGE_BACKEND", "picsum"
        )
    )

    # Style settings - loaded from database settings
    default_tone: str = field(
        default_factory=lambda: get_generation_setting(
            "generation.default_tone", "GENERATION_DEFAULT_TONE", "professional"
        )
    )
    default_style: str = field(
        default_factory=lambda: get_generation_setting(
            "generation.default_style", "GENERATION_DEFAULT_STYLE", "business"
        )
    )

    # Chart generation settings
    auto_charts: bool = field(
        default_factory=lambda: get_generation_setting(
            "generation.auto_charts", "GENERATION_AUTO_CHARTS", False
        )
    )
    chart_style: str = field(
        default_factory=lambda: get_generation_setting(
            "generation.chart_style", "GENERATION_CHART_STYLE", "business"
        )
    )
    chart_dpi: int = field(
        default_factory=lambda: get_generation_setting(
            "generation.chart_dpi", "GENERATION_CHART_DPI", 150
        )
    )

    # Workflow settings
    require_outline_approval: bool = True
    require_section_approval: bool = False
    auto_generate_on_approval: bool = True

    # Quality scoring settings
    enable_quality_review: bool = field(
        default_factory=lambda: get_generation_setting(
            "generation.enable_quality_review", "GENERATION_ENABLE_QUALITY_REVIEW", False
        )
    )
    min_quality_score: float = field(
        default_factory=lambda: get_generation_setting(
            "generation.min_quality_score", "GENERATION_MIN_QUALITY_SCORE", 0.7
        )
    )
    auto_regenerate_low_quality: bool = field(
        default_factory=lambda: get_generation_setting(
            "generation.auto_regenerate_low_quality", "GENERATION_AUTO_REGENERATE", True
        )
    )
    max_regeneration_attempts: int = 2  # Max times to regenerate a low-quality section
