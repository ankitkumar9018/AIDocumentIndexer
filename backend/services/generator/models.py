"""
Document Generation Models

Core data models for document generation workflow.
Extracted from generator.py for modularity.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


# =============================================================================
# Enums
# =============================================================================

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
    SECTIONS_PLANNING = "sections_planning"  # User reviews section plans before generation
    SECTIONS_APPROVED = "sections_approved"  # User has approved sections to generate
    GENERATING = "generating"
    SECTION_REVIEW = "section_review"
    REVISION = "revision"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

class SourceUsageType(str, Enum):
    """How a source was used in document generation."""
    CONTENT = "content"  # Used for actual content/facts
    STYLE = "style"  # Used for writing style/tone
    STRUCTURE = "structure"  # Used for document structure/layout
    INSPIRATION = "inspiration"  # Used as general inspiration
    REFERENCE = "reference"  # Direct citation/reference


@dataclass
class SourceReference:
    """Reference to a source document used in generation."""
    document_id: str
    document_name: str
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    snippet: str = ""
    usage_type: SourceUsageType = SourceUsageType.CONTENT  # How the source was used
    usage_description: Optional[str] = None  # Brief description of what was used
    document_path: Optional[str] = None  # Local file path for hyperlink
    document_url: Optional[str] = None  # URL for web-based documents


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
    # Pre-generation review fields
    description: Optional[str] = None  # Section description/plan from outline
    generation_approved: bool = True  # Whether to generate content for this section
    skipped: bool = False  # Whether user chose to skip this section


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


# =============================================================================
# Configuration
# =============================================================================

def _get_generation_setting(key: str, env_key: str, default: Any) -> Any:
    """
    Get a generation setting with fallback chain:
    1. Database settings (via settings service defaults)
    2. Environment variable
    3. Hardcoded default
    """
    try:
        from backend.services.settings import get_settings_service
        settings = get_settings_service()

        # Try settings service first
        value = settings.get_default_value(key)
        if value is not None:
            return value
    except Exception:
        pass  # Fall through to env/default

    # Fall back to environment variable
    env_value = os.getenv(env_key)
    if env_value is not None:
        # Handle boolean conversion
        if isinstance(default, bool):
            return env_value.lower() in ("true", "1", "yes", "on")
        return env_value

    return default


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
    output_dir: str = str(Path(__file__).resolve().parents[3] / "data" / "generated_docs")
    include_sources: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.include_sources", "GENERATION_INCLUDE_SOURCES", True
        )
    )
    include_toc: bool = True

    # Image generation settings - loaded from database settings
    # Configure via Admin UI: Settings > Document Generation
    include_images: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.include_images", "GENERATION_INCLUDE_IMAGES", True
        )
    )
    # Image backend: "picsum" (free, no API key), "unsplash" (requires API key),
    # "pexels" (requires API key), "openai" (DALL-E, requires API key),
    # "stability" (Stable Diffusion API), "automatic1111" (local SD), or "disabled"
    image_backend: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.image_backend", "GENERATION_IMAGE_BACKEND", "picsum"
        )
    )

    # Style settings - loaded from database settings
    default_tone: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.default_tone", "GENERATION_DEFAULT_TONE", "professional"
        )
    )
    default_style: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.default_style", "GENERATION_DEFAULT_STYLE", "business"
        )
    )

    # Chart generation settings
    auto_charts: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.auto_charts", "GENERATION_AUTO_CHARTS", False
        )
    )
    chart_style: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.chart_style", "GENERATION_CHART_STYLE", "business"
        )
    )
    chart_dpi: int = field(
        default_factory=lambda: _get_generation_setting(
            "generation.chart_dpi", "GENERATION_CHART_DPI", 150
        )
    )

    # Workflow settings
    require_outline_approval: bool = True
    require_section_planning_review: bool = True  # Show section plans for review before generation
    require_section_approval: bool = False  # Review sections after generation (post-gen review)
    auto_generate_on_approval: bool = True

    # Quality scoring settings - enabled by default for LLM-based review of generated content
    enable_quality_review: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.enable_quality_review", "GENERATION_ENABLE_QUALITY_REVIEW", True
        )
    )
    min_quality_score: float = field(
        default_factory=lambda: _get_generation_setting(
            "generation.min_quality_score", "GENERATION_MIN_QUALITY_SCORE", 0.7
        )
    )
    # Vision-based slide review - uses vision LLM to analyze rendered slide images
    # Disabled by default as it's resource-intensive (requires rendering + vision API)
    enable_vision_review: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.enable_vision_review", "GENERATION_ENABLE_VISION_REVIEW", False
        )
    )
    vision_review_model: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.vision_review_model", "GENERATION_VISION_REVIEW_MODEL", "auto"
        )
    )
    vision_review_all_slides: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.vision_review_all_slides", "GENERATION_VISION_REVIEW_ALL_SLIDES", False
        )
    )

    # LLM rewrite settings - for intelligent content condensing
    enable_llm_rewrite: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.enable_llm_rewrite", "GENERATION_ENABLE_LLM_REWRITE", True
        )
    )
    llm_rewrite_model: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.llm_rewrite_model", "GENERATION_LLM_REWRITE_MODEL", "auto"
        )
    )

    # Template analysis settings - for per-slide layout learning
    # When enabled, uses vision model to analyze template slides for styling/layout
    # ENABLED BY DEFAULT for enterprise-grade template matching
    enable_template_vision_analysis: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.enable_template_vision_analysis", "GENERATION_ENABLE_TEMPLATE_VISION_ANALYSIS", True
        )
    )
    template_vision_model: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.template_vision_model", "GENERATION_TEMPLATE_VISION_MODEL", "auto"
        )
    )

    # Per-slide content constraints - ensures content fits each slide's layout
    enable_per_slide_constraints: bool = field(
        default_factory=lambda: _get_generation_setting(
            "generation.enable_per_slide_constraints", "GENERATION_ENABLE_PER_SLIDE_CONSTRAINTS", True
        )
    )


# =============================================================================
# Citation Models
# =============================================================================

@dataclass
class CitationMapping:
    """Maps citation markers to source references."""
    marker: str  # e.g., "[1]"
    source_document: str
    source_page: Optional[int]
    source_snippet: str
    relevance_score: float = 0.0


@dataclass
class ContentWithCitations:
    """Content with inline citations and mapping."""
    content: str
    citations: List[CitationMapping]
    citation_style: str = "numbered"  # "numbered", "superscript", "author_year"


# =============================================================================
# Style Models
# =============================================================================

@dataclass
class StyleProfile:
    """Learned writing style from user documents.

    Captures detailed style characteristics that can be applied
    to new document generation for consistency.
    """
    # Core style attributes
    tone: str = "professional"  # formal, casual, technical, friendly, academic
    vocabulary_level: str = "moderate"  # simple, moderate, advanced, technical
    formality: str = "moderate"  # casual, moderate, formal, highly_formal

    # Structural preferences
    avg_sentence_length: float = 15.0  # Average words per sentence
    avg_paragraph_length: float = 4.0  # Average sentences per paragraph
    structure_pattern: str = "mixed"  # bullet-lists, paragraphs, mixed, headers-heavy
    bullet_preference: bool = False  # Prefers bullet points over prose

    # Language patterns
    uses_passive_voice: float = 0.3  # 0-1 ratio of passive voice usage
    uses_first_person: bool = False  # Uses "I", "we"
    uses_contractions: bool = False  # Uses "don't", "can't", etc.
    heading_style: str = "title_case"  # title_case, sentence_case, all_caps

    # Key phrases and terminology
    key_phrases: List[str] = field(default_factory=list)
    domain_terms: List[str] = field(default_factory=list)

    # Formatting preferences
    uses_bold_emphasis: bool = True
    uses_italic_emphasis: bool = True
    uses_numbered_lists: bool = False

    # Source information
    source_documents: List[str] = field(default_factory=list)
    confidence_score: float = 0.5  # 0-1, how confident we are in the analysis

    # Optional: recommended approach based on analysis
    recommended_approach: Optional[str] = None
    sentence_style: str = "medium"  # short, medium, long


# =============================================================================
# PPTX Template Layout Models
# =============================================================================

@dataclass
class LearnedLayout:
    """Learned layout characteristics from template slide.

    Captures detailed information about a slide layout including:
    - Safe zones for content (avoiding branding elements)
    - Content constraints (max characters, bullets)
    - Visual characteristics from template
    """
    layout_name: str
    layout_type: str  # 'title', 'content', 'two_column', 'image_text', 'section_header', 'blank'

    # Safe zones (in EMUs - English Metric Units, 914400 EMU = 1 inch)
    title_zone: Dict[str, Any] = field(default_factory=dict)  # {'left', 'top', 'width', 'height', 'max_chars'}
    content_zones: List[Dict[str, Any]] = field(default_factory=list)  # Multiple content areas possible
    image_zone: Optional[Dict[str, Any]] = None  # Where images should go

    # Branding elements to avoid
    branding_zones: List[Dict[str, Any]] = field(default_factory=list)

    # Content constraints learned from template
    typical_bullet_count: int = 6
    typical_bullet_length: int = 70
    max_title_chars: int = 50
    has_picture_placeholder: bool = False
    has_footer: bool = False
    has_page_number: bool = False

    # Visual characteristics
    font_size_title: int = 32
    font_size_bullets: int = 18
    bullet_indent: int = 0

    # Visual style (from optional vision analysis)
    visual_style: Dict[str, Any] = field(default_factory=dict)
