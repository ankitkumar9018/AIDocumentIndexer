"""
AIDocumentIndexer - Document Generation Service
================================================

Human-in-the-loop document generation using LangGraph.
Supports PPTX, DOCX, PDF, and other output formats.

LLM provider and model are configured via Admin UI (Operation-Level Config).
Configure the "content_generation" operation in Admin > Settings > LLM Configuration.
"""

import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import structlog

from backend.services.image_generator import (
    ImageGeneratorService,
    ImageGeneratorConfig,
    ImageBackend,
    get_image_generator,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Theme Configuration
# =============================================================================

THEMES = {
    "business": {
        "name": "Business Professional",
        "primary": "#1E3A5F",
        "secondary": "#3D5A80",
        "accent": "#E0E1DD",
        "text": "#2D3A45",
        "light_gray": "#888888",
        "description": "Clean, corporate look ideal for business presentations"
    },
    "creative": {
        "name": "Creative & Bold",
        "primary": "#6B4C9A",
        "secondary": "#9B6B9E",
        "accent": "#F4E4BA",
        "text": "#333333",
        "light_gray": "#666666",
        "description": "Vibrant colors for marketing and creative content"
    },
    "modern": {
        "name": "Modern Minimal",
        "primary": "#212529",
        "secondary": "#495057",
        "accent": "#00B4D8",
        "text": "#212529",
        "light_gray": "#6C757D",
        "description": "Sleek, contemporary design with bold accents"
    },
    "nature": {
        "name": "Nature & Organic",
        "primary": "#2D5016",
        "secondary": "#5A7D3A",
        "accent": "#F5F0E1",
        "text": "#2D3A2E",
        "light_gray": "#7A8B6E",
        "description": "Earthy tones for sustainability and wellness topics"
    },
    "elegant": {
        "name": "Elegant & Refined",
        "primary": "#2C3E50",
        "secondary": "#7F8C8D",
        "accent": "#BDC3C7",
        "text": "#2C3E50",
        "light_gray": "#95A5A6",
        "description": "Sophisticated look for executive presentations"
    },
    "vibrant": {
        "name": "Vibrant & Energetic",
        "primary": "#E74C3C",
        "secondary": "#F39C12",
        "accent": "#FDF2E9",
        "text": "#2D3436",
        "light_gray": "#BDC3C7",
        "description": "Bold colors for high-energy content"
    },
    "tech": {
        "name": "Tech & Digital",
        "primary": "#0984E3",
        "secondary": "#6C5CE7",
        "accent": "#DFE6E9",
        "text": "#2D3436",
        "light_gray": "#B2BEC3",
        "description": "Modern tech aesthetic for digital topics"
    },
    "warm": {
        "name": "Warm & Inviting",
        "primary": "#D35400",
        "secondary": "#E67E22",
        "accent": "#FDEBD0",
        "text": "#2C3E50",
        "light_gray": "#A6ACAF",
        "description": "Cozy colors for community and wellness"
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def strip_markdown(text: str) -> str:
    """Remove markdown formatting, returning clean text.

    Used by PPTX, XLSX, and other generators that don't support markdown.
    """
    if not text:
        return ""
    # Remove headers (# ## ### #### etc.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold **text** or __text__
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Remove italic *text* or _text_ (but not bullet points)
    text = re.sub(r'(?<!\s)\*([^*\n]+)\*(?!\s)', r'\1', text)
    text = re.sub(r'(?<!\s)_([^_\n]+)_(?!\s)', r'\1', text)
    # Remove code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove link syntax [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Clean up multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def smart_truncate(text: str, max_chars: int) -> str:
    """Truncate text at word boundaries to avoid mid-word cuts."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:  # Don't cut too much
        return truncated[:last_space] + '...'
    return truncated + '...'


def sentence_truncate(text: str, max_chars: int) -> str:
    """Truncate text at sentence boundaries, avoiding mid-sentence cuts.

    Handles common abbreviations and ensures meaningful content is preserved.
    Falls back to clause boundaries, then word boundaries.

    Args:
        text: Text to truncate
        max_chars: Maximum character limit

    Returns:
        Truncated text ending at a complete sentence, clause, or word boundary
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Common abbreviations that shouldn't be treated as sentence ends
    # Use negative lookbehind to exclude these
    abbreviations = r'(?<![Mm]r)(?<![Mm]rs)(?<![Mm]s)(?<![Dd]r)(?<![Pp]rof)(?<![Ii]nc)(?<![Ll]td)(?<![Jj]r)(?<![Ss]r)(?<![Vv]s)(?<!etc)(?<!e\.g)(?<!i\.e)'

    # Find sentence endings: period/exclamation/question followed by space or end
    # Exclude common abbreviations
    pattern = abbreviations + r'[.!?](?:\s|$)'
    sentence_ends = list(re.finditer(pattern, truncated))

    if sentence_ends:
        last_end = sentence_ends[-1].end()
        # Use 50% threshold to preserve more complete sentences
        if last_end > max_chars * 0.5:
            return text[:last_end].strip()

    # Fallback: find last complete clause (before comma, semicolon, colon, or dash)
    clause_ends = list(re.finditer(r'[,;:—–-]\s', truncated))
    if clause_ends and clause_ends[-1].start() > max_chars * 0.5:
        return text[:clause_ends[-1].start()].strip() + '...'

    # Final fallback to word boundary
    return smart_truncate(text, max_chars)


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Sanitize title for use as filename.

    Removes/replaces invalid characters for cross-platform compatibility.
    Handles special characters like /, :, |, ?, * that cause issues on various OS.

    Args:
        title: Document title to sanitize
        max_length: Maximum filename length (default 50)

    Returns:
        Safe filename string with only alphanumeric, underscores, and hyphens
    """
    # Normalize unicode characters (e.g., convert é to e)
    title = unicodedata.normalize('NFKD', title)
    title = title.encode('ascii', 'ignore').decode('ascii')
    # Keep only alphanumeric, spaces, hyphens, underscores
    title = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces and multiple hyphens/underscores with single underscore
    title = re.sub(r'[-\s]+', '_', title)
    # Remove leading/trailing underscores
    title = title.strip('_')
    # Limit length
    return title[:max_length] if title else 'document'


def get_theme_colors(theme_key: str = "business") -> dict:
    """Get theme colors, with fallback to business theme."""
    theme = THEMES.get(theme_key, THEMES["business"])
    return theme


def check_spelling(text: str) -> dict:
    """Check spelling and return suggestions for review.

    This flags potential spelling issues for user approval rather than
    auto-correcting, allowing users to decide what to fix.

    Returns:
        dict with:
        - has_issues: bool indicating if spelling issues were found
        - issues: list of dicts with word, position, suggestion, context
        - original: the original text
    """
    if not text:
        return {"has_issues": False, "issues": [], "original": text}

    try:
        from spellchecker import SpellChecker

        spell = SpellChecker()
        words = text.split()
        issues = []

        for i, word in enumerate(words):
            # Clean punctuation from word
            clean_word = word.strip('.,!?:;"\'-()[]{}')
            if not clean_word or len(clean_word) < 3:
                continue

            # Skip common patterns that aren't words
            if clean_word.isdigit():
                continue
            if any(c.isdigit() for c in clean_word):  # Skip alphanumeric codes
                continue
            if clean_word.startswith(('@', '#', 'http', 'www')):
                continue

            # Check if word is known
            if clean_word.lower() not in spell:
                correction = spell.correction(clean_word.lower())
                # Only suggest if we have a different correction
                if correction and correction != clean_word.lower():
                    # Get context (surrounding words)
                    context_start = max(0, i - 2)
                    context_end = min(len(words), i + 3)
                    context = ' '.join(words[context_start:context_end])

                    issues.append({
                        "word": clean_word,
                        "position": i,
                        "suggestion": correction,
                        "context": context
                    })

        return {
            "has_issues": len(issues) > 0,
            "issues": issues[:20],  # Limit to top 20 issues
            "original": text
        }
    except ImportError:
        logger.warning("pyspellchecker not installed, skipping spell check")
        return {"has_issues": False, "issues": [], "original": text}
    except Exception as e:
        logger.warning(f"Spell check failed: {e}")
        return {"has_issues": False, "issues": [], "original": text}


# =============================================================================
# Types
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


def _get_generation_setting(key: str, env_key: str, default: Any) -> Any:
    """
    Get a generation setting with fallback chain:
    1. Database settings (via settings service defaults)
    2. Environment variable
    3. Hardcoded default
    """
    from backend.services.settings import get_settings_service
    settings = get_settings_service()

    # Try settings service first
    value = settings.get_default_value(key)
    if value is not None:
        return value

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
    min_relevance_score: float = 0.5

    # Output settings - loaded from database settings or environment
    output_dir: str = "/tmp/generated_docs"
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
    # Image backend: "unsplash" (free, requires API key), "stability" (Stable Diffusion API),
    # "automatic1111" (local Stable Diffusion), or "disabled"
    image_backend: str = field(
        default_factory=lambda: _get_generation_setting(
            "generation.image_backend", "GENERATION_IMAGE_BACKEND", "unsplash"
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
    require_section_approval: bool = False
    auto_generate_on_approval: bool = True


# =============================================================================
# Document Generation Service
# =============================================================================

class DocumentGenerationService:
    """
    Service for generating documents with human-in-the-loop workflow.

    Features:
    - RAG-powered content generation
    - Outline review and approval
    - Section-by-section generation
    - Source attribution
    - Multiple output formats
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the document generation service."""
        self.config = config or GenerationConfig()
        self._jobs: Dict[str, GenerationJob] = {}

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    async def reload_settings(self) -> None:
        """
        Reload generation settings from database.

        Call this to pick up admin setting changes without service restart.
        Settings are read from Admin UI > Settings > Document Generation.
        """
        from backend.services.settings import get_settings_service
        settings_service = get_settings_service()

        # Get current database settings (async)
        include_images = await settings_service.get_setting("generation.include_images")
        image_backend = await settings_service.get_setting("generation.image_backend")
        include_sources = await settings_service.get_setting("generation.include_sources")
        default_tone = await settings_service.get_setting("generation.default_tone")
        default_style = await settings_service.get_setting("generation.default_style")
        auto_charts = await settings_service.get_setting("generation.auto_charts")
        chart_style = await settings_service.get_setting("generation.chart_style")
        chart_dpi = await settings_service.get_setting("generation.chart_dpi")

        # Update config with database values
        if include_images is not None:
            self.config.include_images = include_images
        if image_backend is not None:
            self.config.image_backend = image_backend
        if include_sources is not None:
            self.config.include_sources = include_sources
        if default_tone is not None:
            self.config.default_tone = default_tone
        if default_style is not None:
            self.config.default_style = default_style
        if auto_charts is not None:
            self.config.auto_charts = auto_charts
        if chart_style is not None:
            self.config.chart_style = chart_style
        if chart_dpi is not None:
            self.config.chart_dpi = int(chart_dpi)

        logger.info(
            "Generation settings reloaded",
            include_images=self.config.include_images,
            image_backend=self.config.image_backend,
            default_tone=self.config.default_tone,
            default_style=self.config.default_style,
            auto_charts=self.config.auto_charts,
        )

    async def create_job(
        self,
        user_id: str,
        title: str,
        description: str,
        output_format: OutputFormat = OutputFormat.DOCX,
        collection_filter: Optional[str] = None,
        folder_id: Optional[str] = None,
        include_subfolders: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        include_images: Optional[bool] = None,  # Override from request
    ) -> GenerationJob:
        """
        Create a new document generation job.

        Args:
            user_id: ID of the user creating the job
            title: Title for the document
            description: Description of what to generate
            output_format: Desired output format
            collection_filter: Optional collection(s) to search (comma-separated for multiple)
            folder_id: Optional folder ID to scope search to
            include_subfolders: Whether to include subfolders when folder_id is set
            metadata: Additional metadata
            include_images: Override image generation setting for this job

        Returns:
            New GenerationJob instance
        """
        # Reload settings from database to pick up admin changes
        await self.reload_settings()

        job_id = str(uuid.uuid4())

        job = GenerationJob(
            id=job_id,
            user_id=user_id,
            title=title,
            description=description,
            output_format=output_format,
            status=GenerationStatus.DRAFT,
            metadata=metadata or {},
        )

        if collection_filter:
            job.metadata["collection_filter"] = collection_filter
        if folder_id:
            job.metadata["folder_id"] = folder_id
            job.metadata["include_subfolders"] = include_subfolders

        # Store image generation setting - use request override or config setting
        job.metadata["include_images"] = (
            include_images if include_images is not None else self.config.include_images
        )
        job.metadata["image_backend"] = self.config.image_backend
        job.metadata["default_tone"] = self.config.default_tone
        job.metadata["default_style"] = self.config.default_style

        self._jobs[job_id] = job

        logger.info(
            "Document generation job created",
            job_id=job_id,
            user_id=user_id,
            title=title,
            include_images=job.metadata["include_images"],
            image_backend=job.metadata["image_backend"],
        )

        return job

    async def generate_outline(
        self,
        job_id: str,
        num_sections: Optional[int] = None,
    ) -> DocumentOutline:
        """
        Generate an outline for the document.

        Args:
            job_id: Job ID
            num_sections: Number of sections to generate. None = auto mode (LLM decides)

        Returns:
            Generated outline
        """
        job = self._get_job(job_id)

        if job.status not in [GenerationStatus.DRAFT, GenerationStatus.OUTLINE_PENDING]:
            raise ValueError(f"Cannot generate outline in status: {job.status}")

        job.status = GenerationStatus.OUTLINE_PENDING
        job.updated_at = datetime.utcnow()

        logger.info(
            "Generating outline",
            job_id=job_id,
            num_sections=num_sections,
            mode="auto" if num_sections is None else "manual",
        )

        # Analyze style from existing documents if enabled
        style_guide = None
        if job.metadata.get("use_existing_docs"):
            logger.info(
                "Analyzing document styles for style-aware generation",
                job_id=job_id,
                style_collections=job.metadata.get("style_collection_filters"),
                style_folder=job.metadata.get("style_folder_id"),
            )
            style_guide = await self._analyze_document_styles(
                collection_filters=job.metadata.get("style_collection_filters"),
                folder_id=job.metadata.get("style_folder_id"),
                include_subfolders=job.metadata.get("include_style_subfolders", True),
            )
            if style_guide:
                # Store for later use in content generation
                job.metadata["style_guide"] = style_guide
                logger.info(
                    "Style analysis complete",
                    job_id=job_id,
                    tone=style_guide.get("tone"),
                    vocabulary=style_guide.get("vocabulary_level"),
                    source_docs=len(style_guide.get("source_documents", [])),
                )
            else:
                logger.warning(
                    "No documents found for style analysis",
                    job_id=job_id,
                )

        # Use RAG to find relevant sources
        sources = []

        # Add style source documents to sources for references slide/page
        # These are the documents used for style, language, and pattern learning
        if style_guide and style_guide.get("source_documents"):
            style_sources = style_guide.get("source_documents", [])
            logger.info(
                "Adding style source documents to references",
                job_id=job_id,
                count=len(style_sources),
            )
            for doc_name in style_sources:
                sources.append(SourceReference(
                    document_id="",  # Style sources identified by name
                    document_name=doc_name,
                    chunk_id="",
                    page_number=None,
                    relevance_score=1.0,
                    snippet="Used for style, language, and pattern learning",
                ))
        if self.config.use_rag:
            rag_sources = await self._search_sources(
                query=f"{job.title}: {job.description}",
                collection_filter=job.metadata.get("collection_filter"),
                max_results=self.config.max_sources,
            )
            sources.extend(rag_sources)  # Add RAG sources to style sources

        # Generate outline using LLM
        outline = await self._generate_outline_with_llm(
            title=job.title,
            description=job.description,
            sources=sources,
            num_sections=num_sections,
            output_format=job.output_format.value,
            style_guide=style_guide,
        )

        job.outline = outline
        job.sources_used = sources
        job.updated_at = datetime.utcnow()

        logger.info(
            "Outline generated",
            job_id=job_id,
            sections=len(outline.sections),
        )

        return outline

    async def approve_outline(
        self,
        job_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> GenerationJob:
        """
        Approve the outline and optionally apply modifications.

        Args:
            job_id: Job ID
            modifications: Optional changes to the outline

        Returns:
            Updated job
        """
        job = self._get_job(job_id)

        if job.status != GenerationStatus.OUTLINE_PENDING:
            raise ValueError(f"Cannot approve outline in status: {job.status}")

        if not job.outline:
            raise ValueError("No outline to approve")

        # Apply modifications if provided
        if modifications:
            if "title" in modifications:
                job.outline.title = modifications["title"]
            if "sections" in modifications:
                job.outline.sections = modifications["sections"]
            if "tone" in modifications:
                job.outline.tone = modifications["tone"]
            # Allow theme change after outline generation
            if "theme" in modifications and modifications["theme"]:
                job.metadata["theme"] = modifications["theme"]
                logger.info(f"Theme changed to: {modifications['theme']}")

        job.status = GenerationStatus.OUTLINE_APPROVED
        job.updated_at = datetime.utcnow()

        logger.info("Outline approved", job_id=job_id)

        # Auto-generate if configured
        if self.config.auto_generate_on_approval:
            await self.generate_content(job_id)

        return job

    async def generate_content(
        self,
        job_id: str,
    ) -> GenerationJob:
        """
        Generate the document content based on approved outline.

        Args:
            job_id: Job ID

        Returns:
            Updated job with generated content
        """
        job = self._get_job(job_id)

        if job.status != GenerationStatus.OUTLINE_APPROVED:
            raise ValueError(f"Cannot generate content in status: {job.status}")

        if not job.outline:
            raise ValueError("No outline available")

        job.status = GenerationStatus.GENERATING
        job.updated_at = datetime.utcnow()

        logger.info("Generating content", job_id=job_id)

        try:
            # Generate each section
            sections = []
            for idx, section_def in enumerate(job.outline.sections):
                section = await self._generate_section(
                    job=job,
                    section_title=section_def.get("title", f"Section {idx + 1}"),
                    section_description=section_def.get("description", ""),
                    order=idx,
                )
                sections.append(section)

                logger.info(
                    "Section generated",
                    job_id=job_id,
                    section_title=section.title,
                    order=idx,
                )

            job.sections = sections

            # Move to review or completed based on config
            if self.config.require_section_approval:
                job.status = GenerationStatus.SECTION_REVIEW
            else:
                job.status = GenerationStatus.COMPLETED
                job.completed_at = datetime.utcnow()

                # Generate output file
                await self._generate_output_file(job)

            job.updated_at = datetime.utcnow()

        except Exception as e:
            logger.error("Content generation failed", job_id=job_id, error=str(e))
            job.status = GenerationStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.utcnow()
            raise

        return job

    async def approve_section(
        self,
        job_id: str,
        section_id: str,
        feedback: Optional[str] = None,
        approved: bool = True,
    ) -> Section:
        """
        Approve or request revision for a section.

        Args:
            job_id: Job ID
            section_id: Section ID
            feedback: Optional feedback for revision
            approved: Whether the section is approved

        Returns:
            Updated section
        """
        job = self._get_job(job_id)

        if job.status not in [GenerationStatus.SECTION_REVIEW, GenerationStatus.REVISION]:
            raise ValueError(f"Cannot approve section in status: {job.status}")

        section = next((s for s in job.sections if s.id == section_id), None)
        if not section:
            raise ValueError(f"Section not found: {section_id}")

        section.approved = approved
        section.feedback = feedback if not approved else None

        if not approved:
            job.status = GenerationStatus.REVISION
        else:
            # Check if all sections are approved
            all_approved = all(s.approved for s in job.sections)
            if all_approved:
                job.status = GenerationStatus.COMPLETED
                job.completed_at = datetime.utcnow()

                # Generate output file
                await self._generate_output_file(job)

        job.updated_at = datetime.utcnow()

        return section

    async def revise_section(
        self,
        job_id: str,
        section_id: str,
    ) -> Section:
        """
        Revise a section based on feedback.

        Args:
            job_id: Job ID
            section_id: Section ID

        Returns:
            Revised section
        """
        job = self._get_job(job_id)

        if job.status != GenerationStatus.REVISION:
            raise ValueError(f"Cannot revise section in status: {job.status}")

        section = next((s for s in job.sections if s.id == section_id), None)
        if not section:
            raise ValueError(f"Section not found: {section_id}")

        if not section.feedback:
            raise ValueError("No feedback provided for revision")

        # Regenerate with feedback
        revised_section = await self._regenerate_section_with_feedback(
            job=job,
            section=section,
        )

        # Update section
        section.revised_content = revised_section.content
        section.sources = revised_section.sources
        section.approved = False
        section.feedback = None

        job.status = GenerationStatus.SECTION_REVIEW
        job.updated_at = datetime.utcnow()

        return section

    async def get_job(self, job_id: str) -> GenerationJob:
        """Get a generation job by ID."""
        return self._get_job(job_id)

    async def list_jobs(
        self,
        user_id: str,
        status: Optional[GenerationStatus] = None,
    ) -> List[GenerationJob]:
        """List jobs for a user."""
        jobs = [j for j in self._jobs.values() if j.user_id == user_id]

        if status:
            jobs = [j for j in jobs if j.status == status]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def cancel_job(self, job_id: str) -> GenerationJob:
        """Cancel a generation job."""
        job = self._get_job(job_id)

        if job.status in [GenerationStatus.COMPLETED, GenerationStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel job in status: {job.status}")

        job.status = GenerationStatus.CANCELLED
        job.updated_at = datetime.utcnow()

        logger.info("Job cancelled", job_id=job_id)

        return job

    async def get_output_file(self, job_id: str) -> tuple[bytes, str, str]:
        """
        Get the generated output file.

        Args:
            job_id: Job ID

        Returns:
            Tuple of (file_bytes, filename, content_type)
        """
        job = self._get_job(job_id)

        if job.status != GenerationStatus.COMPLETED:
            raise ValueError("Job is not completed")

        if not job.output_path:
            raise ValueError("No output file available")

        # Read file
        with open(job.output_path, "rb") as f:
            file_bytes = f.read()

        filename = Path(job.output_path).name
        content_type = self._get_content_type(job.output_format)

        return file_bytes, filename, content_type

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _get_job(self, job_id: str) -> GenerationJob:
        """Get job by ID or raise error."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        return job

    async def _search_sources(
        self,
        query: str,
        collection_filter: Optional[str] = None,
        max_results: int = 10,
    ) -> List[SourceReference]:
        """Search for relevant sources using RAG."""
        try:
            from backend.services.vectorstore import get_vector_store, SearchType

            vector_store = get_vector_store()
            results = await vector_store.search(
                query=query,
                search_type=SearchType.HYBRID,
                top_k=max_results,
            )

            sources = []
            for result in results:
                if result.score >= self.config.min_relevance_score:
                    sources.append(
                        SourceReference(
                            document_id=str(result.document_id),
                            document_name="",  # Would need to look up
                            chunk_id=str(result.chunk_id),
                            page_number=result.page_number,
                            relevance_score=result.score,
                            snippet=result.content[:200],
                        )
                    )

            return sources

        except Exception as e:
            logger.warning("Failed to search sources", error=str(e))
            return []

    async def _analyze_document_styles(
        self,
        collection_filters: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        include_subfolders: bool = True,
        sample_size: int = 5,
    ) -> Optional[dict]:
        """Analyze existing documents to extract style patterns for new document generation.

        This method retrieves sample documents from the specified collections/folders
        and uses LLM to analyze their writing style, tone, vocabulary, and structure.

        Args:
            collection_filters: List of collections to sample from
            folder_id: Folder ID to scope the search
            include_subfolders: Whether to include subfolders
            sample_size: Number of unique documents to sample

        Returns:
            Style analysis dict with tone, vocabulary_level, structure_pattern, etc.
            Returns None if no documents found or analysis fails.
        """
        try:
            import json
            from backend.services.vectorstore import get_vector_store, SearchType
            from backend.db.database import async_session_context
            from backend.db.models import Document as DBDocument, Folder
            from sqlalchemy import select, cast, String, literal, or_

            vector_store = get_vector_store()

            # Build document ID filter based on collection/folder filters
            document_ids = None

            # Get document IDs matching collection filters
            if collection_filters:
                async with async_session_context() as db:
                    all_collection_doc_ids = set()
                    for collection in collection_filters:
                        if collection == "(Untagged)":
                            query_stmt = select(DBDocument.id).where(
                                or_(
                                    DBDocument.tags.is_(None),
                                    DBDocument.tags == [],
                                )
                            )
                        else:
                            # Escape LIKE special characters
                            safe_filter = collection.replace("\\", "\\\\")
                            safe_filter = safe_filter.replace("%", "\\%")
                            safe_filter = safe_filter.replace("_", "\\_")
                            safe_filter = safe_filter.replace('"', '\\"')
                            pattern = f'%"{safe_filter}"%'
                            query_stmt = select(DBDocument.id).where(
                                cast(DBDocument.tags, String).like(literal(pattern))
                            )
                        result = await db.execute(query_stmt)
                        all_collection_doc_ids.update(str(row[0]) for row in result.fetchall())

                    if all_collection_doc_ids:
                        document_ids = list(all_collection_doc_ids)
                        logger.info(
                            "Style analysis filtered by collections",
                            collections=collection_filters,
                            document_count=len(document_ids),
                        )
                    else:
                        logger.info("No documents match collection filters for style analysis")
                        return None

            # Get document IDs from folder filter
            if folder_id:
                async with async_session_context() as db:
                    folder_doc_ids = set()

                    if include_subfolders:
                        # Get folder and all subfolders
                        folder_ids = [folder_id]
                        # Get subfolders recursively
                        result = await db.execute(
                            select(Folder.id).where(Folder.parent_id == folder_id)
                        )
                        subfolder_ids = [str(row[0]) for row in result.fetchall()]
                        folder_ids.extend(subfolder_ids)

                        # Get documents in all folders
                        for fid in folder_ids:
                            result = await db.execute(
                                select(DBDocument.id).where(DBDocument.folder_id == fid)
                            )
                            folder_doc_ids.update(str(row[0]) for row in result.fetchall())
                    else:
                        # Just the specified folder
                        result = await db.execute(
                            select(DBDocument.id).where(DBDocument.folder_id == folder_id)
                        )
                        folder_doc_ids.update(str(row[0]) for row in result.fetchall())

                    if folder_doc_ids:
                        if document_ids:
                            # Intersect with collection filter results
                            document_ids = list(set(document_ids) & folder_doc_ids)
                        else:
                            document_ids = list(folder_doc_ids)
                        logger.info(
                            "Style analysis filtered by folder",
                            folder_id=folder_id,
                            include_subfolders=include_subfolders,
                            document_count=len(document_ids),
                        )
                    else:
                        logger.info("No documents in folder for style analysis")
                        return None

            # Log if searching all documents (no filters)
            if document_ids is None:
                logger.info("Style analysis searching all documents (no filters specified)")

            # Get document samples directly from database (more reliable than search)
            # This approach ensures we get samples even when search terms don't match
            from backend.db.models import Chunk as DBChunk

            samples = []
            async with async_session_context() as db:
                # Build query for chunks with content
                chunk_query = select(
                    DBChunk.content,
                    DBChunk.document_id,
                    DBDocument.name.label("document_name"),
                    DBDocument.tags.label("collection"),
                ).join(DBDocument, DBChunk.document_id == DBDocument.id)

                # Apply document_ids filter if specified
                if document_ids:
                    chunk_query = chunk_query.where(
                        cast(DBChunk.document_id, String).in_(document_ids)
                    )

                # Order by document to get variety, limit results
                chunk_query = chunk_query.order_by(DBDocument.id).limit(sample_size * 5)

                result = await db.execute(chunk_query)
                rows = result.fetchall()

                # Extract unique document samples
                seen_docs = set()
                for row in rows:
                    doc_id = str(row.document_id) if row.document_id else None
                    if doc_id and doc_id not in seen_docs and len(samples) < sample_size:
                        seen_docs.add(doc_id)
                        # Get document name - may be in tags JSON or name field
                        doc_name = row.document_name or ""
                        collection_tags = row.collection if row.collection else []
                        samples.append({
                            "content": row.content or "",
                            "document_name": doc_name,
                            "collection": collection_tags[0] if collection_tags else "",
                        })
                        logger.debug(
                            "Style sample added",
                            doc_id=doc_id,
                            doc_name=doc_name,
                        )

            if not samples:
                logger.info("No document samples found for style analysis")
                return None

            logger.info(
                "Style analysis samples collected",
                sample_count=len(samples),
                document_names=[s["document_name"] for s in samples],
            )

            # Use LLM to analyze style patterns
            sample_text = "\n\n".join([
                f"--- Sample from {s['document_name'] or 'Unknown'} ---\n{s['content'][:1000]}"
                for s in samples
            ])

            analysis_prompt = f"""Analyze the following document excerpts and extract their writing style patterns.

DOCUMENT SAMPLES:
{sample_text}

Analyze and return a JSON object with:
{{
    "tone": "formal" | "casual" | "technical" | "friendly" | "academic",
    "vocabulary_level": "simple" | "moderate" | "advanced",
    "structure_pattern": "bullet-lists" | "paragraphs" | "mixed" | "headers-heavy",
    "sentence_style": "short-concise" | "medium" | "long-detailed",
    "key_phrases": ["common phrases or terms used"],
    "formatting_notes": "any notable formatting patterns observed",
    "recommended_approach": "brief recommendation for matching this style"
}}

Return ONLY valid JSON, no other text."""

            response = await self.llm.ainvoke(analysis_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse the JSON response
            import json
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                style_analysis = json.loads(json_match.group())
            else:
                style_analysis = json.loads(response_text)

            style_analysis["source_documents"] = [s["document_name"] for s in samples if s["document_name"]]

            logger.info(
                "Style analysis completed",
                tone=style_analysis.get("tone"),
                vocabulary=style_analysis.get("vocabulary_level"),
                num_sources=len(style_analysis.get("source_documents", [])),
            )

            return style_analysis

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse style analysis JSON: {e}")
            # Return default style guide
            return {
                "tone": "professional",
                "vocabulary_level": "moderate",
                "structure_pattern": "mixed",
                "sentence_style": "medium",
                "source_documents": [],
            }
        except Exception as e:
            logger.warning(f"Failed to analyze document styles: {e}")
            return None

    async def _generate_outline_with_llm(
        self,
        title: str,
        description: str,
        sources: List[SourceReference],
        num_sections: Optional[int] = None,
        output_format: str = "docx",
        style_guide: Optional[Dict[str, Any]] = None,
    ) -> DocumentOutline:
        """Generate outline using LLM.

        Args:
            title: Document title
            description: Document description
            sources: Relevant sources from RAG
            num_sections: Number of sections. None = auto mode (LLM decides)
            output_format: Target output format (affects section count guidance)
            style_guide: Optional style analysis from existing documents
        """
        # Build context from sources
        context = ""
        if sources:
            context = "Relevant information from the knowledge base:\n\n"
            for source in sources[:5]:
                context += f"- {source.snippet}...\n\n"

        # Build style instructions if available
        style_instructions = ""
        if style_guide:
            style_instructions = f"""
STYLE GUIDE (based on existing documents in your collection):
- Tone: {style_guide.get('tone', 'professional')}
- Vocabulary Level: {style_guide.get('vocabulary_level', 'moderate')}
- Structure Pattern: {style_guide.get('structure_pattern', 'mixed')}
- Sentence Style: {style_guide.get('sentence_style', 'medium')}
{f"- Formatting Notes: {style_guide.get('formatting_notes')}" if style_guide.get('formatting_notes') else ""}
{f"- Key Phrases: {', '.join(style_guide.get('key_phrases', []))}" if style_guide.get('key_phrases') else ""}

IMPORTANT: Match the style, tone, and structure of the existing documents.
The new document should feel like it belongs to the same collection.
Use similar vocabulary and phrasing patterns as found in the source documents.
"""

        # Build section instruction based on mode
        if num_sections is None:
            # Auto mode - LLM decides optimal count
            section_instruction = f"""Analyze the topic complexity and determine the optimal number of sections.

Consider:
- Topic depth and complexity
- Output format: {output_format} (presentations typically need 5-10 slides, documents 3-8 pages)
- Amount of source material available
- Target audience expectations

First, output your recommended section count on a line by itself like:
SECTION_COUNT: N

Where N is between 3 and 15.

Then generate exactly N sections with specific, descriptive titles."""
        else:
            section_instruction = f"Generate exactly {num_sections} sections with specific, descriptive titles."

        # Create prompt for outline generation
        prompt = f"""Create a professional document outline for the following:
{style_instructions}

Title: {title}
Description: {description}

{context}

{section_instruction}

IMPORTANT RULES:
- Each section title MUST be specific and descriptive (e.g., "Market Analysis and Trends", "Implementation Strategy")
- DO NOT use generic titles like "Section 1", "Introduction", "Overview", "Conclusion"
- Titles should clearly indicate what the section covers
- Each title should be 3-7 words long

Format each section EXACTLY like this:
## [Specific Descriptive Title]
Description: [2-3 sentences explaining what this section covers]

Example of good titles:
- "Strategic Growth Opportunities"
- "Technical Architecture Overview"
- "Cost-Benefit Analysis"
- "Implementation Timeline and Milestones"

Example of bad titles (DO NOT USE):
- "Section 1", "Introduction", "Overview", "Summary", "Conclusion"

Generate the outline now:"""

        # Use LLM to generate (database-driven configuration)
        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,  # System-level operation
            )
            response = await llm.ainvoke(prompt)

            # Parse response into sections with improved parsing
            import re
            sections = []
            lines = response.content.split("\n")
            current_section = None

            # In auto mode, extract the section count from LLM response
            target_sections = num_sections
            if num_sections is None:
                # Look for SECTION_COUNT: N pattern
                section_count_match = re.search(r'SECTION_COUNT:\s*(\d+)', response.content)
                if section_count_match:
                    target_sections = int(section_count_match.group(1))
                    # Clamp to valid range
                    target_sections = max(3, min(15, target_sections))
                    logger.info(f"Auto mode: LLM suggested {target_sections} sections")
                else:
                    # Fallback: count sections in response, or default to 5
                    target_sections = 5
                    logger.warning("Auto mode: Could not extract section count, defaulting to 5")

            # Generic title patterns to reject
            generic_patterns = [
                r'^section\s*\d*$',
                r'^introduction$',
                r'^overview$',
                r'^summary$',
                r'^conclusion$',
                r'^part\s*\d+$',
            ]

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for section header (## Title or numbered)
                if line.startswith("##") or (line[0].isdigit() and "." in line[:3]):
                    if current_section and current_section["title"]:
                        sections.append(current_section)

                    # Extract title, removing markdown and numbering
                    title = re.sub(r'^[#\d.\s]+', '', line).strip()

                    # Check if title is generic
                    is_generic = any(
                        re.match(pattern, title.lower().strip())
                        for pattern in generic_patterns
                    )

                    if is_generic or not title:
                        # Generate a better title based on document topic
                        title = f"Key Aspect {len(sections) + 1} of {description[:30].split()[0].title() if description else 'Topic'}"

                    current_section = {"title": title, "description": ""}

                elif current_section and line:
                    # Handle description lines
                    if line.lower().startswith("description:"):
                        current_section["description"] = line[12:].strip() + " "
                    elif not current_section["description"]:
                        current_section["description"] += line + " "

            if current_section and current_section["title"]:
                sections.append(current_section)

            # Ensure we have the requested number of sections with meaningful titles
            topic_words = title.split()[:3] if title else ["Document"]
            topic_prefix = " ".join(topic_words)

            while len(sections) < target_sections:
                section_num = len(sections) + 1
                # Generate contextual fallback titles
                fallback_titles = [
                    f"Key Findings and Insights",
                    f"Analysis and Recommendations",
                    f"Implementation Approach",
                    f"Strategic Considerations",
                    f"Supporting Details",
                    f"Additional Context",
                ]
                fallback_title = fallback_titles[min(section_num - 1, len(fallback_titles) - 1)]
                sections.append({
                    "title": f"{fallback_title} for {topic_prefix}",
                    "description": f"Detailed content covering {fallback_title.lower()}",
                })

            return DocumentOutline(
                title=title,
                description=description,
                sections=sections[:target_sections],
            )

        except Exception as e:
            logger.error("Failed to generate outline with LLM", error=str(e))
            # Return a more descriptive fallback outline
            # Use num_sections if specified, otherwise default to 5
            fallback_count = num_sections if num_sections is not None else 5
            fallback_section_templates = [
                ("Background and Context", f"Overview and background information about {title}"),
                ("Key Analysis and Findings", f"Main analysis and findings related to {title}"),
                ("Strategic Recommendations", f"Recommendations and action items for {title}"),
                ("Implementation Details", f"How to implement the strategies for {title}"),
                ("Conclusion and Next Steps", f"Summary and suggested next steps for {title}"),
                ("Supporting Information", f"Additional details and references for {title}"),
            ]
            return DocumentOutline(
                title=title,
                description=description,
                sections=[
                    {
                        "title": fallback_section_templates[min(i, len(fallback_section_templates) - 1)][0],
                        "description": fallback_section_templates[min(i, len(fallback_section_templates) - 1)][1]
                    }
                    for i in range(fallback_count)
                ],
            )

    async def _generate_section(
        self,
        job: GenerationJob,
        section_title: str,
        section_description: str,
        order: int,
    ) -> Section:
        """Generate content for a section."""
        section_id = str(uuid.uuid4())

        # Search for relevant sources for this section
        sources = await self._search_sources(
            query=f"{section_title}: {section_description}",
            collection_filter=job.metadata.get("collection_filter"),
            max_results=5,
        )

        # Build context
        context = ""
        if sources:
            context = "Use the following information:\n\n"
            for source in sources:
                context += f"- {source.snippet}\n\n"

        # Generate format-specific prompts
        if job.output_format == OutputFormat.PPTX:
            format_instructions = """FORMAT REQUIREMENTS FOR PRESENTATION SLIDES:
- Write 6-10 bullet points maximum
- Each bullet MUST be a COMPLETE sentence under 90 characters
- If a point is too long, split it into two shorter points
- Use simple, concise language - avoid complex sentences
- NO markdown formatting (no **, no ##, no _)
- Focus on key takeaways, not detailed explanations
- Start each point with an action verb or key noun
- NEVER leave a sentence incomplete or cut off

CRITICAL: Every bullet point must be a complete, standalone thought.
If you cannot express an idea in under 90 characters, break it into multiple shorter points.

Example format (each is a complete sentence):
• Revenue increased 25% year-over-year across all regions.
• Customer acquisition cost reduced by 15% through optimization.
• New market expansion in Q3 shows promising early results."""
        elif job.output_format in (OutputFormat.DOCX, OutputFormat.PDF):
            format_instructions = """FORMAT REQUIREMENTS FOR DOCUMENT:
- Write 3-5 well-structured paragraphs
- Use clear topic sentences for each paragraph
- Include relevant details and examples
- Maintain professional, formal tone
- Use **bold** for key terms (will be styled)
- Use _italic_ for emphasis (will be styled)
- Avoid excessive bullet points - prefer prose
- Target 200-400 words per section"""
        elif job.output_format == OutputFormat.XLSX:
            format_instructions = """FORMAT REQUIREMENTS FOR SPREADSHEET:
- Write concise, data-oriented content
- Use short sentences that fit in cells
- NO markdown formatting (no **, no ##, no _)
- Focus on quantifiable information
- Use clear, structured points
- Each point on a new line
- Target 5-10 key points
- Avoid long paragraphs"""
        else:
            format_instructions = """Write clear, well-structured content.
Include relevant details and maintain a professional tone."""

        # Calculate section position context
        total_sections = len(job.outline.sections) if job.outline else 5
        current_section_num = order + 1  # order is 0-indexed

        if current_section_num <= 2:
            position_context = "This is an EARLY section - set the stage and introduce key concepts."
        elif current_section_num >= total_sections - 1:
            position_context = "This is a CONCLUDING section - summarize key points and provide actionable takeaways."
        else:
            position_context = "This is a MIDDLE section - dive into details, analysis, and supporting information."

        # Build style context if available from style analysis
        style_context = ""
        style_guide = job.metadata.get("style_guide")
        if style_guide:
            style_context = f"""
STYLE REQUIREMENTS (based on existing documents):
- Write in a {style_guide.get('tone', 'professional')} tone
- Use {style_guide.get('vocabulary_level', 'moderate')} vocabulary
- Follow {style_guide.get('structure_pattern', 'mixed')} structure
- Keep sentences {style_guide.get('sentence_style', 'medium')}
{f"- Recommended approach: {style_guide.get('recommended_approach')}" if style_guide.get('recommended_approach') else ""}
{f"- Use key phrases like: {', '.join(style_guide.get('key_phrases', [])[:5])}" if style_guide.get('key_phrases') else ""}

IMPORTANT: The new content should match the style and tone of existing documents in the collection.
"""

        # Generate content
        prompt = f"""Write content for the following section:
{style_context}

Document Title: {job.title}
Section: {section_title} (Section {current_section_num} of {total_sections})
Description: {section_description}

Position guidance: {position_context}

{context}

{format_instructions}"""

        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,  # System-level operation
            )
            response = await llm.ainvoke(prompt)
            content = response.content

        except Exception as e:
            logger.error("Failed to generate section", error=str(e))
            content = f"[Content for {section_title} - generation failed]"

        return Section(
            id=section_id,
            title=section_title,
            content=content,
            order=order,
            sources=sources,
            approved=not self.config.require_section_approval,
        )

    async def _regenerate_section_with_feedback(
        self,
        job: GenerationJob,
        section: Section,
    ) -> Section:
        """Regenerate a section based on feedback."""
        prompt = f"""Revise the following section based on the feedback provided:

Section Title: {section.title}
Current Content:
{section.content}

Feedback: {section.feedback}

Please revise the content to address the feedback while maintaining quality and relevance."""

        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,  # System-level operation
            )
            response = await llm.ainvoke(prompt)

            return Section(
                id=section.id,
                title=section.title,
                content=response.content,
                order=section.order,
                sources=section.sources,
                approved=False,
            )

        except Exception as e:
            logger.error("Failed to revise section", error=str(e))
            return section

    async def _generate_output_file(self, job: GenerationJob) -> str:
        """Generate the output file in the requested format."""
        safe_title = sanitize_filename(job.title)
        filename = f"{job.id}_{safe_title}"

        if job.output_format == OutputFormat.PPTX:
            output_path = await self._generate_pptx(job, filename)
        elif job.output_format == OutputFormat.DOCX:
            output_path = await self._generate_docx(job, filename)
        elif job.output_format == OutputFormat.PDF:
            output_path = await self._generate_pdf(job, filename)
        elif job.output_format == OutputFormat.XLSX:
            output_path = await self._generate_xlsx(job, filename)
        elif job.output_format == OutputFormat.MARKDOWN:
            output_path = await self._generate_markdown(job, filename)
        elif job.output_format == OutputFormat.HTML:
            output_path = await self._generate_html(job, filename)
        else:
            output_path = await self._generate_txt(job, filename)

        job.output_path = output_path
        return output_path

    async def _generate_pptx(self, job: GenerationJob, filename: str) -> str:
        """Generate professional PowerPoint file with modern styling."""
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            from pptx.dml.color import RGBColor
            from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
            from pptx.enum.shapes import MSO_SHAPE

            prs = Presentation()
            prs.slide_width = Inches(13.333)  # 16:9 aspect ratio
            prs.slide_height = Inches(7.5)

            # Get theme colors from job metadata or use default
            theme_key = job.metadata.get("theme", "business")
            theme = get_theme_colors(theme_key)

            # Apply theme color scheme
            primary_rgb = hex_to_rgb(theme["primary"])
            secondary_rgb = hex_to_rgb(theme["secondary"])
            accent_rgb = hex_to_rgb(theme["accent"])
            text_rgb = hex_to_rgb(theme["text"])
            light_gray_rgb = hex_to_rgb(theme["light_gray"])

            PRIMARY_COLOR = RGBColor(*primary_rgb)
            SECONDARY_COLOR = RGBColor(*secondary_rgb)
            ACCENT_COLOR = RGBColor(*accent_rgb)
            TEXT_COLOR = RGBColor(*text_rgb)
            LIGHT_GRAY = RGBColor(*light_gray_rgb)
            WHITE = RGBColor(0xFF, 0xFF, 0xFF)

            def apply_title_style(shape, font_size=44, bold=True, color=WHITE):
                """Apply consistent title styling."""
                for paragraph in shape.text_frame.paragraphs:
                    paragraph.font.name = "Calibri"
                    paragraph.font.size = Pt(font_size)
                    paragraph.font.bold = bold
                    paragraph.font.color.rgb = color
                    paragraph.alignment = PP_ALIGN.LEFT

            def apply_body_style(shape, font_size=18, color=TEXT_COLOR):
                """Apply consistent body text styling."""
                for paragraph in shape.text_frame.paragraphs:
                    paragraph.font.name = "Calibri"
                    paragraph.font.size = Pt(font_size)
                    paragraph.font.color.rgb = color
                    paragraph.line_spacing = 1.5

            def add_header_bar(slide, text=""):
                """Add a colored header bar to slide."""
                header = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE,
                    Inches(0), Inches(0),
                    prs.slide_width, Inches(1.2)
                )
                header.fill.solid()
                header.fill.fore_color.rgb = PRIMARY_COLOR
                header.line.fill.background()

            def sanitize_text(text: str) -> str:
                """Sanitize text for PPTX XML compatibility."""
                if not text:
                    return ""
                # Remove control characters that can corrupt XML
                import re
                # Remove ASCII control chars except tab, newline, carriage return
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
                # Replace common problematic characters
                text = text.replace('\x0b', ' ')  # Vertical tab
                text = text.replace('\x0c', ' ')  # Form feed
                return text

            # Note: strip_markdown is now a module-level function

            def add_footer(slide, page_num, total_pages):
                """Add footer with page number."""
                footer_text = f"{page_num} / {total_pages}"
                footer = slide.shapes.add_textbox(
                    Inches(12.3), Inches(7.1),
                    Inches(0.8), Inches(0.3)
                )
                tf = footer.text_frame
                tf.paragraphs[0].text = footer_text
                tf.paragraphs[0].font.size = Pt(10)
                tf.paragraphs[0].font.color.rgb = LIGHT_GRAY
                tf.paragraphs[0].alignment = PP_ALIGN.RIGHT

            # Determine include_sources from job metadata or fall back to config
            include_sources_override = job.metadata.get("include_sources")
            if include_sources_override is not None:
                include_sources = include_sources_override
            else:
                include_sources = self.config.include_sources

            logger.info(f"PPTX include_sources check - override: {include_sources_override}, config: {self.config.include_sources}, resolved: {include_sources}")
            logger.info(f"PPTX sources_used count: {len(job.sources_used) if job.sources_used else 0}")

            # Calculate total slides accurately
            total_slides = 1  # Title slide
            if self.config.include_toc:
                total_slides += 1  # TOC slide
            total_slides += len(job.sections)  # Content slides
            if include_sources and job.sources_used:
                total_slides += 1  # Sources slide only if both conditions met

            current_slide = 0

            # Initialize image generator if images are enabled
            include_images = job.metadata.get("include_images", self.config.include_images)
            section_images = {}
            if include_images:
                try:
                    image_config = ImageGeneratorConfig(
                        enabled=True,
                        backend=ImageBackend(self.config.image_backend),
                        default_width=400,
                        default_height=300,
                    )
                    image_service = get_image_generator(image_config)

                    # Generate images for all sections in parallel
                    sections_data = [
                        (section.title, section.revised_content or section.content)
                        for section in job.sections
                    ]
                    images = await image_service.generate_batch(
                        sections=sections_data,
                        document_title=job.title,
                    )

                    # Map images to sections by index
                    for idx, img in enumerate(images):
                        if img and img.success and img.path:
                            section_images[idx] = img.path

                    logger.info(
                        "Generated images for PPTX",
                        total_sections=len(job.sections),
                        images_generated=len(section_images),
                    )
                except Exception as e:
                    logger.warning(f"Image generation failed, continuing without images: {e}")

            # ========== TITLE SLIDE ==========
            current_slide += 1
            slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout

            # Full background gradient effect
            bg_shape = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(0), Inches(0),
                prs.slide_width, prs.slide_height
            )
            bg_shape.fill.solid()
            bg_shape.fill.fore_color.rgb = PRIMARY_COLOR
            bg_shape.line.fill.background()

            # Accent bar at bottom
            accent_bar = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(0), Inches(6.5),
                prs.slide_width, Inches(1)
            )
            accent_bar.fill.solid()
            accent_bar.fill.fore_color.rgb = SECONDARY_COLOR
            accent_bar.line.fill.background()

            # Title text
            title_box = slide.shapes.add_textbox(
                Inches(0.8), Inches(2.5),
                Inches(11), Inches(1.5)
            )
            tf = title_box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = sanitize_text(job.title) or "Untitled"
            p.font.name = "Calibri"
            p.font.size = Pt(48)
            p.font.bold = True
            p.font.color.rgb = WHITE

            # Subtitle/description
            desc_box = slide.shapes.add_textbox(
                Inches(0.8), Inches(4.2),
                Inches(11), Inches(1)
            )
            tf = desc_box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            desc_text = job.outline.description if job.outline else job.description
            p.text = sanitize_text(desc_text) or ""
            p.font.name = "Calibri"
            p.font.size = Pt(20)
            p.font.color.rgb = ACCENT_COLOR

            # Date
            from datetime import datetime
            date_box = slide.shapes.add_textbox(
                Inches(0.8), Inches(6.7),
                Inches(4), Inches(0.4)
            )
            tf = date_box.text_frame
            p = tf.paragraphs[0]
            p.text = datetime.now().strftime("%B %d, %Y")
            p.font.name = "Calibri"
            p.font.size = Pt(14)
            p.font.color.rgb = WHITE

            # ========== TABLE OF CONTENTS SLIDE ==========
            if self.config.include_toc:
                current_slide += 1
                slide = prs.slides.add_slide(prs.slide_layouts[6])
                add_header_bar(slide)

                # TOC Title
                toc_title = slide.shapes.add_textbox(
                    Inches(0.8), Inches(0.3),
                    Inches(8), Inches(0.8)
                )
                tf = toc_title.text_frame
                p = tf.paragraphs[0]
                p.text = "Contents"
                p.font.name = "Calibri"
                p.font.size = Pt(36)
                p.font.bold = True
                p.font.color.rgb = WHITE

                # TOC items
                toc_box = slide.shapes.add_textbox(
                    Inches(0.8), Inches(1.8),
                    Inches(11), Inches(5)
                )
                tf = toc_box.text_frame
                tf.word_wrap = True

                first_toc_used = False
                for idx, section in enumerate(job.sections):
                    if first_toc_used:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]
                        first_toc_used = True
                    section_title = sanitize_text(section.title) or f"Section {idx + 1}"
                    p.text = f"{idx + 1}.  {section_title}"
                    p.font.name = "Calibri"
                    p.font.size = Pt(20)
                    p.font.color.rgb = TEXT_COLOR
                    p.space_after = Pt(12)

                # Ensure first paragraph is initialized even if no sections
                if not first_toc_used:
                    p = tf.paragraphs[0]
                    p.text = "No sections"
                    p.font.name = "Calibri"
                    p.font.size = Pt(20)

                add_footer(slide, current_slide, total_slides)

            # ========== CONTENT SLIDES ==========
            for section_idx, section in enumerate(job.sections):
                current_slide += 1
                slide = prs.slides.add_slide(prs.slide_layouts[6])
                add_header_bar(slide)

                # Check if we have an image for this section
                has_image = section_idx in section_images

                # Section title
                section_title = slide.shapes.add_textbox(
                    Inches(0.8), Inches(0.3),
                    Inches(11), Inches(0.8)
                )
                tf = section_title.text_frame
                p = tf.paragraphs[0]
                p.text = sanitize_text(section.title) or f"Section {section_idx + 1}"
                p.font.name = "Calibri"
                p.font.size = Pt(32)
                p.font.bold = True
                p.font.color.rgb = WHITE

                # Adjust content area based on whether we have an image
                content = sanitize_text(section.revised_content or section.content) or ""
                # Strip markdown formatting for clean slide content
                content = strip_markdown(content)

                # Limit content for presentation readability with sentence awareness
                # ~450 words = ~3000 chars, but truncate at sentence boundaries
                if len(content) > 3000:
                    content = sentence_truncate(content, 3000)
                    if not content.endswith(('.', '!', '?')):
                        content += '...'

                if has_image:
                    # Content on left, image on right
                    content_box = slide.shapes.add_textbox(
                        Inches(0.8), Inches(1.6),
                        Inches(7.5), Inches(5.2)
                    )
                else:
                    # Full width content
                    content_box = slide.shapes.add_textbox(
                        Inches(0.8), Inches(1.6),
                        Inches(11.5), Inches(5.2)
                    )

                tf = content_box.text_frame
                tf.word_wrap = True

                # Split content into bullet points for cleaner slides
                paragraphs = content.split('\n')
                first_para_used = False
                max_paras = 10 if has_image else 12  # Reasonable limits for readability
                para_count = 0

                # Collect valid paragraphs first to calculate dynamic sizing
                valid_paragraphs = []
                for para_text in paragraphs:
                    para_text = para_text.strip()
                    if para_text:
                        # Format as bullet point for cleaner slides
                        if para_text.startswith(('- ', '• ', '* ', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                            bullet_text = para_text.lstrip('-•* 0123456789.').strip()
                        else:
                            bullet_text = para_text
                        if bullet_text:
                            valid_paragraphs.append(bullet_text)

                # Dynamic font sizing based on content volume
                # Increased max_chars limits to reduce truncation
                total_bullets = min(len(valid_paragraphs), max_paras)
                total_chars = sum(len(p) for p in valid_paragraphs[:max_paras])

                if total_bullets <= 5 and total_chars < 500:
                    font_size = Pt(20)
                    max_chars = 180  # Was 120
                    line_spacing = Pt(10)
                elif total_bullets <= 8 and total_chars < 900:
                    font_size = Pt(18)
                    max_chars = 150  # Was 100
                    line_spacing = Pt(8)
                elif total_bullets <= 10 and total_chars < 1200:
                    font_size = Pt(16)
                    max_chars = 130  # Was 90
                    line_spacing = Pt(6)
                else:
                    font_size = Pt(14)
                    max_chars = 110  # Was 80
                    line_spacing = Pt(4)

                # Adjust max chars if we have an image (narrower content area)
                if has_image:
                    max_chars = int(max_chars * 0.75)  # Was 0.7

                for bullet_text in valid_paragraphs:
                    if para_count >= max_paras:
                        break

                    if first_para_used:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]
                        first_para_used = True
                    para_count += 1

                    # Use sentence-aware truncation to avoid mid-sentence cuts
                    if len(bullet_text) > max_chars:
                        bullet_text = sentence_truncate(bullet_text, max_chars)
                        # Ensure truncated content shows it was cut
                        if not bullet_text.endswith(('.', '!', '?', '...')):
                            bullet_text += '...'

                    p.text = '• ' + bullet_text
                    p.font.name = "Calibri"
                    p.font.size = font_size
                    p.font.color.rgb = TEXT_COLOR
                    p.space_after = line_spacing

                # Ensure first paragraph is initialized even if content was empty
                if not first_para_used:
                    p = tf.paragraphs[0]
                    p.text = ""  # Empty but properly initialized
                    p.font.name = "Calibri"
                    p.font.size = Pt(16)

                # Add image if available
                if has_image:
                    try:
                        image_path = section_images[section_idx]
                        # Position image on right side of slide
                        slide.shapes.add_picture(
                            image_path,
                            Inches(8.5), Inches(1.8),
                            width=Inches(4.3),
                            height=Inches(3.2),
                        )
                        logger.debug(
                            "Added image to PPTX slide",
                            section=section.title,
                            image_path=image_path,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add image to slide: {e}")

                add_footer(slide, current_slide, total_slides)

            # ========== SOURCES SLIDE ==========
            if not include_sources:
                logger.warning(f"Sources slide SKIPPED - include_sources is False")
            elif not job.sources_used:
                logger.warning(f"Sources slide SKIPPED - no sources_used (RAG may have returned no results)")
            if include_sources and job.sources_used:
                current_slide += 1
                slide = prs.slides.add_slide(prs.slide_layouts[6])
                add_header_bar(slide)

                # Sources title
                sources_title = slide.shapes.add_textbox(
                    Inches(0.8), Inches(0.3),
                    Inches(8), Inches(0.8)
                )
                tf = sources_title.text_frame
                p = tf.paragraphs[0]
                p.text = "Sources & References"
                p.font.name = "Calibri"
                p.font.size = Pt(32)
                p.font.bold = True
                p.font.color.rgb = WHITE

                # Sources list
                sources_box = slide.shapes.add_textbox(
                    Inches(0.8), Inches(1.6),
                    Inches(11), Inches(5)
                )
                tf = sources_box.text_frame
                tf.word_wrap = True

                first_source_used = False
                for source in job.sources_used[:10]:
                    if first_source_used:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]
                        first_source_used = True
                    doc_name = sanitize_text(source.document_name or source.document_id[:20])
                    p.text = f"•  {doc_name}"
                    p.font.name = "Calibri"
                    p.font.size = Pt(14)
                    p.font.color.rgb = TEXT_COLOR
                    p.space_after = Pt(6)

                # Ensure first paragraph is initialized even if no sources
                if not first_source_used:
                    p = tf.paragraphs[0]
                    p.text = "No sources available"
                    p.font.name = "Calibri"
                    p.font.size = Pt(14)

                add_footer(slide, current_slide, total_slides)

            output_path = os.path.join(self.config.output_dir, f"{filename}.pptx")
            prs.save(output_path)

            logger.info("Professional PPTX generated", path=output_path)
            return output_path

        except ImportError as e:
            logger.error(f"python-pptx import error: {e}")
            return await self._generate_txt(job, filename)

    async def _generate_docx(self, job: GenerationJob, filename: str) -> str:
        """Generate professional Word document with cover page and proper styling."""
        try:
            from docx import Document
            from docx.shared import Pt, Inches, RGBColor, Cm
            from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
            from docx.enum.style import WD_STYLE_TYPE
            from docx.enum.table import WD_TABLE_ALIGNMENT

            doc = Document()

            # Get theme colors from job metadata or use default
            theme_key = job.metadata.get("theme", "business")
            theme = get_theme_colors(theme_key)

            # Apply theme color scheme
            primary_rgb = hex_to_rgb(theme["primary"])
            secondary_rgb = hex_to_rgb(theme["secondary"])
            text_rgb = hex_to_rgb(theme["text"])
            light_gray_rgb = hex_to_rgb(theme["light_gray"])

            PRIMARY_COLOR = RGBColor(*primary_rgb)
            SECONDARY_COLOR = RGBColor(*secondary_rgb)
            TEXT_COLOR = RGBColor(*text_rgb)
            LIGHT_GRAY = RGBColor(*light_gray_rgb)

            # Determine include_sources from job metadata or fall back to config
            include_sources = job.metadata.get("include_sources", self.config.include_sources)

            # Initialize image generator if images are enabled
            include_images = job.metadata.get("include_images", self.config.include_images)
            section_images = {}
            if include_images:
                try:
                    image_config = ImageGeneratorConfig(
                        enabled=True,
                        backend=ImageBackend(self.config.image_backend),
                        default_width=600,
                        default_height=400,
                    )
                    image_service = get_image_generator(image_config)

                    # Generate images for all sections in parallel
                    sections_data = [
                        (section.title, section.revised_content or section.content)
                        for section in job.sections
                    ]
                    images = await image_service.generate_batch(
                        sections=sections_data,
                        document_title=job.title,
                    )

                    # Map images to sections by index
                    for idx, img in enumerate(images):
                        if img and img.success and img.path:
                            section_images[idx] = img.path

                    logger.info(
                        "Generated images for DOCX",
                        total_sections=len(job.sections),
                        images_generated=len(section_images),
                    )
                except Exception as e:
                    logger.warning(f"Image generation failed, continuing without images: {e}")

            # Configure document margins
            for section in doc.sections:
                section.top_margin = Inches(1)
                section.bottom_margin = Inches(1)
                section.left_margin = Inches(1.25)
                section.right_margin = Inches(1.25)

            # ========== COVER PAGE ==========
            # Add spacing before title
            for _ in range(8):
                doc.add_paragraph()

            # Document title
            title_para = doc.add_paragraph()
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title_para.add_run(job.title)
            title_run.font.name = "Calibri"
            title_run.font.size = Pt(36)
            title_run.font.bold = True
            title_run.font.color.rgb = PRIMARY_COLOR

            # Subtitle / description
            doc.add_paragraph()
            if job.outline:
                desc_para = doc.add_paragraph()
                desc_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                desc_run = desc_para.add_run(job.outline.description)
                desc_run.font.name = "Calibri"
                desc_run.font.size = Pt(14)
                desc_run.font.color.rgb = SECONDARY_COLOR
                desc_run.font.italic = True

            # Add more spacing
            for _ in range(6):
                doc.add_paragraph()

            # Decorative line
            line_para = doc.add_paragraph()
            line_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            line_run = line_para.add_run("─" * 40)
            line_run.font.color.rgb = SECONDARY_COLOR

            # Date
            from datetime import datetime
            doc.add_paragraph()
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_run = date_para.add_run(datetime.now().strftime("%B %d, %Y"))
            date_run.font.name = "Calibri"
            date_run.font.size = Pt(12)
            date_run.font.color.rgb = LIGHT_GRAY

            # Page break after cover
            doc.add_page_break()

            # ========== TABLE OF CONTENTS ==========
            if self.config.include_toc:
                toc_heading = doc.add_heading("Table of Contents", level=1)
                toc_heading.runs[0].font.color.rgb = PRIMARY_COLOR
                toc_heading.runs[0].font.size = Pt(24)

                doc.add_paragraph()

                for idx, section in enumerate(job.sections):
                    toc_entry = doc.add_paragraph()
                    toc_entry.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    toc_entry.paragraph_format.space_after = Pt(6)

                    # Section number
                    num_run = toc_entry.add_run(f"{idx + 1}.  ")
                    num_run.font.name = "Calibri"
                    num_run.font.size = Pt(12)
                    num_run.font.bold = True
                    num_run.font.color.rgb = SECONDARY_COLOR

                    # Section title
                    title_run = toc_entry.add_run(section.title)
                    title_run.font.name = "Calibri"
                    title_run.font.size = Pt(12)
                    title_run.font.color.rgb = TEXT_COLOR

                doc.add_page_break()

            # ========== CONTENT SECTIONS ==========
            for idx, section in enumerate(job.sections):
                # Section heading
                heading = doc.add_heading(section.title, level=1)
                heading.runs[0].font.color.rgb = PRIMARY_COLOR
                heading.runs[0].font.size = Pt(20)
                heading.paragraph_format.space_before = Pt(12)
                heading.paragraph_format.space_after = Pt(12)

                content = section.revised_content or section.content

                # Split content into paragraphs and format
                paragraphs = content.split('\n\n')
                for para_text in paragraphs:
                    para_text = para_text.strip()
                    if not para_text:
                        continue

                    # Check if it's a subheading (starts with ## or ###)
                    if para_text.startswith('###'):
                        sub = doc.add_heading(para_text.lstrip('#').strip(), level=3)
                        sub.runs[0].font.color.rgb = SECONDARY_COLOR
                        sub.runs[0].font.size = Pt(13)
                    elif para_text.startswith('##'):
                        sub = doc.add_heading(para_text.lstrip('#').strip(), level=2)
                        sub.runs[0].font.color.rgb = SECONDARY_COLOR
                        sub.runs[0].font.size = Pt(15)
                    # Check if it's a bullet list
                    elif para_text.startswith(('- ', '• ', '* ')):
                        lines = para_text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith(('- ', '• ', '* ')):
                                bullet_para = doc.add_paragraph(style='List Bullet')
                                bullet_run = bullet_para.add_run(line.lstrip('-•* ').strip())
                                bullet_run.font.name = "Calibri"
                                bullet_run.font.size = Pt(11)
                                bullet_run.font.color.rgb = TEXT_COLOR
                    # Regular paragraph with inline formatting support
                    else:
                        para = doc.add_paragraph()
                        para.paragraph_format.line_spacing = 1.5
                        para.paragraph_format.space_after = Pt(8)

                        # Parse inline bold (**text**) and italic (*text*) formatting
                        import re
                        # Pattern to match **bold**, *italic*, or plain text
                        pattern = r'(\*\*[^*]+\*\*|\*[^*]+\*|[^*]+)'
                        parts = re.findall(pattern, para_text)

                        for part in parts:
                            if part.startswith('**') and part.endswith('**'):
                                # Bold text
                                run = para.add_run(part[2:-2])
                                run.bold = True
                            elif part.startswith('*') and part.endswith('*') and len(part) > 2:
                                # Italic text
                                run = para.add_run(part[1:-1])
                                run.italic = True
                            else:
                                # Plain text
                                run = para.add_run(part)

                            run.font.name = "Calibri"
                            run.font.size = Pt(11)
                            run.font.color.rgb = TEXT_COLOR

                # Add image for this section if available
                if idx in section_images:
                    try:
                        image_path = section_images[idx]
                        doc.add_paragraph()  # Add spacing before image
                        img_para = doc.add_paragraph()
                        img_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        run = img_para.add_run()
                        run.add_picture(image_path, width=Inches(5.0))
                        doc.add_paragraph()  # Add spacing after image
                        logger.debug(
                            "Added image to DOCX section",
                            section=section.title,
                            image_path=image_path,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add image to DOCX: {e}")

                # Add section sources
                if include_sources and section.sources:
                    doc.add_paragraph()
                    source_label = doc.add_paragraph()
                    label_run = source_label.add_run("Sources for this section:")
                    label_run.font.name = "Calibri"
                    label_run.font.size = Pt(9)
                    label_run.font.italic = True
                    label_run.font.color.rgb = LIGHT_GRAY

                    for source in section.sources[:3]:
                        src_para = doc.add_paragraph()
                        src_para.paragraph_format.left_indent = Inches(0.25)
                        src_run = src_para.add_run(f"• {source.document_name or source.document_id}")
                        src_run.font.name = "Calibri"
                        src_run.font.size = Pt(9)
                        src_run.font.color.rgb = LIGHT_GRAY

                # Add page break between sections (except last)
                if idx < len(job.sections) - 1:
                    doc.add_page_break()

            # ========== REFERENCES SECTION ==========
            if include_sources and job.sources_used:
                doc.add_page_break()

                ref_heading = doc.add_heading("References", level=1)
                ref_heading.runs[0].font.color.rgb = PRIMARY_COLOR
                ref_heading.runs[0].font.size = Pt(20)

                doc.add_paragraph()

                for source in job.sources_used:
                    ref_para = doc.add_paragraph()
                    ref_para.paragraph_format.space_after = Pt(4)
                    ref_run = ref_para.add_run(f"• {source.document_name or source.document_id}")
                    ref_run.font.name = "Calibri"
                    ref_run.font.size = Pt(10)
                    ref_run.font.color.rgb = TEXT_COLOR

            output_path = os.path.join(self.config.output_dir, f"{filename}.docx")
            doc.save(output_path)

            logger.info("Professional DOCX generated", path=output_path)
            return output_path

        except ImportError:
            logger.error("python-docx not installed")
            return await self._generate_txt(job, filename)

    async def _generate_pdf(self, job: GenerationJob, filename: str) -> str:
        """Generate professional PDF document with cover page and styling."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor, Color
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                Table, TableStyle, ListFlowable, ListItem, Image
            )
            from reportlab.lib.units import inch, cm

            # Determine include_sources from job metadata or fall back to config
            include_sources = job.metadata.get("include_sources", self.config.include_sources)

            # Initialize image generator if images are enabled
            include_images = job.metadata.get("include_images", self.config.include_images)
            section_images = {}
            if include_images:
                try:
                    image_config = ImageGeneratorConfig(
                        enabled=True,
                        backend=ImageBackend(self.config.image_backend),
                        default_width=600,
                        default_height=400,
                    )
                    image_service = get_image_generator(image_config)

                    # Generate images for all sections in parallel
                    sections_data = [
                        (section.title, section.revised_content or section.content)
                        for section in job.sections
                    ]
                    images = await image_service.generate_batch(
                        sections=sections_data,
                        document_title=job.title,
                    )

                    # Map images to sections by index
                    for idx, img in enumerate(images):
                        if img and img.success and img.path:
                            section_images[idx] = img.path

                    logger.info(
                        "Generated images for PDF",
                        total_sections=len(job.sections),
                        images_generated=len(section_images),
                    )
                except Exception as e:
                    logger.warning(f"Image generation failed, continuing without images: {e}")

            # Get theme colors from job metadata or use default
            theme_key = job.metadata.get("theme", "business")
            theme = get_theme_colors(theme_key)

            # Apply theme color scheme
            PRIMARY_COLOR = HexColor(theme["primary"])
            SECONDARY_COLOR = HexColor(theme["secondary"])
            TEXT_COLOR = HexColor(theme["text"])
            LIGHT_GRAY = HexColor(theme["light_gray"])
            ACCENT_BG = HexColor('#F5F5F5')  # Light background (keep neutral)

            output_path = os.path.join(self.config.output_dir, f"{filename}.pdf")

            # Custom page template for headers/footers
            def add_page_number(canvas, doc):
                """Add page number to footer."""
                canvas.saveState()
                canvas.setFont('Helvetica', 9)
                canvas.setFillColor(LIGHT_GRAY)
                page_num = canvas.getPageNumber()
                text = f"Page {page_num}"
                canvas.drawRightString(letter[0] - 0.75*inch, 0.5*inch, text)
                canvas.restoreState()

            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch,
            )

            # Create custom styles
            styles = getSampleStyleSheet()

            # Cover title style
            cover_title_style = ParagraphStyle(
                'CoverTitle',
                parent=styles['Title'],
                fontSize=36,
                textColor=PRIMARY_COLOR,
                alignment=TA_CENTER,
                spaceAfter=20,
                fontName='Helvetica-Bold',
            )

            # Cover subtitle style
            cover_subtitle_style = ParagraphStyle(
                'CoverSubtitle',
                parent=styles['Normal'],
                fontSize=14,
                textColor=SECONDARY_COLOR,
                alignment=TA_CENTER,
                fontName='Helvetica-Oblique',
                spaceAfter=12,
            )

            # Section heading style
            heading_style = ParagraphStyle(
                'SectionHeading',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=PRIMARY_COLOR,
                spaceBefore=16,
                spaceAfter=12,
                fontName='Helvetica-Bold',
            )

            # Subheading style
            subheading_style = ParagraphStyle(
                'SubHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=SECONDARY_COLOR,
                spaceBefore=12,
                spaceAfter=8,
                fontName='Helvetica-Bold',
            )

            # Body text style
            body_style = ParagraphStyle(
                'BodyText',
                parent=styles['Normal'],
                fontSize=11,
                textColor=TEXT_COLOR,
                alignment=TA_JUSTIFY,
                spaceBefore=4,
                spaceAfter=8,
                leading=16,
                fontName='Helvetica',
            )

            # TOC style
            toc_style = ParagraphStyle(
                'TOCEntry',
                parent=styles['Normal'],
                fontSize=12,
                textColor=TEXT_COLOR,
                spaceBefore=6,
                spaceAfter=6,
                leftIndent=20,
                fontName='Helvetica',
            )

            # Source/reference style
            source_style = ParagraphStyle(
                'SourceStyle',
                parent=styles['Normal'],
                fontSize=9,
                textColor=LIGHT_GRAY,
                fontName='Helvetica-Oblique',
                spaceBefore=4,
                spaceAfter=2,
            )

            story = []

            # ========== COVER PAGE ==========
            story.append(Spacer(1, 2.5*inch))

            # Title
            story.append(Paragraph(job.title, cover_title_style))
            story.append(Spacer(1, 0.3*inch))

            # Decorative line
            line_data = [['─' * 50]]
            line_table = Table(line_data, colWidths=[5*inch])
            line_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TEXTCOLOR', (0, 0), (-1, -1), SECONDARY_COLOR),
            ]))
            story.append(line_table)
            story.append(Spacer(1, 0.3*inch))

            # Description
            if job.outline:
                story.append(Paragraph(job.outline.description, cover_subtitle_style))

            story.append(Spacer(1, 2*inch))

            # Date
            from datetime import datetime
            date_style = ParagraphStyle(
                'DateStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=LIGHT_GRAY,
                alignment=TA_CENTER,
                fontName='Helvetica',
            )
            story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), date_style))

            story.append(PageBreak())

            # ========== TABLE OF CONTENTS ==========
            if self.config.include_toc:
                toc_title_style = ParagraphStyle(
                    'TOCTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=PRIMARY_COLOR,
                    spaceAfter=20,
                    fontName='Helvetica-Bold',
                )
                story.append(Paragraph("Table of Contents", toc_title_style))
                story.append(Spacer(1, 0.3*inch))

                for idx, section in enumerate(job.sections):
                    toc_entry = f"<b><font color='#3D5A80'>{idx + 1}.</font></b>  {section.title}"
                    story.append(Paragraph(toc_entry, toc_style))

                story.append(PageBreak())

            # ========== CONTENT SECTIONS ==========
            for idx, section in enumerate(job.sections):
                # Section heading with number
                heading_text = f"{idx + 1}. {section.title}"
                story.append(Paragraph(heading_text, heading_style))

                content = section.revised_content or section.content

                # Split content and format
                paragraphs = content.split('\n')
                current_list = []

                for para_text in paragraphs:
                    para_text = para_text.strip()
                    if not para_text:
                        # Flush any pending list
                        if current_list:
                            story.append(ListFlowable(
                                current_list,
                                bulletType='bullet',
                                leftIndent=20,
                            ))
                            current_list = []
                        continue

                    # Handle markdown-style headers
                    if para_text.startswith('###'):
                        if current_list:
                            story.append(ListFlowable(current_list, bulletType='bullet', leftIndent=20))
                            current_list = []
                        story.append(Paragraph(para_text.lstrip('#').strip(), subheading_style))
                    elif para_text.startswith('##'):
                        if current_list:
                            story.append(ListFlowable(current_list, bulletType='bullet', leftIndent=20))
                            current_list = []
                        story.append(Paragraph(para_text.lstrip('#').strip(), subheading_style))
                    # Handle bullet points
                    elif para_text.startswith(('- ', '• ', '* ')):
                        bullet_text = para_text.lstrip('-•* ').strip()
                        current_list.append(ListItem(Paragraph(bullet_text, body_style)))
                    # Handle numbered lists
                    elif para_text[:2].replace('.', '').isdigit():
                        if current_list:
                            story.append(ListFlowable(current_list, bulletType='bullet', leftIndent=20))
                            current_list = []
                        import re
                        num_match = re.match(r'^(\d+\.)\s*(.+)', para_text)
                        if num_match:
                            story.append(Paragraph(f"<b>{num_match.group(1)}</b> {num_match.group(2)}", body_style))
                        else:
                            story.append(Paragraph(para_text, body_style))
                    # Regular paragraph
                    else:
                        if current_list:
                            story.append(ListFlowable(current_list, bulletType='bullet', leftIndent=20))
                            current_list = []
                        # Escape XML special characters first
                        formatted = para_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        # Convert markdown bold (**text**) to HTML bold for ReportLab
                        import re
                        formatted = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', formatted)
                        # Convert markdown italic (*text*) to HTML italic
                        formatted = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', formatted)
                        story.append(Paragraph(formatted, body_style))

                # Flush remaining list items
                if current_list:
                    story.append(ListFlowable(current_list, bulletType='bullet', leftIndent=20))

                # Add image for this section if available
                if idx in section_images:
                    try:
                        image_path = section_images[idx]
                        story.append(Spacer(1, 0.2*inch))
                        # Create image with max width of 5 inches
                        img = Image(image_path, width=5*inch, height=3.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                        logger.debug(
                            "Added image to PDF section",
                            section=section.title,
                            image_path=image_path,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add image to PDF: {e}")

                # Section sources
                if include_sources and section.sources:
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph("Sources for this section:", source_style))
                    for source in section.sources[:3]:
                        doc_name = source.document_name or source.document_id
                        story.append(Paragraph(f"• {doc_name}", source_style))

                story.append(Spacer(1, 0.3*inch))

            # ========== REFERENCES ==========
            if include_sources and job.sources_used:
                story.append(PageBreak())
                story.append(Paragraph("References", heading_style))
                story.append(Spacer(1, 0.2*inch))

                for source in job.sources_used:
                    doc_name = source.document_name or source.document_id
                    ref_style = ParagraphStyle(
                        'RefStyle',
                        parent=styles['Normal'],
                        fontSize=10,
                        textColor=TEXT_COLOR,
                        spaceBefore=4,
                        spaceAfter=4,
                        leftIndent=15,
                        fontName='Helvetica',
                    )
                    story.append(Paragraph(f"• {doc_name}", ref_style))

            # Build PDF with page numbers
            doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

            logger.info("Professional PDF generated", path=output_path)
            return output_path

        except ImportError:
            logger.error("reportlab not installed")
            return await self._generate_txt(job, filename)

    async def _generate_markdown(self, job: GenerationJob, filename: str) -> str:
        """Generate Markdown document."""
        # Determine include_sources from job metadata or fall back to config
        include_sources = job.metadata.get("include_sources", self.config.include_sources)

        lines = []

        # Title
        lines.append(f"# {job.title}")
        lines.append("")

        # Description
        if job.outline:
            lines.append(job.outline.description)
            lines.append("")

        # Table of contents
        if self.config.include_toc:
            lines.append("## Table of Contents")
            lines.append("")
            for section in job.sections:
                anchor = section.title.lower().replace(" ", "-")
                lines.append(f"- [{section.title}](#{anchor})")
            lines.append("")

        # Content
        for section in job.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            content = section.revised_content or section.content
            lines.append(content)
            lines.append("")

            # Sources
            if include_sources and section.sources:
                lines.append("**Sources:**")
                for source in section.sources[:3]:
                    lines.append(f"- {source.document_name or source.document_id}")
                lines.append("")

        # References
        if include_sources and job.sources_used:
            lines.append("## References")
            lines.append("")
            for source in job.sources_used:
                lines.append(f"- {source.document_name or source.document_id}")

        output_path = os.path.join(self.config.output_dir, f"{filename}.md")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info("Markdown generated", path=output_path)
        return output_path

    async def _generate_html(self, job: GenerationJob, filename: str) -> str:
        """Generate HTML document."""
        # Determine include_sources from job metadata or fall back to config
        include_sources = job.metadata.get("include_sources", self.config.include_sources)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{job.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        .section {{ margin-bottom: 30px; }}
        .sources {{ font-size: 0.9em; color: #888; }}
    </style>
</head>
<body>
    <h1>{job.title}</h1>
"""

        if job.outline:
            html += f"    <p><em>{job.outline.description}</em></p>\n"

        for section in job.sections:
            content = section.revised_content or section.content
            html += f"""    <div class="section">
        <h2>{section.title}</h2>
        <p>{content.replace(chr(10), '</p><p>')}</p>
"""
            if include_sources and section.sources:
                html += '        <div class="sources">Sources: '
                html += ", ".join(s.document_name or s.document_id for s in section.sources[:3])
                html += "</div>\n"

            html += "    </div>\n"

        html += """</body>
</html>"""

        output_path = os.path.join(self.config.output_dir, f"{filename}.html")
        with open(output_path, "w") as f:
            f.write(html)

        logger.info("HTML generated", path=output_path)
        return output_path

    async def _generate_xlsx(self, job: GenerationJob, filename: str) -> str:
        """
        Generate Excel spreadsheet with document content.

        Creates a structured Excel file with:
        - Summary sheet with document overview
        - Content sheet with all sections
        - Sources sheet with references
        """
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
            from openpyxl.utils import get_column_letter

            # Determine include_sources from job metadata or fall back to config
            include_sources = job.metadata.get("include_sources", self.config.include_sources)

            wb = Workbook()

            # Get theme colors from job metadata or use default
            theme_key = job.metadata.get("theme", "business")
            theme = get_theme_colors(theme_key)

            # Convert theme color to Excel format (without #)
            primary_hex = theme["primary"].lstrip('#')

            # Define styles with theme colors
            header_font = Font(bold=True, size=12, color="FFFFFF")
            header_fill = PatternFill(start_color=primary_hex, end_color=primary_hex, fill_type="solid")
            header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell_alignment = Alignment(vertical="top", wrap_text=True)
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            # Sheet 1: Summary
            ws_summary = wb.active
            ws_summary.title = "Summary"

            # Title row
            ws_summary.merge_cells('A1:B1')
            ws_summary['A1'] = job.title
            ws_summary['A1'].font = Font(bold=True, size=16)
            ws_summary['A1'].alignment = Alignment(horizontal="center")

            # Summary data
            summary_data = [
                ("Description", job.outline.description if job.outline else ""),
                ("Status", job.status.value),
                ("Created", job.created_at.strftime("%Y-%m-%d %H:%M")),
                ("Completed", job.completed_at.strftime("%Y-%m-%d %H:%M") if job.completed_at else "In Progress"),
                ("Total Sections", str(len(job.sections))),
                ("Sources Used", str(len(job.sources_used))),
            ]

            for i, (label, value) in enumerate(summary_data, start=3):
                ws_summary[f'A{i}'] = label
                ws_summary[f'A{i}'].font = Font(bold=True)
                ws_summary[f'B{i}'] = value
                ws_summary[f'A{i}'].border = thin_border
                ws_summary[f'B{i}'].border = thin_border

            # Adjust column widths
            ws_summary.column_dimensions['A'].width = 20
            ws_summary.column_dimensions['B'].width = 60

            # Sheet 2: Content
            ws_content = wb.create_sheet("Content")

            # Headers
            headers = ["Section #", "Title", "Content", "Approved", "Feedback"]
            for col, header in enumerate(headers, start=1):
                cell = ws_content.cell(row=1, column=col, value=header)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
                cell.border = thin_border

            # Content rows
            for i, section in enumerate(job.sections, start=2):
                content = section.revised_content or section.content
                # Strip markdown formatting for clean Excel content
                content = strip_markdown(content) if content else ""

                ws_content.cell(row=i, column=1, value=section.order).border = thin_border
                ws_content.cell(row=i, column=2, value=section.title).border = thin_border
                ws_content.cell(row=i, column=3, value=content).border = thin_border
                ws_content.cell(row=i, column=3).alignment = cell_alignment
                ws_content.cell(row=i, column=4, value="Yes" if section.approved else "No").border = thin_border
                ws_content.cell(row=i, column=5, value=section.feedback or "").border = thin_border

            # Adjust column widths
            ws_content.column_dimensions['A'].width = 12
            ws_content.column_dimensions['B'].width = 30
            ws_content.column_dimensions['C'].width = 80
            ws_content.column_dimensions['D'].width = 12
            ws_content.column_dimensions['E'].width = 40

            # Sheet 3: Sources (only if include_sources is enabled)
            if include_sources and job.sources_used:
                ws_sources = wb.create_sheet("Sources")

                # Headers
                source_headers = ["#", "Document Name", "Page", "Relevance", "Snippet"]
                for col, header in enumerate(source_headers, start=1):
                    cell = ws_sources.cell(row=1, column=col, value=header)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = header_alignment
                    cell.border = thin_border

                # Source rows
                for i, source in enumerate(job.sources_used, start=2):
                    ws_sources.cell(row=i, column=1, value=i-1).border = thin_border
                    ws_sources.cell(row=i, column=2, value=source.document_name).border = thin_border
                    ws_sources.cell(row=i, column=3, value=source.page_number or "N/A").border = thin_border
                    ws_sources.cell(row=i, column=4, value=f"{source.relevance_score:.2f}").border = thin_border
                    ws_sources.cell(row=i, column=5, value=source.snippet[:200] + "..." if len(source.snippet) > 200 else source.snippet).border = thin_border
                    ws_sources.cell(row=i, column=5).alignment = cell_alignment

                # Adjust column widths
                ws_sources.column_dimensions['A'].width = 8
                ws_sources.column_dimensions['B'].width = 40
                ws_sources.column_dimensions['C'].width = 10
                ws_sources.column_dimensions['D'].width = 12
                ws_sources.column_dimensions['E'].width = 60

            # Save workbook
            output_path = os.path.join(self.config.output_dir, f"{filename}.xlsx")
            wb.save(output_path)

            logger.info("XLSX generated", path=output_path)
            return output_path

        except ImportError:
            logger.error("openpyxl not installed, falling back to TXT")
            return await self._generate_txt(job, filename)
        except Exception as e:
            logger.error("XLSX generation failed", error=str(e))
            raise

    async def _generate_txt(self, job: GenerationJob, filename: str) -> str:
        """Generate plain text document."""
        lines = []

        lines.append("=" * 60)
        lines.append(job.title.upper())
        lines.append("=" * 60)
        lines.append("")

        if job.outline:
            lines.append(job.outline.description)
            lines.append("")

        for section in job.sections:
            lines.append("-" * 40)
            lines.append(section.title)
            lines.append("-" * 40)
            lines.append("")
            content = section.revised_content or section.content
            lines.append(content)
            lines.append("")

        output_path = os.path.join(self.config.output_dir, f"{filename}.txt")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info("TXT generated", path=output_path)
        return output_path

    def _get_content_type(self, format: OutputFormat) -> str:
        """Get MIME content type for format."""
        content_types = {
            OutputFormat.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            OutputFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            OutputFormat.PDF: "application/pdf",
            OutputFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            OutputFormat.MARKDOWN: "text/markdown",
            OutputFormat.HTML: "text/html",
            OutputFormat.TXT: "text/plain",
        }
        return content_types.get(format, "application/octet-stream")


# =============================================================================
# Singleton Instance
# =============================================================================

_generation_service: Optional[DocumentGenerationService] = None


def get_generation_service() -> DocumentGenerationService:
    """Get or create document generation service instance."""
    global _generation_service

    if _generation_service is None:
        _generation_service = DocumentGenerationService()

    return _generation_service
