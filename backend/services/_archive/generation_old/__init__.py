"""
AIDocumentIndexer - Document Generation Module
===============================================

Human-in-the-loop document generation using LangGraph.
Supports PPTX, DOCX, PDF, Markdown, HTML, XLSX, and TXT output formats.

This module is organized as follows:

- config.py: Themes, fonts, layouts, language configuration
- models.py: Data models (OutputFormat, Section, GenerationJob, etc.)
- utils.py: Text processing utilities (markdown stripping, truncation, etc.)
- citations.py: Inline citation support
- styles.py: Style learning and application
- service.py: Main DocumentGenerationService class
- formats/: Output format generators (PPTX, DOCX, PDF, etc.)

Usage:
    from backend.services.generation import (
        DocumentGenerationService,
        get_generation_service,
        OutputFormat,
        GenerationConfig,
    )

    service = get_generation_service()
    job = await service.create_job(
        topic="Quarterly Report",
        output_format=OutputFormat.PPTX,
    )
"""

from backend.services.generation.config import (
    LANGUAGE_NAMES,
    THEMES,
    FONT_FAMILIES,
    LAYOUT_TEMPLATES,
    get_theme_colors,
    hex_to_rgb,
)
from backend.services.generation.models import (
    OutputFormat,
    GenerationStatus,
    SourceReference,
    Section,
    DocumentOutline,
    GenerationJob,
    GenerationConfig,
)
from backend.services.generation.utils import (
    strip_markdown,
    filter_llm_metatext,
    filter_title_echo,
    smart_truncate,
    sentence_truncate,
    llm_condense_text,
    smart_condense_content,
    sanitize_filename,
    check_spelling,
)
from backend.services.generation.citations import (
    CitationMapping,
    ContentWithCitations,
    generate_content_with_citations,
    format_citations_for_footnotes,
    format_citations_for_speaker_notes,
    strip_citation_markers,
    convert_citations_to_superscript,
    add_citations_to_section,
)
from backend.services.generation.styles import (
    StyleProfile,
    learn_style_from_documents,
    apply_style_to_prompt,
)
from backend.services.generation.service import (
    DocumentGenerationService,
    get_generation_service,
)

__all__ = [
    # Config
    "LANGUAGE_NAMES",
    "THEMES",
    "FONT_FAMILIES",
    "LAYOUT_TEMPLATES",
    "get_theme_colors",
    "hex_to_rgb",
    # Models
    "OutputFormat",
    "GenerationStatus",
    "SourceReference",
    "Section",
    "DocumentOutline",
    "GenerationJob",
    "GenerationConfig",
    # Utils
    "strip_markdown",
    "filter_llm_metatext",
    "filter_title_echo",
    "smart_truncate",
    "sentence_truncate",
    "llm_condense_text",
    "smart_condense_content",
    "sanitize_filename",
    "check_spelling",
    # Citations
    "CitationMapping",
    "ContentWithCitations",
    "generate_content_with_citations",
    "format_citations_for_footnotes",
    "format_citations_for_speaker_notes",
    "strip_citation_markers",
    "convert_citations_to_superscript",
    "add_citations_to_section",
    # Styles
    "StyleProfile",
    "learn_style_from_documents",
    "apply_style_to_prompt",
    # Service
    "DocumentGenerationService",
    "get_generation_service",
]
