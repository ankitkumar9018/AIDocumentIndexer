"""
Document Generator Package

This package provides template-aware document generation with support for
multiple output formats (PPTX, DOCX, PDF, XLSX, HTML, Markdown).

Key Features:
1. Template Analysis: Extract theme, layouts, and constraints from templates
2. LLM Context: Provide template info to LLM before content generation
3. Content Review: View and edit generated content BEFORE final document rendering
4. Structured Output: Pydantic models ensure content fits template constraints

Workflow:
1. Analyze template -> Get theme, layouts, constraints
2. Build prompt with template context
3. Generate content with LLM
4. **REVIEW STAGE** -> User views/edits each slide/page/sheet
5. Render final document only after approval
"""

# =============================================================================
# MODULAR MODELS - New self-contained implementation
# =============================================================================

# Models (the new modular implementation)
from .models import (
    OutputFormat,
    GenerationStatus,
    SourceReference,
    SourceUsageType,
    Section,
    DocumentOutline,
    GenerationJob,
    GenerationConfig,
    CitationMapping,
    ContentWithCitations,
    StyleProfile,
    LearnedLayout,
)

# Configuration
from .config import (
    LANGUAGE_NAMES,
    THEMES,
    FONT_FAMILIES,
    LAYOUT_TEMPLATES,
    DEFAULT_THEME,
    DEFAULT_FONT_FAMILY,
    DEFAULT_LAYOUT,
)

# Utilities
from .utils import (
    strip_markdown,
    filter_llm_metatext,
    filter_title_echo,
    validate_language_purity,
    is_sentence_complete,
    filter_incomplete_sentences,
    smart_truncate,
    sentence_truncate,
    llm_condense_text,
    smart_condense_content,
    check_spelling,
    hex_to_rgb,
    rgb_to_hex,
    get_contrasting_color,
    lighten_color,
    darken_color,
    sanitize_filename,
    get_theme_colors,
)

# Style learning and application
from .styles import (
    learn_style_from_documents,
    apply_style_to_prompt,
)

# Document verification and repair
from .verification import (
    DocumentVerifier,
    get_document_verifier,
)

# =============================================================================
# TEMPLATE-AWARE GENERATION SYSTEM
# =============================================================================

from .content_models import (
    # Status and actions
    ContentStatus,
    EditAction,
    # PPTX models
    BulletPoint,
    SlideContent,
    PresentationContent,
    # DOCX models
    ParagraphContent,
    DocumentSection,
    DocumentContent,
    # XLSX models
    CellContent,
    RowContent,
    SheetContent,
    SpreadsheetContent,
    # Constraints
    ContentConstraints,
    # Review session
    ContentEditRequest,
    ContentReviewSession,
)

from .content_reviewer import ContentReviewService
from .template_analyzer import TemplateAnalyzer, TemplateAnalysis
from .template_service import TemplateService, TemplateMetadata, get_template_service
from .prompt_builder import PromptBuilder

# Theme system
from .theme import (
    ThemeProfile,
    PPTXTheme,
    DOCXTheme,
    XLSXTheme,
    PDFTheme,
    LayoutInfo,
    PlaceholderSpec,
    ThemeExtractor,
    ThemeExtractorFactory,
    ThemeManager,
    get_theme_manager,
    ThemeApplier,
    create_applier_from_dict,
    create_applier_from_profile,
)

# Citations
from .citations import (
    generate_content_with_citations,
    format_citations_for_footnotes,
    format_citations_for_speaker_notes,
    strip_citation_markers,
    convert_citations_to_superscript,
    add_citations_to_section,
)

# Outline and content generators
from .outline import OutlineGenerator
from .content import ContentGenerator

# Format generators
from .formats import (
    BaseFormatGenerator,
    FormatGeneratorFactory,
    PPTXGenerator,
    DOCXGenerator,
    PDFGenerator,
    XLSXGenerator,
    MarkdownGenerator,
    HTMLGenerator,
    TXTGenerator,
)

# Modular service
from .service import ModularGenerationService, get_modular_generation_service


# =============================================================================
# BACKWARD COMPATIBILITY - DocumentGenerationService
# =============================================================================
# For backward compatibility, we provide a DocumentGenerationService alias
# that wraps the new modular service. The old generator.py functionality
# is now fully implemented in the modular format generators.

# =============================================================================
# DocumentGenerationService - Full implementation using modular components
# =============================================================================
# This class provides all job management methods using the new modular
# components (OutlineGenerator, ContentGenerator, FormatGenerators).

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import uuid
import structlog

_logger = structlog.get_logger(__name__)


class DocumentGenerationService:
    """Document generation service with full job management.

    This class provides the complete interface expected by the API routes,
    using the new modular components for all generation tasks.
    """

    def __init__(self):
        self.config = GenerationConfig()
        self._jobs: Dict[str, GenerationJob] = {}
        self._outline_generator = OutlineGenerator()
        self._content_generator = ContentGenerator(config=self.config)
        self._template_analyzer = TemplateAnalyzer()

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    async def reload_settings(self) -> None:
        """Reload generation settings from database."""
        from backend.services.settings import get_settings_service
        settings_service = get_settings_service()

        include_images = await settings_service.get_setting("generation.include_images")
        image_backend = await settings_service.get_setting("generation.image_backend")
        include_sources = await settings_service.get_setting("generation.include_sources")
        default_tone = await settings_service.get_setting("generation.default_tone")
        default_style = await settings_service.get_setting("generation.default_style")
        auto_charts = await settings_service.get_setting("generation.auto_charts")
        chart_style = await settings_service.get_setting("generation.chart_style")
        chart_dpi = await settings_service.get_setting("generation.chart_dpi")

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

        # Update content generator config
        self._content_generator.config = self.config

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
        include_images: Optional[bool] = None,
    ) -> GenerationJob:
        """Create a new document generation job."""
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

        job.metadata["include_images"] = (
            include_images if include_images is not None else self.config.include_images
        )
        job.metadata["image_backend"] = self.config.image_backend
        job.metadata["default_tone"] = self.config.default_tone
        job.metadata["default_style"] = self.config.default_style

        try:
            from backend.services.llm import LLMConfigManager
            llm_config = await LLMConfigManager.get_config_for_operation("generation")
            job.metadata["llm_model"] = f"{llm_config.provider_type}/{llm_config.model}"
        except Exception:
            job.metadata["llm_model"] = self.config.model

        self._jobs[job_id] = job

        _logger.info(
            "Document generation job created",
            job_id=job_id,
            title=title,
            output_format=output_format.value,
        )

        return job

    def _get_job(self, job_id: str) -> GenerationJob:
        """Get a job by ID, raising ValueError if not found."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        return job

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
        return job

    async def generate_outline(
        self,
        job_id: str,
        num_sections: Optional[int] = None,
    ) -> DocumentOutline:
        """Generate an outline for the job using the modular OutlineGenerator."""
        job = self._get_job(job_id)

        if job.status not in [GenerationStatus.DRAFT, GenerationStatus.OUTLINE_PENDING]:
            raise ValueError(f"Cannot generate outline in status: {job.status}")

        job.status = GenerationStatus.OUTLINE_PENDING
        job.updated_at = datetime.utcnow()

        # Search for relevant sources using RAG
        sources = await self._search_sources(job)
        job.sources_used = sources

        # Style learning from existing documents if enabled
        if job.metadata.get("use_existing_docs"):
            style_guide = await self._learn_style(job)
            job.metadata["style_guide"] = style_guide

        # Determine section count
        effective_num_sections = num_sections or job.metadata.get("page_count") or 5

        # Generate outline using the modular generator
        outline = await self._outline_generator.generate(
            job=job,
            num_sections=effective_num_sections,
        )

        job.outline = outline
        job.status = GenerationStatus.OUTLINE_APPROVED
        job.updated_at = datetime.utcnow()

        _logger.info(
            "Outline generated",
            job_id=job_id,
            sections=len(outline.sections),
        )

        return outline

    async def _search_sources(self, job: GenerationJob) -> List[SourceReference]:
        """Search for relevant sources using RAG."""
        try:
            from backend.services.rag import get_rag_service
            from .models import SourceReference, SourceUsageType
            rag = get_rag_service()

            collection_filter = job.metadata.get("collection_filter")
            folder_id = job.metadata.get("folder_id")
            include_subfolders = job.metadata.get("include_subfolders", True)
            enhance_query = job.metadata.get("enhance_query")

            # Build query from title and description
            query = f"{job.title} {job.description or ''}"

            _logger.info(
                "Searching sources for job",
                job_id=job.id,
                query=query[:100],
                collection_filter=collection_filter,
                folder_id=folder_id,
            )

            # Use rag.query() which returns RAGResponse with .sources attribute
            response = await rag.query(
                question=query,
                collection_filter=collection_filter,
                folder_id=folder_id,
                include_subfolders=include_subfolders,
                top_k=10,
                enhance_query=enhance_query,
            )

            # Extract sources from RAGResponse and convert to SourceReference
            source_references = []
            if response and hasattr(response, 'sources'):
                for source in response.sources:
                    source_ref = SourceReference(
                        document_id=source.document_id,
                        document_name=source.document_name,
                        chunk_id=source.chunk_id,
                        page_number=source.page_number or source.slide_number,
                        relevance_score=source.similarity_score or source.relevance_score,
                        snippet=source.snippet or source.full_content[:200] if source.full_content else "",
                        usage_type=SourceUsageType.CONTENT,
                        usage_description="Content reference from initial search",
                        document_path=source.metadata.get("file_path") if source.metadata else None,
                    )
                    source_references.append(source_ref)

            # If no sources found with filter, try without filter
            if not source_references and (collection_filter or folder_id):
                _logger.info(
                    "No sources found with filters, trying without filters",
                    job_id=job.id,
                    collection_filter=collection_filter,
                    folder_id=folder_id,
                )
                fallback_response = await rag.query(
                    question=query,
                    collection_filter=None,
                    folder_id=None,
                    top_k=10,
                    enhance_query=enhance_query,
                )
                if fallback_response and hasattr(fallback_response, 'sources'):
                    for source in fallback_response.sources:
                        source_ref = SourceReference(
                            document_id=source.document_id,
                            document_name=source.document_name,
                            chunk_id=source.chunk_id,
                            page_number=source.page_number or source.slide_number,
                            relevance_score=source.similarity_score or source.relevance_score,
                            snippet=source.snippet or source.full_content[:200] if source.full_content else "",
                            usage_type=SourceUsageType.CONTENT,
                            usage_description="Content reference from fallback search (all docs)",
                            document_path=source.metadata.get("file_path") if source.metadata else None,
                        )
                        source_references.append(source_ref)

            _logger.info(
                "Sources found for job",
                job_id=job.id,
                sources_count=len(source_references),
                source_names=[s.document_name for s in source_references[:5]] if source_references else [],
            )

            return source_references
        except Exception as e:
            _logger.warning(f"Could not search for sources: {e}", exc_info=True)
            return []

    async def _create_llm_generate_func(self):
        """Create an LLM generate wrapper function for quality reviewers.

        Returns an async function that takes a prompt string and returns the LLM response text.
        This is used by SlideReviewer, DOCXSectionReviewer, XLSXSheetReviewer etc.
        """
        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,
            )

            async def llm_generate(prompt: str) -> str:
                """Generate text from the LLM given a prompt."""
                try:
                    response = await llm.ainvoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    _logger.error(f"LLM generation failed in reviewer: {e}")
                    return ""

            return llm_generate

        except Exception as e:
            _logger.warning(f"Could not create LLM generate function: {e}")
            return None

    async def _learn_style(self, job: GenerationJob) -> Optional[Dict[str, Any]]:
        """Learn style from existing documents and track style sources.

        Uses LLM to intelligently select which documents are good style references
        rather than just picking a fixed number.
        """
        try:
            # Get documents for style learning
            from backend.services.rag import get_rag_service
            rag = get_rag_service()

            collection_filter = job.metadata.get("style_collection_filters") or job.metadata.get("collection_filter")

            # Search for more style reference candidates for LLM to filter
            user_style_limit = job.metadata.get("max_style_sources")
            initial_limit = user_style_limit if user_style_limit else 10

            style_candidates = await rag.search(
                query=f"style examples for {job.title}",
                limit=initial_limit,
                collection_filter=collection_filter,
            )

            if not style_candidates:
                _logger.info("No style reference documents found", job_id=job.id)
                return None

            # Use LLM to filter style sources unless user set a specific limit
            use_smart_filter = job.metadata.get("smart_source_filter", True) if job.metadata else True
            if use_smart_filter and not user_style_limit and len(style_candidates) > 3:
                style_results = await self._filter_style_sources_with_llm(
                    style_candidates, job.title, job.output_format.value
                )
            else:
                style_results = style_candidates[:user_style_limit or 5]

            if not style_results:
                _logger.info("No suitable style reference documents after filtering", job_id=job.id)
                return None

            # Extract document contents for style analysis
            document_contents = [r.get("content", "") for r in style_results if r.get("content")]
            document_names = [r.get("document_name", "Unknown") for r in style_results]

            style_guide = await learn_style_from_documents(
                document_contents=document_contents,
                document_names=document_names,
                use_llm_analysis=True,
            )

            # Track style sources with STYLE usage type
            style_sources = []
            for result in style_results:
                metadata = result.get("metadata", {})
                page_num = metadata.get("page_number") or metadata.get("slide_number")
                doc_path = metadata.get("source") or metadata.get("file_path")
                doc_url = metadata.get("url") or metadata.get("document_url")

                # Use LLM-assigned usage description if available
                usage_desc = result.get("usage_description") or "Writing style and tone reference"

                style_source = SourceReference(
                    document_id=metadata.get("document_id", result.get("chunk_id", "")),
                    document_name=result.get("document_name", "Unknown"),
                    chunk_id=result.get("chunk_id"),
                    page_number=page_num,
                    relevance_score=result.get("score", 0.0),
                    snippet=result.get("content", "")[:200],
                    usage_type=SourceUsageType.STYLE,
                    usage_description=usage_desc,
                    document_path=doc_path,
                    document_url=doc_url,
                )
                style_sources.append(style_source)

            # Add style sources to job.sources_used
            if not job.sources_used:
                job.sources_used = []

            existing_ids = {s.document_id for s in job.sources_used}
            for source in style_sources:
                if source.document_id not in existing_ids:
                    job.sources_used.append(source)
                    existing_ids.add(source.document_id)

            _logger.info(
                "Style learned from existing documents",
                job_id=job.id,
                style_sources_count=len(style_sources),
                source_names=[s.document_name for s in style_sources],
            )
            return style_guide
        except Exception as e:
            _logger.warning(f"Could not learn style: {e}")
            return None

    async def _filter_style_sources_with_llm(
        self,
        candidates: List[dict],
        doc_title: str,
        output_format: str,
    ) -> List[dict]:
        """Use LLM to filter and select documents that are good style references.

        Looks for documents with similar tone, structure, and formatting that
        can serve as style guides for the new document.
        """
        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,
            )

            # Build candidate list for LLM
            candidate_summaries = []
            for i, result in enumerate(candidates):
                doc_name = result.get("document_name", "Unknown")
                snippet = result.get("content", "")[:400]
                candidate_summaries.append(
                    f"[{i}] {doc_name}\n   Sample: {snippet}..."
                )

            candidates_text = "\n\n".join(candidate_summaries)

            prompt = f"""You are selecting documents to use as STYLE REFERENCES for creating a new {output_format.upper()} document.

NEW DOCUMENT TITLE: {doc_title}

CANDIDATE DOCUMENTS:
{candidates_text}

TASK: Select documents that would be GOOD STYLE REFERENCES for the new document.
Consider:
- Writing tone and voice (formal, casual, technical, etc.)
- Document structure and formatting patterns
- Vocabulary level and terminology usage
- How content is organized and presented

Do NOT select documents just based on topic similarity - focus on STYLE compatibility.
A document about a different topic can still be a good style reference if its writing style matches what we need.

For each selected document, explain what style elements it provides.

Return ONLY valid JSON (no markdown code blocks):
{{
    "selected_indices": [0, 3],
    "reasoning": {{
        "0": "Professional tone with clear section structure",
        "3": "Uses technical terminology effectively with good explanations"
    }},
    "overall_style": "Brief description of the combined style from selected documents"
}}

If no documents are good style references: {{"selected_indices": [], "reasoning": {{}}, "overall_style": "No suitable style references found"}}
"""

            response = await llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse LLM response
            import json
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                selected_indices = result.get("selected_indices", [])
                reasoning = result.get("reasoning", {})

                # Filter candidates based on LLM selection
                filtered = []
                for idx in selected_indices:
                    if 0 <= idx < len(candidates):
                        candidate = candidates[idx].copy()
                        # Add LLM's style description
                        if str(idx) in reasoning:
                            candidate["usage_description"] = f"Style: {reasoning[str(idx)]}"
                        filtered.append(candidate)

                _logger.info(
                    "LLM style source filtering complete",
                    candidates_count=len(candidates),
                    selected_count=len(filtered),
                    selected_indices=selected_indices,
                )
                return filtered

            # Fallback to top 3 if parsing fails
            _logger.warning("Could not parse LLM style filter response, using top 3 candidates")
            return candidates[:3]

        except Exception as e:
            _logger.warning(f"LLM style source filtering failed: {e}, using top 3 candidates")
            return candidates[:3]

    async def update_outline(
        self,
        job_id: str,
        sections: List[Dict[str, Any]],
    ) -> DocumentOutline:
        """Update the outline for a job."""
        job = self._get_job(job_id)

        if job.status != GenerationStatus.OUTLINE_APPROVED:
            raise ValueError(f"Cannot update outline in status: {job.status}")

        # Update the outline
        job.outline = DocumentOutline(
            title=job.title,
            description=job.description,
            sections=sections,
        )
        job.updated_at = datetime.utcnow()

        _logger.info(
            "Outline updated",
            job_id=job_id,
            sections=len(sections),
        )

        return job.outline

    async def approve_outline(
        self,
        job_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> GenerationJob:
        """Approve the outline and optionally apply modifications.

        This transitions the job to a state ready for content generation.

        Args:
            job_id: The job ID
            modifications: Optional dict with 'sections' key containing updated section list
        """
        job = self._get_job(job_id)

        if job.status not in [GenerationStatus.OUTLINE_APPROVED, GenerationStatus.DRAFT]:
            raise ValueError(f"Cannot approve outline in status: {job.status}")

        if not job.outline:
            raise ValueError("No outline to approve - generate outline first")

        # Apply modifications if provided
        if modifications is not None:
            sections = modifications.get("sections")
            if sections is not None:
                job.outline = DocumentOutline(
                    title=job.outline.title,
                    description=job.outline.description,
                    sections=sections,
                )

        # Mark as approved and ready for content generation
        job.status = GenerationStatus.SECTIONS_PLANNING
        job.updated_at = datetime.utcnow()

        _logger.info(
            "Outline approved",
            job_id=job_id,
            sections=len(job.outline.sections),
        )

        return job

    async def approve_section_plans(
        self,
        job_id: str,
        section_approvals: Optional[List[Dict[str, Any]]] = None,
    ) -> GenerationJob:
        """Approve section plans and proceed to content generation.

        This allows users to review section plans, skip sections, or modify
        titles/descriptions before generating content.

        Args:
            job_id: The job ID
            section_approvals: Optional list of section approval dicts with keys:
                - section_id: ID of the section
                - approved: Whether to generate this section
                - title: Optional updated title
                - description: Optional updated description
        """
        job = self._get_job(job_id)

        if job.status != GenerationStatus.SECTIONS_PLANNING:
            raise ValueError(f"Cannot approve section plans in status: {job.status}")

        if not job.outline:
            raise ValueError("No outline available")

        # Apply section approvals if provided
        if section_approvals:
            updated_sections = []
            for section_data in job.outline.sections:
                section_id = section_data.get("id") if isinstance(section_data, dict) else getattr(section_data, "id", None)

                # Find matching approval
                approval = next(
                    (a for a in section_approvals if a.get("section_id") == section_id),
                    None
                )

                if approval:
                    # Skip sections that are not approved
                    if not approval.get("approved", True):
                        continue

                    # Update title/description if provided
                    if isinstance(section_data, dict):
                        if approval.get("title"):
                            section_data["title"] = approval["title"]
                        if approval.get("description"):
                            section_data["description"] = approval["description"]

                updated_sections.append(section_data)

            # Update outline with filtered/modified sections
            job.outline = DocumentOutline(
                title=job.outline.title,
                description=job.outline.description,
                sections=updated_sections,
            )

        # Transition to approved status ready for content generation
        job.status = GenerationStatus.SECTIONS_APPROVED
        job.updated_at = datetime.utcnow()

        _logger.info(
            "Section plans approved",
            job_id=job_id,
            sections=len(job.outline.sections),
        )

        return job

    async def generate_content(self, job_id: str) -> GenerationJob:
        """Generate content for all sections using the modular ContentGenerator."""
        job = self._get_job(job_id)

        if job.status not in [GenerationStatus.OUTLINE_APPROVED, GenerationStatus.SECTIONS_PLANNING, GenerationStatus.SECTIONS_APPROVED, GenerationStatus.SECTION_REVIEW]:
            raise ValueError(f"Cannot generate content in status: {job.status}")

        if not job.outline:
            raise ValueError("No outline available - generate outline first")

        job.status = GenerationStatus.GENERATING
        job.updated_at = datetime.utcnow()

        # Analyze template if provided (for any format with template support)
        template_analysis = None
        template_path = job.metadata.get("template_path") if job.metadata else None

        # Supported formats for template analysis
        template_supported_formats = [
            OutputFormat.PPTX,
            OutputFormat.DOCX,
            OutputFormat.PDF,
            OutputFormat.XLSX,
        ]

        if template_path and job.output_format in template_supported_formats:
            try:
                from .template_analyzer import TemplateAnalyzer
                analyzer = TemplateAnalyzer()
                template_analysis = analyzer.analyze(template_path)
                _logger.info(
                    "Template analyzed for content generation",
                    template_path=template_path,
                    format=job.output_format.value,
                    constraints=template_analysis.constraints.model_dump() if template_analysis.constraints else None,
                )
            except Exception as e:
                _logger.warning(f"Could not analyze template for {job.output_format.value}: {e}")

        # Generate content for all sections (with template awareness)
        sections = await self._content_generator.generate_all(
            job=job,
            template_analysis=template_analysis,
        )

        job.sections = sections

        # Aggregate all sources from sections for the sources slide
        all_sources = []
        seen_doc_ids = set()
        for section in sections:
            if section.sources:
                for source in section.sources:
                    if source.document_id not in seen_doc_ids:
                        all_sources.append(source)
                        seen_doc_ids.add(source.document_id)

        _logger.info(
            "Aggregated sources from sections",
            job_id=job_id,
            section_sources_count=len(all_sources),
            existing_job_sources_count=len(job.sources_used) if job.sources_used else 0,
        )

        # Always merge section sources into job.sources_used
        if not job.sources_used:
            job.sources_used = []

        existing_ids = {s.document_id for s in job.sources_used}
        for source in all_sources:
            if source.document_id not in existing_ids:
                job.sources_used.append(source)
                existing_ids.add(source.document_id)

        _logger.info(
            "Final sources for job",
            job_id=job_id,
            total_sources=len(job.sources_used),
        )

        job.status = GenerationStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()

        # Generate the output file
        await self._generate_output_file(job)

        _logger.info(
            "Content generated for all sections",
            job_id=job_id,
            sections=len(sections),
        )

        return job

    async def generate_single_section(
        self,
        job_id: str,
        section_id: str,
    ) -> Section:
        """Generate content for a single section."""
        job = self._get_job(job_id)

        if not job.outline:
            raise ValueError("No outline available")

        # Find the section in outline
        section_info = None
        order = 0
        for i, s in enumerate(job.outline.sections):
            if isinstance(s, dict):
                if s.get("id") == section_id:
                    section_info = s
                    order = i
                    break
            else:
                if getattr(s, "id", None) == section_id:
                    section_info = s
                    order = i
                    break

        if not section_info:
            raise ValueError(f"Section not found: {section_id}")

        title = section_info.get("title") if isinstance(section_info, dict) else section_info.title
        description = section_info.get("description", "") if isinstance(section_info, dict) else getattr(section_info, "description", "")

        section = await self._content_generator.generate_section(
            job=job,
            section_title=title,
            section_description=description,
            order=order,
            existing_section_id=section_id,
        )

        # Update or add to job sections
        existing_idx = next((i for i, s in enumerate(job.sections) if s.id == section_id), None)
        if existing_idx is not None:
            job.sections[existing_idx] = section
        else:
            job.sections.append(section)

        job.updated_at = datetime.utcnow()

        return section

    async def approve_section(
        self,
        job_id: str,
        section_id: str,
        feedback: Optional[str] = None,
        approved: bool = True,
    ) -> Section:
        """Approve or request revision for a section."""
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

    async def revise_section(self, job_id: str, section_id: str) -> Section:
        """Revise a section based on feedback."""
        job = self._get_job(job_id)

        if job.status != GenerationStatus.REVISION:
            raise ValueError(f"Cannot revise section in status: {job.status}")

        section = next((s for s in job.sections if s.id == section_id), None)
        if not section:
            raise ValueError(f"Section not found: {section_id}")

        if not section.feedback:
            raise ValueError("No feedback provided for revision")

        # Regenerate with feedback using modular content generator
        revised_section = await self._content_generator.regenerate_with_feedback(
            section=section,
            job=job,
            feedback=section.feedback,
        )

        # Update section
        section.revised_content = revised_section.content
        section.sources = revised_section.sources
        section.approved = False
        section.feedback = None

        job.status = GenerationStatus.SECTION_REVIEW
        job.updated_at = datetime.utcnow()

        return section

    async def _generate_output_file(self, job: GenerationJob) -> str:
        """Generate the output file using format generators."""
        generator = FormatGeneratorFactory.get(job.output_format)
        if not generator:
            raise ValueError(f"Unsupported output format: {job.output_format}")

        # Inject LLM generate function for quality review if enabled
        enable_quality_review = job.metadata.get("enable_quality_review", self.config.enable_quality_review)
        enable_slide_review = job.metadata.get("enable_slide_review", False)

        if enable_quality_review or enable_slide_review:
            # Create LLM generate wrapper function for reviewers
            llm_generate_func = await self._create_llm_generate_func()
            if llm_generate_func:
                job.metadata["llm_generate_func"] = llm_generate_func
                _logger.info("LLM generate function injected for quality review", job_id=job.id)

        # Generate filename
        import re
        safe_title = re.sub(r'[^\w\s-]', '', job.title)
        safe_title = re.sub(r'\s+', '_', safe_title)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = generator.file_extension
        filename = f"{safe_title}_{timestamp}{ext}"

        # Generate the document
        output_path = await generator.generate(job, filename)

        job.output_path = output_path

        _logger.info(
            "Output file generated",
            job_id=job.id,
            output_path=output_path,
        )

        return output_path

    async def get_output_file(self, job_id: str) -> Tuple[bytes, str, str]:
        """Get the generated output file."""
        import re

        job = self._get_job(job_id)

        if job.status != GenerationStatus.COMPLETED:
            raise ValueError("Job is not completed")

        if not job.output_path:
            raise ValueError("No output file available")

        # Read file
        with open(job.output_path, "rb") as f:
            file_bytes = f.read()

        filename = Path(job.output_path).name

        # Safety check: Ensure filename has proper extension (not .null or missing)
        if not filename or '.null' in filename or not Path(filename).suffix:
            # Fallback to format-based extension
            ext = self._get_extension_for_format(job.output_format)
            safe_title = re.sub(r'[^\w\s-]', '', job.title or 'document')
            safe_title = re.sub(r'\s+', '_', safe_title)[:50]
            filename = f"{safe_title}{ext}"

        content_type = self._get_content_type(job.output_format)

        return file_bytes, filename, content_type

    def _get_extension_for_format(self, output_format: OutputFormat) -> str:
        """Get file extension for output format."""
        extensions = {
            OutputFormat.PPTX: ".pptx",
            OutputFormat.DOCX: ".docx",
            OutputFormat.PDF: ".pdf",
            OutputFormat.XLSX: ".xlsx",
            OutputFormat.MARKDOWN: ".md",
            OutputFormat.HTML: ".html",
            OutputFormat.TXT: ".txt",
        }
        return extensions.get(output_format, ".bin")

    def _get_content_type(self, output_format: OutputFormat) -> str:
        """Get content type for output format."""
        content_types = {
            OutputFormat.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            OutputFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            OutputFormat.PDF: "application/pdf",
            OutputFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            OutputFormat.MARKDOWN: "text/markdown",
            OutputFormat.HTML: "text/html",
            OutputFormat.TXT: "text/plain",
        }
        return content_types.get(output_format, "application/octet-stream")

    async def generate_document(self, job: GenerationJob) -> str:
        """Generate a document from the job specification."""
        # Ensure job is in memory
        if job.id not in self._jobs:
            self._jobs[job.id] = job

        # Generate outline if not present
        if not job.outline:
            await self.generate_outline(job.id)

        # Generate content if not present
        if not job.sections:
            await self.generate_content(job.id)

        # Auto-approve all sections for direct generation
        for section in job.sections:
            section.approved = True

        job.status = GenerationStatus.COMPLETED
        job.completed_at = datetime.utcnow()

        # Generate output file
        output_path = await self._generate_output_file(job)

        return output_path

    async def suggest_theme(
        self,
        title: str,
        description: str,
        document_type: str = "pptx",
    ) -> Dict[str, Any]:
        """Use LLM to suggest optimal theming for a document."""
        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,
            )

            # Build theme options for the prompt
            theme_options = "\n".join([
                f"- {key}: {theme['name']} - {theme['description']}"
                for key, theme in THEMES.items()
            ])

            font_options = "\n".join([
                f"- {key}: {font['name']} - {font['description']}"
                for key, font in FONT_FAMILIES.items()
            ])

            layout_options = "\n".join([
                f"- {key}: {layout['name']} - {layout['description']}"
                for key, layout in LAYOUT_TEMPLATES.items()
            ])

            prompt = f"""Based on the document details below, recommend the best theme, font family, and layout.

Title: {title}
Description: {description}
Document Type: {document_type}

Available Themes:
{theme_options}

Available Font Families:
{font_options}

Available Layouts:
{layout_options}

Respond with JSON only:
{{"theme": "<key>", "font_family": "<key>", "layout": "<key>", "reason": "<brief explanation>"}}"""

            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            import json
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)

        except Exception as e:
            _logger.warning(f"Theme suggestion failed: {e}")
            return {
                "theme": "professional",
                "font_family": "sans_serif",
                "layout": "standard",
                "reason": "Default professional styling",
            }

    async def verify_document(self, job_id: str) -> Dict[str, Any]:
        """Verify the generated document for issues."""
        job = self._get_job(job_id)

        if not job.output_path:
            raise ValueError("No output file to verify")

        verifier = get_document_verifier()
        return await verifier.verify(job.output_path, job.output_format)

    async def handle_verification_action(
        self,
        job_id: str,
        action: str,
    ) -> Dict[str, Any]:
        """Handle user action on verification issues."""
        job = self._get_job(job_id)

        if action == "proceed":
            # User accepts document as-is
            job.metadata["verification_action"] = "proceed"
            return {"success": True, "action": "proceed", "message": "Document accepted"}

        elif action == "auto_repair":
            # Attempt auto-repair
            verifier = get_document_verifier()
            repair_result = await verifier.repair(
                job.output_path,
                job.output_format,
            )

            return {
                "success": repair_result.get("success", False),
                "action": "auto_repair",
                "repairs_made": repair_result.get("repairs_made", []),
                "message": repair_result.get("message", "Repair completed"),
            }

        elif action == "regenerate":
            # User requests full regeneration
            job.metadata["verification_action"] = "regenerate"
            job.metadata["regeneration_requested"] = True
            return {
                "success": True,
                "action": "regenerate",
                "message": "Regeneration requested. Call generate_document() again.",
            }

        else:
            return {
                "success": False,
                "message": f"Unknown action: {action}. Valid: proceed, auto_repair, regenerate",
            }


# Singleton instance
_generation_service: Optional[DocumentGenerationService] = None


def get_generation_service() -> DocumentGenerationService:
    """Get the singleton DocumentGenerationService instance."""
    global _generation_service
    if _generation_service is None:
        _generation_service = DocumentGenerationService()
    return _generation_service


__all__ = [
    # Core models
    "OutputFormat",
    "GenerationStatus",
    "SourceReference",
    "SourceUsageType",
    "Section",
    "DocumentOutline",
    "GenerationJob",
    "GenerationConfig",
    "CitationMapping",
    "ContentWithCitations",
    "StyleProfile",
    "LearnedLayout",
    # Configuration
    "LANGUAGE_NAMES",
    "THEMES",
    "FONT_FAMILIES",
    "LAYOUT_TEMPLATES",
    "DEFAULT_THEME",
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_LAYOUT",
    # Utilities
    "strip_markdown",
    "filter_llm_metatext",
    "filter_title_echo",
    "validate_language_purity",
    "is_sentence_complete",
    "filter_incomplete_sentences",
    "smart_truncate",
    "sentence_truncate",
    "llm_condense_text",
    "smart_condense_content",
    "check_spelling",
    "hex_to_rgb",
    "rgb_to_hex",
    "get_contrasting_color",
    "lighten_color",
    "darken_color",
    "sanitize_filename",
    "get_theme_colors",
    # Style learning
    "learn_style_from_documents",
    "apply_style_to_prompt",
    # Document verification
    "DocumentVerifier",
    "get_document_verifier",
    # Status and actions
    "ContentStatus",
    "EditAction",
    # Content models - PPTX
    "BulletPoint",
    "SlideContent",
    "PresentationContent",
    # Content models - DOCX
    "ParagraphContent",
    "DocumentSection",
    "DocumentContent",
    # Content models - XLSX
    "CellContent",
    "RowContent",
    "SheetContent",
    "SpreadsheetContent",
    # Constraints
    "ContentConstraints",
    # Review system
    "ContentEditRequest",
    "ContentReviewSession",
    "ContentReviewService",
    # Template analysis and service
    "TemplateAnalyzer",
    "TemplateAnalysis",
    "TemplateService",
    "TemplateMetadata",
    "get_template_service",
    "PromptBuilder",
    # Theme system
    "ThemeProfile",
    "PPTXTheme",
    "DOCXTheme",
    "XLSXTheme",
    "PDFTheme",
    "LayoutInfo",
    "PlaceholderSpec",
    "ThemeExtractor",
    "ThemeExtractorFactory",
    "ThemeManager",
    "get_theme_manager",
    "ThemeApplier",
    "create_applier_from_dict",
    "create_applier_from_profile",
    # Citation utilities
    "generate_content_with_citations",
    "format_citations_for_footnotes",
    "format_citations_for_speaker_notes",
    "strip_citation_markers",
    "convert_citations_to_superscript",
    "add_citations_to_section",
    # Outline and content generators
    "OutlineGenerator",
    "ContentGenerator",
    # Format generators
    "BaseFormatGenerator",
    "FormatGeneratorFactory",
    "PPTXGenerator",
    "DOCXGenerator",
    "PDFGenerator",
    "XLSXGenerator",
    "MarkdownGenerator",
    "HTMLGenerator",
    "TXTGenerator",
    # Services
    "DocumentGenerationService",
    "get_generation_service",
    "ModularGenerationService",
    "get_modular_generation_service",
]
