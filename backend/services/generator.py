"""
AIDocumentIndexer - Document Generation Service
================================================

Human-in-the-loop document generation using LangGraph.
Supports PPTX, DOCX, PDF, and other output formats.

LLM provider and model are configured via Admin UI (Operation-Level Config).
Configure the "content_generation" operation in Admin > Settings > LLM Configuration.
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Types
# =============================================================================

class OutputFormat(str, Enum):
    """Supported output formats."""
    PPTX = "pptx"
    DOCX = "docx"
    PDF = "pdf"
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

    # Output settings
    output_dir: str = "/tmp/generated_docs"
    include_sources: bool = True
    include_toc: bool = True

    # Image generation settings (DISABLED by default)
    include_images: bool = False  # Disabled by default
    image_backend: str = "ollama"  # "ollama" or "unsplash"

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

    async def create_job(
        self,
        user_id: str,
        title: str,
        description: str,
        output_format: OutputFormat = OutputFormat.DOCX,
        collection_filter: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GenerationJob:
        """
        Create a new document generation job.

        Args:
            user_id: ID of the user creating the job
            title: Title for the document
            description: Description of what to generate
            output_format: Desired output format
            collection_filter: Optional collection to search
            metadata: Additional metadata

        Returns:
            New GenerationJob instance
        """
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

        self._jobs[job_id] = job

        logger.info(
            "Document generation job created",
            job_id=job_id,
            user_id=user_id,
            title=title,
        )

        return job

    async def generate_outline(
        self,
        job_id: str,
        num_sections: int = 5,
    ) -> DocumentOutline:
        """
        Generate an outline for the document.

        Args:
            job_id: Job ID
            num_sections: Number of sections to generate

        Returns:
            Generated outline
        """
        job = self._get_job(job_id)

        if job.status not in [GenerationStatus.DRAFT, GenerationStatus.OUTLINE_PENDING]:
            raise ValueError(f"Cannot generate outline in status: {job.status}")

        job.status = GenerationStatus.OUTLINE_PENDING
        job.updated_at = datetime.utcnow()

        logger.info("Generating outline", job_id=job_id, num_sections=num_sections)

        # Use RAG to find relevant sources
        sources = []
        if self.config.use_rag:
            sources = await self._search_sources(
                query=f"{job.title}: {job.description}",
                collection_filter=job.metadata.get("collection_filter"),
                max_results=self.config.max_sources,
            )

        # Generate outline using LLM
        outline = await self._generate_outline_with_llm(
            title=job.title,
            description=job.description,
            sources=sources,
            num_sections=num_sections,
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

    async def _generate_outline_with_llm(
        self,
        title: str,
        description: str,
        sources: List[SourceReference],
        num_sections: int,
    ) -> DocumentOutline:
        """Generate outline using LLM."""
        # Build context from sources
        context = ""
        if sources:
            context = "Relevant information from the knowledge base:\n\n"
            for source in sources[:5]:
                context += f"- {source.snippet}...\n\n"

        # Create prompt for outline generation
        prompt = f"""Create a professional document outline for the following:

Title: {title}
Description: {description}

{context}

Generate exactly {num_sections} sections with specific, descriptive titles.

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

            while len(sections) < num_sections:
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
                sections=sections[:num_sections],
            )

        except Exception as e:
            logger.error("Failed to generate outline with LLM", error=str(e))
            # Return a more descriptive fallback outline
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
                    for i in range(num_sections)
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

        # Generate content
        prompt = f"""Write content for the following section of a {job.output_format.value} document:

Document Title: {job.title}
Section: {section_title}
Description: {section_description}

{context}

Write clear, well-structured content that fits the document's purpose.
Include relevant details and maintain a professional tone."""

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
        filename = f"{job.id}_{job.title.replace(' ', '_')[:50]}"

        if job.output_format == OutputFormat.PPTX:
            output_path = await self._generate_pptx(job, filename)
        elif job.output_format == OutputFormat.DOCX:
            output_path = await self._generate_docx(job, filename)
        elif job.output_format == OutputFormat.PDF:
            output_path = await self._generate_pdf(job, filename)
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
            from pptx.util import Inches, Pt, Emu
            from pptx.dml.color import RgbColor
            from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
            from pptx.enum.shapes import MSO_SHAPE

            prs = Presentation()
            prs.slide_width = Inches(13.333)  # 16:9 aspect ratio
            prs.slide_height = Inches(7.5)

            # Define professional color scheme
            PRIMARY_COLOR = RgbColor(0x1E, 0x3A, 0x5F)  # Deep blue
            SECONDARY_COLOR = RgbColor(0x3D, 0x5A, 0x80)  # Medium blue
            ACCENT_COLOR = RgbColor(0xE0, 0xE1, 0xDD)  # Light gray
            TEXT_COLOR = RgbColor(0x2D, 0x3A, 0x45)  # Dark gray
            WHITE = RgbColor(0xFF, 0xFF, 0xFF)

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
                tf.paragraphs[0].font.color.rgb = RgbColor(0x88, 0x88, 0x88)
                tf.paragraphs[0].alignment = PP_ALIGN.RIGHT

            total_slides = len(job.sections) + 2  # Title + sections + sources
            if self.config.include_toc:
                total_slides += 1

            current_slide = 0

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
            p.text = job.title
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
            p.text = job.outline.description if job.outline else job.description
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

                for idx, section in enumerate(job.sections):
                    if idx > 0:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]
                    p.text = f"{idx + 1}.  {section.title}"
                    p.font.name = "Calibri"
                    p.font.size = Pt(20)
                    p.font.color.rgb = TEXT_COLOR
                    p.space_after = Pt(12)

                add_footer(slide, current_slide, total_slides)

            # ========== CONTENT SLIDES ==========
            for section in job.sections:
                current_slide += 1
                slide = prs.slides.add_slide(prs.slide_layouts[6])
                add_header_bar(slide)

                # Section title
                section_title = slide.shapes.add_textbox(
                    Inches(0.8), Inches(0.3),
                    Inches(11), Inches(0.8)
                )
                tf = section_title.text_frame
                p = tf.paragraphs[0]
                p.text = section.title
                p.font.name = "Calibri"
                p.font.size = Pt(32)
                p.font.bold = True
                p.font.color.rgb = WHITE

                # Content
                content = section.revised_content or section.content
                content_box = slide.shapes.add_textbox(
                    Inches(0.8), Inches(1.6),
                    Inches(11.5), Inches(5.2)
                )
                tf = content_box.text_frame
                tf.word_wrap = True

                # Split content into bullet points or paragraphs
                paragraphs = content.split('\n')
                for idx, para_text in enumerate(paragraphs[:12]):  # Limit to 12 items
                    para_text = para_text.strip()
                    if not para_text:
                        continue

                    if idx > 0:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]

                    # Add bullet if content looks like a list item
                    if para_text.startswith(('- ', '• ', '* ', '1.', '2.', '3.')):
                        p.text = para_text.lstrip('-•* 0123456789.')
                        p.level = 0
                    else:
                        p.text = para_text

                    p.font.name = "Calibri"
                    p.font.size = Pt(16)
                    p.font.color.rgb = TEXT_COLOR
                    p.space_after = Pt(8)

                add_footer(slide, current_slide, total_slides)

            # ========== SOURCES SLIDE ==========
            if self.config.include_sources and job.sources_used:
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

                for idx, source in enumerate(job.sources_used[:10]):
                    if idx > 0:
                        p = tf.add_paragraph()
                    else:
                        p = tf.paragraphs[0]
                    doc_name = source.document_name or source.document_id[:20]
                    p.text = f"•  {doc_name}"
                    p.font.name = "Calibri"
                    p.font.size = Pt(14)
                    p.font.color.rgb = TEXT_COLOR
                    p.space_after = Pt(6)

                add_footer(slide, current_slide, total_slides)

            output_path = os.path.join(self.config.output_dir, f"{filename}.pptx")
            prs.save(output_path)

            logger.info("Professional PPTX generated", path=output_path)
            return output_path

        except ImportError:
            logger.error("python-pptx not installed")
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

            # Define professional colors
            PRIMARY_COLOR = RGBColor(0x1E, 0x3A, 0x5F)  # Deep blue
            SECONDARY_COLOR = RGBColor(0x3D, 0x5A, 0x80)  # Medium blue
            TEXT_COLOR = RGBColor(0x2D, 0x3A, 0x45)  # Dark gray
            LIGHT_GRAY = RGBColor(0x88, 0x88, 0x88)

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
                    # Regular paragraph
                    else:
                        para = doc.add_paragraph()
                        para.paragraph_format.line_spacing = 1.5
                        para.paragraph_format.space_after = Pt(8)

                        run = para.add_run(para_text)
                        run.font.name = "Calibri"
                        run.font.size = Pt(11)
                        run.font.color.rgb = TEXT_COLOR

                # Add section sources
                if self.config.include_sources and section.sources:
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
            if self.config.include_sources and job.sources_used:
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
                Table, TableStyle, ListFlowable, ListItem
            )
            from reportlab.lib.units import inch, cm

            # Define professional colors
            PRIMARY_COLOR = HexColor('#1E3A5F')  # Deep blue
            SECONDARY_COLOR = HexColor('#3D5A80')  # Medium blue
            TEXT_COLOR = HexColor('#2D3A45')  # Dark gray
            LIGHT_GRAY = HexColor('#888888')
            ACCENT_BG = HexColor('#F5F5F5')  # Light background

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
                        # Escape XML special characters
                        para_text = para_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(para_text, body_style))

                # Flush remaining list items
                if current_list:
                    story.append(ListFlowable(current_list, bulletType='bullet', leftIndent=20))

                # Section sources
                if self.config.include_sources and section.sources:
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph("Sources for this section:", source_style))
                    for source in section.sources[:3]:
                        doc_name = source.document_name or source.document_id
                        story.append(Paragraph(f"• {doc_name}", source_style))

                story.append(Spacer(1, 0.3*inch))

            # ========== REFERENCES ==========
            if self.config.include_sources and job.sources_used:
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
            if self.config.include_sources and section.sources:
                lines.append("**Sources:**")
                for source in section.sources[:3]:
                    lines.append(f"- {source.document_name or source.document_id}")
                lines.append("")

        # References
        if self.config.include_sources and job.sources_used:
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
            if self.config.include_sources and section.sources:
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
