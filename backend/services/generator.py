"""
AIDocumentIndexer - Document Generation Service
================================================

Human-in-the-loop document generation using LangGraph.
Supports PPTX, DOCX, PDF, and other output formats.
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
        prompt = f"""Create a document outline for the following:

Title: {title}
Description: {description}

{context}

Generate an outline with {num_sections} sections. For each section, provide:
1. A clear, descriptive title
2. A brief description of what the section should cover

Format your response as a structured outline."""

        # Use LLM to generate
        try:
            from backend.services.llm import get_chat_model

            llm = get_chat_model(model=self.config.model)
            response = await llm.ainvoke(prompt)

            # Parse response into sections (simplified parsing)
            sections = []
            lines = response.content.split("\n")
            current_section = None

            for line in lines:
                line = line.strip()
                if line and (line.startswith("Section") or line.startswith("#") or line[0].isdigit()):
                    if current_section:
                        sections.append(current_section)
                    current_section = {"title": line.lstrip("#0123456789. "), "description": ""}
                elif current_section and line:
                    current_section["description"] += line + " "

            if current_section:
                sections.append(current_section)

            # Ensure we have the requested number of sections
            while len(sections) < num_sections:
                sections.append({
                    "title": f"Section {len(sections) + 1}",
                    "description": "Additional content",
                })

            return DocumentOutline(
                title=title,
                description=description,
                sections=sections[:num_sections],
            )

        except Exception as e:
            logger.error("Failed to generate outline with LLM", error=str(e))
            # Return a basic outline
            return DocumentOutline(
                title=title,
                description=description,
                sections=[
                    {"title": f"Section {i+1}", "description": f"Content for section {i+1}"}
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
            from backend.services.llm import get_chat_model

            llm = get_chat_model(model=self.config.model)
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
            from backend.services.llm import get_chat_model

            llm = get_chat_model(model=self.config.model)
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
        """Generate PowerPoint file."""
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt

            prs = Presentation()

            # Title slide
            title_slide_layout = prs.slide_layouts[0]
            slide = prs.slides.add_slide(title_slide_layout)
            title = slide.shapes.title
            subtitle = slide.placeholders[1]
            title.text = job.title
            subtitle.text = job.outline.description if job.outline else job.description

            # Content slides
            bullet_layout = prs.slide_layouts[1]
            for section in job.sections:
                slide = prs.slides.add_slide(bullet_layout)
                title = slide.shapes.title
                body = slide.placeholders[1]

                title.text = section.title

                tf = body.text_frame
                tf.text = section.revised_content or section.content

            # Sources slide if configured
            if self.config.include_sources and job.sources_used:
                slide = prs.slides.add_slide(bullet_layout)
                title = slide.shapes.title
                body = slide.placeholders[1]

                title.text = "Sources"
                tf = body.text_frame
                for source in job.sources_used[:10]:
                    p = tf.add_paragraph()
                    p.text = f"• Document: {source.document_name or source.document_id}"
                    p.level = 0

            output_path = os.path.join(self.config.output_dir, f"{filename}.pptx")
            prs.save(output_path)

            logger.info("PPTX generated", path=output_path)
            return output_path

        except ImportError:
            logger.error("python-pptx not installed")
            return await self._generate_txt(job, filename)

    async def _generate_docx(self, job: GenerationJob, filename: str) -> str:
        """Generate Word document."""
        try:
            from docx import Document
            from docx.shared import Pt, Inches

            doc = Document()

            # Title
            doc.add_heading(job.title, 0)

            # Description
            if job.outline:
                doc.add_paragraph(job.outline.description)

            # Table of contents placeholder
            if self.config.include_toc:
                doc.add_heading("Table of Contents", level=1)
                for section in job.sections:
                    doc.add_paragraph(section.title, style='TOC Heading')
                doc.add_page_break()

            # Content
            for section in job.sections:
                doc.add_heading(section.title, level=1)
                content = section.revised_content or section.content
                doc.add_paragraph(content)

                # Add sources for this section
                if self.config.include_sources and section.sources:
                    doc.add_paragraph("Sources:", style='Intense Quote')
                    for source in section.sources[:3]:
                        doc.add_paragraph(
                            f"• {source.document_name or source.document_id}",
                            style='List Bullet'
                        )

            # References section
            if self.config.include_sources and job.sources_used:
                doc.add_heading("References", level=1)
                for source in job.sources_used:
                    doc.add_paragraph(
                        f"• {source.document_name or source.document_id}",
                        style='List Bullet'
                    )

            output_path = os.path.join(self.config.output_dir, f"{filename}.docx")
            doc.save(output_path)

            logger.info("DOCX generated", path=output_path)
            return output_path

        except ImportError:
            logger.error("python-docx not installed")
            return await self._generate_txt(job, filename)

    async def _generate_pdf(self, job: GenerationJob, filename: str) -> str:
        """Generate PDF document."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.units import inch

            output_path = os.path.join(self.config.output_dir, f"{filename}.pdf")
            doc = SimpleDocTemplate(output_path, pagesize=letter)

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
            )
            heading_style = styles['Heading2']
            body_style = styles['BodyText']

            story = []

            # Title
            story.append(Paragraph(job.title, title_style))
            story.append(Spacer(1, 0.5 * inch))

            # Description
            if job.outline:
                story.append(Paragraph(job.outline.description, body_style))
                story.append(Spacer(1, 0.25 * inch))

            story.append(PageBreak())

            # Content
            for section in job.sections:
                story.append(Paragraph(section.title, heading_style))
                story.append(Spacer(1, 0.1 * inch))

                content = section.revised_content or section.content
                # Split into paragraphs
                for para in content.split('\n\n'):
                    if para.strip():
                        story.append(Paragraph(para.strip(), body_style))
                        story.append(Spacer(1, 0.1 * inch))

                story.append(Spacer(1, 0.25 * inch))

            doc.build(story)

            logger.info("PDF generated", path=output_path)
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
