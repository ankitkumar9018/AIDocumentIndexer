"""
AIDocumentIndexer - Document Generation API Routes
===================================================

Endpoints for generating documents with human-in-the-loop workflow.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.db.database import get_async_session
from backend.api.middleware.auth import AuthenticatedUser
from backend.services.generator import (
    DocumentGenerationService,
    GenerationJob,
    GenerationStatus,
    OutputFormat,
    DocumentOutline,
    Section,
    SourceReference,
    get_generation_service,
)
from backend.services.image_generator import (
    ImageGeneratorService,
    ImageGeneratorConfig,
    ImageBackend,
    GeneratedImage,
    get_image_generator,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class CreateJobRequest(BaseModel):
    """Request to create a new generation job."""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10)
    output_format: str = Field(default="docx")
    collection_filter: Optional[str] = None
    metadata: Optional[dict] = None


class OutlineModifications(BaseModel):
    """Modifications to apply to an outline."""
    title: Optional[str] = None
    sections: Optional[List[dict]] = None
    tone: Optional[str] = None


class SectionFeedback(BaseModel):
    """Feedback for a section."""
    feedback: Optional[str] = None
    approved: bool = True


class SourceReferenceResponse(BaseModel):
    """Source reference in response."""
    document_id: str
    document_name: str
    chunk_id: Optional[str]
    page_number: Optional[int]
    relevance_score: float
    snippet: str


class SectionResponse(BaseModel):
    """Section in response."""
    id: str
    title: str
    content: str
    order: int
    sources: List[SourceReferenceResponse]
    approved: bool
    feedback: Optional[str]


class OutlineResponse(BaseModel):
    """Outline in response."""
    title: str
    description: str
    sections: List[dict]
    target_audience: Optional[str]
    tone: Optional[str]
    word_count_target: Optional[int]


class JobResponse(BaseModel):
    """Generation job response."""
    id: str
    user_id: str
    title: str
    description: str
    output_format: str
    status: str
    outline: Optional[OutlineResponse]
    sections: List[SectionResponse]
    sources_used_count: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    has_output: bool
    error_message: Optional[str]


class JobListResponse(BaseModel):
    """List of generation jobs."""
    jobs: List[JobResponse]
    total: int


# =============================================================================
# Helper Functions
# =============================================================================

def job_to_response(job: GenerationJob) -> JobResponse:
    """Convert GenerationJob to response model."""
    outline_response = None
    if job.outline:
        outline_response = OutlineResponse(
            title=job.outline.title,
            description=job.outline.description,
            sections=job.outline.sections,
            target_audience=job.outline.target_audience,
            tone=job.outline.tone,
            word_count_target=job.outline.word_count_target,
        )

    sections_response = [
        SectionResponse(
            id=s.id,
            title=s.title,
            content=s.revised_content or s.content,
            order=s.order,
            sources=[
                SourceReferenceResponse(
                    document_id=src.document_id,
                    document_name=src.document_name,
                    chunk_id=src.chunk_id,
                    page_number=src.page_number,
                    relevance_score=src.relevance_score,
                    snippet=src.snippet,
                )
                for src in s.sources
            ],
            approved=s.approved,
            feedback=s.feedback,
        )
        for s in job.sections
    ]

    return JobResponse(
        id=job.id,
        user_id=job.user_id,
        title=job.title,
        description=job.description,
        output_format=job.output_format.value,
        status=job.status.value,
        outline=outline_response,
        sections=sections_response,
        sources_used_count=len(job.sources_used),
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        has_output=job.output_path is not None,
        error_message=job.error_message,
    )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_generation_job(
    request: CreateJobRequest,
    user: AuthenticatedUser,
):
    """
    Create a new document generation job.

    This starts the human-in-the-loop workflow:
    1. Create job -> 2. Generate outline -> 3. Review/approve outline
    -> 4. Generate content -> 5. Review/approve sections -> 6. Download
    """
    logger.info(
        "Creating generation job",
        user_id=user.user_id,
        title=request.title,
        format=request.output_format,
    )

    # Validate output format
    try:
        output_format = OutputFormat(request.output_format.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid output format: {request.output_format}. "
                   f"Valid formats: {[f.value for f in OutputFormat]}",
        )

    service = get_generation_service()

    job = await service.create_job(
        user_id=user.user_id,
        title=request.title,
        description=request.description,
        output_format=output_format,
        collection_filter=request.collection_filter,
        metadata=request.metadata,
    )

    return job_to_response(job)


@router.get("/jobs", response_model=JobListResponse)
async def list_generation_jobs(
    user: AuthenticatedUser,
    status_filter: Optional[str] = Query(None, alias="status"),
):
    """
    List all generation jobs for the current user.
    """
    service = get_generation_service()

    # Parse status filter
    gen_status = None
    if status_filter:
        try:
            gen_status = GenerationStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    jobs = await service.list_jobs(
        user_id=user.user_id,
        status=gen_status,
    )

    return JobListResponse(
        jobs=[job_to_response(j) for j in jobs],
        total=len(jobs),
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_generation_job(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Get a specific generation job.
    """
    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    return job_to_response(job)


@router.post("/jobs/{job_id}/outline", response_model=JobResponse)
async def generate_outline(
    job_id: str,
    user: AuthenticatedUser,
    num_sections: int = Query(default=5, ge=1, le=20),
):
    """
    Generate an outline for the document.

    The outline will be generated based on the job description and
    relevant sources from the knowledge base.
    """
    logger.info("Generating outline", job_id=job_id, num_sections=num_sections)

    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        await service.generate_outline(job_id, num_sections=num_sections)
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return job_to_response(job)


@router.post("/jobs/{job_id}/outline/approve", response_model=JobResponse)
async def approve_outline(
    job_id: str,
    user: AuthenticatedUser,
    modifications: Optional[OutlineModifications] = None,
):
    """
    Approve the outline to proceed with content generation.

    Optionally provide modifications to the outline before approval.
    """
    logger.info("Approving outline", job_id=job_id)

    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        mods = modifications.model_dump(exclude_none=True) if modifications else None
        job = await service.approve_outline(job_id, modifications=mods)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return job_to_response(job)


@router.post("/jobs/{job_id}/generate", response_model=JobResponse)
async def generate_content(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Generate the document content.

    Requires an approved outline.
    """
    logger.info("Generating content", job_id=job_id)

    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        job = await service.generate_content(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Content generation failed", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content generation failed",
        )

    return job_to_response(job)


@router.post("/jobs/{job_id}/sections/{section_id}/feedback", response_model=JobResponse)
async def provide_section_feedback(
    job_id: str,
    section_id: str,
    feedback: SectionFeedback,
    user: AuthenticatedUser,
):
    """
    Provide feedback on a section.

    If approved, marks the section as complete.
    If not approved, the section will need revision.
    """
    logger.info(
        "Section feedback",
        job_id=job_id,
        section_id=section_id,
        approved=feedback.approved,
    )

    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        await service.approve_section(
            job_id=job_id,
            section_id=section_id,
            feedback=feedback.feedback,
            approved=feedback.approved,
        )
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return job_to_response(job)


@router.post("/jobs/{job_id}/sections/{section_id}/revise", response_model=JobResponse)
async def revise_section(
    job_id: str,
    section_id: str,
    user: AuthenticatedUser,
):
    """
    Revise a section based on provided feedback.
    """
    logger.info("Revising section", job_id=job_id, section_id=section_id)

    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        await service.revise_section(job_id=job_id, section_id=section_id)
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return job_to_response(job)


@router.get("/jobs/{job_id}/download")
async def download_generated_document(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Download the generated document.

    Only available for completed jobs.
    """
    logger.info("Downloading document", job_id=job_id)

    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        file_bytes, filename, content_type = await service.get_output_file(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return StreamingResponse(
        iter([file_bytes]),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(file_bytes)),
        },
    )


@router.delete("/jobs/{job_id}")
async def cancel_generation_job(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Cancel a generation job.
    """
    logger.info("Cancelling job", job_id=job_id)

    service = get_generation_service()

    try:
        job = await service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Verify ownership
    if job.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        await service.cancel_job(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/formats")
async def list_output_formats():
    """
    List all supported output formats.
    """
    return {
        "formats": [
            {"value": f.value, "name": f.name, "description": _get_format_description(f)}
            for f in OutputFormat
        ]
    }


def _get_format_description(format: OutputFormat) -> str:
    """Get description for output format."""
    descriptions = {
        OutputFormat.PPTX: "PowerPoint presentation (.pptx)",
        OutputFormat.DOCX: "Word document (.docx)",
        OutputFormat.PDF: "PDF document (.pdf)",
        OutputFormat.XLSX: "Excel spreadsheet (.xlsx)",
        OutputFormat.MARKDOWN: "Markdown text (.md)",
        OutputFormat.HTML: "HTML web page (.html)",
        OutputFormat.TXT: "Plain text (.txt)",
    }
    return descriptions.get(format, "")


# =============================================================================
# Image Generation Models
# =============================================================================

class ImageGenerateRequest(BaseModel):
    """Request to generate an image."""
    prompt: str = Field(..., min_length=5, max_length=1000)
    width: int = Field(default=800, ge=256, le=2048)
    height: int = Field(default=600, ge=256, le=2048)


class ImageGenerateResponse(BaseModel):
    """Response from image generation."""
    success: bool
    path: Optional[str] = None
    width: int
    height: int
    prompt: str
    backend: str
    error: Optional[str] = None


class ImageConfigResponse(BaseModel):
    """Image generation configuration."""
    enabled: bool
    backend: str
    available_backends: List[str]
    sd_webui_available: bool


# =============================================================================
# Image Generation Endpoints
# =============================================================================

@router.post("/image", response_model=ImageGenerateResponse)
async def generate_image(
    request: ImageGenerateRequest,
    user: AuthenticatedUser,
):
    """
    Generate an image using AI.

    Uses Stable Diffusion WebUI (Automatic1111) if available,
    falls back to Unsplash placeholder images.
    """
    logger.info(
        "Generating image",
        user_id=user.user_id,
        prompt_preview=request.prompt[:50],
    )

    service = get_image_generator()

    # Check if image generation is enabled
    if not service.config.enabled:
        return ImageGenerateResponse(
            success=False,
            path=None,
            width=request.width,
            height=request.height,
            prompt=request.prompt,
            backend="disabled",
            error="Image generation is disabled. Enable with include_images=true in generation config.",
        )

    try:
        result = await service.generate_for_section(
            section_title="Custom Generation",
            section_content=request.prompt,
            width=request.width,
            height=request.height,
        )

        if result and result.success:
            return ImageGenerateResponse(
                success=True,
                path=result.path,
                width=result.width,
                height=result.height,
                prompt=result.prompt,
                backend=result.backend.value,
            )
        else:
            return ImageGenerateResponse(
                success=False,
                path=None,
                width=request.width,
                height=request.height,
                prompt=request.prompt,
                backend=result.backend.value if result else "unknown",
                error=result.error if result else "Unknown error",
            )

    except Exception as e:
        logger.error("Image generation failed", error=str(e))
        return ImageGenerateResponse(
            success=False,
            path=None,
            width=request.width,
            height=request.height,
            prompt=request.prompt,
            backend="error",
            error=str(e),
        )


@router.get("/image/config", response_model=ImageConfigResponse)
async def get_image_config(user: AuthenticatedUser):
    """
    Get image generation configuration and availability.
    """
    service = get_image_generator()

    # Check SD WebUI availability
    sd_available = await service.check_sd_webui_available()

    return ImageConfigResponse(
        enabled=service.config.enabled,
        backend=service.config.backend.value,
        available_backends=[b.value for b in ImageBackend],
        sd_webui_available=sd_available,
    )
