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
    THEMES,
    check_spelling,
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
    collection_filter: Optional[str] = None  # Single collection filter (legacy)
    collection_filters: Optional[List[str]] = None  # Multiple collection filters
    folder_id: Optional[str] = Field(default=None, description="Folder ID to scope query to")
    include_subfolders: bool = Field(default=True, description="Include documents in subfolders")
    metadata: Optional[dict] = None
    # Image generation - overrides admin setting if provided
    include_images: Optional[bool] = Field(
        default=None,
        description="Include images in generated document. If not set, uses admin setting."
    )
    # Theme selection - defaults to business if not specified
    theme: Optional[str] = Field(
        default="business",
        description="Visual theme for the document. Options: business, creative, modern, nature"
    )
    # Page/slide count control - None means auto (LLM decides)
    page_count: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of pages/slides (1-20). None = auto (LLM decides based on content)"
    )
    # Include sources page/slide/sheet
    include_sources: Optional[bool] = Field(
        default=None,
        description="Include sources page in generated document. If not set, uses admin setting."
    )
    # Style learning from existing documents
    use_existing_docs: bool = Field(
        default=False,
        description="Learn style and formatting from existing documents via RAG"
    )
    style_collection_filters: Optional[List[str]] = Field(
        default=None,
        description="Collections to learn style from (multi-select)"
    )
    style_folder_id: Optional[str] = Field(
        default=None,
        description="Folder to scope style learning"
    )
    include_style_subfolders: bool = Field(
        default=True,
        description="Include subfolders in style learning"
    )

    @property
    def effective_collection_filter(self) -> Optional[str]:
        """Get effective collection filter - prioritize collection_filters list."""
        if self.collection_filters:
            return ",".join(self.collection_filters)  # Join multiple as comma-separated
        return self.collection_filter


class OutlineModifications(BaseModel):
    """Modifications to apply to an outline."""
    title: Optional[str] = None
    sections: Optional[List[dict]] = None
    tone: Optional[str] = None
    theme: Optional[str] = None  # Allow changing theme after outline generation


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
    # Generation settings used
    include_images: bool = False
    image_backend: Optional[str] = None


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
        # Image generation settings from job metadata
        include_images=job.metadata.get("include_images", False),
        image_backend=job.metadata.get("image_backend"),
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

    # Build metadata with theme and page_count
    metadata = request.metadata or {}
    if request.theme:
        # Validate theme exists
        if request.theme not in THEMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid theme: {request.theme}. Valid themes: {list(THEMES.keys())}",
            )
        metadata["theme"] = request.theme

    # Store page_count in metadata for outline generation
    if request.page_count is not None:
        metadata["page_count"] = request.page_count
        metadata["page_count_mode"] = "manual"
    else:
        metadata["page_count_mode"] = "auto"

    # Store include_sources preference (None means use admin setting)
    if request.include_sources is not None:
        metadata["include_sources"] = request.include_sources

    # Store style learning settings if enabled
    if request.use_existing_docs:
        metadata["use_existing_docs"] = True
        metadata["style_collection_filters"] = request.style_collection_filters
        metadata["style_folder_id"] = request.style_folder_id
        metadata["include_style_subfolders"] = request.include_style_subfolders

    job = await service.create_job(
        user_id=user.user_id,
        title=request.title,
        description=request.description,
        output_format=output_format,
        collection_filter=request.effective_collection_filter,
        folder_id=request.folder_id,
        include_subfolders=request.include_subfolders,
        metadata=metadata,
        include_images=request.include_images,  # Pass through to override admin setting
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
    num_sections: Optional[int] = Query(default=None, ge=1, le=20),
):
    """
    Generate an outline for the document.

    The outline will be generated based on the job description and
    relevant sources from the knowledge base.

    num_sections priority:
    1. Query parameter (if provided) - highest priority, allows override
    2. Job metadata page_count (from CreateJobRequest)
    3. Auto mode (None) - LLM decides optimal count based on content
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
    if job.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    # Determine effective num_sections from priority order
    effective_num_sections = num_sections
    if effective_num_sections is None:
        # Fall back to job metadata page_count
        effective_num_sections = job.metadata.get("page_count")
    # If still None, auto mode - service will handle LLM-based determination

    logger.info(
        "Generating outline",
        job_id=job_id,
        num_sections=effective_num_sections,
        mode="auto" if effective_num_sections is None else "manual",
    )

    try:
        await service.generate_outline(job_id, num_sections=effective_num_sections)
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
            "Content-Disposition": f'attachment; filename="{filename}"',
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
    backend: Optional[str] = Field(None, description="Image generation backend: openai, stability, automatic1111, unsplash")
    model: Optional[str] = Field(None, description="Model to use (backend-specific, e.g., dall-e-3)")
    quality: Optional[str] = Field(None, description="Quality for DALL-E 3: standard or hd")
    style: Optional[str] = Field(None, description="Style for DALL-E 3: vivid or natural")


class ImageGenerateResponse(BaseModel):
    """Response from image generation."""
    success: bool
    path: Optional[str] = None
    image_base64: Optional[str] = None  # Base64-encoded image data
    width: int
    height: int
    prompt: str
    revised_prompt: Optional[str] = None  # DALL-E 3 may revise the prompt
    backend: str
    model: Optional[str] = None
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

    Supports multiple backends:
    - openai: OpenAI DALL-E 3/2 (requires OPENAI_API_KEY)
    - stability: Stability AI (requires STABILITY_API_KEY)
    - automatic1111: Local Stable Diffusion WebUI
    - unsplash: Placeholder images (free, always works)

    If no backend is specified, uses the configured default.
    """
    logger.info(
        "Generating image",
        user_id=user.user_id,
        prompt_preview=request.prompt[:50],
        backend=request.backend,
    )

    service = get_image_generator()

    # Check if image generation is enabled (unless explicitly requesting a backend)
    if not service.config.enabled and not request.backend:
        return ImageGenerateResponse(
            success=False,
            path=None,
            width=request.width,
            height=request.height,
            prompt=request.prompt,
            backend="disabled",
            error="Image generation is disabled. Enable with include_images=true in generation config, or specify a backend explicitly.",
        )

    try:
        import base64

        # Temporarily update config if backend/model specified
        original_backend = service.config.backend
        original_model = service.config.openai_model
        original_quality = service.config.openai_quality
        original_style = service.config.openai_style

        if request.backend:
            try:
                service.config.backend = ImageBackend(request.backend)
            except ValueError:
                return ImageGenerateResponse(
                    success=False,
                    path=None,
                    width=request.width,
                    height=request.height,
                    prompt=request.prompt,
                    backend=request.backend,
                    error=f"Invalid backend: {request.backend}. Valid options: openai, stability, automatic1111, unsplash",
                )

        if request.model:
            service.config.openai_model = request.model
        if request.quality:
            service.config.openai_quality = request.quality
        if request.style:
            service.config.openai_style = request.style

        # Generate image directly with the backend
        result = await service._generate_with_backend(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            backend=service.config.backend,
        )

        # Restore original config
        service.config.backend = original_backend
        service.config.openai_model = original_model
        service.config.openai_quality = original_quality
        service.config.openai_style = original_style

        if result and result.success:
            # Read image file and convert to base64
            image_base64 = None
            if result.path:
                try:
                    with open(result.path, "rb") as f:
                        image_base64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception as e:
                    logger.warning(f"Could not read generated image file: {e}")

            return ImageGenerateResponse(
                success=True,
                path=result.path,
                image_base64=image_base64,
                width=result.width,
                height=result.height,
                prompt=result.prompt,
                revised_prompt=result.metadata.get("revised_prompt") if result.metadata else None,
                backend=result.backend.value,
                model=result.metadata.get("model") if result.metadata else None,
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


@router.get("/image/backends")
async def get_image_backends(user: AuthenticatedUser):
    """
    Get list of available image generation backends with their configuration status.

    Returns which backends are configured and ready to use (have API keys, etc.)
    """
    service = get_image_generator()
    backends = service.get_available_backends()

    return {
        "backends": backends,
        "current_backend": service.config.backend.value,
        "enabled": service.config.enabled,
    }


# =============================================================================
# Theme Models and Endpoints
# =============================================================================

class ThemeInfo(BaseModel):
    """Information about a theme."""
    key: str
    name: str
    description: str
    primary: str
    secondary: str
    accent: str
    text: str


class ThemeSuggestionRequest(BaseModel):
    """Request for theme suggestions."""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10)


class ThemeSuggestionResponse(BaseModel):
    """Response with theme suggestions."""
    recommended: str
    reason: str
    alternatives: List[str]
    themes: dict  # Full theme info for all themes


@router.get("/themes")
async def list_themes():
    """
    Get all available themes for document generation.

    Returns a list of themes with their colors and descriptions.
    """
    themes_list = []
    for key, theme in THEMES.items():
        themes_list.append(ThemeInfo(
            key=key,
            name=theme["name"],
            description=theme["description"],
            primary=theme["primary"],
            secondary=theme["secondary"],
            accent=theme["accent"],
            text=theme["text"],
        ))

    return {"themes": themes_list}


@router.post("/themes/suggest", response_model=ThemeSuggestionResponse)
async def suggest_theme(
    request: ThemeSuggestionRequest,
    user: AuthenticatedUser,
):
    """
    Get LLM-suggested themes based on document topic.

    The LLM analyzes the document title and description to recommend
    the most appropriate theme. User can still override with any theme.
    """
    logger.info(
        "Suggesting themes",
        user_id=user.user_id,
        title=request.title,
    )

    try:
        from backend.services.llm import EnhancedLLMFactory

        prompt = f"""Based on this document topic, recommend the best visual theme for a presentation/document:

Topic: {request.title}
Description: {request.description}

Available themes:
- business: Corporate, professional presentations (blue tones)
- creative: Marketing, design, artistic content (purple/warm tones)
- modern: Tech, startups, contemporary topics (dark with cyan accent)
- nature: Sustainability, wellness, environmental (green/earth tones)

Respond with ONLY valid JSON in this exact format:
{{"recommended": "theme_key", "reason": "Brief 1-sentence explanation", "alternatives": ["theme2", "theme3"]}}

Choose the most appropriate theme for this content. Be decisive."""

        llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="content_generation",
            user_id=None,
        )
        response = await llm.ainvoke(prompt)

        import json
        # Extract JSON from response - it may have extra text
        response_text = response.content.strip()

        # Try to find JSON in the response
        try:
            # First try direct parse
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback to business theme
                result = {
                    "recommended": "business",
                    "reason": "Default professional theme",
                    "alternatives": ["modern", "creative"]
                }

        # Validate the recommended theme exists
        if result.get("recommended") not in THEMES:
            result["recommended"] = "business"

        # Validate alternatives
        valid_alternatives = [alt for alt in result.get("alternatives", []) if alt in THEMES]
        if not valid_alternatives:
            valid_alternatives = [k for k in THEMES.keys() if k != result["recommended"]][:2]
        result["alternatives"] = valid_alternatives

        return ThemeSuggestionResponse(
            recommended=result["recommended"],
            reason=result.get("reason", "Recommended based on content analysis"),
            alternatives=result["alternatives"],
            themes=THEMES,
        )

    except Exception as e:
        logger.error("Theme suggestion failed", error=str(e))
        # Return default on error
        return ThemeSuggestionResponse(
            recommended="business",
            reason="Default professional theme (LLM unavailable)",
            alternatives=["modern", "creative"],
            themes=THEMES,
        )


# =============================================================================
# Spell Checking Models and Endpoints
# =============================================================================

class SpellCheckRequest(BaseModel):
    """Request to check spelling in text."""
    text: str = Field(..., min_length=1, max_length=50000)


class SpellIssue(BaseModel):
    """A spelling issue found in text."""
    word: str
    position: int
    suggestion: str
    context: str


class SpellCheckResponse(BaseModel):
    """Response from spell checking."""
    has_issues: bool
    issues: List[SpellIssue]


class SpellApplyRequest(BaseModel):
    """Request to apply spelling corrections."""
    job_id: str
    corrections: dict = Field(
        ...,
        description="Map of original word to corrected word"
    )


@router.post("/spell-check", response_model=SpellCheckResponse)
async def check_text_spelling(
    request: SpellCheckRequest,
    user: AuthenticatedUser,
):
    """
    Check spelling in provided text.

    Returns a list of potential spelling issues for user review.
    The user can then decide which corrections to apply.
    """
    logger.info(
        "Checking spelling",
        user_id=user.user_id,
        text_length=len(request.text),
    )

    result = check_spelling(request.text)

    return SpellCheckResponse(
        has_issues=result["has_issues"],
        issues=[
            SpellIssue(
                word=issue["word"],
                position=issue["position"],
                suggestion=issue["suggestion"],
                context=issue["context"],
            )
            for issue in result["issues"]
        ],
    )


@router.post("/jobs/{job_id}/spell-check", response_model=SpellCheckResponse)
async def check_job_spelling(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Check spelling in a generation job's content.

    Checks the title and all section titles and content for spelling issues.
    Returns a list of issues for user review before final document generation.
    """
    logger.info("Checking job spelling", job_id=job_id, user_id=user.user_id)

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

    # Collect all text to check
    all_issues = []

    # Check title
    title_result = check_spelling(job.title)
    for issue in title_result["issues"]:
        issue["source"] = "title"
        all_issues.append(issue)

    # Check section titles and content
    for section in job.sections:
        # Check section title
        title_result = check_spelling(section.title)
        for issue in title_result["issues"]:
            issue["source"] = f"section_{section.id}_title"
            all_issues.append(issue)

        # Check section content
        content = section.revised_content or section.content
        content_result = check_spelling(content)
        for issue in content_result["issues"]:
            issue["source"] = f"section_{section.id}_content"
            all_issues.append(issue)

    # Limit total issues
    all_issues = all_issues[:50]

    return SpellCheckResponse(
        has_issues=len(all_issues) > 0,
        issues=[
            SpellIssue(
                word=issue["word"],
                position=issue["position"],
                suggestion=issue["suggestion"],
                context=issue["context"],
            )
            for issue in all_issues
        ],
    )
