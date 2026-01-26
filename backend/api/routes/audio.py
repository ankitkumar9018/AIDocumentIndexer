"""
AIDocumentIndexer - Audio Overview API Routes
==============================================

REST API endpoints for audio overviews (NotebookLM-inspired feature).

Endpoints:
- POST   /audio/overviews              - Create new audio overview
- GET    /audio/overviews              - List audio overviews
- GET    /audio/overviews/{id}         - Get specific overview
- POST   /audio/overviews/{id}/generate - Generate audio for overview
- POST   /audio/overviews/{id}/generate/stream - Generate with streaming updates
- DELETE /audio/overviews/{id}         - Delete overview
- GET    /audio/files/{filename}       - Serve audio file
- GET    /audio/formats                - List available formats
- GET    /audio/voices                 - List available voices
- POST   /audio/estimate-cost          - Estimate generation cost
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_async_session, get_current_organization_id
from backend.db.models import AudioOverviewFormat, AudioOverviewStatus
from backend.services.audio import AudioOverviewService
from backend.services.audio.tts_service import TTSProvider

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class VoiceSelection(BaseModel):
    """Voice selection for audio generation."""
    host1_voice: Optional[str] = Field(None, description="Voice ID for host 1 (e.g., 'en-US-AndrewMultilingualNeural')")
    host2_voice: Optional[str] = Field(None, description="Voice ID for host 2 (e.g., 'en-US-AvaMultilingualNeural')")
    host1_name: Optional[str] = Field(None, description="Custom name for host 1 (default: Alex)")
    host2_name: Optional[str] = Field(None, description="Custom name for host 2 (default: Jordan)")


# Duration preferences
DURATION_PREFERENCES = ["short", "standard", "extended"]


class CreateAudioOverviewRequest(BaseModel):
    """Request to create a new audio overview."""
    document_ids: List[str] = Field(..., description="List of document UUIDs")
    format: str = Field(default="deep_dive", description="Audio format")
    title: Optional[str] = Field(None, description="Custom title")
    custom_instructions: Optional[str] = Field(None, description="Additional instructions")
    tts_provider: str = Field(default="openai", description="TTS provider to use")
    voices: Optional[VoiceSelection] = Field(None, description="Voice selection for hosts")
    duration_preference: str = Field(default="standard", description="Duration preference: 'short', 'standard', or 'extended'")


class AudioOverviewResponse(BaseModel):
    """Response model for audio overview."""
    id: str
    document_ids: List[str]
    format: str
    title: Optional[str]
    status: str
    audio_url: Optional[str]
    duration_seconds: Optional[int]
    script: Optional[Dict[str, Any]]
    error_message: Optional[str]
    config: Optional[Dict[str, Any]]
    tts_provider: Optional[str] = None
    created_by_id: Optional[str]
    created_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class AudioOverviewListResponse(BaseModel):
    """Response model for listing audio overviews."""
    overviews: List[AudioOverviewResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class EstimateCostRequest(BaseModel):
    """Request for cost estimation."""
    document_ids: List[str]
    format: str = "deep_dive"
    tts_provider: str = "openai"


class EstimateCostResponse(BaseModel):
    """Response for cost estimation."""
    document_count: int
    total_document_chars: int
    estimated_duration: Dict[str, int]
    estimated_costs: Dict[str, float]
    tts_provider: str


class UpdateAudioOverviewRequest(BaseModel):
    """Request to update an audio overview."""
    tts_provider: Optional[str] = Field(None, description="TTS provider to use")
    title: Optional[str] = Field(None, description="Custom title")


class FormatInfo(BaseModel):
    """Information about an audio format."""
    id: str
    name: str
    description: str
    typical_duration_minutes: str


class VoiceInfo(BaseModel):
    """Information about a TTS voice."""
    id: str
    name: str
    provider: str
    gender: Optional[str] = None
    style: Optional[str] = None
    preview_url: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def to_response(overview) -> AudioOverviewResponse:
    """Convert AudioOverview model to response."""
    # Handle format - could be enum or string from database
    format_val = overview.format
    if format_val is None:
        format_val = "deep_dive"
    elif hasattr(format_val, 'value'):
        format_val = format_val.value
    # else it's already a string

    # Handle status - could be enum or string from database
    status_val = overview.status
    if status_val is None:
        status_val = "pending"
    elif hasattr(status_val, 'value'):
        status_val = status_val.value
    # else it's already a string

    # Convert document_ids to list of strings
    doc_ids = []
    if overview.document_ids:
        doc_ids = [str(d) for d in overview.document_ids]

    # Handle tts_provider - could be enum or string from database
    tts_provider_val = getattr(overview, 'tts_provider', None)
    if tts_provider_val is not None and hasattr(tts_provider_val, 'value'):
        tts_provider_val = tts_provider_val.value

    return AudioOverviewResponse(
        id=str(overview.id),
        document_ids=doc_ids,
        format=format_val,
        title=overview.title,
        status=status_val,
        audio_url=overview.audio_url,
        duration_seconds=overview.duration_seconds,
        script=overview.script,
        error_message=overview.error_message,
        config=getattr(overview, 'host_config', None),  # Model uses host_config, response uses config
        tts_provider=tts_provider_val,
        created_by_id=str(overview.created_by_id) if overview.created_by_id else None,
        created_at=overview.created_at,
        completed_at=getattr(overview, 'completed_at', None),
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/overviews", response_model=AudioOverviewResponse)
async def create_audio_overview(
    request: CreateAudioOverviewRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """
    Create a new audio overview from documents.

    This creates the overview record but does not generate the audio.
    Call the /generate endpoint to start generation.
    """
    try:
        format_enum = AudioOverviewFormat(request.format)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format: {request.format}. Valid formats: {[f.value for f in AudioOverviewFormat]}",
        )

    try:
        tts_provider = TTSProvider(request.tts_provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid TTS provider: {request.tts_provider}. Valid providers: {[p.value for p in TTSProvider]}",
        )

    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    document_ids = [uuid.UUID(did) for did in request.document_ids]

    # Validate duration preference
    if request.duration_preference not in DURATION_PREFERENCES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid duration preference: {request.duration_preference}. Valid options: {DURATION_PREFERENCES}",
        )

    # Build host config from voice selection and duration preference
    host_config = {}
    if request.voices:
        host_config["host1_voice"] = request.voices.host1_voice
        host_config["host2_voice"] = request.voices.host2_voice
        host_config["host1_name"] = request.voices.host1_name or "Alex"
        host_config["host2_name"] = request.voices.host2_name or "Jordan"
    host_config["duration_preference"] = request.duration_preference

    overview = await service.create_overview(
        document_ids=document_ids,
        format=format_enum,
        title=request.title,
        custom_instructions=request.custom_instructions,
        tts_provider=tts_provider,
        host_config=host_config if host_config else None,
    )

    return to_response(overview)


@router.get("/overviews", response_model=AudioOverviewListResponse)
async def list_audio_overviews(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    format: Optional[str] = Query(None, description="Filter by format"),
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """List audio overviews with optional filtering."""
    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    filters = {}
    if status:
        try:
            filters["status"] = AudioOverviewStatus(status)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status: {status}")
    if format:
        try:
            filters["format"] = AudioOverviewFormat(format)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid format: {format}")

    if document_id:
        # Use document-specific listing
        overviews, total = await service.list_by_documents(
            document_ids=[uuid.UUID(document_id)],
            page=page,
            page_size=page_size,
        )
    else:
        overviews, total = await service.list(
            page=page,
            page_size=page_size,
            filters=filters if filters else None,
        )

    return AudioOverviewListResponse(
        overviews=[to_response(o) for o in overviews],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/overviews/{overview_id}", response_model=AudioOverviewResponse)
async def get_audio_overview(
    overview_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """Get a specific audio overview by ID."""
    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    overview = await service.get_by_id(uuid.UUID(overview_id))
    if not overview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio overview not found")

    return to_response(overview)


@router.post("/overviews/{overview_id}/generate", response_model=AudioOverviewResponse)
async def generate_audio_overview(
    overview_id: str,
    background: bool = Query(False, description="Run generation in background"),
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """
    Generate audio for an overview.

    If background=True, returns immediately and generates in background.
    Poll the GET endpoint to check status.
    """
    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    overview = await service.get_by_id(uuid.UUID(overview_id))
    if not overview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio overview not found")

    # Only return early if status is ready AND audio file actually exists
    if overview.status == AudioOverviewStatus.READY.value and overview.storage_path:
        # Verify the file exists
        import os
        if os.path.exists(overview.storage_path):
            return to_response(overview)
        # File doesn't exist, reset status to allow regeneration
        overview.status = AudioOverviewStatus.PENDING.value
        await session.commit()

    if overview.status in [AudioOverviewStatus.GENERATING_SCRIPT.value, AudioOverviewStatus.GENERATING_AUDIO.value]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Generation already in progress",
        )

    overview = await service.generate_overview(
        overview_id=uuid.UUID(overview_id),
        background=background,
    )

    return to_response(overview)


@router.post("/overviews/{overview_id}/generate/stream")
async def generate_audio_overview_stream(
    overview_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """
    Generate audio with streaming status updates.

    Returns Server-Sent Events with generation progress.
    """
    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    overview = await service.get_by_id(uuid.UUID(overview_id))
    if not overview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio overview not found")

    async def generate():
        async for event in service.generate_overview_streaming(uuid.UUID(overview_id)):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.patch("/overviews/{overview_id}", response_model=AudioOverviewResponse)
async def update_audio_overview(
    overview_id: str,
    request: UpdateAudioOverviewRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """
    Update an audio overview's settings.

    Only pending or failed overviews can have their TTS provider changed.
    """
    from backend.db.models import AudioOverview

    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    overview = await service.get_by_id_or_raise(uuid.UUID(overview_id))

    # Check if the overview can be modified
    if overview.status not in [AudioOverviewStatus.PENDING.value, AudioOverviewStatus.FAILED.value, "pending", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only update pending or failed audio overviews",
        )

    # Update fields
    if request.tts_provider is not None:
        try:
            tts_provider = TTSProvider(request.tts_provider)
            overview.tts_provider = tts_provider.value
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid TTS provider: {request.tts_provider}. Valid providers: {[p.value for p in TTSProvider]}",
            )

    if request.title is not None:
        overview.title = request.title

    # Reset error state if it was failed
    if overview.status == AudioOverviewStatus.FAILED.value or overview.status == "failed":
        overview.status = AudioOverviewStatus.PENDING.value
        overview.error_message = None

    await session.commit()
    await session.refresh(overview)

    return to_response(overview)


@router.delete("/overviews/{overview_id}")
async def delete_audio_overview(
    overview_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """Delete an audio overview and its audio file."""
    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    # Get file path before deletion
    file_path = await service.get_audio_file_path(uuid.UUID(overview_id))

    # Delete record
    await service.delete(uuid.UUID(overview_id), soft=False)

    # Delete audio file if exists
    if file_path and file_path.exists():
        file_path.unlink()

    return {"message": "Audio overview deleted", "id": overview_id}


@router.get("/files/{filename}")
async def serve_audio_file(
    filename: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
):
    """Serve an audio file."""
    from backend.core.config import settings

    # Get the backend directory (backend/api/routes -> backend)
    backend_root = Path(__file__).resolve().parent.parent.parent
    audio_storage_setting = getattr(settings, "AUDIO_STORAGE_PATH", None)
    storage_path = Path(audio_storage_setting) if audio_storage_setting else (backend_root / "storage" / "audio")
    file_path = storage_path / filename

    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {file_path}")

    # Validate file is within storage path (security)
    try:
        file_path.resolve().relative_to(storage_path.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename,
    )


@router.get("/formats", response_model=List[FormatInfo])
async def list_audio_formats():
    """List available audio overview formats."""
    formats = [
        FormatInfo(
            id="deep_dive",
            name="Deep Dive",
            description="Comprehensive exploration with two hosts discussing in depth",
            typical_duration_minutes="15-20",
        ),
        FormatInfo(
            id="brief",
            name="Brief Summary",
            description="Quick, focused summary hitting key points",
            typical_duration_minutes="5",
        ),
        FormatInfo(
            id="critique",
            name="Critique",
            description="Thoughtful analysis discussing strengths and weaknesses",
            typical_duration_minutes="10-15",
        ),
        FormatInfo(
            id="debate",
            name="Debate",
            description="Two hosts with contrasting viewpoints",
            typical_duration_minutes="12-15",
        ),
        FormatInfo(
            id="lecture",
            name="Lecture",
            description="Educational single-speaker presentation",
            typical_duration_minutes="10-15",
        ),
        FormatInfo(
            id="interview",
            name="Interview",
            description="Q&A style with interviewer and expert",
            typical_duration_minutes="12-15",
        ),
    ]
    return formats


@router.get("/voices")
async def list_available_voices(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """List available TTS voices."""
    from backend.services.audio.tts_service import TTSService

    tts_service = TTSService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    provider_enum = None
    if provider:
        try:
            provider_enum = TTSProvider(provider)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {provider}. Valid: {[p.value for p in TTSProvider]}",
            )

    voices = await tts_service.get_voices(provider_enum)
    return voices


@router.post("/estimate-cost", response_model=EstimateCostResponse)
async def estimate_generation_cost(
    request: EstimateCostRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """Estimate the cost of generating an audio overview."""
    try:
        format_enum = AudioOverviewFormat(request.format)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid format: {request.format}")

    try:
        tts_provider = TTSProvider(request.tts_provider)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid TTS provider: {request.tts_provider}")

    service = AudioOverviewService(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    document_ids = [uuid.UUID(did) for did in request.document_ids]

    estimate = await service.estimate_cost(
        document_ids=document_ids,
        format=format_enum,
        tts_provider=tts_provider,
    )

    return EstimateCostResponse(**estimate)


# =============================================================================
# TTS Settings & Model Management
# =============================================================================

class TTSSettingsResponse(BaseModel):
    """TTS settings response."""
    default_provider: str
    available_providers: List[Dict[str, Any]]


class TTSSettingsUpdateRequest(BaseModel):
    """Request to update TTS settings."""
    default_provider: str = Field(..., description="Default TTS provider")


class CoquiModelInfo(BaseModel):
    """Info about a Coqui TTS model."""
    name: str
    language: str
    description: str
    is_installed: bool
    size_mb: Optional[float] = None


class CoquiModelActionResponse(BaseModel):
    """Response for Coqui model actions."""
    success: bool
    message: str
    model_name: Optional[str] = None


@router.get("/settings/tts", response_model=TTSSettingsResponse)
async def get_tts_settings(
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """Get TTS settings including default provider and available providers."""
    import os
    from backend.db.models import SystemSettings
    from sqlalchemy import select

    # Try to get default from database first, then fallback to env var, then "edge"
    setting_key = "audio.default_tts_provider"
    result = await session.execute(
        select(SystemSettings).where(SystemSettings.key == setting_key)
    )
    setting = result.scalar_one_or_none()
    default_provider = setting.value if setting else os.getenv("DEFAULT_TTS_PROVIDER", "edge")

    # Build list of available providers with their status
    available_providers = [
        {
            "id": "edge",
            "name": "Microsoft Edge TTS",
            "description": "Free high-quality voices from Microsoft",
            "requires_api_key": False,
            "is_available": True,  # Edge TTS is always available (uses internet)
            "cost": "Free",
        },
        {
            "id": "openai",
            "name": "OpenAI TTS",
            "description": "High quality, natural sounding voices",
            "requires_api_key": True,
            "is_available": bool(os.getenv("OPENAI_API_KEY")),
            "cost": "$0.015/1K chars (tts-1) or $0.030/1K chars (tts-1-hd)",
        },
        {
            "id": "elevenlabs",
            "name": "ElevenLabs",
            "description": "Premium voices with voice cloning",
            "requires_api_key": True,
            "is_available": bool(os.getenv("ELEVENLABS_API_KEY")),
            "cost": "Varies by plan",
        },
        {
            "id": "coqui",
            "name": "Coqui TTS (Local)",
            "description": "Self-hosted, runs locally with no external dependencies",
            "requires_api_key": False,
            "is_available": _check_coqui_available(),
            "cost": "Free (requires GPU for best performance)",
        },
    ]

    return TTSSettingsResponse(
        default_provider=default_provider,
        available_providers=available_providers,
    )


@router.put("/settings/tts")
async def update_tts_settings(
    request: TTSSettingsUpdateRequest,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """Update TTS settings (default provider)."""
    from backend.db.models import SystemSettings
    from sqlalchemy import select

    # Validate provider
    valid_providers = ["edge", "openai", "elevenlabs", "coqui"]
    if request.default_provider not in valid_providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}",
        )

    # Update or create the setting
    setting_key = "audio.default_tts_provider"
    result = await session.execute(
        select(SystemSettings).where(SystemSettings.key == setting_key)
    )
    setting = result.scalar_one_or_none()

    if setting:
        setting.value = request.default_provider
    else:
        setting = SystemSettings(key=setting_key, value=request.default_provider)
        session.add(setting)

    await session.commit()

    return {
        "message": "TTS settings updated successfully",
        "default_provider": request.default_provider,
    }


def _check_coqui_available() -> bool:
    """Check if Coqui TTS is installed."""
    try:
        import TTS
        return True
    except ImportError:
        return False


@router.get("/settings/tts/coqui/models", response_model=List[CoquiModelInfo])
async def list_coqui_models(
    current_user: dict = Depends(get_current_user),
):
    """List available and installed Coqui TTS models."""
    if not _check_coqui_available():
        # Return empty list if Coqui is not installed (not an error)
        return []

    try:
        from TTS.api import TTS

        # Get list of available models
        tts = TTS()
        available_models = tts.list_models()

        # Common English TTS models
        recommended_models = [
            {
                "name": "tts_models/en/ljspeech/tacotron2-DDC",
                "language": "English",
                "description": "LJSpeech Tacotron2 - Good quality, fast",
                "size_mb": 150,
            },
            {
                "name": "tts_models/en/ljspeech/vits",
                "language": "English",
                "description": "LJSpeech VITS - High quality",
                "size_mb": 120,
            },
            {
                "name": "tts_models/en/vctk/vits",
                "language": "English",
                "description": "VCTK VITS - Multi-speaker",
                "size_mb": 150,
            },
            {
                "name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "language": "Multilingual",
                "description": "XTTS v2 - Best quality, voice cloning",
                "size_mb": 1800,
            },
        ]

        # Check which models are installed
        import os
        from pathlib import Path

        tts_home = Path.home() / ".local" / "share" / "tts"

        models = []
        for model_info in recommended_models:
            model_name = model_info["name"]
            # Check if model directory exists
            model_path = tts_home / model_name.replace("/", "--")
            is_installed = model_path.exists() if tts_home.exists() else False

            models.append(CoquiModelInfo(
                name=model_name,
                language=model_info["language"],
                description=model_info["description"],
                is_installed=is_installed,
                size_mb=model_info["size_mb"],
            ))

        return models

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list models: {str(e)}")


@router.post("/settings/tts/coqui/models/download", response_model=CoquiModelActionResponse)
async def download_coqui_model(
    model_name: str = Query(..., description="Model name to download"),
    current_user: dict = Depends(get_current_user),
):
    """Download a Coqui TTS model."""
    if not _check_coqui_available():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coqui TTS is not installed. Install with: pip install TTS",
        )

    try:
        from TTS.api import TTS

        # This will download the model if not already present
        tts = TTS(model_name=model_name, progress_bar=True)

        return CoquiModelActionResponse(
            success=True,
            message=f"Model '{model_name}' downloaded successfully",
            model_name=model_name,
        )

    except Exception as e:
        return CoquiModelActionResponse(
            success=False,
            message=f"Failed to download model: {str(e)}",
            model_name=model_name,
        )


@router.delete("/settings/tts/coqui/models/{model_name:path}", response_model=CoquiModelActionResponse)
async def delete_coqui_model(
    model_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a downloaded Coqui TTS model."""
    import shutil
    from pathlib import Path

    try:
        tts_home = Path.home() / ".local" / "share" / "tts"
        model_path = tts_home / model_name.replace("/", "--")

        if not model_path.exists():
            return CoquiModelActionResponse(
                success=False,
                message=f"Model '{model_name}' is not installed",
                model_name=model_name,
            )

        shutil.rmtree(model_path)

        return CoquiModelActionResponse(
            success=True,
            message=f"Model '{model_name}' deleted successfully",
            model_name=model_name,
        )

    except Exception as e:
        return CoquiModelActionResponse(
            success=False,
            message=f"Failed to delete model: {str(e)}",
            model_name=model_name,
        )


@router.get("/settings/tts/providers")
async def list_tts_providers(
    current_user: dict = Depends(get_current_user),
):
    """List all TTS providers with their availability status."""
    import os

    providers = []

    # Edge TTS (always available)
    providers.append({
        "id": "edge",
        "name": "Microsoft Edge TTS",
        "status": "available",
        "requires_setup": False,
        "setup_instructions": None,
    })

    # OpenAI TTS
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    providers.append({
        "id": "openai",
        "name": "OpenAI TTS",
        "status": "available" if has_openai else "needs_api_key",
        "requires_setup": not has_openai,
        "setup_instructions": "Set OPENAI_API_KEY environment variable" if not has_openai else None,
    })

    # ElevenLabs
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY"))
    providers.append({
        "id": "elevenlabs",
        "name": "ElevenLabs",
        "status": "available" if has_elevenlabs else "needs_api_key",
        "requires_setup": not has_elevenlabs,
        "setup_instructions": "Set ELEVENLABS_API_KEY environment variable" if not has_elevenlabs else None,
    })

    # Local Coqui TTS
    has_coqui = _check_coqui_available()
    providers.append({
        "id": "local",
        "name": "Local (Coqui TTS)",
        "status": "available" if has_coqui else "not_installed",
        "requires_setup": not has_coqui,
        "setup_instructions": "Install with: pip install TTS" if not has_coqui else None,
    })

    # PHASE 45: Ultra-Fast TTS providers
    # Murf Falcon (55ms latency)
    has_murf = bool(os.getenv("MURF_API_KEY"))
    providers.append({
        "id": "murf",
        "name": "Murf Falcon (Ultra-Fast)",
        "status": "available" if has_murf else "needs_api_key",
        "requires_setup": not has_murf,
        "setup_instructions": "Set MURF_API_KEY environment variable" if not has_murf else None,
        "features": ["55ms latency", "$0.01/min", "Ultra-low latency"],
    })

    # Smallest.ai Lightning (RTF 0.01)
    has_smallest = bool(os.getenv("SMALLEST_API_KEY"))
    providers.append({
        "id": "smallest",
        "name": "Smallest.ai Lightning (Fastest)",
        "status": "available" if has_smallest else "needs_api_key",
        "requires_setup": not has_smallest,
        "setup_instructions": "Set SMALLEST_API_KEY environment variable" if not has_smallest else None,
        "features": ["RTF 0.01", "Fastest TTS available", "Sub-100ms synthesis"],
    })

    # Fish Speech (ELO 1339)
    has_fish = bool(os.getenv("FISH_API_KEY"))
    providers.append({
        "id": "fish_speech",
        "name": "Fish Speech 1.5",
        "status": "available" if has_fish else "needs_api_key",
        "requires_setup": not has_fish,
        "setup_instructions": "Set FISH_API_KEY environment variable" if not has_fish else None,
        "features": ["ELO 1339", "DualAR architecture", "High quality"],
    })

    return {"providers": providers}


# =============================================================================
# PHASE 45: Ultra-Fast TTS Endpoints
# =============================================================================

class UltraFastTTSRequest(BaseModel):
    """Request for ultra-fast TTS synthesis."""
    text: str = Field(..., description="Text to synthesize", max_length=5000)
    provider: str = Field(default="smallest", description="Ultra-fast TTS provider: murf, smallest, fish_speech")
    voice_id: Optional[str] = Field(None, description="Voice ID for the provider")
    stream: bool = Field(default=False, description="Enable streaming response")


class UltraFastTTSResponse(BaseModel):
    """Response from ultra-fast TTS synthesis."""
    audio_url: Optional[str] = None
    audio_data_base64: Optional[str] = None
    latency_ms: float
    provider: str
    sample_rate: int
    duration_estimate_ms: Optional[float] = None


@router.post("/ultra-fast/synthesize", response_model=UltraFastTTSResponse)
async def ultra_fast_tts_synthesize(
    request: UltraFastTTSRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Synthesize text using ultra-fast TTS providers (Phase 45).

    Providers:
    - murf: Murf Falcon (55ms latency, $0.01/min)
    - smallest: Smallest.ai Lightning (RTF 0.01, fastest available)
    - fish_speech: Fish Speech 1.5 (ELO 1339, high quality)
    """
    import base64
    import time
    from backend.core.config import settings

    # Check if ultra-fast TTS is enabled
    if not getattr(settings, "ENABLE_ULTRA_FAST_TTS", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ultra-fast TTS is not enabled. Set ENABLE_ULTRA_FAST_TTS=true in config.",
        )

    start_time = time.time()

    try:
        from backend.services.audio.ultra_fast_tts import get_ultra_fast_tts, UltraFastTTSProvider

        # Validate provider
        try:
            provider_enum = UltraFastTTSProvider(request.provider)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {request.provider}. Valid: murf, smallest, fish_speech",
            )

        tts_service = await get_ultra_fast_tts(provider=provider_enum)

        # Synthesize
        audio_data = await tts_service.synthesize(
            text=request.text,
            voice_id=request.voice_id,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Encode audio as base64
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        # Estimate duration (rough: ~150 words/min at 24kHz)
        word_count = len(request.text.split())
        duration_estimate_ms = (word_count / 150) * 60 * 1000

        return UltraFastTTSResponse(
            audio_data_base64=audio_base64,
            latency_ms=latency_ms,
            provider=request.provider,
            sample_rate=24000,
            duration_estimate_ms=duration_estimate_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"TTS synthesis failed: {str(e)}")


@router.post("/ultra-fast/stream")
async def ultra_fast_tts_stream(
    request: UltraFastTTSRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Stream ultra-fast TTS audio in real-time.

    Returns audio chunks as they are generated for minimal latency.
    """
    from backend.core.config import settings

    if not getattr(settings, "ENABLE_ULTRA_FAST_TTS", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ultra-fast TTS is not enabled. Set ENABLE_ULTRA_FAST_TTS=true in config.",
        )

    async def generate_audio_stream():
        try:
            from backend.services.audio.ultra_fast_tts import get_ultra_fast_tts, UltraFastTTSProvider

            provider_enum = UltraFastTTSProvider(request.provider)
            tts_service = await get_ultra_fast_tts(provider=provider_enum)

            async for chunk in tts_service.stream(
                text=request.text,
                voice_id=request.voice_id,
            ):
                yield chunk.audio_data

        except Exception as e:
            # Log error but don't break the stream
            import structlog
            logger = structlog.get_logger(__name__)
            logger.error("Stream error", error=str(e))

    return StreamingResponse(
        generate_audio_stream(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-TTS-Provider": request.provider,
        },
    )


@router.get("/ultra-fast/providers")
async def list_ultra_fast_providers(
    current_user: dict = Depends(get_current_user),
):
    """List available ultra-fast TTS providers with status."""
    import os
    from backend.core.config import settings

    enabled = getattr(settings, "ENABLE_ULTRA_FAST_TTS", False)

    providers = []

    # Murf Falcon
    has_murf = bool(os.getenv("MURF_API_KEY"))
    providers.append({
        "id": "murf",
        "name": "Murf Falcon",
        "available": enabled and has_murf,
        "latency_ms": 55,
        "cost_per_min": 0.01,
        "features": ["Ultra-low latency", "Natural voices", "130ms TTFA"],
    })

    # Smallest.ai
    has_smallest = bool(os.getenv("SMALLEST_API_KEY"))
    providers.append({
        "id": "smallest",
        "name": "Smallest.ai Lightning",
        "available": enabled and has_smallest,
        "latency_ms": 30,
        "cost_per_min": 0.02,
        "features": ["Fastest TTS", "RTF 0.01", "Sub-100ms synthesis"],
    })

    # Fish Speech
    has_fish = bool(os.getenv("FISH_API_KEY"))
    providers.append({
        "id": "fish_speech",
        "name": "Fish Speech 1.5",
        "available": enabled and has_fish,
        "latency_ms": 100,
        "cost_per_min": 0.01,
        "features": ["High quality", "ELO 1339", "DualAR architecture"],
    })

    return {
        "enabled": enabled,
        "providers": providers,
        "default_provider": getattr(settings, "ULTRA_FAST_TTS_PROVIDER", "smallest"),
    }
