"""
AIDocumentIndexer - Multi-LLM Collaboration API Routes
=======================================================

Endpoints for multi-LLM collaboration workflows.
Multiple LLMs work together for higher quality output.
"""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import AuthenticatedUser
from backend.services.llm import llm_config
from backend.services.collaboration import (
    CollaborationService,
    CollaborationSession,
    CollaborationConfig,
    CollaborationMode,
    CollaborationStatus,
    ModelConfig,
    get_collaboration_service,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class ModelConfigRequest(BaseModel):
    """Configuration for a specific model."""
    provider: Optional[str] = Field(default=None, description="LLM provider (auto-detected from settings if not specified)")
    model: Optional[str] = Field(default=None, description="Model name (uses provider default if not specified)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(default=2000, ge=100, le=8000, description="Max tokens to generate")


class CreateSessionRequest(BaseModel):
    """Request to create a collaboration session."""
    prompt: str = Field(..., min_length=10, description="The prompt/request for generation")
    context: Optional[str] = Field(None, description="Optional context (from RAG, documents, etc.)")
    mode: str = Field(default="review", description="Collaboration mode: single, review, full, debate")
    generator: Optional[ModelConfigRequest] = None
    critic: Optional[ModelConfigRequest] = None
    synthesizer: Optional[ModelConfigRequest] = None
    max_iterations: int = Field(default=2, ge=1, le=5, description="Max revision cycles")


class GenerationResultResponse(BaseModel):
    """Result from a generation step."""
    role: str
    model: str
    content: str
    timestamp: datetime


class CritiqueResultResponse(BaseModel):
    """Result from a critique step."""
    overall_score: float
    should_revise: bool
    raw_content: str


class SessionResponse(BaseModel):
    """Collaboration session response."""
    id: str
    user_id: str
    status: str
    mode: str
    prompt: str
    has_context: bool
    initial_generation: Optional[GenerationResultResponse]
    critiques_count: int
    revisions_count: int
    final_synthesis: Optional[GenerationResultResponse]
    final_output: Optional[str]
    total_tokens: int
    total_cost: float
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


class SessionListResponse(BaseModel):
    """List of collaboration sessions."""
    sessions: List[SessionResponse]
    total: int


class CostEstimateResponse(BaseModel):
    """Cost estimate for collaboration."""
    estimated_cost: float
    currency: str
    breakdown: List[dict]
    mode: str
    models: dict


# =============================================================================
# Helper Functions
# =============================================================================

def session_to_response(session: CollaborationSession) -> SessionResponse:
    """Convert CollaborationSession to response model."""
    initial_gen = None
    if session.initial_generation:
        initial_gen = GenerationResultResponse(
            role=session.initial_generation.role.value,
            model=session.initial_generation.model,
            content=session.initial_generation.content,
            timestamp=session.initial_generation.timestamp,
        )

    final_synth = None
    if session.final_synthesis:
        final_synth = GenerationResultResponse(
            role=session.final_synthesis.role.value,
            model=session.final_synthesis.model,
            content=session.final_synthesis.content,
            timestamp=session.final_synthesis.timestamp,
        )

    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        status=session.status.value,
        mode=session.config.mode.value,
        prompt=session.prompt,
        has_context=session.context is not None,
        initial_generation=initial_gen,
        critiques_count=len(session.critiques),
        revisions_count=len(session.revisions),
        final_synthesis=final_synth,
        final_output=session.final_output,
        total_tokens=session.total_tokens,
        total_cost=session.total_cost,
        created_at=session.created_at,
        completed_at=session.completed_at,
        error_message=session.error_message,
    )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_collaboration_session(
    request: CreateSessionRequest,
    user: AuthenticatedUser,
):
    """
    Create a new multi-LLM collaboration session.

    Collaboration modes:
    - single: No collaboration, single LLM (fastest, cheapest)
    - review: Generator + Critic + Synthesizer (recommended)
    - full: Generator + Critic + Revise cycles + Synthesizer (highest quality)
    - debate: Multiple generators debate, synthesizer decides (experimental)
    """
    logger.info(
        "Creating collaboration session",
        user_id=user.user_id,
        mode=request.mode,
    )

    # Parse mode
    try:
        mode = CollaborationMode(request.mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode: {request.mode}. Valid modes: single, review, full, debate",
        )

    # Build config — use system default provider, not hardcoded "openai"
    _default_provider = llm_config.default_provider

    generator_config = ModelConfig(
        provider=request.generator.provider if request.generator else _default_provider,
        model=request.generator.model if request.generator else None,
        temperature=request.generator.temperature if request.generator else 0.7,
        max_tokens=request.generator.max_tokens if request.generator else 2000,
    )

    critic_config = ModelConfig(
        provider=request.critic.provider if request.critic else _default_provider,
        model=request.critic.model if request.critic else None,
        temperature=request.critic.temperature if request.critic else 0.3,
        max_tokens=request.critic.max_tokens if request.critic else 2000,
    )

    synthesizer_config = ModelConfig(
        provider=request.synthesizer.provider if request.synthesizer else _default_provider,
        model=request.synthesizer.model if request.synthesizer else None,
        temperature=request.synthesizer.temperature if request.synthesizer else 0.5,
        max_tokens=request.synthesizer.max_tokens if request.synthesizer else 2000,
    )

    config = CollaborationConfig(
        mode=mode,
        generator=generator_config,
        critic=critic_config,
        synthesizer=synthesizer_config,
        max_iterations=request.max_iterations,
    )

    service = get_collaboration_service()

    session = await service.create_session(
        user_id=user.user_id,
        prompt=request.prompt,
        context=request.context,
        config=config,
    )

    return session_to_response(session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_collaboration_sessions(
    user: AuthenticatedUser,
    status_filter: Optional[str] = None,
):
    """
    List all collaboration sessions for the current user.
    """
    service = get_collaboration_service()

    # Parse status filter
    collab_status = None
    if status_filter:
        try:
            collab_status = CollaborationStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    sessions = service.list_sessions(
        user_id=user.user_id,
        status=collab_status,
    )

    return SessionListResponse(
        sessions=[session_to_response(s) for s in sessions],
        total=len(sessions),
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_collaboration_session(
    session_id: str,
    user: AuthenticatedUser,
):
    """
    Get a specific collaboration session.
    """
    service = get_collaboration_service()

    session = service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Verify ownership
    if session.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session",
        )

    return session_to_response(session)


@router.post("/sessions/{session_id}/run", response_model=SessionResponse)
async def run_collaboration(
    session_id: str,
    user: AuthenticatedUser,
):
    """
    Run the collaboration workflow for a session.

    This will execute the multi-LLM collaboration based on the session's mode.
    """
    logger.info("Running collaboration", session_id=session_id)

    service = get_collaboration_service()

    session = service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Verify ownership
    if session.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session",
        )

    # Check if already running or completed
    if session.status not in [CollaborationStatus.PENDING, CollaborationStatus.FAILED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is not in a runnable state: {session.status.value}",
        )

    try:
        session = await service.run_collaboration(session_id)
    except Exception as e:
        logger.error("Collaboration failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Collaboration failed: {str(e)}",
        )

    return session_to_response(session)


@router.get("/sessions/{session_id}/stream")
async def stream_collaboration(
    session_id: str,
    user: AuthenticatedUser,
):
    """
    Stream collaboration progress updates.

    Returns Server-Sent Events with progress updates during collaboration.
    """
    import json

    service = get_collaboration_service()

    session = service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Verify ownership
    if session.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session",
        )

    async def event_generator():
        try:
            async for update in service.stream_collaboration(session_id):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions/{session_id}/critiques")
async def get_session_critiques(
    session_id: str,
    user: AuthenticatedUser,
):
    """
    Get all critiques for a collaboration session.

    Returns detailed critique information including strengths, weaknesses, and suggestions.
    """
    service = get_collaboration_service()

    session = service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Verify ownership
    if session.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session",
        )

    critiques = [
        {
            "index": i,
            "overall_score": c.overall_score,
            "should_revise": c.should_revise,
            "strengths": c.strengths,
            "weaknesses": c.weaknesses,
            "suggestions": c.suggestions,
            "raw_content": c.raw_content,
        }
        for i, c in enumerate(session.critiques)
    ]

    return {
        "session_id": session_id,
        "critiques": critiques,
        "total": len(critiques),
    }


@router.get("/sessions/{session_id}/generations")
async def get_session_generations(
    session_id: str,
    user: AuthenticatedUser,
):
    """
    Get all generations for a collaboration session.

    Returns the initial generation, revisions, and final synthesis.
    """
    service = get_collaboration_service()

    session = service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Verify ownership
    if session.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this session",
        )

    generations = []

    if session.initial_generation:
        generations.append({
            "type": "initial",
            "role": session.initial_generation.role.value,
            "model": session.initial_generation.model,
            "content": session.initial_generation.content,
            "timestamp": session.initial_generation.timestamp.isoformat(),
        })

    for i, rev in enumerate(session.revisions):
        generations.append({
            "type": "revision",
            "index": i + 1,
            "role": rev.role.value,
            "model": rev.model,
            "content": rev.content,
            "timestamp": rev.timestamp.isoformat(),
        })

    if session.final_synthesis:
        generations.append({
            "type": "synthesis",
            "role": session.final_synthesis.role.value,
            "model": session.final_synthesis.model,
            "content": session.final_synthesis.content,
            "timestamp": session.final_synthesis.timestamp.isoformat(),
        })

    return {
        "session_id": session_id,
        "generations": generations,
        "total": len(generations),
    }


@router.post("/estimate-cost", response_model=CostEstimateResponse)
async def estimate_collaboration_cost(
    request: CreateSessionRequest,
    user: AuthenticatedUser,
):
    """
    Estimate the cost of a collaboration session.

    Provides a rough estimate based on the prompt, context, mode, and models.
    """
    service = get_collaboration_service()

    # Parse mode
    try:
        mode = CollaborationMode(request.mode.lower())
    except ValueError:
        mode = CollaborationMode.REVIEW

    # Build config
    config = CollaborationConfig(
        mode=mode,
        generator=ModelConfig(
            model=request.generator.model if request.generator else None,
        ),
        critic=ModelConfig(
            model=request.critic.model if request.critic else None,
        ),
        synthesizer=ModelConfig(
            model=request.synthesizer.model if request.synthesizer else None,
        ),
        max_iterations=request.max_iterations,
    )

    estimate = service.estimate_cost(
        prompt=request.prompt,
        context=request.context,
        config=config,
    )

    return CostEstimateResponse(**estimate)


@router.get("/modes")
async def list_collaboration_modes():
    """
    List all available collaboration modes.
    """
    return {
        "modes": [
            {
                "value": "single",
                "name": "Single",
                "description": "Single LLM, no collaboration. Fastest and cheapest.",
            },
            {
                "value": "review",
                "name": "Review",
                "description": "Generator + Critic + Synthesizer. Recommended for most use cases.",
            },
            {
                "value": "full",
                "name": "Full",
                "description": "Generator + iterative Critic/Revise cycles + Synthesizer. Highest quality.",
            },
            {
                "value": "debate",
                "name": "Debate",
                "description": "Multiple generators debate, synthesizer decides. Experimental.",
            },
        ]
    }


@router.get("/models")
async def list_available_models():
    """
    List all available models for collaboration.
    """
    return {
        "models": [
            {
                "provider": "openai",
                "models": [
                    {"id": "gpt-4", "name": "GPT-4", "description": "Most capable OpenAI model"},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Faster, cheaper GPT-4"},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Fast and affordable"},
                ],
            },
            {
                "provider": "anthropic",
                "models": [
                    {"id": "claude-2", "name": "Claude 2", "description": "Anthropic's flagship model"},
                    {"id": "claude-instant", "name": "Claude Instant", "description": "Faster Claude variant"},
                ],
            },
            {
                "provider": "ollama",
                "models": [
                    {"id": "llama2", "name": "Llama 2", "description": "Meta's open source model"},
                    {"id": "mistral", "name": "Mistral", "description": "Efficient open source model"},
                    {"id": "codellama", "name": "Code Llama", "description": "Code-specialized Llama"},
                ],
            },
        ]
    }
