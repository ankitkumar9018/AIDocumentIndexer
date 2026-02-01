"""
AIDocumentIndexer - Ensemble Voting API Routes
===============================================

Endpoints for multi-model ensemble voting to improve accuracy.

Endpoints:
- POST /query - Query multiple models and vote on answer
- GET /strategies - List available voting strategies
- POST /compare - Compare answers from different models
"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.ensemble_voting import (
    get_ensemble_voting_service,
    EnsembleVotingService,
    VotingStrategy,
)
from backend.services.llm_provider import PROVIDER_TYPES, LLMProviderService
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.db.models import LLMProvider
from backend.api.middleware.auth import AuthenticatedUser
from sqlalchemy import select

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class ModelConfig(BaseModel):
    """Configuration for a single model."""
    provider: str = Field(..., description="Provider (openai, anthropic, ollama)")
    model: str = Field(..., description="Model name")


class EnsembleQueryRequest(BaseModel):
    """Request for ensemble query."""
    question: str = Field(..., description="Question to answer")
    context: Optional[str] = Field("", description="Optional context for RAG")
    models: Optional[List[ModelConfig]] = Field(
        None,
        description="Models to use (defaults to system config)"
    )
    strategy: str = Field(
        default="confidence",
        description="Voting strategy: majority, confidence, consensus, best_of_n, synthesis"
    )
    min_agreement: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="Minimum agreement threshold for consensus"
    )


class ModelAnswerResponse(BaseModel):
    """Response from a single model."""
    model: str
    provider: str
    answer: str
    confidence: float
    reasoning: str
    latency_ms: int


class EnsembleQueryResponse(BaseModel):
    """Response from ensemble query."""
    question: str
    final_answer: str
    confidence: float
    strategy: str
    agreement_level: str
    disagreements: List[str]
    model_answers: List[ModelAnswerResponse]
    total_latency_ms: int
    models_used: List[str]


class CompareRequest(BaseModel):
    """Request to compare model answers."""
    question: str = Field(..., description="Question to compare answers for")
    answer1: str = Field(..., description="First answer")
    answer2: str = Field(..., description="Second answer")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/query", response_model=EnsembleQueryResponse)
async def ensemble_query(
    request: EnsembleQueryRequest,
    user: AuthenticatedUser,
):
    """
    Query multiple models and vote on the best answer.

    This endpoint:
    1. Sends the question to multiple LLMs in parallel
    2. Collects and analyzes their answers
    3. Detects agreement/disagreement
    4. Votes on the best answer using the chosen strategy

    Useful for:
    - Fact-checking (compare multiple model opinions)
    - Reducing hallucinations (consensus detection)
    - Complex questions (synthesis from multiple perspectives)
    """
    service = get_ensemble_voting_service()

    # Parse strategy
    try:
        strategy = VotingStrategy(request.strategy)
    except ValueError:
        strategy = VotingStrategy.CONFIDENCE

    # Parse models
    models = None
    if request.models:
        models = [{"provider": m.provider, "model": m.model} for m in request.models]

    logger.info(
        "Ensemble query request",
        strategy=strategy.value,
        model_count=len(models) if models else "default",
        user_id=user.user_id,
    )

    try:
        result = await service.query(
            question=request.question,
            context=request.context or "",
            models=models,
            strategy=strategy,
            min_agreement=request.min_agreement,
        )

        return EnsembleQueryResponse(
            question=result.query,
            final_answer=result.final_answer,
            confidence=result.confidence,
            strategy=result.strategy.value,
            agreement_level=result.agreement_level,
            disagreements=result.disagreements,
            model_answers=[
                ModelAnswerResponse(
                    model=a.model,
                    provider=a.provider,
                    answer=a.answer,
                    confidence=a.confidence,
                    reasoning=a.reasoning,
                    latency_ms=a.latency_ms,
                )
                for a in result.model_answers
            ],
            total_latency_ms=result.total_latency_ms,
            models_used=result.models_used,
        )

    except Exception as e:
        logger.error("Ensemble query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble query failed: {str(e)}",
        )


@router.get("/strategies")
async def get_voting_strategies(
    user: AuthenticatedUser,
):
    """
    Get available voting strategies and their descriptions.
    """
    return {
        "strategies": [
            {
                "id": "majority",
                "name": "Majority Voting",
                "description": "Pick the answer chosen by most models",
                "best_for": "Simple factual questions with clear answers",
            },
            {
                "id": "confidence",
                "name": "Confidence Weighted",
                "description": "Weight answers by model confidence scores",
                "best_for": "Questions where some models may be more confident",
            },
            {
                "id": "consensus",
                "name": "Consensus Required",
                "description": "Require agreement, synthesize if disagreement",
                "best_for": "Critical questions where accuracy is paramount",
            },
            {
                "id": "best_of_n",
                "name": "Best of N",
                "description": "Pick the answer with highest confidence",
                "best_for": "Quick decisions when one model may be clearly better",
            },
            {
                "id": "synthesis",
                "name": "AI Synthesis",
                "description": "Use an LLM to synthesize all answers into one",
                "best_for": "Complex questions benefiting from multiple perspectives",
            },
        ],
        "default_models": [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            {"provider": "ollama", "model": "llama3.2"},
        ],
    }


@router.post("/compare")
async def compare_answers(
    request: CompareRequest,
    user: AuthenticatedUser,
):
    """
    Compare two answers to determine if they agree.

    Returns:
    - agree: yes/no/partial
    - explanation: Why they agree or disagree
    """
    service = get_ensemble_voting_service()

    agrees, explanation = await service._check_agreement(
        question=request.question,
        answer1=request.answer1,
        answer2=request.answer2,
    )

    return {
        "question": request.question,
        "agree": agrees,
        "explanation": explanation,
    }


@router.get("/models")
async def get_available_models(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get list of models available for ensemble queries.

    Returns models from:
    1. Configured providers in the database (with API keys)
    2. Default provider type configurations
    """
    # Get active providers from database
    query = select(LLMProvider).where(LLMProvider.is_active == True)
    result = await db.execute(query)
    db_providers = result.scalars().all()

    configured_providers = set()
    models_list = []

    # Add models from configured providers
    for provider in db_providers:
        configured_providers.add(provider.provider_type)
        type_config = PROVIDER_TYPES.get(provider.provider_type, {})
        chat_models = type_config.get("chat_models", [])

        # Handle dynamic models (like Ollama)
        if chat_models == "dynamic":
            chat_models = ["Check Ollama for available models"]

        models_list.append({
            "provider": provider.provider_type,
            "provider_name": type_config.get("name", provider.provider_type),
            "models": chat_models if isinstance(chat_models, list) else [],
            "available": True,
            "configured": True,
            "default_model": provider.default_chat_model,
        })

    # Add unconfigured providers as unavailable
    for provider_type, config in PROVIDER_TYPES.items():
        if provider_type not in configured_providers:
            chat_models = config.get("chat_models", [])
            if chat_models == "dynamic":
                chat_models = []

            models_list.append({
                "provider": provider_type,
                "provider_name": config.get("name", provider_type),
                "models": chat_models if isinstance(chat_models, list) else [],
                "available": False,
                "configured": False,
                "default_model": config.get("default_chat_model"),
            })

    return {"models": models_list}
