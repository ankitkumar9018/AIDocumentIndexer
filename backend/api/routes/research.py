"""
AIDocumentIndexer - Deep Research API Routes
=============================================

Endpoints for multi-LLM verification and fact-checking research.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import json

from sqlalchemy import select, func, and_

from backend.services.llm import LLMFactory
from backend.services.llm_provider import LLMProviderService
from backend.services.rag import RAGService, get_rag_service_dependency
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.db.models import ResearchResult, ResearchStatus
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class VerificationStep(BaseModel):
    """A single verification step result."""
    id: str
    model: str
    status: str = "completed"
    claim: str
    verdict: str  # "supported", "contradicted", "uncertain"
    confidence: float
    reasoning: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: Optional[int] = None


class ResearchRequest(BaseModel):
    """Request for deep research verification."""
    query: str = Field(..., description="Research question or claim to verify")
    provider_ids: List[str] = Field(default_factory=list, description="Provider IDs to use for verification")
    verification_rounds: int = Field(default=3, ge=1, le=5, description="Number of verification rounds")
    use_multiple_llms: bool = Field(default=True, description="Use multiple LLMs for cross-verification")
    include_sources: bool = Field(default=True, description="Include source citations")


class ResearchResponse(BaseModel):
    """Response from deep research verification."""
    id: str
    query: str
    final_answer: str
    overall_confidence: float
    verification_steps: List[VerificationStep]
    sources: List[Dict[str, Any]]
    has_conflicts: bool
    total_time_ms: int
    models_used: List[str]


# =============================================================================
# Research Prompts
# =============================================================================

VERIFICATION_PROMPT = """You are a fact-checking research assistant. Your task is to verify the following claim or answer the research question.

**Research Question/Claim:**
{query}

**Context from Knowledge Base:**
{context}

Analyze the information carefully and provide:
1. Your verdict: Is the claim SUPPORTED, CONTRADICTED, or UNCERTAIN based on the evidence?
2. Your confidence level (0.0 to 1.0)
3. Your reasoning explaining why you reached this conclusion
4. Key evidence that supports your verdict

Respond in JSON format:
{{
  "verdict": "supported|contradicted|uncertain",
  "confidence": 0.0-1.0,
  "reasoning": "Your detailed reasoning",
  "key_evidence": ["evidence point 1", "evidence point 2"],
  "claim_identified": "The specific claim you verified"
}}"""


SYNTHESIS_PROMPT = """You are a research synthesis expert. Multiple AI models have analyzed the following research question:

**Research Question:**
{query}

**Verification Results from Multiple Models:**
{verification_results}

Synthesize these results into a final, authoritative answer:

1. Consider the consensus and conflicts between models
2. Weight higher-confidence responses more heavily
3. Note any areas of disagreement

Provide a comprehensive final answer that:
- States the conclusion clearly
- Acknowledges certainty/uncertainty
- Summarizes the key supporting evidence

Respond in JSON format:
{{
  "final_answer": "Your synthesized answer",
  "overall_confidence": 0.0-1.0,
  "has_conflicts": true|false,
  "conflict_summary": "Description of any conflicts (if has_conflicts is true)",
  "consensus_points": ["point1", "point2"]
}}"""


# =============================================================================
# Helper Functions
# =============================================================================

async def run_verification_step(
    query: str,
    context: str,
    provider_type: str,
    model_name: str,
    step_id: str,
) -> VerificationStep:
    """Run a single verification step with one LLM."""
    start_time = datetime.utcnow()

    try:
        llm = LLMFactory.get_chat_model(
            provider=provider_type,
            model=model_name,
            temperature=0.3,  # Lower temperature for more consistent fact-checking
            max_tokens=2048,
        )

        prompt = VERIFICATION_PROMPT.format(
            query=query,
            context=context or "No specific context provided. Use your knowledge to assess the claim.",
        )

        response = await llm.ainvoke(prompt)
        output = response.content

        # Parse JSON response
        try:
            output_clean = output.strip()
            if output_clean.startswith("```json"):
                output_clean = output_clean[7:]
            if output_clean.startswith("```"):
                output_clean = output_clean[3:]
            if output_clean.endswith("```"):
                output_clean = output_clean[:-3]

            result = json.loads(output_clean.strip())
        except json.JSONDecodeError:
            # Fallback parsing
            result = {
                "verdict": "uncertain",
                "confidence": 0.5,
                "reasoning": output,
                "key_evidence": [],
                "claim_identified": query,
            }

        duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return VerificationStep(
            id=step_id,
            model=f"{provider_type}/{model_name}",
            status="completed",
            claim=result.get("claim_identified", query),
            verdict=result.get("verdict", "uncertain"),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            sources=[{"type": "model_knowledge", "evidence": e} for e in result.get("key_evidence", [])],
            duration_ms=duration,
        )

    except Exception as e:
        logger.error(f"Verification step failed: {e}")
        return VerificationStep(
            id=step_id,
            model=f"{provider_type}/{model_name}",
            status="failed",
            claim=query,
            verdict="uncertain",
            confidence=0.0,
            reasoning=f"Verification failed: {str(e)}",
            sources=[],
            duration_ms=0,
        )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/verify", response_model=ResearchResponse)
async def verify_research(
    request: ResearchRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    rag_service: RAGService = Depends(get_rag_service_dependency),
):
    """
    Perform deep research verification using multiple LLMs.

    This endpoint:
    1. Retrieves relevant context from the knowledge base
    2. Runs multiple verification rounds with different LLMs
    3. Synthesizes the results into a final answer with confidence scoring
    """
    start_time = datetime.utcnow()
    research_id = str(uuid4())

    logger.info(
        "Starting deep research verification",
        research_id=research_id,
        query=request.query[:100],
        user_id=user.user_id,
        verification_rounds=request.verification_rounds,
        provider_count=len(request.provider_ids),
    )

    # Get providers to use
    providers = []
    if request.provider_ids:
        for provider_id in request.provider_ids:
            try:
                provider = await LLMProviderService.get_provider(db, provider_id)
                if provider and provider.is_active:
                    providers.append({
                        "type": provider.provider_type,
                        "model": provider.default_chat_model or "gpt-4o",
                        "name": provider.name,
                    })
            except Exception as e:
                logger.warning(f"Could not load provider {provider_id}: {e}")

    # Default providers if none specified
    if not providers:
        providers = [
            {"type": "openai", "model": "gpt-4o", "name": "GPT-4o"},
        ]

    # If not using multiple LLMs, use only the first
    if not request.use_multiple_llms:
        providers = providers[:1]

    # Get context from RAG if available
    context = ""
    rag_sources = []
    if request.include_sources:
        try:
            rag_result = await rag_service.query(
                query=request.query,
                top_k=5,
                include_metadata=True,
            )
            if rag_result and rag_result.get("sources"):
                context_parts = []
                for source in rag_result["sources"][:5]:
                    context_parts.append(source.get("content", ""))
                    rag_sources.append({
                        "document": source.get("document_name", "Unknown"),
                        "content": source.get("content", "")[:200] + "...",
                        "score": source.get("score", 0),
                    })
                context = "\n\n---\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    # Run verification steps
    verification_steps = []
    step_count = 0

    for round_num in range(request.verification_rounds):
        for provider in providers:
            step_id = f"step-{round_num}-{provider['name']}"
            step_count += 1

            step = await run_verification_step(
                query=request.query,
                context=context,
                provider_type=provider["type"],
                model_name=provider["model"],
                step_id=step_id,
            )
            verification_steps.append(step)

    # Synthesize final answer
    synthesis_provider = providers[0]

    try:
        llm = LLMFactory.get_chat_model(
            provider=synthesis_provider["type"],
            model=synthesis_provider["model"],
            temperature=0.3,
            max_tokens=2048,
        )

        verification_summary = "\n\n".join([
            f"**Model: {step.model}**\n"
            f"Verdict: {step.verdict} (Confidence: {step.confidence:.2f})\n"
            f"Reasoning: {step.reasoning}"
            for step in verification_steps
        ])

        synthesis_prompt = SYNTHESIS_PROMPT.format(
            query=request.query,
            verification_results=verification_summary,
        )

        response = await llm.ainvoke(synthesis_prompt)
        output = response.content

        # Parse synthesis response
        try:
            output_clean = output.strip()
            if output_clean.startswith("```json"):
                output_clean = output_clean[7:]
            if output_clean.startswith("```"):
                output_clean = output_clean[3:]
            if output_clean.endswith("```"):
                output_clean = output_clean[:-3]

            synthesis = json.loads(output_clean.strip())
        except json.JSONDecodeError:
            # Calculate from steps
            avg_confidence = sum(s.confidence for s in verification_steps) / len(verification_steps)
            verdicts = [s.verdict for s in verification_steps]
            has_conflicts = len(set(verdicts)) > 1

            synthesis = {
                "final_answer": output,
                "overall_confidence": avg_confidence,
                "has_conflicts": has_conflicts,
            }

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        avg_confidence = sum(s.confidence for s in verification_steps) / len(verification_steps) if verification_steps else 0.5
        synthesis = {
            "final_answer": "Unable to synthesize results. Please review individual verification steps.",
            "overall_confidence": avg_confidence,
            "has_conflicts": False,
        }

    # Calculate total time
    total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    # Compile all sources
    all_sources = rag_sources.copy()
    for step in verification_steps:
        all_sources.extend(step.sources)

    logger.info(
        "Deep research completed",
        research_id=research_id,
        steps_completed=len(verification_steps),
        overall_confidence=synthesis.get("overall_confidence", 0),
        total_time_ms=total_time,
    )

    return ResearchResponse(
        id=research_id,
        query=request.query,
        final_answer=synthesis.get("final_answer", ""),
        overall_confidence=float(synthesis.get("overall_confidence", 0.5)),
        verification_steps=verification_steps,
        sources=all_sources,
        has_conflicts=synthesis.get("has_conflicts", False),
        total_time_ms=total_time,
        models_used=list(set(p["model"] for p in providers)),
    )


@router.post("/verify/stream")
async def verify_research_stream(
    request: ResearchRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Perform deep research verification with streaming updates.

    Returns Server-Sent Events with progress updates for each verification step.
    """
    async def generate_events():
        research_id = str(uuid4())

        yield f"data: {json.dumps({'type': 'start', 'research_id': research_id})}\n\n"

        # Get providers
        providers = []
        if request.provider_ids:
            for provider_id in request.provider_ids:
                try:
                    provider = await LLMProviderService.get_provider(db, provider_id)
                    if provider and provider.is_active:
                        providers.append({
                            "type": provider.provider_type,
                            "model": provider.default_chat_model or "gpt-4o",
                            "name": provider.name,
                        })
                except Exception:
                    pass

        if not providers:
            providers = [{"type": "openai", "model": "gpt-4o", "name": "GPT-4o"}]

        if not request.use_multiple_llms:
            providers = providers[:1]

        # Run verification steps with progress updates
        verification_steps = []
        for round_num in range(request.verification_rounds):
            for provider in providers:
                step_id = f"step-{round_num}-{provider['name']}"

                # Send start event
                yield f"data: {json.dumps({'type': 'step_start', 'step_id': step_id, 'model': provider['name']})}\n\n"

                step = await run_verification_step(
                    query=request.query,
                    context="",
                    provider_type=provider["type"],
                    model_name=provider["model"],
                    step_id=step_id,
                )
                verification_steps.append(step)

                # Send completion event
                yield f"data: {json.dumps({'type': 'step_complete', 'step': step.model_dump()})}\n\n"

        # Send final result
        avg_confidence = sum(s.confidence for s in verification_steps) / len(verification_steps) if verification_steps else 0.5
        verdicts = [s.verdict for s in verification_steps]
        has_conflicts = len(set(verdicts)) > 1

        yield f"data: {json.dumps({'type': 'complete', 'overall_confidence': avg_confidence, 'has_conflicts': has_conflicts})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/history")
async def get_research_history(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    limit: int = 20,
    offset: int = 0,
    starred: Optional[bool] = None,
):
    """
    Get research history for the current user.
    """
    conditions = [ResearchResult.user_id == user.user_id]
    if starred is not None:
        conditions.append(ResearchResult.is_starred == starred)

    # Get total count
    count_query = select(func.count(ResearchResult.id)).where(and_(*conditions))
    total = (await db.execute(count_query)).scalar() or 0

    # Get results
    query = (
        select(ResearchResult)
        .where(and_(*conditions))
        .order_by(ResearchResult.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    results = result.scalars().all()

    return {
        "research_history": [
            {
                "id": str(r.id),
                "query": r.query,
                "status": r.status,
                "summary": r.summary,
                "confidence_score": r.confidence_score,
                "models_used": r.models_used or [],
                "is_starred": r.is_starred,
                "created_at": r.created_at.isoformat(),
            }
            for r in results
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{research_id}")
async def get_research_result(
    research_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get a specific research result by ID.
    """
    from uuid import UUID
    try:
        research_uuid = UUID(research_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid research ID")

    result = await db.execute(
        select(ResearchResult).where(
            ResearchResult.id == research_uuid,
            ResearchResult.user_id == user.user_id,
        )
    )
    research = result.scalar_one_or_none()

    if not research:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research result {research_id} not found"
        )

    return {
        "id": str(research.id),
        "query": research.query,
        "context": research.context,
        "status": research.status,
        "findings": research.findings or [],
        "sources": research.sources or [],
        "verification": research.verification,
        "summary": research.summary,
        "confidence_score": research.confidence_score,
        "models_used": research.models_used or [],
        "execution_time_ms": research.execution_time_ms,
        "is_starred": research.is_starred,
        "created_at": research.created_at.isoformat(),
        "updated_at": research.updated_at.isoformat(),
    }


@router.patch("/{research_id}/star")
async def toggle_research_star(
    research_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Toggle research result starred status."""
    from uuid import UUID
    try:
        research_uuid = UUID(research_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid research ID")

    result = await db.execute(
        select(ResearchResult).where(
            ResearchResult.id == research_uuid,
            ResearchResult.user_id == user.user_id,
        )
    )
    research = result.scalar_one_or_none()

    if not research:
        raise HTTPException(status_code=404, detail="Research result not found")

    research.is_starred = not research.is_starred
    await db.commit()

    return {"id": str(research.id), "is_starred": research.is_starred}


@router.delete("/{research_id}")
async def delete_research_result(
    research_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Delete a research result."""
    from uuid import UUID
    try:
        research_uuid = UUID(research_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid research ID")

    result = await db.execute(
        select(ResearchResult).where(
            ResearchResult.id == research_uuid,
            ResearchResult.user_id == user.user_id,
        )
    )
    research = result.scalar_one_or_none()

    if not research:
        raise HTTPException(status_code=404, detail="Research result not found")

    await db.delete(research)
    await db.commit()

    return {"message": "Research result deleted successfully"}
