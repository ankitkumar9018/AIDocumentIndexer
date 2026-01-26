"""
AIDocumentIndexer - DSPy Prompt Optimization API Routes
=======================================================

Phase 93: Admin API for DSPy prompt optimization.

Endpoints:
- POST /optimize: Trigger optimization for a signature
- GET /status/{job_id}: Check optimization job status
- GET /examples: List training examples
- POST /examples: Add manual training example
- DELETE /examples/{id}: Remove training example
- GET /example-counts: Get counts by source
- POST /deploy/{job_id}: Deploy optimization result
"""

import uuid
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import get_session
from backend.api.routes.admin import get_current_admin_user
from backend.db.models import DSPyTrainingExample, DSPyOptimizationJob

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/dspy", tags=["DSPy Optimization"])


# =============================================================================
# Request/Response Models
# =============================================================================

class OptimizeRequest(BaseModel):
    signature: str = Field(
        ..., description="Signature to optimize: rag_answer, query_expansion, query_decomposition, react_reasoning, answer_synthesis"
    )
    optimizer: str = Field(
        default="bootstrap_few_shot",
        description="Optimizer type: bootstrap_few_shot or miprov2",
    )
    max_examples: int = Field(default=50, ge=5, le=1000)
    agent_id: Optional[str] = Field(default=None, description="Agent ID for prompt version export")


class AddExampleRequest(BaseModel):
    signature_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ExampleResponse(BaseModel):
    id: str
    signature_name: str
    inputs: Optional[Dict[str, Any]]
    outputs: Optional[Dict[str, Any]]
    source: str
    quality_score: float
    is_active: bool
    created_at: Optional[str]


class JobResponse(BaseModel):
    id: str
    signature_name: str
    optimizer_type: str
    status: str
    num_train_examples: int
    num_dev_examples: int
    baseline_score: Optional[float]
    optimized_score: Optional[float]
    improvement_pct: Optional[float]
    error_message: Optional[str]
    prompt_version_id: Optional[str]
    created_at: Optional[str]


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/optimize", response_model=JobResponse)
async def trigger_optimization(
    request: OptimizeRequest,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """
    Trigger DSPy prompt optimization for a signature.

    Creates a background optimization job that:
    1. Collects training examples from DB
    2. Runs BootstrapFewShot or MIPROv2 optimizer
    3. Evaluates improvement
    4. Optionally exports to prompt version manager
    """
    valid_signatures = [
        "rag_answer", "query_expansion", "query_decomposition",
        "react_reasoning", "answer_synthesis",
    ]
    if request.signature not in valid_signatures:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid signature: {request.signature}. Valid: {valid_signatures}",
        )

    # Create job record
    job = DSPyOptimizationJob(
        id=uuid.uuid4(),
        signature_name=request.signature,
        optimizer_type=request.optimizer,
        status="running",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Run optimization (in-request for now; could be background task)
    try:
        from backend.services.prompt_optimization.dspy_optimizer import (
            DSPyOptimizer,
        )

        optimizer = DSPyOptimizer(db)
        result = await optimizer.optimize(
            signature_name=request.signature,
            optimizer_type=request.optimizer,
            max_examples=request.max_examples,
            agent_id=request.agent_id,
        )

        # Update job with results
        job.status = "completed" if not result.error else "failed"
        job.num_train_examples = result.num_training_examples
        job.num_dev_examples = result.num_dev_examples
        job.baseline_score = result.baseline_score
        job.optimized_score = result.optimized_score
        job.improvement_pct = result.improvement_pct
        job.error_message = result.error
        if result.compiled_demos or result.compiled_instructions:
            job.compiled_state = {
                "instructions": result.compiled_instructions,
                "demos": result.compiled_demos,
            }
        if result.prompt_version_id:
            job.prompt_version_id = uuid.UUID(result.prompt_version_id)

        await db.commit()
        await db.refresh(job)

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        await db.commit()
        await db.refresh(job)
        logger.error("DSPy optimization failed", error=str(e), job_id=str(job.id))

    return JobResponse(
        id=str(job.id),
        signature_name=job.signature_name,
        optimizer_type=job.optimizer_type,
        status=job.status,
        num_train_examples=job.num_train_examples,
        num_dev_examples=job.num_dev_examples,
        baseline_score=job.baseline_score,
        optimized_score=job.optimized_score,
        improvement_pct=job.improvement_pct,
        error_message=job.error_message,
        prompt_version_id=str(job.prompt_version_id) if job.prompt_version_id else None,
        created_at=str(job.created_at) if job.created_at else None,
    )


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """Get status of a DSPy optimization job."""
    result = await db.execute(
        select(DSPyOptimizationJob).where(
            DSPyOptimizationJob.id == uuid.UUID(job_id)
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        id=str(job.id),
        signature_name=job.signature_name,
        optimizer_type=job.optimizer_type,
        status=job.status,
        num_train_examples=job.num_train_examples,
        num_dev_examples=job.num_dev_examples,
        baseline_score=job.baseline_score,
        optimized_score=job.optimized_score,
        improvement_pct=job.improvement_pct,
        error_message=job.error_message,
        prompt_version_id=str(job.prompt_version_id) if job.prompt_version_id else None,
        created_at=str(job.created_at) if job.created_at else None,
    )


@router.get("/jobs")
async def list_jobs(
    signature: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """List DSPy optimization jobs."""
    query = select(DSPyOptimizationJob).order_by(
        DSPyOptimizationJob.created_at.desc()
    )
    if signature:
        query = query.where(DSPyOptimizationJob.signature_name == signature)
    query = query.limit(limit)

    result = await db.execute(query)
    jobs = result.scalars().all()

    return [
        JobResponse(
            id=str(j.id),
            signature_name=j.signature_name,
            optimizer_type=j.optimizer_type,
            status=j.status,
            num_train_examples=j.num_train_examples,
            num_dev_examples=j.num_dev_examples,
            baseline_score=j.baseline_score,
            optimized_score=j.optimized_score,
            improvement_pct=j.improvement_pct,
            error_message=j.error_message,
            prompt_version_id=str(j.prompt_version_id) if j.prompt_version_id else None,
            created_at=str(j.created_at) if j.created_at else None,
        )
        for j in jobs
    ]


@router.get("/examples")
async def list_examples(
    signature: Optional[str] = None,
    source: Optional[str] = None,
    active_only: bool = True,
    limit: int = 50,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """List training examples."""
    query = select(DSPyTrainingExample).order_by(
        DSPyTrainingExample.quality_score.desc()
    )
    if signature:
        query = query.where(DSPyTrainingExample.signature_name == signature)
    if source:
        query = query.where(DSPyTrainingExample.source == source)
    if active_only:
        query = query.where(DSPyTrainingExample.is_active == True)
    query = query.limit(limit)

    result = await db.execute(query)
    examples = result.scalars().all()

    return [
        ExampleResponse(
            id=str(e.id),
            signature_name=e.signature_name,
            inputs=e.inputs,
            outputs=e.outputs,
            source=e.source,
            quality_score=e.quality_score,
            is_active=e.is_active,
            created_at=str(e.created_at) if e.created_at else None,
        )
        for e in examples
    ]


@router.post("/examples", response_model=ExampleResponse)
async def add_example(
    request: AddExampleRequest,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """Add a manual training example."""
    example = DSPyTrainingExample(
        id=uuid.uuid4(),
        signature_name=request.signature_name,
        inputs=request.inputs,
        outputs=request.outputs,
        source="manual",
        quality_score=request.quality_score,
        is_active=True,
    )
    db.add(example)
    await db.commit()
    await db.refresh(example)

    return ExampleResponse(
        id=str(example.id),
        signature_name=example.signature_name,
        inputs=example.inputs,
        outputs=example.outputs,
        source=example.source,
        quality_score=example.quality_score,
        is_active=example.is_active,
        created_at=str(example.created_at) if example.created_at else None,
    )


@router.delete("/examples/{example_id}")
async def delete_example(
    example_id: str,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """Delete (deactivate) a training example."""
    result = await db.execute(
        select(DSPyTrainingExample).where(
            DSPyTrainingExample.id == uuid.UUID(example_id)
        )
    )
    example = result.scalar_one_or_none()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    example.is_active = False
    await db.commit()

    return {"status": "deactivated", "id": example_id}


@router.get("/example-counts")
async def get_example_counts(
    signature: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """Get training example counts by source."""
    try:
        from backend.services.prompt_optimization.dspy_example_collector import (
            DSPyExampleCollector,
        )

        collector = DSPyExampleCollector(db)
        counts = await collector.get_example_count(
            signature_name=signature or "",
        )
        return counts

    except Exception as e:
        logger.warning("Failed to get example counts", error=str(e))
        return {"chat_feedback": 0, "trajectory": 0, "manual": 0, "total": 0}


@router.post("/deploy/{job_id}")
async def deploy_optimization(
    job_id: str,
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    _admin=Depends(get_current_admin_user),
):
    """
    Deploy a completed optimization result.

    Creates a new prompt version via PromptVersionManager for A/B testing.
    """
    result = await db.execute(
        select(DSPyOptimizationJob).where(
            DSPyOptimizationJob.id == uuid.UUID(job_id)
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job.status}', must be 'completed' to deploy",
        )

    if not job.compiled_state:
        raise HTTPException(
            status_code=400,
            detail="No compiled state available for deployment",
        )

    try:
        from backend.services.prompt_optimization.dspy_optimizer import (
            DSPyOptimizer,
            DSPyOptimizationResult,
        )

        optimizer = DSPyOptimizer(db)
        opt_result = DSPyOptimizationResult(
            signature_name=job.signature_name,
            optimizer_used=job.optimizer_type,
            num_training_examples=job.num_train_examples,
            num_dev_examples=job.num_dev_examples,
            baseline_score=job.baseline_score or 0,
            optimized_score=job.optimized_score or 0,
            improvement_pct=job.improvement_pct or 0,
            compiled_instructions=job.compiled_state.get("instructions", ""),
            compiled_demos=job.compiled_state.get("demos", []),
        )

        version_id = await optimizer._export_to_prompt_version(opt_result, agent_id)

        if version_id:
            job.status = "deployed"
            job.prompt_version_id = uuid.UUID(version_id)
            await db.commit()

            return {
                "status": "deployed",
                "job_id": job_id,
                "prompt_version_id": version_id,
                "improvement_pct": job.improvement_pct,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create prompt version",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Deployment failed", error=str(e), job_id=job_id)
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")
