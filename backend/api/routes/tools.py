"""
AIDocumentIndexer - Tool Augmentation API Routes
=================================================

Endpoints for tool-augmented reasoning that helps small LLMs
overcome their limitations.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.tool_augmentation import (
    ToolAugmentationService,
    get_tool_augmentation_service,
    ToolType,
)
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.db.models import Document
from backend.api.middleware.auth import AuthenticatedUser
from sqlalchemy import select
from uuid import UUID

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class CalculatorRequest(BaseModel):
    """Request for calculator tool."""
    expression: str = Field(..., description="Mathematical expression to evaluate")


class CodeExecutorRequest(BaseModel):
    """Request for code executor tool."""
    code: str = Field(..., description="Python code to execute")
    timeout: Optional[float] = Field(None, ge=0.1, le=30, description="Timeout in seconds")


class FactCheckRequest(BaseModel):
    """Request for fact checker tool."""
    claim: str = Field(..., description="Claim to verify")
    claims: Optional[List[str]] = Field(None, description="Multiple claims to verify")
    context: Optional[str] = Field("", description="Additional context")
    source_ids: Optional[List[str]] = Field(None, description="Document IDs to check against")


class DateCalculatorRequest(BaseModel):
    """Request for date calculator tool."""
    operation: str = Field(..., description="Operation: diff, add, subtract, weekday, is_leap, days_in_month, now")
    date1: Optional[str] = Field(None, description="First date (YYYY-MM-DD)")
    date2: Optional[str] = Field(None, description="Second date for diff operation")
    days: Optional[int] = Field(None, description="Days to add/subtract")
    months: Optional[int] = Field(None, description="Months to add/subtract")
    years: Optional[int] = Field(None, description="Years to add/subtract")


class UnitConvertRequest(BaseModel):
    """Request for unit converter tool."""
    value: float = Field(..., description="Value to convert")
    from_unit: str = Field(..., description="Source unit")
    to_unit: str = Field(..., description="Target unit")
    category: Optional[str] = Field(None, description="Category (auto-detected if not provided)")


class AugmentedQueryRequest(BaseModel):
    """Request for augmented query processing."""
    query: str = Field(..., description="User query to augment")
    context: Optional[str] = Field("", description="Additional context")
    auto_detect_tools: bool = Field(True, description="Auto-detect which tools to use")
    tools: Optional[List[str]] = Field(None, description="Specific tools to use")
    calculator_expression: Optional[str] = Field(None, description="Expression for calculator")
    code_to_run: Optional[str] = Field(None, description="Code to execute")
    facts_to_check: Optional[List[str]] = Field(None, description="Facts to verify")
    source_ids: Optional[List[str]] = Field(None, description="Document IDs to use as sources")


class ToolResultResponse(BaseModel):
    """Response from a tool execution."""
    tool: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: int
    metadata: Dict[str, Any] = {}


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/calculate", response_model=ToolResultResponse)
async def calculate(
    request: CalculatorRequest,
    user: AuthenticatedUser,
):
    """
    Evaluate a mathematical expression.

    Supports:
    - Basic arithmetic: +, -, *, /, **, %, //
    - Functions: sqrt, abs, round, floor, ceil, sin, cos, tan, log, exp
    - Constants: pi, e
    """
    service = get_tool_augmentation_service()
    result = service.calculate(request.expression)

    return ToolResultResponse(
        tool=result.tool.value,
        success=result.success,
        result=result.result,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
        metadata=result.metadata,
    )


@router.post("/execute-code", response_model=ToolResultResponse)
async def execute_code(
    request: CodeExecutorRequest,
    user: AuthenticatedUser,
):
    """
    Execute Python code in a sandboxed environment.

    Safety features:
    - Forbidden imports (os, sys, subprocess, etc.)
    - Timeout protection
    - Limited builtins
    """
    service = get_tool_augmentation_service()
    result = await service.execute_code(request.code, request.timeout)

    return ToolResultResponse(
        tool=result.tool.value,
        success=result.success,
        result=result.result,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
        metadata=result.metadata,
    )


@router.post("/fact-check", response_model=ToolResultResponse)
async def fact_check(
    request: FactCheckRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Verify a claim against available evidence.

    Returns:
    - verdict: supported, contradicted, or unverifiable
    - confidence: 0.0 to 1.0
    - evidence: supporting evidence points
    """
    service = get_tool_augmentation_service()

    # Load sources if IDs provided
    sources = []
    if request.source_ids:
        try:
            source_uuids = [UUID(sid) for sid in request.source_ids]
            query = select(Document).where(Document.id.in_(source_uuids))
            result = await db.execute(query)
            docs = result.scalars().all()
            sources = [
                {
                    "id": str(doc.id),
                    "title": doc.title,
                    "content": doc.content_text or "",
                }
                for doc in docs
            ]
        except (ValueError, Exception) as e:
            logger.warning("Failed to load source documents", error=str(e))

    if request.claims:
        # Multiple claims
        results = await service.fact_checker.check_multiple_facts(
            request.claims,
            context=request.context or "",
            sources=sources,
        )
        # Return combined result
        return ToolResultResponse(
            tool=ToolType.FACT_CHECKER.value,
            success=all(r.success for r in results),
            result=[r.to_dict() for r in results],
            error=None,
            execution_time_ms=sum(r.execution_time_ms for r in results),
            metadata={"claims_checked": len(request.claims)},
        )
    else:
        # Single claim
        result = await service.check_fact(
            request.claim,
            context=request.context or "",
            sources=sources,
        )

        return ToolResultResponse(
            tool=result.tool.value,
            success=result.success,
            result=result.result,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            metadata=result.metadata,
        )


@router.post("/date-calculate", response_model=ToolResultResponse)
async def date_calculate(
    request: DateCalculatorRequest,
    user: AuthenticatedUser,
):
    """
    Perform date/time calculations.

    Operations:
    - diff: Calculate difference between two dates
    - add: Add days/months/years to a date
    - subtract: Subtract days/months/years from a date
    - weekday: Get day of week for a date
    - is_leap: Check if a year is a leap year
    - days_in_month: Get days in a month
    - now: Get current date/time
    """
    service = get_tool_augmentation_service()
    result = service.calculate_date(
        operation=request.operation,
        date1=request.date1,
        date2=request.date2,
        days=request.days,
        months=request.months,
        years=request.years,
    )

    return ToolResultResponse(
        tool=result.tool.value,
        success=result.success,
        result=result.result,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
        metadata=result.metadata,
    )


@router.post("/convert-units", response_model=ToolResultResponse)
async def convert_units(
    request: UnitConvertRequest,
    user: AuthenticatedUser,
):
    """
    Convert between units of measurement.

    Supported categories:
    - length: m, km, cm, mm, mi, yd, ft, in
    - mass: g, kg, mg, lb, oz, ton, tonne
    - volume: l, ml, gal, qt, pt, cup, fl_oz, tbsp, tsp
    - temperature: c, f, k
    - time: s, ms, min, h, d, wk, mo, yr
    - data: b, kb, mb, gb, tb, pb
    - speed: m/s, km/h, mph, kn, ft/s
    - area: m2, km2, cm2, ft2, yd2, acre, ha
    """
    service = get_tool_augmentation_service()
    result = service.convert_units(
        value=request.value,
        from_unit=request.from_unit,
        to_unit=request.to_unit,
    )

    return ToolResultResponse(
        tool=result.tool.value,
        success=result.success,
        result=result.result,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
        metadata=result.metadata,
    )


@router.post("/augment-query")
async def augment_query(
    request: AugmentedQueryRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Augment a query with tool results.

    This endpoint:
    1. Analyzes the query to detect needed tools
    2. Executes relevant tools
    3. Returns enhanced context with tool results

    Useful for:
    - Pre-processing queries before RAG
    - Adding verified calculations to responses
    - Fact-checking claims in questions
    """
    service = get_tool_augmentation_service()

    # Load sources if IDs provided
    sources = None
    if request.source_ids:
        try:
            source_uuids = [UUID(sid) for sid in request.source_ids]
            query = select(Document).where(Document.id.in_(source_uuids))
            result_db = await db.execute(query)
            docs = result_db.scalars().all()
            sources = [
                {
                    "id": str(doc.id),
                    "title": doc.title,
                    "content": doc.content_text or "",
                }
                for doc in docs
            ]
        except (ValueError, Exception) as e:
            logger.warning("Failed to load source documents for augmentation", error=str(e))

    result = await service.augment_query(
        query=request.query,
        context=request.context or "",
        sources=sources,
        auto_detect_tools=request.auto_detect_tools,
    )

    return {
        "original_query": result["original_query"],
        "tools_used": result["tools_used"],
        "tool_results": result["tool_results"],
        "enhanced_context": result["enhanced_context"],
    }


@router.get("/available")
async def get_available_tools(user: AuthenticatedUser):
    """
    Get list of available augmentation tools.
    """
    return {
        "tools": [
            {
                "id": "calculator",
                "name": "Calculator",
                "description": "Evaluate mathematical expressions safely",
                "compensates_for": "Arithmetic errors in small LLMs",
                "example": "sqrt(16) + 2**3",
            },
            {
                "id": "code_executor",
                "name": "Code Executor",
                "description": "Execute Python code in a sandboxed environment",
                "compensates_for": "Logic verification limitations",
                "example": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nprint(factorial(5))",
            },
            {
                "id": "fact_checker",
                "name": "Fact Checker",
                "description": "Verify claims against available evidence",
                "compensates_for": "Hallucination in small LLMs",
                "example": "The Earth orbits the Sun in 365.25 days",
            },
            {
                "id": "date_calculator",
                "name": "Date Calculator",
                "description": "Perform date and time calculations",
                "compensates_for": "Temporal reasoning errors",
                "example": "Days between 2024-01-01 and 2024-12-31",
            },
            {
                "id": "unit_converter",
                "name": "Unit Converter",
                "description": "Convert between units of measurement",
                "compensates_for": "Measurement and conversion errors",
                "example": "100 km to miles",
            },
        ],
        "categories": [
            {
                "id": "math",
                "name": "Mathematical",
                "tools": ["calculator", "unit_converter"],
            },
            {
                "id": "verification",
                "name": "Verification",
                "tools": ["code_executor", "fact_checker"],
            },
            {
                "id": "temporal",
                "name": "Temporal",
                "tools": ["date_calculator"],
            },
        ],
    }
