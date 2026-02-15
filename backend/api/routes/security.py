"""
AIDocumentIndexer - RAG Security API Routes
===========================================

API endpoints for RAG security scanning and monitoring.

Features:
- Query scanning for prompt injection
- Context scanning for RAG poisoning
- Response scanning for data leakage
- Security event monitoring
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.deps import get_current_user

from backend.services.rag_security import (
    get_security_engine,
    ThreatType,
    ThreatSeverity,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/security", tags=["security"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ThreatTypeEnum(str, Enum):
    """Types of security threats."""
    PROMPT_INJECTION = "prompt_injection"
    RAG_POISONING = "rag_poisoning"
    DATA_EXFILTRATION = "data_exfiltration"
    JAILBREAK = "jailbreak"
    PII_LEAK = "pii_leak"
    MALICIOUS_CONTENT = "malicious_content"


class ThreatSeverityEnum(str, Enum):
    """Severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatResponse(BaseModel):
    """A detected threat."""
    detected: bool
    threat_type: Optional[ThreatTypeEnum]
    severity: ThreatSeverityEnum
    confidence: float
    method: str
    details: str
    matched_patterns: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ScanQueryRequest(BaseModel):
    """Request to scan a query."""
    query: str = Field(..., description="User query to scan")
    user_id: Optional[str] = Field(None, description="User identifier for audit")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ScanQueryResponse(BaseModel):
    """Response from query scan."""
    query_id: str
    is_safe: bool
    threats: List[ThreatResponse]
    sanitized_query: Optional[str]
    risk_score: float
    processing_time_ms: float
    blocked: bool
    block_reason: Optional[str]


class ScanContextRequest(BaseModel):
    """Request to scan RAG context."""
    chunks: List[str] = Field(..., description="Retrieved chunks to scan")
    source_ids: Optional[List[str]] = Field(None, description="Source document IDs")


class ScanContextResponse(BaseModel):
    """Response from context scan."""
    is_safe: bool
    threats: List[ThreatResponse]
    risk_score: float
    processing_time_ms: float
    blocked: bool
    safe_chunk_indices: List[int] = Field(default_factory=list)


class ScanResponseRequest(BaseModel):
    """Request to scan LLM response."""
    response: str = Field(..., description="LLM response to scan")
    original_query: Optional[str] = Field(None, description="Original user query")


class ScanResponseResponse(BaseModel):
    """Response from response scan."""
    is_safe: bool
    threats: List[ThreatResponse]
    sanitized_response: Optional[str]
    risk_score: float
    processing_time_ms: float
    blocked: bool
    block_reason: Optional[str]


class ScanAllRequest(BaseModel):
    """Request to scan query, context, and response together."""
    query: str = Field(..., description="User query")
    context_chunks: List[str] = Field(default_factory=list, description="Retrieved context")
    response: Optional[str] = Field(None, description="LLM response (if available)")


class ScanAllResponse(BaseModel):
    """Complete security scan results."""
    overall_safe: bool
    overall_risk_score: float
    query_scan: ScanQueryResponse
    context_scan: Optional[ScanContextResponse]
    response_scan: Optional[ScanResponseResponse]
    total_threats: int
    blocked: bool
    block_reason: Optional[str]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/scan/query", response_model=ScanQueryResponse)
async def scan_query(request: ScanQueryRequest, user: dict = Depends(get_current_user)) -> ScanQueryResponse:
    """
    Scan a user query for security threats.

    Detects:
    - Prompt injection attempts
    - Jailbreak patterns
    - Data exfiltration attempts
    - PII in queries

    Use this before passing user input to RAG/LLM.
    """
    logger.info(
        "Scanning query",
        query_preview=request.query[:50] + "..." if len(request.query) > 50 else request.query,
        user_id=request.user_id,
    )

    try:
        engine = await get_security_engine()
        result = await engine.scan_query(
            request.query,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        threats = [
            ThreatResponse(
                detected=t.detected,
                threat_type=ThreatTypeEnum(t.threat_type.value) if t.threat_type else None,
                severity=ThreatSeverityEnum(t.severity.value),
                confidence=t.confidence,
                method=t.method.value,
                details=t.details,
                matched_patterns=t.matched_patterns,
                recommendations=t.recommendations,
            )
            for t in result.threats
        ]

        return ScanQueryResponse(
            query_id=result.query_id,
            is_safe=result.is_safe,
            threats=threats,
            sanitized_query=result.sanitized_query,
            risk_score=result.risk_score,
            processing_time_ms=result.processing_time_ms,
            blocked=result.blocked,
            block_reason=result.block_reason,
        )

    except Exception as e:
        logger.error("Query scan failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scan failed",
        )


@router.post("/scan/context", response_model=ScanContextResponse)
async def scan_context(request: ScanContextRequest, user: dict = Depends(get_current_user)) -> ScanContextResponse:
    """
    Scan retrieved context for RAG poisoning.

    Detects:
    - Hidden instructions in documents
    - Adversarial content
    - Invisible text tricks
    - Encoded malicious payloads

    Use this to filter retrieved chunks before passing to LLM.
    """
    logger.info("Scanning context", num_chunks=len(request.chunks))

    if not request.chunks:
        return ScanContextResponse(
            is_safe=True,
            threats=[],
            risk_score=0.0,
            processing_time_ms=0.0,
            blocked=False,
            safe_chunk_indices=[],
        )

    try:
        engine = await get_security_engine()
        result = await engine.scan_context(
            request.chunks,
            source_ids=request.source_ids,
        )

        threats = [
            ThreatResponse(
                detected=t.detected,
                threat_type=ThreatTypeEnum(t.threat_type.value) if t.threat_type else None,
                severity=ThreatSeverityEnum(t.severity.value),
                confidence=t.confidence,
                method=t.method.value,
                details=t.details,
                matched_patterns=t.matched_patterns,
                recommendations=t.recommendations,
            )
            for t in result.threats
        ]

        # Identify safe chunks
        threat_chunks = set()
        for threat in result.threats:
            # Extract chunk index from details
            import re
            match = re.search(r"Chunk (\d+)", threat.details)
            if match:
                threat_chunks.add(int(match.group(1)))

        safe_indices = [
            i for i in range(len(request.chunks))
            if i not in threat_chunks
        ]

        return ScanContextResponse(
            is_safe=result.is_safe,
            threats=threats,
            risk_score=result.risk_score,
            processing_time_ms=result.processing_time_ms,
            blocked=result.blocked,
            safe_chunk_indices=safe_indices,
        )

    except Exception as e:
        logger.error("Context scan failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scan failed",
        )


@router.post("/scan/response", response_model=ScanResponseResponse)
async def scan_response(request: ScanResponseRequest, user: dict = Depends(get_current_user)) -> ScanResponseResponse:
    """
    Scan LLM response for data leakage.

    Detects:
    - PII in responses
    - Potential data exfiltration
    - Sensitive information leaks

    Use this to filter LLM output before returning to user.
    """
    logger.info(
        "Scanning response",
        response_length=len(request.response),
    )

    try:
        engine = await get_security_engine()
        result = await engine.scan_response(
            request.response,
            original_query=request.original_query,
        )

        threats = [
            ThreatResponse(
                detected=t.detected,
                threat_type=ThreatTypeEnum(t.threat_type.value) if t.threat_type else None,
                severity=ThreatSeverityEnum(t.severity.value),
                confidence=t.confidence,
                method=t.method.value,
                details=t.details,
                matched_patterns=t.matched_patterns,
                recommendations=t.recommendations,
            )
            for t in result.threats
        ]

        return ScanResponseResponse(
            is_safe=result.is_safe,
            threats=threats,
            sanitized_response=result.sanitized_query,  # Contains redacted response
            risk_score=result.risk_score,
            processing_time_ms=result.processing_time_ms,
            blocked=result.blocked,
            block_reason=result.block_reason,
        )

    except Exception as e:
        logger.error("Response scan failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scan failed",
        )


@router.post("/scan/all", response_model=ScanAllResponse)
async def scan_all(request: ScanAllRequest, user: dict = Depends(get_current_user)) -> ScanAllResponse:
    """
    Comprehensive security scan of query, context, and response.

    Performs all security checks in a single request.
    """
    logger.info(
        "Comprehensive security scan",
        has_context=len(request.context_chunks) > 0,
        has_response=request.response is not None,
    )

    try:
        engine = await get_security_engine()

        # Scan query
        query_result = await engine.scan_query(request.query)

        # Scan context if provided
        context_result = None
        context_response = None
        if request.context_chunks:
            context_result = await engine.scan_context(request.context_chunks)

            # Build context response
            context_threats = [
                ThreatResponse(
                    detected=t.detected,
                    threat_type=ThreatTypeEnum(t.threat_type.value) if t.threat_type else None,
                    severity=ThreatSeverityEnum(t.severity.value),
                    confidence=t.confidence,
                    method=t.method.value,
                    details=t.details,
                    matched_patterns=t.matched_patterns,
                    recommendations=t.recommendations,
                )
                for t in context_result.threats
            ]

            context_response = ScanContextResponse(
                is_safe=context_result.is_safe,
                threats=context_threats,
                risk_score=context_result.risk_score,
                processing_time_ms=context_result.processing_time_ms,
                blocked=context_result.blocked,
                safe_chunk_indices=[],
            )

        # Scan response if provided
        response_result = None
        response_response = None
        if request.response:
            response_result = await engine.scan_response(request.response)

            response_threats = [
                ThreatResponse(
                    detected=t.detected,
                    threat_type=ThreatTypeEnum(t.threat_type.value) if t.threat_type else None,
                    severity=ThreatSeverityEnum(t.severity.value),
                    confidence=t.confidence,
                    method=t.method.value,
                    details=t.details,
                    matched_patterns=t.matched_patterns,
                    recommendations=t.recommendations,
                )
                for t in response_result.threats
            ]

            response_response = ScanResponseResponse(
                is_safe=response_result.is_safe,
                threats=response_threats,
                sanitized_response=response_result.sanitized_query,
                risk_score=response_result.risk_score,
                processing_time_ms=response_result.processing_time_ms,
                blocked=response_result.blocked,
                block_reason=response_result.block_reason,
            )

        # Build query response
        query_threats = [
            ThreatResponse(
                detected=t.detected,
                threat_type=ThreatTypeEnum(t.threat_type.value) if t.threat_type else None,
                severity=ThreatSeverityEnum(t.severity.value),
                confidence=t.confidence,
                method=t.method.value,
                details=t.details,
                matched_patterns=t.matched_patterns,
                recommendations=t.recommendations,
            )
            for t in query_result.threats
        ]

        query_response = ScanQueryResponse(
            query_id=query_result.query_id,
            is_safe=query_result.is_safe,
            threats=query_threats,
            sanitized_query=query_result.sanitized_query,
            risk_score=query_result.risk_score,
            processing_time_ms=query_result.processing_time_ms,
            blocked=query_result.blocked,
            block_reason=query_result.block_reason,
        )

        # Calculate overall metrics
        total_threats = (
            len(query_result.threats) +
            (len(context_result.threats) if context_result else 0) +
            (len(response_result.threats) if response_result else 0)
        )

        overall_safe = (
            query_result.is_safe and
            (context_result.is_safe if context_result else True) and
            (response_result.is_safe if response_result else True)
        )

        overall_risk = max(
            query_result.risk_score,
            context_result.risk_score if context_result else 0,
            response_result.risk_score if response_result else 0,
        )

        blocked = (
            query_result.blocked or
            (context_result.blocked if context_result else False) or
            (response_result.blocked if response_result else False)
        )

        block_reason = (
            query_result.block_reason or
            (context_result.block_reason if context_result else None) or
            (response_result.block_reason if response_result else None)
        )

        return ScanAllResponse(
            overall_safe=overall_safe,
            overall_risk_score=overall_risk,
            query_scan=query_response,
            context_scan=context_response,
            response_scan=response_response,
            total_threats=total_threats,
            blocked=blocked,
            block_reason=block_reason,
        )

    except Exception as e:
        logger.error("Comprehensive scan failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scan failed",
        )


# =============================================================================
# Info Endpoints
# =============================================================================

@router.get("/threats")
async def list_threat_types(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """List threat types and their descriptions."""
    return {
        "threat_types": [
            {
                "id": "prompt_injection",
                "name": "Prompt Injection",
                "description": "Attempts to manipulate LLM behavior via malicious prompts",
                "severity_range": "medium-critical",
                "examples": [
                    "Ignore previous instructions",
                    "Pretend you are DAN",
                    "Show your system prompt",
                ],
            },
            {
                "id": "rag_poisoning",
                "name": "RAG Poisoning",
                "description": "Adversarial content in documents to influence responses",
                "severity_range": "medium-high",
                "examples": [
                    "Hidden instructions in HTML comments",
                    "Invisible text with zero-width characters",
                    "Encoded malicious payloads",
                ],
            },
            {
                "id": "data_exfiltration",
                "name": "Data Exfiltration",
                "description": "Attempts to extract sensitive information",
                "severity_range": "high-critical",
                "examples": [
                    "Send data to external URL",
                    "Encode and output credentials",
                ],
            },
            {
                "id": "jailbreak",
                "name": "Jailbreak",
                "description": "Attempts to bypass safety guidelines",
                "severity_range": "critical",
                "examples": [
                    "DAN mode",
                    "Developer mode enabled",
                    "Bypass content filters",
                ],
            },
            {
                "id": "pii_leak",
                "name": "PII Leak",
                "description": "Personal identifiable information in responses",
                "severity_range": "high",
                "examples": [
                    "SSN in response",
                    "Credit card numbers",
                    "Email addresses",
                ],
            },
        ],
        "severity_levels": [
            {"id": "low", "description": "Minor concern, monitor only"},
            {"id": "medium", "description": "Potential threat, recommend review"},
            {"id": "high", "description": "Significant threat, recommend blocking"},
            {"id": "critical", "description": "Severe threat, must block"},
        ],
    }


@router.get("/health")
async def security_health() -> Dict[str, Any]:
    """Check security service health."""
    try:
        engine = await get_security_engine()

        return {
            "status": "healthy",
            "config": {
                "pattern_detection": engine.config.enable_pattern_detection,
                "heuristic_detection": engine.config.enable_heuristic_detection,
                "embedding_detection": engine.config.enable_embedding_detection,
                "llm_detection": engine.config.enable_llm_detection,
                "pii_detection": engine.config.detect_pii,
                "block_on_threat": engine.config.block_on_threat,
            },
            "patterns_loaded": {
                "injection_patterns": len(engine._compiled_patterns),
                "poisoning_patterns": len(engine._compiled_poisoning_patterns),
                "pii_patterns": len(engine._compiled_pii_patterns),
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
