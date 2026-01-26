"""
AIDocumentIndexer - RAG Security Service
=========================================

Security hardening for RAG systems against adversarial attacks.

Threats addressed:
1. Prompt Injection - Malicious prompts trying to manipulate LLM behavior
2. RAG Poisoning - Adversarial content in documents to influence responses
3. Data Exfiltration - Attempts to extract sensitive information
4. Jailbreaking - Attempts to bypass safety guidelines

Detection methods:
- Pattern-based detection (fast, low false positives)
- Embedding-based anomaly detection (semantic analysis)
- LLM-based classification (high accuracy, slower)
- Heuristic scoring (balanced)

Research:
- "Ignore This Title and HackAPrompt" (2023) - Prompt injection taxonomy
- "Poisoning RAG Systems" (2024) - Document-level attacks
- OWASP LLM Top 10 - Security guidelines
- NeMo Guardrails - NVIDIA's approach

Best practices implemented:
- Input sanitization before RAG
- Output filtering after LLM response
- Query complexity limits
- Rate limiting integration
- Audit logging for security events
"""

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ThreatType(str, Enum):
    """Types of security threats."""
    PROMPT_INJECTION = "prompt_injection"
    RAG_POISONING = "rag_poisoning"
    DATA_EXFILTRATION = "data_exfiltration"
    JAILBREAK = "jailbreak"
    PII_LEAK = "pii_leak"
    MALICIOUS_CONTENT = "malicious_content"


class ThreatSeverity(str, Enum):
    """Severity levels for detected threats."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    """Detection methods available."""
    PATTERN = "pattern"        # Regex pattern matching
    HEURISTIC = "heuristic"    # Rule-based scoring
    EMBEDDING = "embedding"    # Semantic similarity
    LLM = "llm"               # LLM-based classification


@dataclass
class RAGSecurityConfig:
    """Configuration for RAG security."""
    # Detection settings
    enable_pattern_detection: bool = True
    enable_heuristic_detection: bool = True
    enable_embedding_detection: bool = False  # Requires embedding model
    enable_llm_detection: bool = False        # Requires LLM calls

    # Thresholds
    pattern_block_threshold: float = 0.8      # Block if score >= threshold
    heuristic_block_threshold: float = 0.7
    embedding_anomaly_threshold: float = 0.85

    # Actions
    block_on_threat: bool = True              # Block detected threats
    log_all_queries: bool = True              # Audit log all queries
    alert_on_critical: bool = True            # Alert on critical threats

    # Limits
    max_query_length: int = 10000             # Max query characters
    max_context_length: int = 100000          # Max context characters

    # PII detection
    detect_pii: bool = True
    pii_patterns: List[str] = field(default_factory=lambda: [
        r"\b\d{3}-\d{2}-\d{4}\b",              # SSN
        r"\b\d{16}\b",                          # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",      # Phone
    ])


@dataclass
class ThreatDetection:
    """Result of threat detection."""
    detected: bool
    threat_type: Optional[ThreatType]
    severity: ThreatSeverity
    confidence: float
    method: DetectionMethod
    details: str
    matched_patterns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SecurityScanResult:
    """Complete security scan result."""
    query_id: str
    timestamp: datetime
    is_safe: bool
    threats: List[ThreatDetection]
    sanitized_query: Optional[str]
    risk_score: float
    processing_time_ms: float
    blocked: bool
    block_reason: Optional[str] = None


# =============================================================================
# Prompt Injection Patterns
# =============================================================================

# Known prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    # Direct instructions
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", ThreatSeverity.HIGH),
    (r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?)", ThreatSeverity.HIGH),
    (r"forget\s+(everything|all|your)\s+(previous|instructions?)", ThreatSeverity.HIGH),

    # Role manipulation
    (r"you\s+are\s+now\s+(?:a|an|the)\s+(?:different|new|evil)", ThreatSeverity.HIGH),
    (r"pretend\s+(to\s+be|you\s+are)\s+(?:a|an)", ThreatSeverity.MEDIUM),
    (r"act\s+as\s+(?:if|though)\s+you\s+(?:are|were)", ThreatSeverity.MEDIUM),
    (r"roleplay\s+as\s+", ThreatSeverity.MEDIUM),

    # System prompt extraction
    (r"(?:what|show|reveal|display|print|output)\s+(?:is\s+)?(?:your|the)\s+(?:system\s+)?prompt", ThreatSeverity.HIGH),
    (r"(?:repeat|echo|print)\s+(?:your\s+)?(?:instructions?|rules?|guidelines?)", ThreatSeverity.HIGH),
    (r"(?:show|display)\s+(?:your\s+)?(?:initial|original|system)\s+(?:message|prompt)", ThreatSeverity.HIGH),

    # Jailbreak attempts
    (r"(?:DAN|STAN|DUDE|KEVIN)\s+(?:mode|jailbreak)", ThreatSeverity.CRITICAL),
    (r"developer\s+mode\s+enabled", ThreatSeverity.CRITICAL),
    (r"(?:enable|activate|enter)\s+(?:god|admin|root|sudo)\s+mode", ThreatSeverity.CRITICAL),
    (r"bypass\s+(?:your\s+)?(?:safety|content|ethical)\s+(?:filters?|guidelines?)", ThreatSeverity.CRITICAL),

    # Code injection
    (r"```(?:python|javascript|bash|shell)\s*\n.*(?:exec|eval|system|os\.)", ThreatSeverity.HIGH),
    (r"<script[^>]*>", ThreatSeverity.HIGH),
    (r"\$\{.*\}", ThreatSeverity.MEDIUM),  # Template injection

    # Data exfiltration
    (r"(?:send|post|transmit|exfiltrate)\s+(?:to|data\s+to)", ThreatSeverity.HIGH),
    (r"(?:curl|wget|fetch)\s+https?://", ThreatSeverity.HIGH),

    # Encoding evasion
    (r"(?:base64|hex|rot13|unicode)\s*(?:decode|encode)", ThreatSeverity.MEDIUM),
]

# RAG poisoning patterns (in documents)
RAG_POISONING_PATTERNS = [
    # Hidden instructions
    (r"<\!--.*(?:ignore|forget|disregard).*-->", ThreatSeverity.HIGH),
    (r"\[hidden\].*\[/hidden\]", ThreatSeverity.HIGH),
    (r"<!-- SYSTEM:.*-->", ThreatSeverity.CRITICAL),

    # Invisible text tricks
    (r"[\u200b-\u200f\u2028-\u202f\ufeff]", ThreatSeverity.MEDIUM),  # Zero-width chars
    (r"color:\s*(?:white|#fff|transparent)", ThreatSeverity.MEDIUM),

    # Adversarial content
    (r"when\s+asked\s+about.*(?:say|respond|answer)", ThreatSeverity.HIGH),
    (r"if\s+(?:user|anyone)\s+asks.*(?:tell|respond)", ThreatSeverity.HIGH),

    # Fake context
    (r"(?:fact|truth|important):\s*the\s+(?:real|actual|true)\s+answer", ThreatSeverity.MEDIUM),
]


# =============================================================================
# RAG Security Engine
# =============================================================================

class RAGSecurityEngine:
    """
    Security engine for RAG systems.

    Usage:
        engine = RAGSecurityEngine()

        # Scan query before RAG
        result = await engine.scan_query(user_query)
        if result.blocked:
            return "Query blocked for security reasons"

        # Scan retrieved context
        context_result = await engine.scan_context(retrieved_chunks)

        # Scan final response
        response_result = await engine.scan_response(llm_response)
    """

    def __init__(self, config: Optional[RAGSecurityConfig] = None):
        self.config = config or RAGSecurityConfig()
        self._compiled_patterns: List[Tuple[re.Pattern, ThreatSeverity]] = []
        self._compiled_poisoning_patterns: List[Tuple[re.Pattern, ThreatSeverity]] = []
        self._compiled_pii_patterns: List[re.Pattern] = []
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), severity)
            for pattern, severity in PROMPT_INJECTION_PATTERNS
        ]

        self._compiled_poisoning_patterns = [
            (re.compile(pattern, re.IGNORECASE), severity)
            for pattern, severity in RAG_POISONING_PATTERNS
        ]

        self._compiled_pii_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.pii_patterns
        ]

    async def scan_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SecurityScanResult:
        """
        Scan a user query for security threats.

        Args:
            query: User query to scan
            user_id: Optional user identifier for audit
            session_id: Optional session identifier

        Returns:
            SecurityScanResult with threat analysis
        """
        start_time = time.time()
        query_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:12]

        threats: List[ThreatDetection] = []

        # Length check
        if len(query) > self.config.max_query_length:
            threats.append(ThreatDetection(
                detected=True,
                threat_type=ThreatType.MALICIOUS_CONTENT,
                severity=ThreatSeverity.MEDIUM,
                confidence=1.0,
                method=DetectionMethod.HEURISTIC,
                details=f"Query exceeds maximum length ({len(query)} > {self.config.max_query_length})",
                recommendations=["Shorten the query"],
            ))

        # Pattern-based detection
        if self.config.enable_pattern_detection:
            pattern_threats = self._detect_patterns(query, self._compiled_patterns)
            threats.extend(pattern_threats)

        # Heuristic detection
        if self.config.enable_heuristic_detection:
            heuristic_threats = self._detect_heuristics(query)
            threats.extend(heuristic_threats)

        # PII detection
        if self.config.detect_pii:
            pii_threats = self._detect_pii(query)
            threats.extend(pii_threats)

        # Calculate risk score
        risk_score = self._calculate_risk_score(threats)

        # Determine if blocked
        blocked = False
        block_reason = None

        if self.config.block_on_threat:
            critical_threats = [t for t in threats if t.severity == ThreatSeverity.CRITICAL]
            high_threats = [t for t in threats if t.severity == ThreatSeverity.HIGH]

            if critical_threats:
                blocked = True
                block_reason = f"Critical threat detected: {critical_threats[0].details}"
            elif len(high_threats) >= 2:
                blocked = True
                block_reason = f"Multiple high-severity threats detected"
            elif risk_score >= self.config.pattern_block_threshold:
                blocked = True
                block_reason = f"Risk score {risk_score:.2f} exceeds threshold"

        # Sanitize query
        sanitized = self._sanitize_query(query) if not blocked else None

        processing_time = (time.time() - start_time) * 1000

        # Log security event
        if self.config.log_all_queries or threats:
            await self._log_security_event(
                event_type="query_scan",
                query_id=query_id,
                user_id=user_id,
                session_id=session_id,
                threats=threats,
                blocked=blocked,
                risk_score=risk_score,
            )

        return SecurityScanResult(
            query_id=query_id,
            timestamp=datetime.now(timezone.utc),
            is_safe=len(threats) == 0,
            threats=threats,
            sanitized_query=sanitized,
            risk_score=risk_score,
            processing_time_ms=processing_time,
            blocked=blocked,
            block_reason=block_reason,
        )

    async def scan_context(
        self,
        chunks: List[str],
        source_ids: Optional[List[str]] = None,
    ) -> SecurityScanResult:
        """
        Scan retrieved context for RAG poisoning.

        Args:
            chunks: Retrieved document chunks
            source_ids: Optional document identifiers

        Returns:
            SecurityScanResult for the context
        """
        start_time = time.time()
        query_id = hashlib.md5(f"ctx_{time.time()}".encode()).hexdigest()[:12]

        threats: List[ThreatDetection] = []

        for idx, chunk in enumerate(chunks):
            # Length check
            if len(chunk) > self.config.max_context_length:
                threats.append(ThreatDetection(
                    detected=True,
                    threat_type=ThreatType.MALICIOUS_CONTENT,
                    severity=ThreatSeverity.LOW,
                    confidence=1.0,
                    method=DetectionMethod.HEURISTIC,
                    details=f"Chunk {idx} exceeds maximum length",
                    recommendations=["Truncate or split the chunk"],
                ))

            # RAG poisoning patterns
            poisoning_threats = self._detect_patterns(
                chunk, self._compiled_poisoning_patterns, ThreatType.RAG_POISONING
            )
            for threat in poisoning_threats:
                threat.details = f"Chunk {idx}: {threat.details}"
            threats.extend(poisoning_threats)

            # Hidden content detection
            hidden_threats = self._detect_hidden_content(chunk, idx)
            threats.extend(hidden_threats)

        risk_score = self._calculate_risk_score(threats)
        processing_time = (time.time() - start_time) * 1000

        blocked = (
            self.config.block_on_threat and
            any(t.severity == ThreatSeverity.CRITICAL for t in threats)
        )

        return SecurityScanResult(
            query_id=query_id,
            timestamp=datetime.now(timezone.utc),
            is_safe=len(threats) == 0,
            threats=threats,
            sanitized_query=None,
            risk_score=risk_score,
            processing_time_ms=processing_time,
            blocked=blocked,
            block_reason="Poisoned context detected" if blocked else None,
        )

    async def scan_response(
        self,
        response: str,
        original_query: Optional[str] = None,
    ) -> SecurityScanResult:
        """
        Scan LLM response for data leakage or policy violations.

        Args:
            response: LLM response to scan
            original_query: Original user query for context

        Returns:
            SecurityScanResult for the response
        """
        start_time = time.time()
        query_id = hashlib.md5(f"resp_{time.time()}".encode()).hexdigest()[:12]

        threats: List[ThreatDetection] = []

        # PII leak detection
        if self.config.detect_pii:
            pii_threats = self._detect_pii(response, is_response=True)
            threats.extend(pii_threats)

        # Check for potential data exfiltration
        exfil_threats = self._detect_data_exfiltration(response)
        threats.extend(exfil_threats)

        risk_score = self._calculate_risk_score(threats)
        processing_time = (time.time() - start_time) * 1000

        blocked = (
            self.config.block_on_threat and
            any(t.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH] for t in threats)
        )

        return SecurityScanResult(
            query_id=query_id,
            timestamp=datetime.now(timezone.utc),
            is_safe=len(threats) == 0,
            threats=threats,
            sanitized_query=self._redact_pii(response) if not blocked else None,
            risk_score=risk_score,
            processing_time_ms=processing_time,
            blocked=blocked,
            block_reason="Response contains sensitive data" if blocked else None,
        )

    def _detect_patterns(
        self,
        text: str,
        patterns: List[Tuple[re.Pattern, ThreatSeverity]],
        threat_type: ThreatType = ThreatType.PROMPT_INJECTION,
    ) -> List[ThreatDetection]:
        """Detect threats using regex patterns."""
        threats = []

        for pattern, severity in patterns:
            matches = pattern.findall(text)
            if matches:
                threats.append(ThreatDetection(
                    detected=True,
                    threat_type=threat_type,
                    severity=severity,
                    confidence=0.9 if severity == ThreatSeverity.CRITICAL else 0.8,
                    method=DetectionMethod.PATTERN,
                    details=f"Pattern matched: {pattern.pattern[:50]}...",
                    matched_patterns=[str(m) for m in matches[:5]],
                    recommendations=[
                        "Review query for malicious intent",
                        "Consider blocking or sanitizing",
                    ],
                ))

        return threats

    def _detect_heuristics(self, text: str) -> List[ThreatDetection]:
        """Detect threats using heuristic rules."""
        threats = []
        text_lower = text.lower()

        # Instruction density check
        instruction_words = ["ignore", "forget", "disregard", "pretend", "act", "bypass", "override"]
        instruction_count = sum(1 for word in instruction_words if word in text_lower)

        if instruction_count >= 3:
            threats.append(ThreatDetection(
                detected=True,
                threat_type=ThreatType.PROMPT_INJECTION,
                severity=ThreatSeverity.HIGH,
                confidence=0.7,
                method=DetectionMethod.HEURISTIC,
                details=f"High density of instruction words ({instruction_count})",
                recommendations=["Review for prompt injection attempt"],
            ))

        # Special character density
        special_chars = sum(1 for c in text if c in "{}[]<>|\\")
        if len(text) > 0 and special_chars / len(text) > 0.1:
            threats.append(ThreatDetection(
                detected=True,
                threat_type=ThreatType.MALICIOUS_CONTENT,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.6,
                method=DetectionMethod.HEURISTIC,
                details=f"High special character density ({special_chars / len(text):.2%})",
                recommendations=["May contain code injection"],
            ))

        # Excessive repetition (potential DoS)
        words = text_lower.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2:
                threats.append(ThreatDetection(
                    detected=True,
                    threat_type=ThreatType.MALICIOUS_CONTENT,
                    severity=ThreatSeverity.LOW,
                    confidence=0.5,
                    method=DetectionMethod.HEURISTIC,
                    details=f"Excessive repetition ({unique_ratio:.2%} unique words)",
                    recommendations=["Potential resource exhaustion attempt"],
                ))

        return threats

    def _detect_pii(
        self,
        text: str,
        is_response: bool = False,
    ) -> List[ThreatDetection]:
        """Detect PII in text."""
        threats = []

        for pattern in self._compiled_pii_patterns:
            matches = pattern.findall(text)
            if matches:
                threat_type = ThreatType.PII_LEAK if is_response else ThreatType.DATA_EXFILTRATION
                threats.append(ThreatDetection(
                    detected=True,
                    threat_type=threat_type,
                    severity=ThreatSeverity.HIGH if is_response else ThreatSeverity.MEDIUM,
                    confidence=0.9,
                    method=DetectionMethod.PATTERN,
                    details=f"PII detected: {len(matches)} matches",
                    matched_patterns=["[REDACTED]"],  # Don't log actual PII
                    recommendations=[
                        "Redact PII before processing" if not is_response else
                        "Response contains sensitive data - consider filtering"
                    ],
                ))

        return threats

    def _detect_hidden_content(self, text: str, chunk_idx: int) -> List[ThreatDetection]:
        """Detect hidden or obfuscated content in documents."""
        threats = []

        # Zero-width characters
        zero_width = re.findall(r"[\u200b-\u200f\u2028-\u202f\ufeff]", text)
        if len(zero_width) > 5:
            threats.append(ThreatDetection(
                detected=True,
                threat_type=ThreatType.RAG_POISONING,
                severity=ThreatSeverity.HIGH,
                confidence=0.85,
                method=DetectionMethod.PATTERN,
                details=f"Chunk {chunk_idx}: Zero-width characters detected ({len(zero_width)})",
                recommendations=["Strip invisible characters"],
            ))

        # Base64 encoded content (potential hidden instructions)
        base64_pattern = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
        base64_matches = base64_pattern.findall(text)
        if base64_matches:
            threats.append(ThreatDetection(
                detected=True,
                threat_type=ThreatType.RAG_POISONING,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.6,
                method=DetectionMethod.PATTERN,
                details=f"Chunk {chunk_idx}: Potential encoded content ({len(base64_matches)} matches)",
                recommendations=["Decode and inspect encoded content"],
            ))

        return threats

    def _detect_data_exfiltration(self, response: str) -> List[ThreatDetection]:
        """Detect potential data exfiltration in response."""
        threats = []

        # URLs in response
        url_pattern = re.compile(r"https?://[^\s]+")
        urls = url_pattern.findall(response)
        suspicious_urls = [u for u in urls if not any(
            safe in u for safe in ["wikipedia", "github.com", "docs.", "example.com"]
        )]

        if suspicious_urls:
            threats.append(ThreatDetection(
                detected=True,
                threat_type=ThreatType.DATA_EXFILTRATION,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.5,
                method=DetectionMethod.PATTERN,
                details=f"Response contains {len(suspicious_urls)} potentially suspicious URLs",
                recommendations=["Review URLs for data exfiltration risk"],
            ))

        return threats

    def _calculate_risk_score(self, threats: List[ThreatDetection]) -> float:
        """Calculate overall risk score from detected threats."""
        if not threats:
            return 0.0

        severity_weights = {
            ThreatSeverity.LOW: 0.1,
            ThreatSeverity.MEDIUM: 0.3,
            ThreatSeverity.HIGH: 0.6,
            ThreatSeverity.CRITICAL: 1.0,
        }

        scores = [
            severity_weights[t.severity] * t.confidence
            for t in threats
        ]

        # Use max + diminishing returns for additional threats
        scores.sort(reverse=True)
        total = scores[0] if scores else 0.0
        for score in scores[1:]:
            total += score * 0.5  # Diminishing returns

        return min(total, 1.0)

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query by removing potentially harmful content."""
        sanitized = query

        # Remove zero-width characters
        sanitized = re.sub(r"[\u200b-\u200f\u2028-\u202f\ufeff]", "", sanitized)

        # Escape potential template injections
        sanitized = re.sub(r"\$\{[^}]+\}", "[REMOVED]", sanitized)

        # Remove script tags
        sanitized = re.sub(r"<script[^>]*>.*?</script>", "[REMOVED]", sanitized, flags=re.IGNORECASE | re.DOTALL)

        return sanitized

    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text

        for pattern in self._compiled_pii_patterns:
            redacted = pattern.sub("[REDACTED]", redacted)

        return redacted

    async def _log_security_event(
        self,
        event_type: str,
        query_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        threats: List[ThreatDetection],
        blocked: bool,
        risk_score: float,
    ) -> None:
        """Log security event for audit."""
        log_data = {
            "event_type": event_type,
            "query_id": query_id,
            "user_id": user_id,
            "session_id": session_id,
            "threat_count": len(threats),
            "blocked": blocked,
            "risk_score": risk_score,
            "threats": [
                {
                    "type": t.threat_type.value,
                    "severity": t.severity.value,
                    "confidence": t.confidence,
                    "method": t.method.value,
                }
                for t in threats
            ],
        }

        if threats:
            if any(t.severity == ThreatSeverity.CRITICAL for t in threats):
                logger.error("SECURITY: Critical threat detected", **log_data)
            elif any(t.severity == ThreatSeverity.HIGH for t in threats):
                logger.warning("SECURITY: High severity threat detected", **log_data)
            else:
                logger.info("SECURITY: Threat detected", **log_data)
        else:
            logger.debug("SECURITY: Query scanned (clean)", query_id=query_id)


# =============================================================================
# Singleton Management
# =============================================================================

_security_engine: Optional[RAGSecurityEngine] = None
_engine_lock = asyncio.Lock()


async def get_security_engine(
    config: Optional[RAGSecurityConfig] = None,
) -> RAGSecurityEngine:
    """Get or create RAG security engine singleton."""
    global _security_engine

    async with _engine_lock:
        if _security_engine is None:
            _security_engine = RAGSecurityEngine(config)

        return _security_engine


async def scan_query(query: str) -> SecurityScanResult:
    """
    Convenience function to scan a query.

    Usage:
        from backend.services.rag_security import scan_query

        result = await scan_query(user_input)
        if result.blocked:
            return "Your query has been blocked for security reasons."
    """
    engine = await get_security_engine()
    return await engine.scan_query(query)


async def scan_context(chunks: List[str]) -> SecurityScanResult:
    """
    Convenience function to scan RAG context.

    Usage:
        result = await scan_context(retrieved_chunks)
        if not result.is_safe:
            # Filter out poisoned chunks
            pass
    """
    engine = await get_security_engine()
    return await engine.scan_context(chunks)


async def scan_response(response: str) -> SecurityScanResult:
    """
    Convenience function to scan LLM response.

    Usage:
        result = await scan_response(llm_output)
        if result.blocked:
            return "Response filtered for containing sensitive information."
    """
    engine = await get_security_engine()
    return await engine.scan_response(response)
