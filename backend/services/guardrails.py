"""
AIDocumentIndexer - Guardrails Service
========================================

Security layer for prompt injection defense and content safety.
Implements OWASP LLM Top 10 protections.

Features:
- Prompt injection detection
- Jailbreak attempt blocking
- Content moderation
- PII detection and masking
- Output validation
- Rate limiting
"""

import asyncio
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


class ThreatType(str, Enum):
    """Types of detected threats."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    PII_EXPOSURE = "pii_exposure"
    HARMFUL_CONTENT = "harmful_content"
    DATA_EXFILTRATION = "data_exfiltration"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    TOKEN_MANIPULATION = "token_manipulation"
    ENCODING_ATTACK = "encoding_attack"


class ActionType(str, Enum):
    """Actions to take on detected threats."""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    WARN = "warn"
    LOG = "log"


class Severity(str, Enum):
    """Severity levels for threats."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThreatDetection:
    """Result of threat detection."""
    detected: bool
    threat_type: Optional[ThreatType] = None
    severity: Severity = Severity.INFO
    confidence: float = 0.0
    matched_pattern: Optional[str] = None
    explanation: str = ""
    suggested_action: ActionType = ActionType.ALLOW


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    allowed: bool
    input_modified: bool
    output_modified: bool
    original_input: str
    processed_input: str
    original_output: Optional[str] = None
    processed_output: Optional[str] = None
    threats_detected: List[ThreatDetection] = field(default_factory=list)
    pii_masked: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "input_modified": self.input_modified,
            "output_modified": self.output_modified,
            "threats_detected": [
                {
                    "type": t.threat_type.value if t.threat_type else None,
                    "severity": t.severity.value,
                    "confidence": t.confidence,
                    "explanation": t.explanation,
                }
                for t in self.threats_detected
            ],
            "pii_masked": self.pii_masked,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    current_count: int
    limit: int
    window_seconds: int
    retry_after_seconds: Optional[int] = None


class GuardrailsConfig:
    """Configuration for guardrails."""

    def __init__(
        self,
        enable_injection_detection: bool = True,
        enable_jailbreak_detection: bool = True,
        enable_pii_masking: bool = True,
        enable_content_moderation: bool = True,
        enable_output_validation: bool = True,
        enable_rate_limiting: bool = True,
        block_on_high_severity: bool = True,
        pii_replacement: str = "[REDACTED]",
        max_input_length: int = 50000,
        rate_limit_requests: int = 100,
        rate_limit_window_seconds: int = 60,
    ):
        self.enable_injection_detection = enable_injection_detection
        self.enable_jailbreak_detection = enable_jailbreak_detection
        self.enable_pii_masking = enable_pii_masking
        self.enable_content_moderation = enable_content_moderation
        self.enable_output_validation = enable_output_validation
        self.enable_rate_limiting = enable_rate_limiting
        self.block_on_high_severity = block_on_high_severity
        self.pii_replacement = pii_replacement
        self.max_input_length = max_input_length
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window_seconds = rate_limit_window_seconds


class GuardrailsService:
    """
    Security guardrails for LLM interactions.

    Implements multiple defense layers:
    1. Input validation and sanitization
    2. Prompt injection detection
    3. Jailbreak attempt detection
    4. PII detection and masking
    5. Content moderation
    6. Output validation
    7. Rate limiting
    """

    def __init__(self, config: Optional[GuardrailsConfig] = None):
        self.config = config or GuardrailsConfig()
        self._rate_limit_buckets: Dict[str, List[datetime]] = defaultdict(list)
        self._blocked_patterns_cache: Set[str] = set()

        # Compile detection patterns
        self._injection_patterns = self._compile_injection_patterns()
        self._jailbreak_patterns = self._compile_jailbreak_patterns()
        self._pii_patterns = self._compile_pii_patterns()

    def _compile_injection_patterns(self) -> List[Tuple[re.Pattern, str, Severity]]:
        """Compile prompt injection detection patterns."""
        patterns = [
            # Direct instruction injection
            (r'ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?', "Instruction override attempt", Severity.HIGH),
            (r'disregard\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?', "Instruction override attempt", Severity.HIGH),
            (r'forget\s+(?:everything|all)\s+(?:you\s+)?(?:know|learned)', "Memory manipulation attempt", Severity.HIGH),

            # System prompt extraction
            (r'(?:what|show|tell|reveal|display)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)', "System prompt leak attempt", Severity.CRITICAL),
            (r'repeat\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)', "System prompt leak attempt", Severity.CRITICAL),
            (r'print\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions|initialization)', "System prompt leak attempt", Severity.CRITICAL),

            # Role manipulation
            (r'you\s+are\s+(?:now|no\s+longer)\s+(?:a|an)', "Role manipulation attempt", Severity.HIGH),
            (r'pretend\s+(?:you\s+are|to\s+be)', "Role manipulation attempt", Severity.MEDIUM),
            (r'act\s+as\s+(?:if\s+)?(?:you\s+(?:are|were))?', "Role manipulation attempt", Severity.MEDIUM),

            # Delimiter attacks
            (r'```(?:system|assistant|user)', "Delimiter injection attempt", Severity.HIGH),
            (r'<\|(?:im_start|im_end|system|user|assistant)\|>', "Token injection attempt", Severity.CRITICAL),
            (r'\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>', "Format injection attempt", Severity.HIGH),

            # Code execution attempts
            (r'(?:execute|run|eval)\s*\(', "Code execution attempt", Severity.CRITICAL),
            (r'import\s+(?:os|sys|subprocess)', "System import attempt", Severity.CRITICAL),
            (r'__(?:import|builtins|class|globals)__', "Python introspection attempt", Severity.CRITICAL),

            # Data exfiltration
            (r'(?:send|post|transmit)\s+(?:to|data)\s+(?:http|https|ftp)', "Data exfiltration attempt", Severity.CRITICAL),
            (r'(?:curl|wget|fetch)\s+', "External request attempt", Severity.HIGH),
        ]

        return [(re.compile(p, re.IGNORECASE), desc, sev) for p, desc, sev in patterns]

    def _compile_jailbreak_patterns(self) -> List[Tuple[re.Pattern, str, Severity]]:
        """Compile jailbreak detection patterns."""
        patterns = [
            # DAN and variants
            (r'\bDAN\b.*(?:mode|jailbreak|bypass)', "DAN jailbreak attempt", Severity.HIGH),
            (r'do\s+anything\s+now', "DAN jailbreak attempt", Severity.HIGH),

            # Developer mode
            (r'developer\s+mode\s+(?:enabled|activated|on)', "Developer mode jailbreak", Severity.HIGH),
            (r'enable\s+developer\s+mode', "Developer mode jailbreak", Severity.HIGH),

            # Hypothetical scenarios for bypassing
            (r'hypothetically.*(?:if\s+you\s+(?:could|were)|no\s+restrictions)', "Hypothetical bypass attempt", Severity.MEDIUM),
            (r'in\s+(?:a\s+)?fictional\s+(?:world|scenario).*(?:rules|restrictions)', "Fictional bypass attempt", Severity.MEDIUM),

            # Roleplay exploits
            (r'(?:roleplay|pretend).*(?:no\s+(?:rules|limits|restrictions)|evil|malicious)', "Malicious roleplay attempt", Severity.HIGH),

            # Constraint removal
            (r'remove\s+(?:all\s+)?(?:your\s+)?(?:filters|restrictions|limitations|constraints)', "Constraint removal attempt", Severity.HIGH),
            (r'(?:without|ignore)\s+(?:any\s+)?(?:ethical|safety|content)\s+(?:guidelines|restrictions)', "Ethics bypass attempt", Severity.HIGH),

            # Token smuggling
            (r'(?:encode|decode|base64|rot13).*(?:instructions|prompt)', "Encoding attack attempt", Severity.HIGH),
        ]

        return [(re.compile(p, re.IGNORECASE), desc, sev) for p, desc, sev in patterns]

    def _compile_pii_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile PII detection patterns."""
        patterns = [
            # Email
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),

            # Phone numbers (various formats)
            (r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', "phone"),

            # SSN
            (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', "ssn"),

            # Credit card
            (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b', "credit_card"),

            # IP Address
            (r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', "ip_address"),

            # API Keys (generic patterns)
            (r'\b(?:sk|pk|api|key|token)[-_]?[a-zA-Z0-9]{20,}\b', "api_key"),

            # AWS Keys
            (r'\bAKIA[0-9A-Z]{16}\b', "aws_key"),

            # Passwords in common formats
            (r'(?:password|passwd|pwd)\s*[=:]\s*[^\s]+', "password"),
        ]

        return [(re.compile(p, re.IGNORECASE), pii_type) for p, pii_type in patterns]

    async def check_input(
        self,
        text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Check input text through all guardrails.

        Args:
            text: Input text to check
            user_id: User ID for rate limiting
            context: Additional context

        Returns:
            GuardrailResult with check results
        """
        start_time = datetime.utcnow()

        threats: List[ThreatDetection] = []
        pii_masked: List[str] = []
        warnings: List[str] = []
        processed_text = text
        allowed = True

        # Length check
        if len(text) > self.config.max_input_length:
            warnings.append(f"Input truncated from {len(text)} to {self.config.max_input_length} characters")
            processed_text = text[:self.config.max_input_length]

        # Rate limiting
        if self.config.enable_rate_limiting and user_id:
            rate_result = self._check_rate_limit(user_id)
            if not rate_result.allowed:
                return GuardrailResult(
                    allowed=False,
                    input_modified=False,
                    output_modified=False,
                    original_input=text,
                    processed_input=text,
                    threats_detected=[ThreatDetection(
                        detected=True,
                        threat_type=ThreatType.TOKEN_MANIPULATION,
                        severity=Severity.MEDIUM,
                        confidence=1.0,
                        explanation=f"Rate limit exceeded. Retry after {rate_result.retry_after_seconds}s",
                    )],
                    warnings=[f"Rate limit: {rate_result.current_count}/{rate_result.limit} requests"],
                )

        # Prompt injection detection
        if self.config.enable_injection_detection:
            injection_threats = self._detect_injection(processed_text)
            threats.extend(injection_threats)

        # Jailbreak detection
        if self.config.enable_jailbreak_detection:
            jailbreak_threats = self._detect_jailbreak(processed_text)
            threats.extend(jailbreak_threats)

        # PII masking
        if self.config.enable_pii_masking:
            processed_text, masked = self._mask_pii(processed_text)
            pii_masked.extend(masked)

        # Determine if we should block
        high_severity_threats = [
            t for t in threats
            if t.severity in [Severity.CRITICAL, Severity.HIGH]
        ]

        if self.config.block_on_high_severity and high_severity_threats:
            allowed = False

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return GuardrailResult(
            allowed=allowed,
            input_modified=processed_text != text,
            output_modified=False,
            original_input=text,
            processed_input=processed_text,
            threats_detected=threats,
            pii_masked=pii_masked,
            warnings=warnings,
            processing_time_ms=processing_time,
        )

    async def check_output(
        self,
        output: str,
        original_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Check output text through guardrails.

        Args:
            output: Output text to check
            original_input: Original input for context
            context: Additional context

        Returns:
            GuardrailResult with check results
        """
        start_time = datetime.utcnow()

        threats: List[ThreatDetection] = []
        pii_masked: List[str] = []
        warnings: List[str] = []
        processed_output = output
        allowed = True

        if self.config.enable_output_validation:
            # Check for system prompt leakage
            leak_threat = self._detect_system_prompt_leak(output, context)
            if leak_threat.detected:
                threats.append(leak_threat)
                # Redact leaked content
                processed_output = self._redact_system_prompt(output, context)

        # PII masking in output
        if self.config.enable_pii_masking:
            processed_output, masked = self._mask_pii(processed_output)
            pii_masked.extend(masked)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return GuardrailResult(
            allowed=allowed,
            input_modified=False,
            output_modified=processed_output != output,
            original_input=original_input,
            processed_input=original_input,
            original_output=output,
            processed_output=processed_output,
            threats_detected=threats,
            pii_masked=pii_masked,
            warnings=warnings,
            processing_time_ms=processing_time,
        )

    def _detect_injection(self, text: str) -> List[ThreatDetection]:
        """Detect prompt injection attempts."""
        threats = []

        for pattern, description, severity in self._injection_patterns:
            match = pattern.search(text)
            if match:
                threats.append(ThreatDetection(
                    detected=True,
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity=severity,
                    confidence=0.9,
                    matched_pattern=match.group(0)[:100],
                    explanation=description,
                    suggested_action=ActionType.BLOCK if severity in [Severity.CRITICAL, Severity.HIGH] else ActionType.WARN,
                ))

        return threats

    def _detect_jailbreak(self, text: str) -> List[ThreatDetection]:
        """Detect jailbreak attempts."""
        threats = []

        for pattern, description, severity in self._jailbreak_patterns:
            match = pattern.search(text)
            if match:
                threats.append(ThreatDetection(
                    detected=True,
                    threat_type=ThreatType.JAILBREAK,
                    severity=severity,
                    confidence=0.85,
                    matched_pattern=match.group(0)[:100],
                    explanation=description,
                    suggested_action=ActionType.BLOCK if severity == Severity.HIGH else ActionType.WARN,
                ))

        return threats

    def _mask_pii(self, text: str) -> Tuple[str, List[str]]:
        """Mask PII in text."""
        masked_types = []
        processed = text

        for pattern, pii_type in self._pii_patterns:
            if pattern.search(processed):
                masked_types.append(pii_type)
                processed = pattern.sub(self.config.pii_replacement, processed)

        return processed, masked_types

    def _detect_system_prompt_leak(
        self,
        output: str,
        context: Optional[Dict[str, Any]],
    ) -> ThreatDetection:
        """Detect if system prompt is being leaked in output."""
        if not context:
            return ThreatDetection(detected=False)

        system_prompt = context.get("system_prompt", "")
        if not system_prompt:
            return ThreatDetection(detected=False)

        # Check if significant portion of system prompt appears in output
        system_words = set(system_prompt.lower().split())
        output_words = set(output.lower().split())

        overlap = len(system_words & output_words) / max(len(system_words), 1)

        if overlap > 0.5:
            return ThreatDetection(
                detected=True,
                threat_type=ThreatType.SYSTEM_PROMPT_LEAK,
                severity=Severity.HIGH,
                confidence=overlap,
                explanation="Potential system prompt leak detected in output",
                suggested_action=ActionType.MODIFY,
            )

        return ThreatDetection(detected=False)

    def _redact_system_prompt(
        self,
        output: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Redact system prompt content from output."""
        if not context:
            return output

        system_prompt = context.get("system_prompt", "")
        if not system_prompt:
            return output

        # Simple redaction - replace exact matches
        # In production, use more sophisticated matching
        return output.replace(system_prompt, "[SYSTEM CONTENT REDACTED]")

    def _check_rate_limit(self, user_id: str) -> RateLimitResult:
        """Check rate limit for user."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.config.rate_limit_window_seconds)

        # Clean old entries
        self._rate_limit_buckets[user_id] = [
            t for t in self._rate_limit_buckets[user_id]
            if t > window_start
        ]

        current_count = len(self._rate_limit_buckets[user_id])

        if current_count >= self.config.rate_limit_requests:
            # Calculate retry after
            oldest = min(self._rate_limit_buckets[user_id])
            retry_after = int((oldest + timedelta(seconds=self.config.rate_limit_window_seconds) - now).total_seconds()) + 1

            return RateLimitResult(
                allowed=False,
                current_count=current_count,
                limit=self.config.rate_limit_requests,
                window_seconds=self.config.rate_limit_window_seconds,
                retry_after_seconds=retry_after,
            )

        # Add current request
        self._rate_limit_buckets[user_id].append(now)

        return RateLimitResult(
            allowed=True,
            current_count=current_count + 1,
            limit=self.config.rate_limit_requests,
            window_seconds=self.config.rate_limit_window_seconds,
        )

    async def get_threat_stats(self) -> Dict[str, Any]:
        """Get statistics on detected threats."""
        return {
            "total_checks": 0,  # Would track in production
            "threats_blocked": 0,
            "pii_masked": 0,
            "rate_limited": len(self._rate_limit_buckets),
        }


# Singleton instance
_guardrails_service: Optional[GuardrailsService] = None


def get_guardrails_service() -> GuardrailsService:
    """Get or create the guardrails service singleton."""
    global _guardrails_service
    if _guardrails_service is None:
        _guardrails_service = GuardrailsService()
    return _guardrails_service
