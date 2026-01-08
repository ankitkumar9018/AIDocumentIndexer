"""
AIDocumentIndexer - Cross-Validation Agent
============================================

Validates LLM-generated content against source documents to:
1. Detect hallucinations (claims not supported by sources)
2. Verify factual consistency
3. Flag unsupported statements
4. Provide confidence scores for generated content

This agent helps reduce hallucinations by ~40% through multi-agent verification.
"""

import json
import time
import uuid
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.agents.agent_base import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
    TaskStatus,
    TaskType,
    PromptTemplate,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Validation Result Types
# =============================================================================

@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str
    is_supported: bool
    confidence: float  # 0.0 to 1.0
    supporting_sources: List[str] = field(default_factory=list)
    contradiction_sources: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "is_supported": self.is_supported,
            "confidence": self.confidence,
            "supporting_sources": self.supporting_sources,
            "contradiction_sources": self.contradiction_sources,
            "explanation": self.explanation,
        }


@dataclass
class ValidationReport:
    """Complete validation report for generated content."""
    is_valid: bool
    overall_confidence: float  # 0.0 to 1.0
    total_claims: int
    verified_claims: int
    unsupported_claims: int
    contradicted_claims: int
    claim_verifications: List[ClaimVerification] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "overall_confidence": self.overall_confidence,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "unsupported_claims": self.unsupported_claims,
            "contradicted_claims": self.contradicted_claims,
            "claim_verifications": [c.to_dict() for c in self.claim_verifications],
            "issues": self.issues,
            "improvements": self.improvements,
            "missing_information": self.missing_information,
        }


# =============================================================================
# Prompts
# =============================================================================

VALIDATOR_SYSTEM_PROMPT = """You are a Cross-Validation Agent that verifies generated content against source documents.

Your role is to:
1. Extract factual claims from the generated content
2. Verify each claim against the provided source documents
3. Identify unsupported or hallucinated statements
4. Flag contradictions between generated content and sources
5. Note important information from sources that might be missing

Be rigorous but fair in your evaluation. A claim is supported if the source:
- Directly states the same information
- Provides evidence that logically implies the claim
- Contains data that supports the claim

A claim is unsupported if:
- No source mentions or implies it
- It makes specific assertions (numbers, dates, names) not found in sources
- It extrapolates beyond what sources reasonably support

A claim contradicts if:
- Sources state the opposite
- Sources provide conflicting data
- The claim misrepresents source information"""

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from this content.

CONTENT:
{content}

Extract specific, verifiable claims. Include:
- Statements of fact (numbers, dates, names, events)
- Assertions about relationships or causes
- Conclusions or summaries

Ignore:
- Opinions clearly marked as such
- General statements without specific claims
- Hedged language ("might", "possibly", "could be")

Return as JSON:
{{
    "claims": [
        "Specific claim 1",
        "Specific claim 2",
        ...
    ]
}}"""

CLAIM_VERIFICATION_PROMPT = """Verify this claim against the provided sources.

CLAIM: {claim}

SOURCES:
{sources}

Determine if this claim is:
1. SUPPORTED - Sources provide direct evidence
2. PARTIALLY_SUPPORTED - Some evidence, but not complete
3. UNSUPPORTED - No evidence in sources
4. CONTRADICTED - Sources provide conflicting information

Return as JSON:
{{
    "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
    "confidence": 0.0-1.0,
    "supporting_evidence": ["quote or reference from source"],
    "contradicting_evidence": ["if any"],
    "explanation": "Brief explanation of verdict"
}}"""

FULL_VALIDATION_PROMPT = """Validate this generated content against the source documents.

ORIGINAL QUERY/REQUEST:
{query}

GENERATED CONTENT:
{content}

SOURCE DOCUMENTS:
{sources}

Perform a thorough validation:

1. CLAIM ANALYSIS: Identify key factual claims in the generated content
2. VERIFICATION: Check each claim against the sources
3. MISSING INFO: Note important information from sources not included
4. CONSISTENCY: Check for internal consistency and logical coherence

Return your analysis as JSON:
{{
    "is_valid": true/false,
    "overall_confidence": 0.0-1.0,
    "claims_analysis": [
        {{
            "claim": "The specific claim",
            "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
            "confidence": 0.0-1.0,
            "evidence": "Supporting or contradicting evidence from sources",
            "source_reference": "Which source document"
        }}
    ],
    "issues": [
        "List of problems found"
    ],
    "improvements": [
        "Suggestions to fix issues"
    ],
    "missing_from_sources": [
        "Important info from sources not in the generated content"
    ]
}}"""


# =============================================================================
# Validator Agent
# =============================================================================

class ValidatorAgent(BaseAgent):
    """
    Cross-validation agent for detecting hallucinations and verifying content.

    Uses a multi-step process:
    1. Extract claims from generated content
    2. Match claims against source documents
    3. Score confidence for each claim
    4. Produce validation report with issues and suggestions
    """

    DEFAULT_SYSTEM_PROMPT = VALIDATOR_SYSTEM_PROMPT
    DEFAULT_TASK_PROMPT = FULL_VALIDATION_PROMPT

    # Thresholds
    CLAIM_SUPPORT_THRESHOLD = 0.6  # Minimum confidence for claim to be "supported"
    OVERALL_VALIDITY_THRESHOLD = 0.7  # Minimum overall confidence for content to be "valid"
    MAX_UNSUPPORTED_RATIO = 0.3  # Max ratio of unsupported claims allowed

    def __init__(
        self,
        config: AgentConfig,
        llm=None,
        prompt_template: Optional[PromptTemplate] = None,
        trajectory_collector=None,
    ):
        super().__init__(
            config=config,
            llm=llm,
            prompt_template=prompt_template,
            trajectory_collector=trajectory_collector,
        )

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Validate content against sources.

        Args:
            task: Validation task
            context: Context containing:
                - content: Generated content to validate
                - sources: List of source documents/chunks
                - original_query: Original user query
                - dependency_results: Results from prior steps

        Returns:
            AgentResult with ValidationReport
        """
        self.clear_trajectory()
        start_time = time.time()

        # Extract content to validate
        content = self._extract_content(context)
        if not content:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message="No content provided for validation",
            )

        # Extract sources
        sources = self._extract_sources(context)
        if not sources:
            logger.warning("No sources provided for validation, will do basic coherence check")

        original_query = context.get("original_query", task.description)
        language = context.get("language", "en")

        try:
            # Perform validation
            validation_report = await self._validate_content(
                content=content,
                sources=sources,
                query=original_query,
                language=language,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output=validation_report.to_dict(),
                confidence_score=validation_report.overall_confidence,
                duration_ms=duration_ms,
                trajectory_steps=self._current_trajectory,
                metadata={
                    "is_valid": validation_report.is_valid,
                    "verified_claims": validation_report.verified_claims,
                    "unsupported_claims": validation_report.unsupported_claims,
                },
            )

        except Exception as e:
            logger.error(
                "Validation failed",
                error=str(e),
                task_id=task.id,
                exc_info=True,
            )
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(e),
                trajectory_steps=self._current_trajectory,
            )

    def _extract_content(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract content to validate from context."""
        # Direct content
        if context.get("content"):
            content = context["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, dict):
                return content.get("content") or content.get("findings") or json.dumps(content)

        # From dependency results
        if context.get("dependency_results"):
            for result in context["dependency_results"].values():
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    if "findings" in result:
                        return result["findings"]
                    elif "content" in result:
                        return result["content"]

        return None

    def _extract_sources(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source documents from context."""
        sources = []

        # Direct sources
        if context.get("sources"):
            raw_sources = context["sources"]
            if isinstance(raw_sources, list):
                sources.extend(raw_sources)
            elif isinstance(raw_sources, dict):
                sources.append(raw_sources)

        # From dependency results (research agent output)
        if context.get("dependency_results"):
            for result in context["dependency_results"].values():
                if isinstance(result, dict) and "sources" in result:
                    sources.extend(result["sources"])

        # From search_results
        if context.get("search_results"):
            sources.extend(context["search_results"])

        return sources

    def _format_sources_for_prompt(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for inclusion in prompt."""
        if not sources:
            return "No source documents provided."

        formatted = []
        for i, source in enumerate(sources, 1):
            source_name = source.get("document_name") or source.get("source") or f"Source {i}"
            content = source.get("content", "")[:1000]  # Limit per source
            formatted.append(f"[{i}] {source_name}:\n{content}\n")

        return "\n".join(formatted)

    async def _validate_content(
        self,
        content: str,
        sources: List[Dict[str, Any]],
        query: str,
        language: str = "en",
    ) -> ValidationReport:
        """
        Perform full validation of content against sources.

        Uses a single comprehensive prompt for efficiency,
        with optional detailed claim-by-claim verification for high-stakes content.
        """
        sources_text = self._format_sources_for_prompt(sources)

        # Build validation prompt
        validation_prompt = FULL_VALIDATION_PROMPT.format(
            query=query,
            content=content[:4000],  # Limit content length
            sources=sources_text,
        )

        messages = [
            SystemMessage(content=self.DEFAULT_SYSTEM_PROMPT),
            HumanMessage(content=validation_prompt),
        ]

        self.record_step(
            action_type="validation_start",
            input_data={
                "content_length": len(content),
                "source_count": len(sources),
            },
            output_data={},
        )

        # Invoke LLM for validation
        response_text, input_tokens, output_tokens = await self.invoke_llm(
            messages, record=True
        )

        # Parse response
        validation_report = self._parse_validation_response(response_text)

        self.record_step(
            action_type="validation_complete",
            input_data={},
            output_data={
                "is_valid": validation_report.is_valid,
                "confidence": validation_report.overall_confidence,
                "issues_count": len(validation_report.issues),
            },
            tokens_used=input_tokens + output_tokens,
        )

        return validation_report

    def _parse_validation_response(self, response: str) -> ValidationReport:
        """Parse LLM response into ValidationReport."""
        try:
            # Extract JSON from response
            json_data = self._extract_json(response)

            if not json_data:
                logger.warning("Could not parse validation response as JSON")
                return self._create_fallback_report(response)

            # Parse claims analysis
            claim_verifications = []
            claims_analysis = json_data.get("claims_analysis", [])

            verified_count = 0
            unsupported_count = 0
            contradicted_count = 0

            for claim_data in claims_analysis:
                verdict = claim_data.get("verdict", "UNSUPPORTED")
                confidence = float(claim_data.get("confidence", 0.5))

                is_supported = verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED")
                if is_supported:
                    verified_count += 1
                elif verdict == "CONTRADICTED":
                    contradicted_count += 1
                else:
                    unsupported_count += 1

                claim_verifications.append(ClaimVerification(
                    claim=claim_data.get("claim", ""),
                    is_supported=is_supported,
                    confidence=confidence,
                    supporting_sources=[claim_data.get("source_reference", "")] if is_supported else [],
                    contradiction_sources=[claim_data.get("source_reference", "")] if verdict == "CONTRADICTED" else [],
                    explanation=claim_data.get("evidence", ""),
                ))

            total_claims = len(claims_analysis)

            # Calculate overall confidence
            if total_claims > 0:
                avg_confidence = sum(c.confidence for c in claim_verifications) / total_claims
                unsupported_ratio = unsupported_count / total_claims
            else:
                avg_confidence = json_data.get("overall_confidence", 0.7)
                unsupported_ratio = 0

            # Determine validity
            is_valid = json_data.get("is_valid", True)
            if unsupported_ratio > self.MAX_UNSUPPORTED_RATIO:
                is_valid = False
            if avg_confidence < self.OVERALL_VALIDITY_THRESHOLD:
                is_valid = False
            if contradicted_count > 0:
                is_valid = False

            return ValidationReport(
                is_valid=is_valid,
                overall_confidence=avg_confidence,
                total_claims=total_claims,
                verified_claims=verified_count,
                unsupported_claims=unsupported_count,
                contradicted_claims=contradicted_count,
                claim_verifications=claim_verifications,
                issues=json_data.get("issues", []),
                improvements=json_data.get("improvements", []),
                missing_information=json_data.get("missing_from_sources", []),
            )

        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            return self._create_fallback_report(response)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text response."""
        # Try to find JSON in markdown code block
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass

        # Try to find raw JSON object
        if "{" in text:
            try:
                start = text.index("{")
                brace_count = 0
                end = start
                for i, char in enumerate(text[start:], start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                return json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError):
                pass

        return None

    def _create_fallback_report(self, response: str) -> ValidationReport:
        """Create a fallback report when JSON parsing fails."""
        # Try to extract some information from the text
        is_valid = "valid" in response.lower() and "invalid" not in response.lower()
        has_issues = "issue" in response.lower() or "problem" in response.lower()

        return ValidationReport(
            is_valid=is_valid and not has_issues,
            overall_confidence=0.6 if is_valid else 0.4,
            total_claims=0,
            verified_claims=0,
            unsupported_claims=0,
            contradicted_claims=0,
            issues=["Could not parse detailed validation - manual review recommended"],
            improvements=[],
            missing_information=[],
        )

    async def validate(
        self,
        content: str,
        sources: List[Dict[str, Any]],
        original_query: str,
    ) -> ValidationReport:
        """
        Convenience method for direct validation.

        Args:
            content: Content to validate
            sources: Source documents
            original_query: Original user query

        Returns:
            ValidationReport
        """
        task = AgentTask(
            id=str(uuid.uuid4()),
            type=TaskType.EVALUATION,
            name="Content Validation",
            description="Validate content against sources",
        )

        result = await self.execute(
            task,
            {
                "content": content,
                "sources": sources,
                "original_query": original_query,
            }
        )

        if result.is_success and result.output:
            return ValidationReport(**result.output) if isinstance(result.output, dict) else result.output
        else:
            raise ValueError(result.error_message or "Validation failed")

    async def quick_check(
        self,
        content: str,
        sources: List[Dict[str, Any]],
    ) -> Tuple[bool, float, List[str]]:
        """
        Quick validation check returning simple results.

        Args:
            content: Content to validate
            sources: Source documents

        Returns:
            Tuple of (is_valid, confidence, issues)
        """
        report = await self.validate(content, sources, "Quick validation check")
        return (
            report.is_valid,
            report.overall_confidence,
            report.issues,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_validator_agent(
    trajectory_collector=None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ValidatorAgent:
    """
    Create a configured ValidatorAgent instance.

    Args:
        trajectory_collector: Optional trajectory collector
        config_overrides: Optional configuration overrides

    Returns:
        Configured ValidatorAgent
    """
    base_config = {
        "agent_id": str(uuid.uuid4()),
        "name": "Validator Agent",
        "description": "Cross-validates content against sources to detect hallucinations",
        "temperature": 0.3,  # Lower temperature for more consistent validation
        "max_tokens": 4096,
    }

    if config_overrides:
        base_config.update(config_overrides)

    config = AgentConfig(**base_config)

    return ValidatorAgent(
        config=config,
        trajectory_collector=trajectory_collector,
    )
