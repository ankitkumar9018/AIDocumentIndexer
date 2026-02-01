"""
AIDocumentIndexer - Self-Verification Service
==============================================

Makes small LLMs verify their own answers to catch errors.
Implements reflection, fact-checking, and correction loops.

Enhanced with:
- XML-structured verification for reliable parsing
- Few-shot verification examples
- Recursive self-improvement loops
- Multi-pass refinement
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import re

import structlog

from backend.services.llm import LLMFactory
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class VerificationStatus(str, Enum):
    """Status of verification."""
    VERIFIED = "verified"           # Answer is correct
    CORRECTED = "corrected"         # Answer was wrong, now fixed
    UNCERTAIN = "uncertain"         # Can't verify confidently
    FAILED = "failed"               # Verification process failed


@dataclass
class VerificationIssue:
    """An issue found during verification."""
    issue_type: str  # factual_error, logical_error, incomplete, unclear
    description: str
    severity: str  # high, medium, low
    suggested_fix: Optional[str] = None


@dataclass
class VerifiedAnswer:
    """Result of answer verification."""
    original_answer: str
    verified_answer: str
    status: VerificationStatus
    confidence: float
    issues_found: List[VerificationIssue] = field(default_factory=list)
    verification_reasoning: str = ""
    corrections_made: List[str] = field(default_factory=list)
    source_support: float = 0.0  # How well sources support the answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_answer": self.original_answer,
            "verified_answer": self.verified_answer,
            "status": self.status.value,
            "confidence": self.confidence,
            "issues_found": [
                {
                    "type": i.issue_type,
                    "description": i.description,
                    "severity": i.severity,
                    "fix": i.suggested_fix,
                }
                for i in self.issues_found
            ],
            "verification_reasoning": self.verification_reasoning,
            "corrections_made": self.corrections_made,
            "source_support": self.source_support,
        }


class SelfVerificationService:
    """
    Have the model verify its own answers to catch errors.

    This service implements several verification techniques:
    1. Reflection: Ask model to critique its own answer
    2. Source checking: Verify claims against provided sources
    3. Logical consistency: Check for contradictions
    4. Completeness: Ensure question is fully answered
    5. Correction loop: Fix identified issues
    """

    VERIFICATION_PROMPT = """You are a rigorous fact-checker reviewing an AI-generated answer.

**Original Question**: {question}

**Proposed Answer**: {answer}

**Source Documents** (if available):
{sources}

**Verification Checklist**:

1. **Factual Accuracy**: Is every claim in the answer supported by the sources or verifiable facts?
   - [ ] All claims are accurate
   - [ ] Some claims are unverified
   - [ ] Contains factual errors

2. **Logical Consistency**: Is the reasoning sound with no contradictions?
   - [ ] Logic is sound
   - [ ] Minor logical gaps
   - [ ] Contains contradictions

3. **Completeness**: Does the answer fully address the question?
   - [ ] Fully complete
   - [ ] Partially complete
   - [ ] Significantly incomplete

4. **Clarity**: Is the answer clear and understandable?
   - [ ] Very clear
   - [ ] Somewhat unclear
   - [ ] Confusing

5. **Source Support**: How well do the sources support this answer?
   - Strong support (0.8-1.0)
   - Moderate support (0.5-0.7)
   - Weak support (0.2-0.4)
   - No support (0.0-0.1)

**Response Format** (JSON):
```json
{{
    "factual_accuracy": "accurate|unverified|errors",
    "logical_consistency": "sound|gaps|contradictions",
    "completeness": "complete|partial|incomplete",
    "clarity": "clear|unclear|confusing",
    "source_support_score": 0.0-1.0,
    "issues": [
        {{
            "type": "factual_error|logical_error|incomplete|unclear",
            "description": "Description of the issue",
            "severity": "high|medium|low",
            "fix": "How to fix it"
        }}
    ],
    "overall_verdict": "correct|needs_correction|unreliable",
    "confidence": 0-100,
    "corrected_answer": "The corrected answer if needed, or null"
}}
```"""

    REFLECTION_PROMPT = """Before finalizing your answer, step back and reflect.

**Question**: {question}

**Your Draft Answer**: {answer}

**Reflection Questions**:
1. Did I actually answer what was asked, or did I go off-topic?
2. Am I making any assumptions that aren't stated in the sources?
3. Could my answer be misinterpreted? How can I be clearer?
4. What's the weakest part of my answer?
5. If I were the user, would I be satisfied with this answer?

**After reflection, provide**:
- One sentence summary of your answer's quality
- List any issues you found
- Your improved answer (if needed)

**Format**:
QUALITY: [Good/Needs Work/Poor]
ISSUES: [List of issues or "None"]
IMPROVED ANSWER: [Your improved answer or "No changes needed"]"""

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        temperature: float = 0.2,  # Low temperature for verification
    ):
        """Initialize the verification service."""
        self.provider = provider or settings.DEFAULT_LLM_PROVIDER
        self.model = model or settings.DEFAULT_CHAT_MODEL
        self.temperature = temperature

    async def verify(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        max_correction_rounds: int = 2,
    ) -> VerifiedAnswer:
        """
        Verify an answer and correct if needed.

        Args:
            question: The original question
            answer: The answer to verify
            sources: Source documents used to generate the answer
            max_correction_rounds: Maximum correction iterations

        Returns:
            VerifiedAnswer with verification results
        """
        logger.info(
            "Starting self-verification",
            question_length=len(question),
            answer_length=len(answer),
            sources_count=len(sources) if sources else 0,
        )

        # Format sources for prompt
        sources_text = self._format_sources(sources) if sources else "No sources provided."

        # Get verification from LLM
        try:
            verification_result = await self._run_verification(
                question, answer, sources_text
            )

            # If issues found and corrections allowed, try to fix
            current_answer = answer
            corrections_made = []
            correction_round = 0

            while (
                verification_result.get("overall_verdict") == "needs_correction"
                and correction_round < max_correction_rounds
                and verification_result.get("corrected_answer")
            ):
                corrections_made.append(
                    f"Round {correction_round + 1}: {len(verification_result.get('issues', []))} issues fixed"
                )
                current_answer = verification_result["corrected_answer"]

                # Re-verify the corrected answer
                verification_result = await self._run_verification(
                    question, current_answer, sources_text
                )
                correction_round += 1

            # Parse issues
            issues = [
                VerificationIssue(
                    issue_type=i.get("type", "unknown"),
                    description=i.get("description", ""),
                    severity=i.get("severity", "medium"),
                    suggested_fix=i.get("fix"),
                )
                for i in verification_result.get("issues", [])
            ]

            # Determine final status
            verdict = verification_result.get("overall_verdict", "correct")
            if verdict == "correct":
                status = VerificationStatus.VERIFIED
            elif corrections_made:
                status = VerificationStatus.CORRECTED
            elif verdict == "unreliable":
                status = VerificationStatus.UNCERTAIN
            else:
                status = VerificationStatus.UNCERTAIN

            result = VerifiedAnswer(
                original_answer=answer,
                verified_answer=current_answer,
                status=status,
                confidence=verification_result.get("confidence", 70) / 100.0,
                issues_found=issues,
                verification_reasoning=self._build_reasoning(verification_result),
                corrections_made=corrections_made,
                source_support=verification_result.get("source_support_score", 0.5),
            )

            logger.info(
                "Verification complete",
                status=status.value,
                confidence=result.confidence,
                issues_count=len(issues),
                corrections=len(corrections_made),
            )

            return result

        except Exception as e:
            logger.error("Verification failed", error=str(e))
            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=answer,
                status=VerificationStatus.FAILED,
                confidence=0.5,
                verification_reasoning=f"Verification failed: {str(e)}",
            )

    async def _run_verification(
        self,
        question: str,
        answer: str,
        sources_text: str,
    ) -> Dict[str, Any]:
        """Run the verification LLM call."""
        prompt = self.VERIFICATION_PROMPT.format(
            question=question,
            answer=answer,
            sources=sources_text,
        )

        llm = LLMFactory.get_chat_model(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=2048,
        )

        response = await llm.ainvoke(prompt)
        output = response.content

        # Parse JSON response
        return self._parse_verification_response(output)

    async def reflect(
        self,
        question: str,
        answer: str,
    ) -> VerifiedAnswer:
        """
        Quick reflection pass on an answer.

        Lighter weight than full verification, useful for
        checking answers before sending to user.
        """
        logger.info(
            "Running reflection",
            question_length=len(question),
            answer_length=len(answer),
        )

        prompt = self.REFLECTION_PROMPT.format(
            question=question,
            answer=answer,
        )

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=1024,
            )

            response = await llm.ainvoke(prompt)
            output = response.content

            # Parse reflection response
            quality_match = re.search(r'QUALITY:\s*\[?(Good|Needs Work|Poor)\]?', output, re.IGNORECASE)
            quality = quality_match.group(1).lower() if quality_match else "unknown"

            issues_match = re.search(r'ISSUES:\s*\[?(.*?)\]?\s*(?=IMPROVED|$)', output, re.DOTALL | re.IGNORECASE)
            issues_text = issues_match.group(1).strip() if issues_match else ""

            improved_match = re.search(r'IMPROVED ANSWER:\s*(.*?)$', output, re.DOTALL | re.IGNORECASE)
            improved_answer = improved_match.group(1).strip() if improved_match else None

            # Determine if we have an improved answer
            has_improvements = improved_answer and "no changes" not in improved_answer.lower()

            issues = []
            if issues_text and issues_text.lower() != "none":
                issues.append(VerificationIssue(
                    issue_type="reflection",
                    description=issues_text,
                    severity="medium" if quality == "needs work" else "high" if quality == "poor" else "low",
                ))

            # Map quality to confidence
            confidence_map = {"good": 0.85, "needs work": 0.65, "poor": 0.4, "unknown": 0.6}
            confidence = confidence_map.get(quality, 0.6)

            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=improved_answer if has_improvements else answer,
                status=VerificationStatus.CORRECTED if has_improvements else VerificationStatus.VERIFIED,
                confidence=confidence,
                issues_found=issues,
                verification_reasoning=f"Reflection quality: {quality}",
                corrections_made=["Improved via reflection"] if has_improvements else [],
            )

        except Exception as e:
            logger.error("Reflection failed", error=str(e))
            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=answer,
                status=VerificationStatus.FAILED,
                confidence=0.5,
                verification_reasoning=f"Reflection failed: {str(e)}",
            )

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format source documents for the prompt."""
        if not sources:
            return "No sources provided."

        formatted = []
        for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
            content = source.get("content", source.get("text", ""))[:500]
            title = source.get("title", source.get("document_name", f"Source {i}"))
            formatted.append(f"**Source {i}: {title}**\n{content}")

        return "\n\n".join(formatted)

    def _parse_verification_response(self, output: str) -> Dict[str, Any]:
        """Parse the JSON verification response."""
        # Try to extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object directly
        json_match = re.search(r'\{[^{}]*"overall_verdict"[^{}]*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: parse manually
        result = {
            "overall_verdict": "correct",
            "confidence": 70,
            "issues": [],
        }

        # Check for keywords indicating problems
        lower_output = output.lower()
        if any(word in lower_output for word in ["error", "incorrect", "wrong", "inaccurate"]):
            result["overall_verdict"] = "needs_correction"
            result["confidence"] = 50
        elif any(word in lower_output for word in ["uncertain", "unclear", "cannot verify"]):
            result["overall_verdict"] = "uncertain"
            result["confidence"] = 40

        return result

    def _build_reasoning(self, verification_result: Dict[str, Any]) -> str:
        """Build a human-readable verification reasoning."""
        parts = []

        accuracy = verification_result.get("factual_accuracy", "unknown")
        parts.append(f"Factual accuracy: {accuracy}")

        logic = verification_result.get("logical_consistency", "unknown")
        parts.append(f"Logical consistency: {logic}")

        completeness = verification_result.get("completeness", "unknown")
        parts.append(f"Completeness: {completeness}")

        source_score = verification_result.get("source_support_score", 0)
        parts.append(f"Source support: {source_score:.0%}")

        return ". ".join(parts)

    async def verify_with_xml(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> VerifiedAnswer:
        """
        Verify an answer using XML-structured output for reliable parsing.

        Args:
            question: The original question
            answer: The answer to verify
            sources: Source documents used

        Returns:
            VerifiedAnswer with verification results
        """
        from backend.services.enhanced_prompts import (
            XML_STRUCTURED_VERIFICATION,
            parse_xml_response,
        )

        logger.info(
            "Starting XML-structured verification",
            question_length=len(question),
            answer_length=len(answer),
        )

        sources_text = self._format_sources(sources) if sources else "No sources provided."

        prompt = XML_STRUCTURED_VERIFICATION.format(
            question=question,
            answer=answer,
            sources=sources_text,
        )

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=2048,
            )

            response = await llm.ainvoke(prompt)
            output = response.content

            # Parse XML response
            parsed = parse_xml_response(output)

            # Extract claims verification
            issues = []
            if parsed.get('claims'):
                for claim in parsed['claims']:
                    if claim['support'] in ['unsupported', 'partial']:
                        issues.append(VerificationIssue(
                            issue_type="factual_error" if claim['support'] == 'unsupported' else "incomplete",
                            description=f"Claim: {claim['claim']}",
                            severity="high" if claim['support'] == 'unsupported' else "medium",
                        ))

            # Determine status
            status_str = parsed.get('status', 'correct')
            if status_str == 'correct':
                status = VerificationStatus.VERIFIED
            elif status_str == 'needs_correction':
                status = VerificationStatus.CORRECTED
            else:
                status = VerificationStatus.UNCERTAIN

            confidence = parsed.get('confidence', 70) / 100.0
            corrected = parsed.get('corrected_answer')
            verified_answer = corrected if corrected and corrected != 'N/A' else answer

            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=verified_answer,
                status=status,
                confidence=confidence,
                issues_found=issues,
                verification_reasoning=f"XML-structured verification: {len(issues)} issues found",
                corrections_made=["XML verification correction"] if corrected and corrected != 'N/A' else [],
                source_support=0.8 if not issues else 0.5,
            )

        except Exception as e:
            logger.error("XML verification failed", error=str(e))
            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=answer,
                status=VerificationStatus.FAILED,
                confidence=0.5,
                verification_reasoning=f"XML verification failed: {str(e)}",
            )

    async def verify_with_few_shot(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> VerifiedAnswer:
        """
        Verify an answer using few-shot examples for guidance.

        Args:
            question: The original question
            answer: The answer to verify
            sources: Source documents used

        Returns:
            VerifiedAnswer with verification results
        """
        from backend.services.enhanced_prompts import FEW_SHOT_VERIFICATION

        logger.info(
            "Starting few-shot verification",
            question_length=len(question),
            answer_length=len(answer),
        )

        sources_text = self._format_sources(sources) if sources else "No sources provided."

        prompt = FEW_SHOT_VERIFICATION.format(
            question=question,
            answer=answer,
            sources=sources_text,
        )

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=2048,
            )

            response = await llm.ainvoke(prompt)
            output = response.content

            # Parse the response
            verification_result = self._parse_verification_response(output)

            # Look for SUPPORTED/UNSUPPORTED markers
            issues = []
            unsupported_count = output.lower().count('unsupported')
            partial_count = output.lower().count('partial')

            if unsupported_count > 0:
                issues.append(VerificationIssue(
                    issue_type="factual_error",
                    description=f"{unsupported_count} claims not supported by sources",
                    severity="high",
                ))

            if partial_count > 0:
                issues.append(VerificationIssue(
                    issue_type="incomplete",
                    description=f"{partial_count} claims only partially supported",
                    severity="medium",
                ))

            # Extract confidence from response
            confidence_match = re.search(r'Confidence:\s*(\d+)', output, re.IGNORECASE)
            confidence = int(confidence_match.group(1)) / 100.0 if confidence_match else 0.7

            # Check for corrected answer
            corrected_match = re.search(r'Corrected:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
            corrected = corrected_match.group(1).strip() if corrected_match else None

            # Determine status
            if 'VERIFIED' in output.upper() and unsupported_count == 0:
                status = VerificationStatus.VERIFIED
            elif corrected:
                status = VerificationStatus.CORRECTED
            elif 'NEEDS CORRECTION' in output.upper():
                status = VerificationStatus.CORRECTED if corrected else VerificationStatus.UNCERTAIN
            else:
                status = VerificationStatus.VERIFIED if confidence > 0.7 else VerificationStatus.UNCERTAIN

            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=corrected if corrected else answer,
                status=status,
                confidence=confidence,
                issues_found=issues,
                verification_reasoning=f"Few-shot verification: {len(issues)} issues identified",
                corrections_made=["Few-shot verification correction"] if corrected else [],
            )

        except Exception as e:
            logger.error("Few-shot verification failed", error=str(e))
            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=answer,
                status=VerificationStatus.FAILED,
                confidence=0.5,
                verification_reasoning=f"Few-shot verification failed: {str(e)}",
            )

    async def verify_recursive(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 3,
    ) -> VerifiedAnswer:
        """
        Verify and improve an answer using recursive self-improvement.

        This method asks the model to:
        1. Generate/verify the answer
        2. Critique its own response
        3. Improve based on critique
        4. Repeat until satisfied or max iterations reached

        Args:
            question: The original question
            answer: The answer to verify
            sources: Source documents used
            max_iterations: Maximum improvement iterations

        Returns:
            VerifiedAnswer with verification results
        """
        from backend.services.enhanced_prompts import (
            RECURSIVE_IMPROVEMENT_SYSTEM,
            RECURSIVE_IMPROVEMENT_PROMPT,
            parse_recursive_response,
        )

        logger.info(
            "Starting recursive verification",
            question_length=len(question),
            answer_length=len(answer),
            max_iterations=max_iterations,
        )

        sources_text = self._format_sources(sources) if sources else "No sources provided."

        # Modified prompt to verify existing answer instead of generating new one
        prompt = f"""Question: {question}

Current Answer to Verify:
{answer}

Sources:
{sources_text}

Your task:
1. Evaluate the current answer for accuracy, completeness, and clarity
2. If issues found, critique and improve it
3. You may do up to {max_iterations} iterations of improvement
4. Use the XML structure provided

<attempt number="1">
  <response>[Evaluate: is the current answer correct?]</response>
  <self_critique>
    <accuracy>[What might be inaccurate?]</accuracy>
    <completeness>[What's missing?]</completeness>
    <clarity>[What's unclear?]</clarity>
  </self_critique>
  <needs_improvement>[yes/no]</needs_improvement>
</attempt>

... (more attempts if needed)

<final_answer confidence="[0-100]">
[The verified/improved answer]
</final_answer>"""

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=3000,  # More tokens for recursive process
            )

            messages = [
                {"role": "system", "content": RECURSIVE_IMPROVEMENT_SYSTEM},
                {"role": "user", "content": prompt},
            ]

            response = await llm.ainvoke(messages)
            output = response.content

            # Parse recursive response
            parsed = parse_recursive_response(output)

            # Collect issues from all attempts
            issues = []
            corrections_made = []

            for attempt in parsed['attempts']:
                if attempt.get('improvements'):
                    corrections_made.append(f"Iteration {attempt['number']}: {attempt['improvements']}")

            # Determine status based on iterations and final confidence
            num_iterations = len(parsed['attempts'])
            confidence = parsed.get('confidence', 70) / 100.0
            final_answer = parsed.get('final_answer', answer)

            if num_iterations == 1 and confidence > 0.8:
                status = VerificationStatus.VERIFIED
            elif corrections_made:
                status = VerificationStatus.CORRECTED
            elif confidence < 0.5:
                status = VerificationStatus.UNCERTAIN
            else:
                status = VerificationStatus.VERIFIED

            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=final_answer if final_answer else answer,
                status=status,
                confidence=confidence,
                issues_found=issues,
                verification_reasoning=f"Recursive verification: {num_iterations} iterations, {len(corrections_made)} improvements",
                corrections_made=corrections_made,
            )

        except Exception as e:
            logger.error("Recursive verification failed", error=str(e))
            return VerifiedAnswer(
                original_answer=answer,
                verified_answer=answer,
                status=VerificationStatus.FAILED,
                confidence=0.5,
                verification_reasoning=f"Recursive verification failed: {str(e)}",
            )

    async def verify_enhanced(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        use_xml: bool = True,
        use_few_shot: bool = True,
        use_recursive: bool = False,
        max_correction_rounds: int = 2,
    ) -> VerifiedAnswer:
        """
        Enhanced verification combining multiple techniques.

        Automatically selects the best verification approach based on
        the answer complexity and available options.

        Args:
            question: The original question
            answer: The answer to verify
            sources: Source documents used
            use_xml: Whether to use XML structured verification
            use_few_shot: Whether to use few-shot examples
            use_recursive: Whether to use recursive improvement
            max_correction_rounds: Maximum correction iterations

        Returns:
            VerifiedAnswer with verification results
        """
        logger.info(
            "Starting enhanced verification",
            use_xml=use_xml,
            use_few_shot=use_few_shot,
            use_recursive=use_recursive,
        )

        # Use recursive if requested (most thorough)
        if use_recursive:
            return await self.verify_recursive(
                question, answer, sources, max_iterations=max_correction_rounds + 1
            )

        # Use XML structured verification
        if use_xml:
            result = await self.verify_with_xml(question, answer, sources)

            # If issues found and few-shot available, double-check with few-shot
            if result.issues_found and use_few_shot:
                few_shot_result = await self.verify_with_few_shot(question, answer, sources)

                # Merge results - take the more conservative assessment
                if few_shot_result.confidence < result.confidence:
                    result.confidence = few_shot_result.confidence
                result.issues_found.extend(few_shot_result.issues_found)

            return result

        # Use few-shot verification
        if use_few_shot:
            return await self.verify_with_few_shot(question, answer, sources)

        # Fall back to standard verification
        return await self.verify(question, answer, sources, max_correction_rounds)


# Singleton instance
_verification_service: Optional[SelfVerificationService] = None


def get_verification_service() -> SelfVerificationService:
    """Get or create the verification service singleton."""
    global _verification_service
    if _verification_service is None:
        _verification_service = SelfVerificationService()
    return _verification_service
