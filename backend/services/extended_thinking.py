"""
AIDocumentIndexer - Extended Thinking Service
==============================================

Enables deep reasoning for complex queries.
Allows models to "think longer" before answering.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import re

import structlog

from backend.services.llm import LLMFactory
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class ThinkingLevel(str, Enum):
    """Configurable thinking depth levels."""
    OFF = "off"           # No extended thinking
    MINIMAL = "minimal"   # Quick reflection
    LOW = "low"           # Basic reasoning
    MEDIUM = "medium"     # Standard extended thinking
    HIGH = "high"         # Deep reasoning
    MAX = "max"           # Maximum reasoning depth


@dataclass
class ThinkingStep:
    """A single step in the thinking process."""
    step_number: int
    title: str
    content: str
    confidence: float = 0.0


@dataclass
class ThinkingResult:
    """Result of extended thinking."""
    query: str
    thinking_level: ThinkingLevel
    thinking_steps: List[ThinkingStep] = field(default_factory=list)
    thinking_summary: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    tokens_used: int = 0
    thinking_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "thinking_level": self.thinking_level.value,
            "thinking_steps": [
                {
                    "step": s.step_number,
                    "title": s.title,
                    "content": s.content,
                    "confidence": s.confidence,
                }
                for s in self.thinking_steps
            ],
            "thinking_summary": self.thinking_summary,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
        }


class ExtendedThinkingService:
    """
    Enable extended reasoning for complex queries.

    This service allows smaller LLMs to achieve deeper reasoning
    by explicitly allocating "thinking tokens" before generating
    the final answer.

    Techniques:
    1. Structured thinking prompts
    2. Progressive reasoning depth
    3. Self-questioning
    4. Multiple perspective analysis
    5. Confidence calibration
    """

    # Token budgets for each thinking level
    THINKING_BUDGETS = {
        ThinkingLevel.OFF: 0,
        ThinkingLevel.MINIMAL: 256,
        ThinkingLevel.LOW: 512,
        ThinkingLevel.MEDIUM: 1024,
        ThinkingLevel.HIGH: 2048,
        ThinkingLevel.MAX: 4096,
    }

    # Thinking templates for different levels
    THINKING_TEMPLATES = {
        ThinkingLevel.MINIMAL: """<thinking>
Quick check: {query}
Key point: [one sentence]
Confidence: [low/medium/high]
</thinking>

Answer:""",

        ThinkingLevel.LOW: """<thinking>
Question: {query}

Step 1 - Understand: What exactly is being asked?
Step 2 - Key Info: What do I know that's relevant?
Step 3 - Answer: Based on this...

Confidence: [1-10]
</thinking>

Answer:""",

        ThinkingLevel.MEDIUM: """<thinking>
Question: {query}

**Step 1: Understanding**
Let me make sure I understand the question correctly...
The core question is asking about...

**Step 2: Relevant Knowledge**
What I know about this topic:
- Key fact 1...
- Key fact 2...
- Key fact 3...

**Step 3: Analysis**
Connecting these facts...
The relationship between...
This implies that...

**Step 4: Consider Alternatives**
Could there be other interpretations?
What about edge cases?

**Step 5: Formulate Answer**
Based on my analysis, the answer is...

**Step 6: Confidence Assessment**
I'm [X]% confident because...
Potential uncertainties include...
</thinking>

Answer:""",

        ThinkingLevel.HIGH: """<thinking>
Question: {query}

**PHASE 1: DEEP UNDERSTANDING**

1.1 Surface-level: What is literally being asked?
...

1.2 Implicit: What underlying question might this represent?
...

1.3 Context: What background knowledge is assumed?
...

**PHASE 2: KNOWLEDGE ACTIVATION**

2.1 Direct Knowledge: What do I know directly about this?
...

2.2 Related Knowledge: What related concepts might help?
...

2.3 Counterexamples: What could contradict my initial thoughts?
...

**PHASE 3: MULTI-PERSPECTIVE ANALYSIS**

3.1 Perspective A: Looking at this from [angle 1]...
...

3.2 Perspective B: From [angle 2]...
...

3.3 Synthesis: Combining these perspectives...
...

**PHASE 4: CRITICAL EVALUATION**

4.1 Strength of Evidence: How strong is my evidence?
...

4.2 Logical Validity: Is my reasoning sound?
...

4.3 Potential Biases: What biases might affect my answer?
...

**PHASE 5: ANSWER CONSTRUCTION**

5.1 Core Answer: The main answer is...
...

5.2 Nuances: Important caveats include...
...

5.3 Confidence: I'm [X]% confident because...
...
</thinking>

Answer:""",

        ThinkingLevel.MAX: """<thinking>
Question: {query}

═══════════════════════════════════════════════════════════════
STAGE 1: COMPREHENSIVE UNDERSTANDING
═══════════════════════════════════════════════════════════════

1.1 LITERAL INTERPRETATION
What exactly is being asked, word by word?
...

1.2 DEEPER INTENT
What problem is the asker really trying to solve?
...

1.3 SCOPE ANALYSIS
What's in scope? What's explicitly or implicitly out of scope?
...

1.4 ASSUMPTIONS AUDIT
What assumptions am I making? Are they justified?
...

═══════════════════════════════════════════════════════════════
STAGE 2: KNOWLEDGE MAPPING
═══════════════════════════════════════════════════════════════

2.1 CORE DOMAIN KNOWLEDGE
What fundamental concepts apply here?
...

2.2 ADJACENT KNOWLEDGE
What related domains might provide insight?
...

2.3 HISTORICAL CONTEXT
What historical or evolutionary context is relevant?
...

2.4 CUTTING-EDGE CONSIDERATIONS
What recent developments might affect this?
...

═══════════════════════════════════════════════════════════════
STAGE 3: MULTI-ANGLE ANALYSIS
═══════════════════════════════════════════════════════════════

3.1 FIRST-PRINCIPLES REASONING
Starting from basics, what can we derive?
...

3.2 ANALOGICAL REASONING
What similar problems have known solutions?
...

3.3 CONTRARIAN ANALYSIS
What if the opposite of my initial intuition is true?
...

3.4 EXPERT PERSPECTIVES
How would different experts approach this?
...

═══════════════════════════════════════════════════════════════
STAGE 4: SYNTHESIS AND INTEGRATION
═══════════════════════════════════════════════════════════════

4.1 CONVERGENT EVIDENCE
What conclusions are supported by multiple angles?
...

4.2 CONFLICT RESOLUTION
Where do different approaches disagree? How to resolve?
...

4.3 UNIFIED THEORY
What overarching framework explains all observations?
...

═══════════════════════════════════════════════════════════════
STAGE 5: CRITICAL EVALUATION
═══════════════════════════════════════════════════════════════

5.1 EVIDENCE QUALITY
How reliable is each piece of supporting evidence?
...

5.2 LOGICAL AUDIT
Step through the logic chain - any gaps?
...

5.3 BIAS CHECK
What cognitive biases might be affecting this analysis?
...

5.4 UNCERTAINTY QUANTIFICATION
What are the main sources of uncertainty?
...

═══════════════════════════════════════════════════════════════
STAGE 6: FINAL SYNTHESIS
═══════════════════════════════════════════════════════════════

6.1 CORE ANSWER
The fundamental answer is...
...

6.2 CONFIDENCE LEVEL
I'm [X]% confident, because...
...

6.3 KEY CAVEATS
Important limitations and exceptions...
...

6.4 ACTIONABLE RECOMMENDATIONS
If applicable, what should be done...
...
</thinking>

Answer:"""
    }

    def __init__(
        self,
        provider: str = None,
        model: str = None,
    ):
        """Initialize the extended thinking service."""
        self.provider = provider or settings.DEFAULT_LLM_PROVIDER
        self.model = model or settings.DEFAULT_CHAT_MODEL

    def should_use_extended_thinking(
        self,
        query: str,
        context_length: int = 0,
    ) -> ThinkingLevel:
        """
        Automatically determine appropriate thinking level.

        Based on:
        - Query complexity
        - Question type
        - Context availability
        """
        query_lower = query.lower()

        # Simple factual questions -> minimal
        simple_patterns = [
            r'^what is\s',
            r'^who is\s',
            r'^when did\s',
            r'^where is\s',
            r'^define\s',
        ]
        for pattern in simple_patterns:
            if re.match(pattern, query_lower):
                return ThinkingLevel.MINIMAL

        # Complex analysis questions -> high
        complex_patterns = [
            r'why\s.*\?',
            r'how does.*work',
            r'compare.*and\s',
            r'analyze\s',
            r'evaluate\s',
            r'what are the implications',
            r'pros and cons',
        ]
        for pattern in complex_patterns:
            if re.search(pattern, query_lower):
                return ThinkingLevel.HIGH

        # Very complex -> max
        very_complex_patterns = [
            r'comprehensive analysis',
            r'detailed explanation',
            r'multiple perspectives',
            r'synthesize',
            r'critically evaluate',
        ]
        for pattern in very_complex_patterns:
            if re.search(pattern, query_lower):
                return ThinkingLevel.MAX

        # Default based on query length
        if len(query) > 200:
            return ThinkingLevel.MEDIUM
        elif len(query) > 100:
            return ThinkingLevel.LOW
        else:
            return ThinkingLevel.MINIMAL

    async def think(
        self,
        query: str,
        context: str = "",
        level: Optional[ThinkingLevel] = None,
        max_tokens: int = None,
    ) -> ThinkingResult:
        """
        Apply extended thinking to a query.

        Args:
            query: The question to think about
            context: Additional context (from RAG, etc.)
            level: Thinking depth level (auto-detected if None)
            max_tokens: Override default token budget

        Returns:
            ThinkingResult with thinking steps and answer
        """
        import time
        start_time = time.time()

        # Auto-detect level if not specified
        if level is None:
            level = self.should_use_extended_thinking(query)

        # Handle OFF level
        if level == ThinkingLevel.OFF:
            return await self._direct_answer(query, context)

        logger.info(
            "Starting extended thinking",
            level=level.value,
            query_length=len(query),
        )

        # Get token budget
        thinking_budget = max_tokens or self.THINKING_BUDGETS[level]

        # Get template
        template = self.THINKING_TEMPLATES.get(level, self.THINKING_TEMPLATES[ThinkingLevel.MEDIUM])

        # Format prompt
        prompt = template.format(query=query)

        if context:
            prompt = f"**Context:**\n{context}\n\n{prompt}"

        # Generate with thinking
        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.4,  # Balanced for reasoning
                max_tokens=thinking_budget + 1024,  # Extra for final answer
            )

            response = await llm.ainvoke(prompt)
            output = response.content

            # Parse thinking and answer
            result = self._parse_thinking_response(query, output, level)

            result.tokens_used = thinking_budget
            result.thinking_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Extended thinking complete",
                level=level.value,
                steps=len(result.thinking_steps),
                confidence=result.confidence,
                time_ms=result.thinking_time_ms,
            )

            return result

        except Exception as e:
            logger.error("Extended thinking failed", error=str(e))
            return ThinkingResult(
                query=query,
                thinking_level=level,
                final_answer=f"Thinking process failed: {str(e)}",
                confidence=0.0,
            )

    async def _direct_answer(
        self,
        query: str,
        context: str,
    ) -> ThinkingResult:
        """Generate direct answer without extended thinking."""
        prompt = query
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}"

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
            )

            response = await llm.ainvoke(prompt)

            return ThinkingResult(
                query=query,
                thinking_level=ThinkingLevel.OFF,
                final_answer=response.content,
                confidence=0.7,  # Default confidence for direct answers
            )

        except Exception as e:
            return ThinkingResult(
                query=query,
                thinking_level=ThinkingLevel.OFF,
                final_answer=f"Error: {str(e)}",
                confidence=0.0,
            )

    def _parse_thinking_response(
        self,
        query: str,
        output: str,
        level: ThinkingLevel,
    ) -> ThinkingResult:
        """Parse the thinking response into structured result."""
        # Extract thinking block
        thinking_match = re.search(
            r'<thinking>(.*?)</thinking>',
            output,
            re.DOTALL | re.IGNORECASE
        )

        thinking_text = thinking_match.group(1) if thinking_match else ""

        # Extract final answer
        answer_match = re.search(
            r'(?:Answer:|</thinking>)\s*(.*?)$',
            output,
            re.DOTALL | re.IGNORECASE
        )
        final_answer = answer_match.group(1).strip() if answer_match else output

        # Parse thinking steps
        steps = self._extract_thinking_steps(thinking_text, level)

        # Extract confidence
        confidence = 0.7  # Default
        confidence_match = re.search(
            r"(\d+)%?\s*confident",
            thinking_text,
            re.IGNORECASE
        )
        if confidence_match:
            confidence = min(100, int(confidence_match.group(1))) / 100.0

        # Create summary
        summary = self._create_thinking_summary(steps)

        return ThinkingResult(
            query=query,
            thinking_level=level,
            thinking_steps=steps,
            thinking_summary=summary,
            final_answer=final_answer,
            confidence=confidence,
        )

    def _extract_thinking_steps(
        self,
        thinking_text: str,
        level: ThinkingLevel,
    ) -> List[ThinkingStep]:
        """Extract structured thinking steps from text."""
        steps = []

        # Different patterns for different levels
        if level in [ThinkingLevel.HIGH, ThinkingLevel.MAX]:
            # Look for STAGE/PHASE patterns
            pattern = r'(?:STAGE|PHASE|Step)\s*\d+[:\s]*([^\n]+)\n(.*?)(?=(?:STAGE|PHASE|Step)\s*\d+|═|$)'
        else:
            # Look for Step patterns
            pattern = r'(?:\*\*)?Step\s*\d+[:\s-]*([^\n*]+)(?:\*\*)?\s*(.*?)(?=(?:\*\*)?Step\s*\d+|$)'

        matches = re.findall(pattern, thinking_text, re.DOTALL | re.IGNORECASE)

        for i, (title, content) in enumerate(matches, 1):
            steps.append(ThinkingStep(
                step_number=i,
                title=title.strip(),
                content=content.strip()[:500],  # Limit content length
            ))

        # If no structured steps found, create basic steps
        if not steps and thinking_text:
            paragraphs = [p.strip() for p in thinking_text.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs[:5], 1):
                steps.append(ThinkingStep(
                    step_number=i,
                    title=f"Thought {i}",
                    content=para[:500],
                ))

        return steps

    def _create_thinking_summary(
        self,
        steps: List[ThinkingStep],
    ) -> str:
        """Create a brief summary of the thinking process."""
        if not steps:
            return "No structured thinking captured."

        summary_parts = [f"{s.title}" for s in steps[:5]]
        return "Reasoning covered: " + " → ".join(summary_parts)


# Singleton instance
_extended_thinking: Optional[ExtendedThinkingService] = None


def get_extended_thinking_service() -> ExtendedThinkingService:
    """Get or create the extended thinking service singleton."""
    global _extended_thinking
    if _extended_thinking is None:
        _extended_thinking = ExtendedThinkingService()
    return _extended_thinking
