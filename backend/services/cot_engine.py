"""
AIDocumentIndexer - Chain-of-Thought Engine
============================================

Forces small LLMs to reason step-by-step for improved accuracy.
Implements multiple CoT strategies for different query types.

Enhanced with:
- XML-structured output for reliable parsing
- Few-shot examples for complex reasoning
- Recursive self-improvement option
- Role-based prompting
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import re

import structlog

from backend.services.llm import LLMFactory
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class ReasoningStrategy(str, Enum):
    """Different reasoning strategies for different query types."""
    STANDARD = "standard"           # General step-by-step reasoning
    ANALYTICAL = "analytical"       # For analysis and comparison tasks
    FACTUAL = "factual"            # For fact-based questions
    MATHEMATICAL = "mathematical"   # For calculations and logic
    CREATIVE = "creative"          # For open-ended/creative tasks
    SYNTHESIS = "synthesis"        # For combining multiple sources


@dataclass
class ReasoningResult:
    """Result of chain-of-thought reasoning."""
    question: str
    thinking_steps: List[str]
    final_answer: str
    confidence: float
    strategy_used: ReasoningStrategy
    verification_notes: Optional[str] = None
    sources_cited: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "thinking_steps": self.thinking_steps,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used.value,
            "verification_notes": self.verification_notes,
            "sources_cited": self.sources_cited or [],
        }


class ChainOfThoughtEngine:
    """
    Force small LLMs to reason step-by-step for improved accuracy.

    This engine applies various chain-of-thought prompting strategies
    to help smaller models achieve Claude-level reasoning quality.

    Key techniques:
    1. Step-by-step decomposition
    2. Self-consistency checking
    3. Multiple reasoning paths
    4. Verification steps
    """

    # Prompt templates for different reasoning strategies
    PROMPTS = {
        ReasoningStrategy.STANDARD: """You are a careful analytical thinker. Think through this problem step by step.

**Question**: {question}

**Relevant Context**:
{context}

**Instructions**: Break down your reasoning into clear numbered steps. After each step, verify it makes sense before proceeding.

**Step 1: Understand the Question**
First, let me understand exactly what is being asked...

**Step 2: Identify Key Information**
The relevant facts from the context are...

**Step 3: Analyze and Connect**
Connecting these facts, I can see that...

**Step 4: Formulate Answer**
Based on my analysis...

**Step 5: Verify Reasoning**
Let me double-check my logic...

**Final Answer**:
[Your clear, concise answer]

**Confidence**: [0-100]%""",

        ReasoningStrategy.ANALYTICAL: """You are an expert analyst. Compare and analyze this systematically.

**Question**: {question}

**Context**:
{context}

**Analysis Framework**:

**Step 1: Define Criteria**
What are the key dimensions to analyze?

**Step 2: Gather Evidence**
What does the context tell us about each dimension?

**Step 3: Compare and Contrast**
How do the elements compare across dimensions?

**Step 4: Identify Patterns**
What patterns or trends emerge?

**Step 5: Draw Conclusions**
What can we conclude from this analysis?

**Step 6: Assess Confidence**
How confident am I, and what uncertainties remain?

**Final Analysis**:
[Your structured analysis]

**Confidence**: [0-100]%""",

        ReasoningStrategy.FACTUAL: """You are a fact-checker. Answer based strictly on the provided evidence.

**Question**: {question}

**Source Documents**:
{context}

**Fact-Checking Process**:

**Step 1: Identify the Claim**
The specific factual question being asked is...

**Step 2: Search for Evidence**
Looking through the sources, I find...

**Step 3: Evaluate Source Quality**
These sources are [reliable/somewhat reliable/unreliable] because...

**Step 4: Cross-Reference**
Multiple sources [agree/disagree] on...

**Step 5: Assess Certainty**
Based on the evidence, I am [certain/fairly certain/uncertain] because...

**Factual Answer**:
[Your evidence-based answer]

**Sources Used**: [List sources]
**Confidence**: [0-100]%""",

        ReasoningStrategy.MATHEMATICAL: """You are a precise calculator. Solve this step by step, showing all work.

**Problem**: {question}

**Given Information**:
{context}

**Solution Process**:

**Step 1: Identify Known Values**
The given values are...

**Step 2: Determine What to Find**
We need to calculate/determine...

**Step 3: Choose Method**
The appropriate method/formula is...

**Step 4: Perform Calculations**
Calculating step by step:
- First: ...
- Then: ...
- Finally: ...

**Step 5: Verify Result**
Let me check by...

**Answer**: [Your numerical/logical answer]
**Units**: [If applicable]
**Confidence**: [0-100]%""",

        ReasoningStrategy.CREATIVE: """You are a creative thinker. Explore this thoughtfully.

**Prompt**: {question}

**Context/Inspiration**:
{context}

**Creative Process**:

**Step 1: Understand the Goal**
The creative objective is...

**Step 2: Brainstorm Approaches**
Possible directions include:
- Option A: ...
- Option B: ...
- Option C: ...

**Step 3: Develop Best Approach**
I'll develop Option [X] because...

**Step 4: Add Depth and Detail**
Enriching with specifics...

**Step 5: Refine and Polish**
Making it more [engaging/clear/impactful]...

**Creative Output**:
[Your creative response]

**Reasoning for Choices**: [Why you made key creative decisions]""",

        ReasoningStrategy.SYNTHESIS: """You are a synthesis expert. Combine these sources into a coherent answer.

**Question**: {question}

**Sources to Synthesize**:
{context}

**Synthesis Process**:

**Step 1: Extract Key Points**
From each source:
- Source 1: Main points are...
- Source 2: Main points are...
- Source 3: Main points are...

**Step 2: Identify Agreements**
The sources agree on...

**Step 3: Identify Conflicts**
The sources disagree on... My resolution is...

**Step 4: Fill Gaps**
Information missing from all sources includes...

**Step 5: Create Unified Answer**
Combining all insights...

**Synthesized Answer**:
[Your comprehensive answer drawing from all sources]

**Key Sources**: [Which sources contributed most]
**Confidence**: [0-100]%"""
    }

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        temperature: float = 0.3,
    ):
        """Initialize the CoT engine with specified LLM."""
        self.provider = provider or settings.DEFAULT_LLM_PROVIDER
        self.model = model or settings.DEFAULT_CHAT_MODEL
        self.temperature = temperature

    def _classify_query(self, question: str) -> ReasoningStrategy:
        """
        Automatically classify the query to select the best reasoning strategy.
        """
        question_lower = question.lower()

        # Mathematical indicators
        math_patterns = [
            r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Arithmetic
            r'calculate|compute|solve|equation',
            r'how many|how much|percentage|ratio',
            r'sum|total|average|mean|median',
        ]
        for pattern in math_patterns:
            if re.search(pattern, question_lower):
                return ReasoningStrategy.MATHEMATICAL

        # Factual indicators
        factual_keywords = [
            'is it true', 'did', 'was', 'were', 'has',
            'when did', 'where is', 'who is', 'what is',
            'fact', 'verify', 'confirm', 'true or false',
        ]
        if any(kw in question_lower for kw in factual_keywords):
            return ReasoningStrategy.FACTUAL

        # Analytical indicators
        analytical_keywords = [
            'compare', 'contrast', 'analyze', 'evaluate',
            'pros and cons', 'advantages', 'disadvantages',
            'difference between', 'similarities',
        ]
        if any(kw in question_lower for kw in analytical_keywords):
            return ReasoningStrategy.ANALYTICAL

        # Creative indicators
        creative_keywords = [
            'write', 'create', 'imagine', 'design',
            'story', 'poem', 'generate', 'brainstorm',
            'ideas for', 'suggest', 'creative',
        ]
        if any(kw in question_lower for kw in creative_keywords):
            return ReasoningStrategy.CREATIVE

        # Synthesis indicators
        synthesis_keywords = [
            'summarize', 'combine', 'integrate', 'synthesize',
            'based on all', 'considering all', 'overall',
        ]
        if any(kw in question_lower for kw in synthesis_keywords):
            return ReasoningStrategy.SYNTHESIS

        # Default to standard reasoning
        return ReasoningStrategy.STANDARD

    async def reason(
        self,
        question: str,
        context: str = "",
        strategy: Optional[ReasoningStrategy] = None,
        max_tokens: int = 2048,
    ) -> ReasoningResult:
        """
        Apply chain-of-thought reasoning to a question.

        Args:
            question: The question to reason about
            context: Relevant context/documents
            strategy: Reasoning strategy (auto-detected if None)
            max_tokens: Maximum tokens for response

        Returns:
            ReasoningResult with thinking steps and final answer
        """
        # Auto-classify if strategy not specified
        if strategy is None:
            strategy = self._classify_query(question)

        logger.info(
            "Applying CoT reasoning",
            strategy=strategy.value,
            question_length=len(question),
            context_length=len(context),
        )

        # Get the appropriate prompt template
        prompt_template = self.PROMPTS[strategy]

        # Format the prompt
        formatted_prompt = prompt_template.format(
            question=question,
            context=context if context else "No additional context provided.",
        )

        # Get LLM and generate response
        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            response = await llm.ainvoke(formatted_prompt)
            output = response.content

            # Parse the response
            result = self._parse_response(question, output, strategy)

            logger.info(
                "CoT reasoning complete",
                strategy=strategy.value,
                confidence=result.confidence,
                steps_count=len(result.thinking_steps),
            )

            return result

        except Exception as e:
            logger.error("CoT reasoning failed", error=str(e))
            # Return a basic result on error
            return ReasoningResult(
                question=question,
                thinking_steps=["Error during reasoning"],
                final_answer=f"Unable to complete reasoning: {str(e)}",
                confidence=0.0,
                strategy_used=strategy,
            )

    def _parse_response(
        self,
        question: str,
        output: str,
        strategy: ReasoningStrategy,
    ) -> ReasoningResult:
        """Parse the LLM response into structured reasoning result."""

        # Extract thinking steps (lines starting with Step or **)
        thinking_steps = []
        step_pattern = r'\*\*Step \d+[:\s].*?\*\*[:\s]*(.*?)(?=\*\*Step|\*\*Final|$)'
        matches = re.findall(step_pattern, output, re.DOTALL | re.IGNORECASE)
        for match in matches:
            step_text = match.strip()
            if step_text:
                thinking_steps.append(step_text[:500])  # Limit step length

        # If no steps found, split by newlines
        if not thinking_steps:
            lines = output.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('**Final') and not line.startswith('**Confidence'):
                    thinking_steps.append(line.strip()[:500])

        # Extract final answer
        final_answer = ""
        answer_patterns = [
            r'\*\*Final Answer\*\*[:\s]*(.*?)(?=\*\*Confidence|$)',
            r'\*\*Synthesized Answer\*\*[:\s]*(.*?)(?=\*\*Key|$)',
            r'\*\*Factual Answer\*\*[:\s]*(.*?)(?=\*\*Sources|$)',
            r'\*\*Creative Output\*\*[:\s]*(.*?)(?=\*\*Reasoning|$)',
            r'\*\*Answer\*\*[:\s]*(.*?)(?=\*\*|$)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                break

        # If no structured answer found, use last paragraph
        if not final_answer:
            paragraphs = output.split('\n\n')
            final_answer = paragraphs[-1].strip() if paragraphs else output

        # Extract confidence
        confidence = 0.7  # Default confidence
        confidence_match = re.search(r'\*\*Confidence\*\*[:\s]*(\d+)', output, re.IGNORECASE)
        if confidence_match:
            confidence = min(100, int(confidence_match.group(1))) / 100.0

        return ReasoningResult(
            question=question,
            thinking_steps=thinking_steps[:10],  # Limit to 10 steps
            final_answer=final_answer[:2000],  # Limit answer length
            confidence=confidence,
            strategy_used=strategy,
        )

    async def reason_with_multiple_paths(
        self,
        question: str,
        context: str = "",
        num_paths: int = 3,
    ) -> ReasoningResult:
        """
        Apply self-consistency by generating multiple reasoning paths.

        Uses majority voting to select the most consistent answer.
        """
        import asyncio

        logger.info(
            "Multi-path reasoning",
            question_length=len(question),
            num_paths=num_paths,
        )

        # Generate multiple reasoning paths in parallel
        tasks = [
            self.reason(question, context, temperature=0.5 + i * 0.1)
            for i in range(num_paths)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = [r for r in results if isinstance(r, ReasoningResult)]

        if not valid_results:
            return ReasoningResult(
                question=question,
                thinking_steps=["All reasoning paths failed"],
                final_answer="Unable to reach a conclusion",
                confidence=0.0,
                strategy_used=ReasoningStrategy.STANDARD,
            )

        # Use the result with highest confidence
        best_result = max(valid_results, key=lambda r: r.confidence)

        # Boost confidence if multiple paths agree
        agreement_count = sum(
            1 for r in valid_results
            if self._answers_similar(r.final_answer, best_result.final_answer)
        )

        confidence_boost = (agreement_count / len(valid_results)) * 0.2
        best_result.confidence = min(1.0, best_result.confidence + confidence_boost)

        best_result.verification_notes = (
            f"Verified with {num_paths} reasoning paths. "
            f"{agreement_count}/{len(valid_results)} paths agreed."
        )

        return best_result

    def _answers_similar(self, answer1: str, answer2: str, threshold: float = 0.7) -> bool:
        """Check if two answers are semantically similar."""
        # Simple word overlap check
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= threshold

    async def reason_with_few_shot(
        self,
        question: str,
        context: str = "",
        examples: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2048,
    ) -> ReasoningResult:
        """
        Apply chain-of-thought reasoning with few-shot examples.

        Uses the enhanced prompting system for better results on small models.

        Args:
            question: The question to reason about
            context: Relevant context/documents
            examples: Optional custom examples, otherwise uses defaults
            max_tokens: Maximum tokens for response

        Returns:
            ReasoningResult with thinking steps and final answer
        """
        from backend.services.enhanced_prompts import (
            FEW_SHOT_REASONING,
            ANALYSIS_EXAMPLES,
            parse_xml_response,
        )

        logger.info(
            "Applying few-shot CoT reasoning",
            question_length=len(question),
            context_length=len(context),
            num_examples=len(examples) if examples else 2,
        )

        # Build few-shot examples section
        if examples:
            examples_text = "\n\n---\n".join([
                f"Question: {ex.get('question', '')}\n"
                f"Context: {ex.get('context', '')}\n"
                f"Reasoning: {ex.get('reasoning', '')}\n"
                f"Answer: {ex.get('answer', '')}"
                for ex in examples
            ])
        else:
            # Use default examples
            examples_text = "\n\n---\n".join([
                f"Question: {ex.question}\n"
                f"Context: {ex.context}\n"
                f"Reasoning: {ex.reasoning}\n"
                f"Answer: {ex.answer}"
                for ex in ANALYSIS_EXAMPLES[:2]
            ])

        # Format prompt with examples
        prompt = FEW_SHOT_REASONING.format(
            question=question,
            context=context if context else "No additional context provided.",
        )

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            response = await llm.ainvoke(prompt)
            output = response.content

            # Parse the response
            result = self._parse_response(question, output, ReasoningStrategy.ANALYTICAL)

            logger.info(
                "Few-shot CoT reasoning complete",
                confidence=result.confidence,
                steps_count=len(result.thinking_steps),
            )

            return result

        except Exception as e:
            logger.error("Few-shot CoT reasoning failed", error=str(e))
            return ReasoningResult(
                question=question,
                thinking_steps=["Error during reasoning"],
                final_answer=f"Unable to complete reasoning: {str(e)}",
                confidence=0.0,
                strategy_used=ReasoningStrategy.ANALYTICAL,
            )

    async def reason_with_xml_structure(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 2048,
    ) -> ReasoningResult:
        """
        Apply chain-of-thought reasoning with XML-structured output.

        Uses XML tags for more reliable parsing of the response.

        Args:
            question: The question to reason about
            context: Relevant context/documents
            max_tokens: Maximum tokens for response

        Returns:
            ReasoningResult with thinking steps and final answer
        """
        from backend.services.enhanced_prompts import (
            XML_STRUCTURED_ANALYSIS,
            parse_xml_response,
        )

        logger.info(
            "Applying XML-structured CoT reasoning",
            question_length=len(question),
            context_length=len(context),
        )

        prompt = XML_STRUCTURED_ANALYSIS.format(
            input_content=context if context else "No additional context provided.",
            instructions=question,
        )

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            response = await llm.ainvoke(prompt)
            output = response.content

            # Parse XML response
            parsed = parse_xml_response(output)

            # Extract thinking steps from evidence items
            thinking_steps = []
            if parsed.get('understanding'):
                thinking_steps.append(f"Understanding: {parsed['understanding']}")
            if parsed.get('evidence'):
                for item in parsed['evidence']:
                    thinking_steps.append(f"Evidence from {item['source']}: {item['content']}")
            if parsed.get('reasoning'):
                thinking_steps.append(f"Reasoning: {parsed['reasoning']}")

            # Get confidence and answer
            confidence = parsed.get('confidence', 70) / 100.0
            final_answer = parsed.get('answer', output)

            result = ReasoningResult(
                question=question,
                thinking_steps=thinking_steps[:10],
                final_answer=final_answer[:2000],
                confidence=confidence,
                strategy_used=ReasoningStrategy.ANALYTICAL,
                verification_notes=parsed.get('caveats'),
            )

            logger.info(
                "XML-structured CoT reasoning complete",
                confidence=result.confidence,
                steps_count=len(result.thinking_steps),
            )

            return result

        except Exception as e:
            logger.error("XML-structured CoT reasoning failed", error=str(e))
            return ReasoningResult(
                question=question,
                thinking_steps=["Error during reasoning"],
                final_answer=f"Unable to complete reasoning: {str(e)}",
                confidence=0.0,
                strategy_used=ReasoningStrategy.ANALYTICAL,
            )

    async def reason_enhanced(
        self,
        question: str,
        context: str = "",
        use_few_shot: bool = True,
        use_xml: bool = True,
        strategy: Optional[ReasoningStrategy] = None,
        max_tokens: int = 2048,
    ) -> ReasoningResult:
        """
        Apply enhanced chain-of-thought reasoning combining multiple techniques.

        This method combines:
        - Role-based prompting
        - Chain-of-thought reasoning
        - Few-shot examples (optional)
        - XML-structured output (optional)

        Recommended for small LLMs where reliability is important.

        Args:
            question: The question to reason about
            context: Relevant context/documents
            use_few_shot: Whether to include few-shot examples
            use_xml: Whether to use XML structure
            strategy: Reasoning strategy (auto-detected if None)
            max_tokens: Maximum tokens for response

        Returns:
            ReasoningResult with thinking steps and final answer
        """
        from backend.services.enhanced_prompts import (
            EnhancedPromptBuilder,
            ANALYSIS_EXAMPLES,
            FewShotExample,
            parse_xml_response,
        )

        # Auto-classify if strategy not specified
        if strategy is None:
            strategy = self._classify_query(question)

        logger.info(
            "Applying enhanced CoT reasoning",
            strategy=strategy.value,
            use_few_shot=use_few_shot,
            use_xml=use_xml,
        )

        # Build enhanced prompt
        builder = EnhancedPromptBuilder()
        builder.with_role("analyst").with_cot(True)

        if use_few_shot:
            builder.with_few_shot([
                FewShotExample(
                    question=ex.question,
                    context=ex.context,
                    reasoning=ex.reasoning,
                    answer=ex.answer,
                )
                for ex in ANALYSIS_EXAMPLES[:2]
            ])

        if use_xml:
            builder.with_xml_output(True)

        system_prompt, user_prompt = builder.build(question, context)

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            # Format as messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await llm.ainvoke(messages)
            output = response.content

            # Parse response based on format
            if use_xml:
                parsed = parse_xml_response(output)

                thinking_steps = []
                # Extract thinking block
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', output, re.DOTALL)
                if thinking_match:
                    thinking_text = thinking_match.group(1)
                    steps = [s.strip() for s in thinking_text.split('\n') if s.strip()]
                    thinking_steps = steps[:10]

                confidence = parsed.get('confidence', 70) / 100.0
                final_answer = parsed.get('answer', output)
            else:
                # Fall back to standard parsing
                result = self._parse_response(question, output, strategy)
                return result

            result = ReasoningResult(
                question=question,
                thinking_steps=thinking_steps,
                final_answer=final_answer[:2000] if isinstance(final_answer, str) else str(final_answer)[:2000],
                confidence=confidence,
                strategy_used=strategy,
            )

            logger.info(
                "Enhanced CoT reasoning complete",
                strategy=strategy.value,
                confidence=result.confidence,
                steps_count=len(result.thinking_steps),
            )

            return result

        except Exception as e:
            logger.error("Enhanced CoT reasoning failed", error=str(e))
            return ReasoningResult(
                question=question,
                thinking_steps=["Error during reasoning"],
                final_answer=f"Unable to complete reasoning: {str(e)}",
                confidence=0.0,
                strategy_used=strategy,
            )


# Singleton instance
_cot_engine: Optional[ChainOfThoughtEngine] = None


def get_cot_engine() -> ChainOfThoughtEngine:
    """Get or create the CoT engine singleton."""
    global _cot_engine
    if _cot_engine is None:
        _cot_engine = ChainOfThoughtEngine()
    return _cot_engine
