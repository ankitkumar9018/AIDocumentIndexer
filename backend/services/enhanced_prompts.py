"""
AIDocumentIndexer - Enhanced Prompting System
==============================================

Advanced prompting techniques that boost small LLM performance:
1. XML-structured output for reliable parsing
2. Few-shot examples for complex tasks
3. Recursive self-improvement loops
4. Role + Chain-of-Thought + Examples (combined technique)

Based on research that shows these techniques improve small model output quality
by 20-40% on complex tasks.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# XML-Structured Output Templates
# =============================================================================
# Research: XML tags improve output parsing reliability by 95%+ vs free-form text
# Small models especially benefit from explicit structure they can follow

XML_STRUCTURED_ANALYSIS = """Analyze the following and provide a structured response.

<input>
{input_content}
</input>

<instructions>
{instructions}
</instructions>

Provide your response in this EXACT structure:

<analysis>
  <understanding>
    [What is being asked - one paragraph]
  </understanding>

  <evidence>
    <item source="[Source Name]">[Key fact 1]</item>
    <item source="[Source Name]">[Key fact 2]</item>
    <item source="[Source Name]">[Key fact 3]</item>
  </evidence>

  <reasoning>
    [How the evidence connects to answer the question - 2-3 paragraphs]
  </reasoning>

  <answer confidence="[0-100]">
    [Your final answer]
  </answer>

  <caveats>
    [Any important limitations or uncertainties]
  </caveats>
</analysis>"""


XML_STRUCTURED_COMPARISON = """Compare the following items based on the provided context.

<context>
{context}
</context>

<question>
{question}
</question>

Provide your response in this EXACT structure:

<comparison>
  <items>
    <item name="[Item 1 Name]">
      <aspect name="[Aspect 1]">[Value/Description]</aspect>
      <aspect name="[Aspect 2]">[Value/Description]</aspect>
    </item>
    <item name="[Item 2 Name]">
      <aspect name="[Aspect 1]">[Value/Description]</aspect>
      <aspect name="[Aspect 2]">[Value/Description]</aspect>
    </item>
  </items>

  <similarities>
    <point>[Shared characteristic 1]</point>
    <point>[Shared characteristic 2]</point>
  </similarities>

  <differences>
    <point>[Key difference 1]</point>
    <point>[Key difference 2]</point>
  </differences>

  <conclusion confidence="[0-100]">
    [Your synthesized conclusion]
  </conclusion>
</comparison>"""


XML_STRUCTURED_VERIFICATION = """Verify the following answer against the provided sources.

<original_question>
{question}
</original_question>

<proposed_answer>
{answer}
</proposed_answer>

<sources>
{sources}
</sources>

Provide your verification in this EXACT structure:

<verification>
  <factual_check>
    <claim source_support="[supported/unsupported/partial]">[Claim from answer]</claim>
    <claim source_support="[supported/unsupported/partial]">[Another claim]</claim>
  </factual_check>

  <logic_check>
    <assessment>[sound/minor_gaps/flawed]</assessment>
    <explanation>[Why the logic is/isn't sound]</explanation>
  </logic_check>

  <completeness_check>
    <assessment>[complete/partial/incomplete]</assessment>
    <missing>[What's missing, if anything]</missing>
  </completeness_check>

  <verdict confidence="[0-100]">
    <status>[correct/needs_correction/unreliable]</status>
    <corrected_answer>[Improved answer if needed, or "N/A" if correct]</corrected_answer>
  </verdict>
</verification>"""


# =============================================================================
# Few-Shot Enhanced Templates
# =============================================================================
# Research: Few-shot examples improve performance by 15-30% on complex tasks
# Most effective when examples match the task type closely

FEW_SHOT_REASONING = """You are an analytical assistant. Think through problems step-by-step.

Here are examples of how to analyze questions:

---
EXAMPLE 1:
Question: "What caused the 2008 financial crisis?"
Context: "The 2008 crisis was triggered by the collapse of the housing bubble. Subprime mortgages were packaged into complex securities (MBS/CDOs). Banks had excessive leverage ratios of 30:1 or higher."

Reasoning:
Step 1 - Identify the core question: Looking for causes of a specific event
Step 2 - Gather evidence: Housing bubble, subprime mortgages, MBS/CDOs, bank leverage
Step 3 - Connect the dots: Housing bubble → subprime defaults → MBS collapse → overleveraged banks fail
Step 4 - Synthesize: Multiple interrelated causes with a chain reaction

Answer: The 2008 financial crisis was caused by the collapse of the housing bubble, which triggered defaults on subprime mortgages that had been packaged into complex securities (MBS/CDOs). This was amplified by banks' excessive leverage ratios of 30:1+, causing systemic failures when the housing-backed securities lost value.

---
EXAMPLE 2:
Question: "Is remote work more productive than office work?"
Context: "Stanford study showed 13% productivity increase for remote workers. However, collaboration and mentoring suffered. A Microsoft study found 'weak ties' between teams decreased 25%."

Reasoning:
Step 1 - Identify the core question: Comparing two work modes on productivity
Step 2 - Gather evidence: +13% individual productivity, -collaboration, -25% weak ties
Step 3 - Connect the dots: Individual tasks benefit, collaborative tasks suffer
Step 4 - Synthesize: Not a simple yes/no - depends on task type

Answer: Remote work shows mixed productivity effects. Individual focused work increases 13% (Stanford), but collaboration and cross-team connections suffer, with weak ties decreasing 25% (Microsoft). The answer depends on the nature of the work being performed.

---
Now analyze this question:

Question: {question}
Context: {context}

Follow the same step-by-step reasoning pattern. Be specific and cite sources."""


FEW_SHOT_VERIFICATION = """You are a fact-checker. Verify answers against source documents.

Here are examples of verification:

---
EXAMPLE 1:
Question: "What was Company X's Q3 revenue?"
Answer: "Q3 revenue was $5.2 million, up 15% YoY"
Source: "Q3 Financial Report: Revenue reached $5.2M. YoY growth: 15%."

Verification:
- Claim "$5.2 million": ✓ SUPPORTED (exact match in source)
- Claim "up 15% YoY": ✓ SUPPORTED (exact match in source)
- Logic: Sound - claims directly stated in source
- Completeness: Complete - answered what was asked
Verdict: CORRECT (Confidence: 95%)

---
EXAMPLE 2:
Question: "Who founded the company and when?"
Answer: "John Smith founded the company in 2010 with his brother"
Source: "Founded in 2010 by John Smith."

Verification:
- Claim "John Smith founded": ✓ SUPPORTED
- Claim "in 2010": ✓ SUPPORTED
- Claim "with his brother": ✗ UNSUPPORTED (not in source)
- Logic: Mostly sound, but includes unverified information
- Completeness: Over-answered with unsupported detail
Verdict: NEEDS CORRECTION (Confidence: 70%)
Corrected: "John Smith founded the company in 2010."

---
Now verify this:

Question: {question}
Answer: {answer}
Sources: {sources}

Follow the same verification pattern. Mark each claim as SUPPORTED/UNSUPPORTED/PARTIAL."""


FEW_SHOT_SYNTHESIS = """You are a research synthesizer. Combine information from multiple sources.

Here are examples of synthesis:

---
EXAMPLE 1:
Question: "What are the effects of caffeine on sleep?"
Sources:
- Source A: "Caffeine blocks adenosine receptors, reducing sleepiness"
- Source B: "Half-life of caffeine is 5-6 hours in adults"
- Source C: "Studies show caffeine within 6 hours of bedtime reduces sleep by 1 hour"

Synthesis:
From Source A, I understand the mechanism (adenosine blocking).
From Source B, I understand the duration of effects (5-6 hour half-life).
From Source C, I understand the practical impact (1 hour less sleep).

Combined Answer: Caffeine affects sleep by blocking adenosine receptors [A], making you feel less sleepy. With a 5-6 hour half-life [B], consuming caffeine within 6 hours of bedtime can reduce total sleep by approximately 1 hour [C].

---
EXAMPLE 2:
Question: "What's the relationship between exercise and mental health?"
Sources:
- Source A: "Regular exercise increases serotonin and endorphin production"
- Source B: "Meta-analysis: Exercise as effective as antidepressants for mild depression"
- Source C: "30 minutes of moderate exercise 3x/week shows significant mood improvement"

Synthesis:
From Source A, I get the biochemical mechanism.
From Source B, I get comparative effectiveness data.
From Source C, I get specific dosage recommendations.

Combined Answer: Exercise improves mental health through increased serotonin and endorphin production [A]. Research shows it can be as effective as antidepressants for mild depression [B]. The effective dose appears to be 30 minutes of moderate exercise, 3 times per week [C].

---
Now synthesize this:

Question: {question}
Sources:
{sources}

Follow the same synthesis pattern. Cite each source explicitly."""


# =============================================================================
# Recursive Self-Improvement Template
# =============================================================================
# Research: Multiple revision passes improve output quality by 10-25%
# Most effective when each pass has a specific focus

RECURSIVE_IMPROVEMENT_SYSTEM = """You are a self-improving assistant. After each response, you'll critically evaluate and improve it.

PROCESS:
1. Generate initial response
2. Critique focusing on: accuracy, completeness, clarity
3. Improve based on critique
4. Repeat if needed (max 3 iterations)

Output your thinking in this structure:

<attempt number="1">
  <response>[Your initial answer]</response>
  <self_critique>
    <accuracy>[What might be inaccurate?]</accuracy>
    <completeness>[What's missing?]</completeness>
    <clarity>[What's unclear?]</clarity>
  </self_critique>
  <needs_improvement>[yes/no]</needs_improvement>
</attempt>

<attempt number="2">
  <improvements_made>[What you fixed]</improvements_made>
  <response>[Your improved answer]</response>
  <self_critique>...</self_critique>
  <needs_improvement>[yes/no]</needs_improvement>
</attempt>

<final_answer confidence="[0-100]">
[Your best answer after all improvements]
</final_answer>"""


RECURSIVE_IMPROVEMENT_PROMPT = """Question: {question}

Context: {context}

Instructions:
1. Generate your best answer
2. Critically evaluate it for accuracy, completeness, and clarity
3. Improve it if needed (you may do up to 3 iterations)
4. Provide your final answer with confidence level

Remember to show your self-improvement process using the XML structure."""


# =============================================================================
# Role + CoT + Few-Shot Combined Template (Technique 7)
# =============================================================================
# Research: Combining all three techniques produces best results for complex tasks

ADVANCED_COMBINED_TEMPLATE = """<role>
You are a {role_name}: {role_description}
Your expertise includes: {expertise_areas}
Your approach is: {approach_style}
</role>

<examples>
{few_shot_examples}
</examples>

<methodology>
For every question, follow this Chain-of-Thought process:

1. UNDERSTAND: What exactly is being asked? What's the goal?
2. GATHER: What relevant information do I have? What sources apply?
3. ANALYZE: How do the pieces connect? What patterns emerge?
4. SYNTHESIZE: What's the complete answer considering all angles?
5. VERIFY: Is my answer accurate? Complete? Well-supported?
6. PRESENT: Clear, structured, with citations and confidence level
</methodology>

<task>
{task_description}
</task>

<context>
{context}
</context>

<output_format>
Provide your response as:

**Understanding**: [1-2 sentences on what's being asked]

**Evidence**:
- [Key fact 1] [Source]
- [Key fact 2] [Source]
- [Key fact 3] [Source]

**Analysis**: [Your reasoning connecting the evidence]

**Answer**: [Your final answer]

**Confidence**: [0-100]% - [Brief justification]

**Follow-up Questions**: [2-3 related questions]
</output_format>"""


# =============================================================================
# Role Definitions for Different Task Types
# =============================================================================

@dataclass
class RoleDefinition:
    """Defines a role with expertise and approach."""
    name: str
    description: str
    expertise: List[str]
    approach: str


ROLES = {
    "analyst": RoleDefinition(
        name="Senior Research Analyst",
        description="You analyze complex information objectively, finding patterns and insights.",
        expertise=["data analysis", "pattern recognition", "critical evaluation", "synthesis"],
        approach="methodical, evidence-based, thorough yet concise"
    ),
    "fact_checker": RoleDefinition(
        name="Professional Fact-Checker",
        description="You verify claims against sources with meticulous attention to detail.",
        expertise=["source verification", "claim assessment", "bias detection", "accuracy evaluation"],
        approach="skeptical, precise, citation-focused"
    ),
    "synthesizer": RoleDefinition(
        name="Information Synthesizer",
        description="You combine information from multiple sources into coherent insights.",
        expertise=["multi-source integration", "conflict resolution", "gap identification", "summary creation"],
        approach="integrative, balanced, comprehensive"
    ),
    "teacher": RoleDefinition(
        name="Expert Educator",
        description="You explain complex topics clearly, adapting to the audience's level.",
        expertise=["explanation", "analogies", "step-by-step instruction", "concept clarification"],
        approach="clear, patient, engaging, example-rich"
    ),
    "advisor": RoleDefinition(
        name="Strategic Advisor",
        description="You provide actionable recommendations based on thorough analysis.",
        expertise=["strategic thinking", "risk assessment", "opportunity identification", "decision support"],
        approach="practical, forward-thinking, risk-aware"
    ),
}


# =============================================================================
# Few-Shot Example Templates
# =============================================================================

@dataclass
class FewShotExample:
    """A single few-shot example."""
    question: str
    context: str
    reasoning: str
    answer: str


# Pre-built examples for common task types
ANALYSIS_EXAMPLES = [
    FewShotExample(
        question="What factors contributed to the project delay?",
        context="Project timeline: Original deadline March 15. Actual: April 30. Issues: vendor delivery 2 weeks late, 3 team members sick for combined 15 days, requirements changed twice.",
        reasoning="Step 1: Identify delay = 45 days (March 15 → April 30). Step 2: Quantify causes - vendor (14 days), illness (15 days), requirements (unquantified but caused rework). Step 3: 14+15 = 29 days from known causes, ~16 days from requirements changes.",
        answer="The 45-day delay resulted from: vendor delivery issues (14 days), team illness (15 days), and requirements changes (approximately 16 days of rework). The largest contributor was requirement changes, followed closely by vendor issues."
    ),
    FewShotExample(
        question="Should we expand to the European market?",
        context="Current revenue: $50M (North America only). EU market size: $30M addressable. EU regulatory compliance cost: $2M upfront. Competitor X entered EU last year, achieved $8M revenue.",
        reasoning="Step 1: Opportunity = $30M addressable, competitor achieved $8M. Step 2: Costs = $2M upfront compliance. Step 3: ROI potential = if we capture even 25% like competitor, $7.5M revenue vs $2M cost. Step 4: Risk = regulatory complexity, competition.",
        answer="EU expansion appears favorable. With $30M addressable market and competitor X achieving $8M in year one, capturing 25% would yield $7.5M revenue against $2M compliance costs. Recommend phased entry starting with regulatory compliance, then targeted launch."
    ),
]


VERIFICATION_EXAMPLES = [
    FewShotExample(
        question="Verify: 'Tesla sold 1.3 million cars in 2022'",
        context="Tesla 2022 Annual Report: 'Total vehicle deliveries in 2022 reached 1,313,851 units, a 40% increase over 2021.'",
        reasoning="Claim: 1.3 million. Source: 1,313,851. Difference: 13,851 (1% rounding). Claim is accurate within normal rounding.",
        answer="VERIFIED. The claim of 1.3 million is accurate - Tesla's annual report shows 1,313,851 deliveries, which rounds to 1.3 million. Confidence: 98%"
    ),
    FewShotExample(
        question="Verify: 'Amazon started as an online bookstore in 1995'",
        context="Amazon History: 'Jeff Bezos founded Amazon.com in Bellevue, Washington, in July 1994. The site launched online in July 1995, initially selling books.'",
        reasoning="Claim 1: Online bookstore - correct, initially sold books. Claim 2: 1995 - correct for launch date. Note: Founded 1994, launched 1995.",
        answer="VERIFIED with nuance. Amazon launched online in 1995 selling books (correct), though the company was founded in 1994. If the claim means 'started selling online,' it's accurate. Confidence: 90%"
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================

def build_advanced_prompt(
    task_type: str,
    question: str,
    context: str,
    role: str = "analyst",
    include_examples: bool = True,
    num_examples: int = 2,
) -> str:
    """
    Build an advanced prompt combining role, CoT, and few-shot examples.

    Args:
        task_type: Type of task (analysis, verification, synthesis)
        question: The user's question
        context: Relevant context/documents
        role: Role to use from ROLES dictionary
        include_examples: Whether to include few-shot examples
        num_examples: Number of examples to include

    Returns:
        Complete prompt string
    """
    role_def = ROLES.get(role, ROLES["analyst"])

    # Select appropriate examples
    examples_text = ""
    if include_examples:
        if task_type == "verification":
            examples = VERIFICATION_EXAMPLES[:num_examples]
        else:
            examples = ANALYSIS_EXAMPLES[:num_examples]

        examples_text = "\n\n".join([
            f"---\nQuestion: {ex.question}\nContext: {ex.context}\nReasoning: {ex.reasoning}\nAnswer: {ex.answer}"
            for ex in examples
        ])

    return ADVANCED_COMBINED_TEMPLATE.format(
        role_name=role_def.name,
        role_description=role_def.description,
        expertise_areas=", ".join(role_def.expertise),
        approach_style=role_def.approach,
        few_shot_examples=examples_text if examples_text else "[No examples for this task]",
        task_description=f"Answer the following {task_type} question",
        context=context,
    ) + f"\n\n**Question**: {question}"


def parse_xml_response(response: str) -> Dict[str, Any]:
    """
    Parse XML-structured response into a dictionary.

    Args:
        response: LLM response with XML tags

    Returns:
        Dictionary with parsed values
    """
    result = {}

    # Extract common XML elements
    patterns = [
        (r'<understanding>(.*?)</understanding>', 'understanding'),
        (r'<reasoning>(.*?)</reasoning>', 'reasoning'),
        (r'<answer[^>]*>(.*?)</answer>', 'answer'),
        (r'<conclusion[^>]*>(.*?)</conclusion>', 'conclusion'),
        (r'<verdict[^>]*>(.*?)</verdict>', 'verdict'),
        (r'<caveats>(.*?)</caveats>', 'caveats'),
        (r'confidence="(\d+)"', 'confidence'),
        (r'<status>(.*?)</status>', 'status'),
        (r'<corrected_answer>(.*?)</corrected_answer>', 'corrected_answer'),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == 'confidence':
                result[key] = int(value)
            else:
                result[key] = value

    # Extract evidence items
    evidence_items = re.findall(
        r'<item[^>]*source="([^"]*)"[^>]*>(.*?)</item>',
        response,
        re.DOTALL
    )
    if evidence_items:
        result['evidence'] = [
            {'source': source, 'content': content.strip()}
            for source, content in evidence_items
        ]

    # Extract claims for verification
    claims = re.findall(
        r'<claim[^>]*source_support="([^"]*)"[^>]*>(.*?)</claim>',
        response,
        re.DOTALL
    )
    if claims:
        result['claims'] = [
            {'support': support, 'claim': claim.strip()}
            for support, claim in claims
        ]

    return result


def parse_recursive_response(response: str) -> Dict[str, Any]:
    """
    Parse recursive improvement response.

    Args:
        response: LLM response with attempt/final_answer tags

    Returns:
        Dictionary with attempts and final answer
    """
    result = {
        'attempts': [],
        'final_answer': None,
        'confidence': 0,
    }

    # Extract attempts
    attempt_pattern = r'<attempt\s+number="(\d+)">(.*?)</attempt>'
    attempts = re.findall(attempt_pattern, response, re.DOTALL)

    for num, content in attempts:
        attempt_data = {
            'number': int(num),
            'response': '',
            'improvements': '',
            'needs_improvement': True,
        }

        response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
        if response_match:
            attempt_data['response'] = response_match.group(1).strip()

        improvements_match = re.search(r'<improvements_made>(.*?)</improvements_made>', content, re.DOTALL)
        if improvements_match:
            attempt_data['improvements'] = improvements_match.group(1).strip()

        needs_match = re.search(r'<needs_improvement>\s*(yes|no)\s*</needs_improvement>', content, re.IGNORECASE)
        if needs_match:
            attempt_data['needs_improvement'] = needs_match.group(1).lower() == 'yes'

        result['attempts'].append(attempt_data)

    # Extract final answer
    final_match = re.search(
        r'<final_answer[^>]*confidence="(\d+)"[^>]*>(.*?)</final_answer>',
        response,
        re.DOTALL
    )
    if final_match:
        result['confidence'] = int(final_match.group(1))
        result['final_answer'] = final_match.group(2).strip()

    return result


def get_enhanced_prompt_for_task(
    task_type: str,
    question: str,
    context: str,
    model_name: Optional[str] = None,
    use_xml: bool = True,
    use_few_shot: bool = True,
    use_recursive: bool = False,
) -> Tuple[str, str]:
    """
    Get the optimal enhanced prompt for a task.

    Combines techniques based on task requirements and model capability.

    Args:
        task_type: Type of task (analysis, verification, synthesis, comparison)
        question: The user's question
        context: Relevant context
        model_name: Model being used (for capability detection)
        use_xml: Whether to use XML structured output
        use_few_shot: Whether to include few-shot examples
        use_recursive: Whether to use recursive improvement

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    from backend.services.rag_module.prompts import is_tiny_model, is_llama_weak

    # Determine model capabilities
    is_small = model_name and (is_tiny_model(model_name) or is_llama_weak(model_name))

    # Small models benefit most from structured prompts
    if is_small:
        use_xml = True
        use_few_shot = True

    # Select system prompt based on task and options
    if use_recursive:
        system_prompt = RECURSIVE_IMPROVEMENT_SYSTEM
        user_prompt = RECURSIVE_IMPROVEMENT_PROMPT.format(
            question=question,
            context=context,
        )
    elif use_few_shot and task_type in ("analysis", "reasoning"):
        system_prompt = "You are an expert analyst. Follow the examples provided."
        user_prompt = FEW_SHOT_REASONING.format(
            question=question,
            context=context,
        )
    elif use_few_shot and task_type == "verification":
        system_prompt = "You are a meticulous fact-checker. Follow the examples provided."
        user_prompt = FEW_SHOT_VERIFICATION.format(
            question=question,
            answer="[Answer to verify would go here]",
            sources=context,
        )
    elif use_few_shot and task_type == "synthesis":
        system_prompt = "You are an expert at combining information from multiple sources."
        user_prompt = FEW_SHOT_SYNTHESIS.format(
            question=question,
            sources=context,
        )
    elif use_xml and task_type == "comparison":
        system_prompt = "You are an analytical assistant. Use the exact XML structure provided."
        user_prompt = XML_STRUCTURED_COMPARISON.format(
            question=question,
            context=context,
        )
    elif use_xml and task_type == "verification":
        system_prompt = "You are a fact-checker. Use the exact XML structure provided."
        user_prompt = XML_STRUCTURED_VERIFICATION.format(
            question=question,
            answer="[Answer to verify]",
            sources=context,
        )
    elif use_xml:
        system_prompt = "You are an analytical assistant. Use the exact XML structure provided."
        user_prompt = XML_STRUCTURED_ANALYSIS.format(
            input_content=context,
            instructions=question,
        )
    else:
        # Use advanced combined template
        system_prompt = ""
        user_prompt = build_advanced_prompt(
            task_type=task_type,
            question=question,
            context=context,
            role="analyst",
            include_examples=use_few_shot,
        )

    return system_prompt, user_prompt


# =============================================================================
# Integration Functions
# =============================================================================

class EnhancedPromptBuilder:
    """
    Builder class for constructing enhanced prompts.

    Usage:
        prompt = EnhancedPromptBuilder()
            .with_role("analyst")
            .with_cot(True)
            .with_few_shot(examples)
            .with_xml_output(True)
            .build(question, context)
    """

    def __init__(self):
        self._role = "analyst"
        self._use_cot = True
        self._few_shot_examples: List[FewShotExample] = []
        self._use_xml = False
        self._use_recursive = False
        self._custom_instructions: List[str] = []

    def with_role(self, role: str) -> 'EnhancedPromptBuilder':
        """Set the role for the prompt."""
        self._role = role
        return self

    def with_cot(self, enabled: bool = True) -> 'EnhancedPromptBuilder':
        """Enable/disable chain-of-thought reasoning."""
        self._use_cot = enabled
        return self

    def with_few_shot(self, examples: List[FewShotExample]) -> 'EnhancedPromptBuilder':
        """Add few-shot examples."""
        self._few_shot_examples = examples
        return self

    def with_xml_output(self, enabled: bool = True) -> 'EnhancedPromptBuilder':
        """Enable XML-structured output."""
        self._use_xml = enabled
        return self

    def with_recursive_improvement(self, enabled: bool = True) -> 'EnhancedPromptBuilder':
        """Enable recursive self-improvement."""
        self._use_recursive = enabled
        return self

    def with_custom_instruction(self, instruction: str) -> 'EnhancedPromptBuilder':
        """Add a custom instruction."""
        self._custom_instructions.append(instruction)
        return self

    def build(self, question: str, context: str) -> Tuple[str, str]:
        """
        Build the final prompt.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        role_def = ROLES.get(self._role, ROLES["analyst"])

        # Build system prompt
        system_parts = [
            f"You are a {role_def.name}. {role_def.description}",
            f"Your expertise: {', '.join(role_def.expertise)}",
            f"Your approach: {role_def.approach}",
        ]

        if self._use_cot:
            system_parts.append("""
Always think step-by-step:
1. Understand what's being asked
2. Gather relevant evidence
3. Analyze and connect the facts
4. Synthesize your answer
5. Verify accuracy and completeness
""")

        if self._use_recursive:
            system_parts.append("""
After your initial answer, critically evaluate it:
- Is it accurate and well-supported?
- Is it complete?
- Is it clear?
If improvements are needed, revise and show your work.
""")

        for instruction in self._custom_instructions:
            system_parts.append(instruction)

        system_prompt = "\n\n".join(system_parts)

        # Build user prompt
        user_parts = []

        # Add few-shot examples
        if self._few_shot_examples:
            examples_text = "\n\n---\n\n".join([
                f"Example Question: {ex.question}\n"
                f"Example Context: {ex.context}\n"
                f"Example Reasoning: {ex.reasoning}\n"
                f"Example Answer: {ex.answer}"
                for ex in self._few_shot_examples
            ])
            user_parts.append(f"Learn from these examples:\n\n{examples_text}\n\n---\n\nNow answer:")

        # Add context and question
        user_parts.append(f"Context:\n{context}")
        user_parts.append(f"Question: {question}")

        # Add output format if XML
        if self._use_xml:
            user_parts.append("""
Provide your response in this structure:
<thinking>
[Your step-by-step reasoning]
</thinking>

<answer confidence="[0-100]">
[Your final answer]
</answer>

<sources>
[List sources used]
</sources>
""")

        user_prompt = "\n\n".join(user_parts)

        return system_prompt, user_prompt


# Export key components
__all__ = [
    'EnhancedPromptBuilder',
    'build_advanced_prompt',
    'parse_xml_response',
    'parse_recursive_response',
    'get_enhanced_prompt_for_task',
    'ROLES',
    'RoleDefinition',
    'FewShotExample',
    'ANALYSIS_EXAMPLES',
    'VERIFICATION_EXAMPLES',
    'XML_STRUCTURED_ANALYSIS',
    'XML_STRUCTURED_COMPARISON',
    'XML_STRUCTURED_VERIFICATION',
    'FEW_SHOT_REASONING',
    'FEW_SHOT_VERIFICATION',
    'FEW_SHOT_SYNTHESIS',
    'RECURSIVE_IMPROVEMENT_SYSTEM',
    'RECURSIVE_IMPROVEMENT_PROMPT',
    'ADVANCED_COMBINED_TEMPLATE',
]
