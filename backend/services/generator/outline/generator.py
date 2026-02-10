"""
Outline Generator

Generates document outlines using LLM with optional template context.
Migrated from generator.py for modularity.
"""

import re
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import structlog

from ..config import LANGUAGE_NAMES

if TYPE_CHECKING:
    from ..models import GenerationJob, DocumentOutline, SourceReference
    from ..template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


# Generic title patterns to reject
GENERIC_PATTERNS = [
    r'^section\s*\d*$',
    r'^introduction$',
    r'^overview$',
    r'^summary$',
    r'^conclusion$',
    r'^part\s*\d+$',
    r'^section_count',
    r'^content\s+covering',
    r'^tone[:\s]',
    r'^vocabulary[:\s]',
    r'^user$',
    r'^---',
    r'^internal\s+style',
    r'^use\s+these\s+style',
    r'^match\s+the\s+style',
    r'^end\s+internal',
    r'^create\s+a\s+professional',
]

# Lines to completely skip (style guidance and internal prompts)
SKIP_PATTERNS = [
    r'---INTERNAL STYLE GUIDANCE',
    r'---END INTERNAL GUIDANCE',
    r'DO NOT OUTPUT THIS AS CONTENT',
    r'Use these style hints when writing',
    r'Match the style of existing documents',
    r'This is internal guidance only',
    r'^- Tone:',
    r'^- Vocabulary:',
    r'^Tone:',
    r'^Vocabulary:',
]


class OutlineGenerator:
    """Generates document outlines using LLM.

    This class provides outline generation with optional template context,
    ensuring the generated outline fits the template constraints.
    """

    def __init__(self):
        self._llm = None

    async def _get_llm(self):
        """Get the LLM instance for outline generation."""
        if self._llm is None:
            from backend.services.llm import EnhancedLLMFactory
            self._llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,
            )
        return self._llm

    async def generate(
        self,
        job: "GenerationJob",
        template_analysis: Optional["TemplateAnalysis"] = None,
        num_sections: int = 5,
    ) -> "DocumentOutline":
        """Generate an outline for a document.

        Args:
            job: The generation job
            template_analysis: Optional template analysis for constraints
            num_sections: Number of sections to generate

        Returns:
            DocumentOutline with title, description, and sections
        """
        # Get sources for context
        sources = job.sources_used if hasattr(job, 'sources_used') else []

        # Get style guide from metadata
        style_guide = job.metadata.get("style_guide") if job.metadata else None

        # Get output language
        output_language = job.metadata.get("output_language", "en") if job.metadata else "en"

        return await self.generate_with_llm(
            title=job.title,
            description=job.description,
            sources=sources,
            num_sections=num_sections,
            output_format=job.output_format.value if hasattr(job.output_format, 'value') else str(job.output_format),
            style_guide=style_guide,
            output_language=output_language,
        )

    async def generate_with_llm(
        self,
        title: str,
        description: str,
        sources: List["SourceReference"],
        num_sections: Optional[int] = None,
        output_format: str = "docx",
        style_guide: Optional[Dict[str, Any]] = None,
        output_language: str = "en",
    ) -> "DocumentOutline":
        """Generate outline using LLM.

        Args:
            title: Document title
            description: Document description
            sources: Relevant sources from RAG
            num_sections: Number of sections. None = auto mode (LLM decides)
            output_format: Target output format (affects section count guidance)
            style_guide: Optional style analysis from existing documents
            output_language: Language code for generated content (default: en)
        """
        from ..models import DocumentOutline

        # Build context from sources
        context = self._build_source_context(sources)

        # Build style instructions if available
        style_instructions = self._build_style_instructions(style_guide)

        # Build language instruction
        language_instruction = self._build_language_instruction(output_language)

        # Build section instruction based on mode
        section_instruction = self._build_section_instruction(num_sections, output_format)

        # Create prompt for outline generation
        prompt = self._build_outline_prompt(
            title=title,
            description=description,
            context=context,
            style_instructions=style_instructions,
            section_instruction=section_instruction,
            language_instruction=language_instruction,
        )

        # Use LLM to generate
        try:
            llm = await self._get_llm()
            response = await llm.ainvoke(prompt)

            # Log the raw LLM response for debugging
            logger.info(
                "LLM outline response received",
                response_length=len(response.content) if response.content else 0,
                response_preview=response.content[:500] if response.content else "empty",
            )

            # Parse response into sections
            sections, target_sections = self._parse_outline_response(
                response.content,
                title,
                description,
                num_sections,
            )

            return DocumentOutline(
                title=title,
                description=description,
                sections=sections[:target_sections],
            )

        except Exception as e:
            logger.error("Failed to generate outline with LLM", error=str(e))
            return self._create_fallback_outline(title, description, num_sections)

    def _build_source_context(self, sources: List["SourceReference"]) -> str:
        """Build context from sources."""
        if not sources:
            logger.warning(
                "No sources available for outline context - LLM will generate without document knowledge"
            )
            return ""

        context = "Relevant information from the knowledge base:\n\n"
        for source in sources[:5]:
            context += f"- {source.snippet}...\n\n"

        logger.debug(
            "Outline context built from sources",
            sources_used=len(sources[:5]),
            context_length=len(context),
            first_snippet=sources[0].snippet[:100] if sources else "none",
        )
        return context

    def _build_style_instructions(self, style_guide: Optional[Any]) -> str:
        """Build style instructions from style guide.

        Args:
            style_guide: Either a StyleProfile dataclass or a dict with style info
        """
        if not style_guide:
            return ""

        # Handle both StyleProfile dataclass and dict
        if hasattr(style_guide, 'tone'):
            # It's a StyleProfile dataclass
            tone = getattr(style_guide, 'tone', 'professional')
            vocabulary = getattr(style_guide, 'vocabulary_level', 'moderate')
        elif isinstance(style_guide, dict):
            # It's a dict
            tone = style_guide.get('tone', 'professional')
            vocabulary = style_guide.get('vocabulary_level', 'moderate')
        else:
            return ""

        return f"""
---INTERNAL STYLE GUIDANCE (DO NOT OUTPUT THIS AS CONTENT)---
Use these style hints when writing, but DO NOT include them as sections:
- Tone: {tone}
- Vocabulary: {vocabulary}
Match the style of existing documents. This is internal guidance only.
---END INTERNAL GUIDANCE---
"""

    def _build_language_instruction(self, output_language: str) -> str:
        """Build language instruction based on output language."""
        if output_language == "auto":
            return """
LANGUAGE REQUIREMENT:
1. CRITICAL: Detect the language from the USER'S PROMPT/DESCRIPTION - this is what they wrote and what they expect.
2. The user's prompt language takes PRIORITY over any source documents language.
3. If the user wrote their prompt in English, respond in English - even if source documents are in German, French, etc.
4. If the user wrote in German, respond in German.
5. If the title contains Hinglish (Hindi+English mix with clear Hindi words like "ke", "ka", "hai", "mein"), respond in Hinglish.
6. DEFAULT TO ENGLISH if:
   - The user's prompt uses standard English words
   - The language is unclear or ambiguous
   - The prompt contains only technical terms or proper nouns
   - You are unsure about the language
7. IGNORE the language of source/reference documents - only the USER'S prompt matters for output language.

IMPORTANT: The user's prompt language determines output language. Source documents in other languages should be translated to match the user's language.
"""
        elif output_language == "en":
            return """
---LANGUAGE REQUIREMENT---
OUTPUT LANGUAGE: English

IMPORTANT: The user has explicitly selected ENGLISH as the output language.
- Generate ALL section titles and descriptions in English.
- Even if the title/topic contains words from other languages (Hindi, German, etc.), output must be in English.
- Translate any non-English terms in the title to English concepts.
- Do NOT use Hinglish, Hindi, or any other language - only English.
---END LANGUAGE REQUIREMENT---
"""
        else:
            language_name = LANGUAGE_NAMES.get(output_language, output_language.upper())
            return f"""
---LANGUAGE REQUIREMENT---
OUTPUT LANGUAGE: {language_name}

IMPORTANT: The user has explicitly selected {language_name.upper()} as the output language.
- Generate ALL section titles and descriptions in {language_name}.
- Even if the title/topic contains words from other languages, output must be in {language_name}.
- Translate any terms from other languages to {language_name}.
- Do NOT use any other language - only {language_name}.
- Technical terms may remain in English if commonly used that way in {language_name}.
---END LANGUAGE REQUIREMENT---
"""

    def _build_section_instruction(
        self,
        num_sections: Optional[int],
        output_format: str,
    ) -> str:
        """Build section instruction based on mode."""
        if num_sections is None:
            # Auto mode - LLM decides optimal count
            return f"""Analyze the topic complexity and determine the optimal number of sections.

Consider:
- Topic depth and complexity
- Output format: {output_format} (presentations typically need 5-10 slides, documents 3-8 pages)
- Amount of source material available
- Target audience expectations

First, output your recommended section count on a line by itself like:
SECTION_COUNT: N

Where N is between 3 and 15.

Then generate exactly N sections with specific, descriptive titles."""
        else:
            return f"Generate exactly {num_sections} sections with specific, descriptive titles."

    def _build_outline_prompt(
        self,
        title: str,
        description: str,
        context: str,
        style_instructions: str,
        section_instruction: str,
        language_instruction: str,
    ) -> str:
        """Build the full outline generation prompt."""
        return f"""Create a professional document outline for the following:
{style_instructions}
Title: {title}
Description: {description}

{context}

{section_instruction}

CRITICAL RULES:
1. Each section title MUST contain specific keywords from the topic "{title}"
2. NEVER use generic template titles - every title must be unique to THIS specific topic
3. Include specific nouns, brands, products, or concepts from the topic in each title
4. Descriptions must explain WHAT SPECIFIC CONTENT will be in that section

ABSOLUTELY FORBIDDEN GENERIC TITLES (never use these or similar):
- "Background and Context", "Key Analysis", "Strategic Recommendations"
- "Implementation Details", "Conclusion and Next Steps", "Overview"
- "Introduction", "Summary", "Key Findings", "Analysis and Findings"
- "Action Items", "Next Steps", "Recommendations"
- Any title that could apply to ANY document topic

Format each section EXACTLY like this:
## [Title with specific topic keywords]
Description: [Specific content about THIS topic, not generic filler]

GOOD EXAMPLE for "marketing strategy for upcoming shoe launch":
## Target Demographics for Athletic Footwear
Description: Analysis of key customer segments including runners, casual athletes, and fashion-conscious buyers aged 18-35.

## Social Media Campaign for Shoe Release
Description: Instagram, TikTok, and influencer partnership strategies to build pre-launch excitement.

## Retail and E-commerce Distribution Plan
Description: Store placement strategy, online launch timing, and inventory allocation across channels.

## Competitive Pricing Analysis vs Nike and Adidas
Description: Price point positioning relative to competitors, value proposition, and promotional pricing strategy.

## Launch Event and PR Timeline
Description: Press release schedule, influencer seeding dates, and launch day activation plans.

BAD EXAMPLE (NEVER DO THIS):
## Background and Context ❌
## Key Analysis and Findings ❌
## Strategic Recommendations ❌
## Implementation Details ❌
## Conclusion and Next Steps ❌

{language_instruction}

Generate the outline for "{title}" with SPECIFIC, NON-GENERIC titles now:"""

    def _parse_outline_response(
        self,
        response_content: str,
        title: str,
        description: str,
        num_sections: Optional[int],
    ) -> tuple:
        """Parse LLM response into sections.

        Returns:
            Tuple of (sections list, target section count)
        """
        sections = []
        lines = response_content.split("\n")
        current_section = None

        # In auto mode, extract the section count from LLM response
        target_sections = num_sections
        if num_sections is None:
            # Look for SECTION_COUNT: N pattern
            section_count_match = re.search(r'SECTION_COUNT:\s*(\d+)', response_content)
            if section_count_match:
                target_sections = int(section_count_match.group(1))
                # Clamp to valid range
                target_sections = max(3, min(15, target_sections))
                logger.info(f"Auto mode: LLM suggested {target_sections} sections")
            else:
                # Fallback: count sections in response, or default to 5
                target_sections = 5
                logger.warning("Auto mode: Could not extract section count, defaulting to 5")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip internal style guidance and prompt echoes
            should_skip = any(
                re.search(pattern, line, re.IGNORECASE)
                for pattern in SKIP_PATTERNS
            )
            if should_skip:
                continue

            # Skip SECTION_COUNT metadata line
            if re.search(r'SECTION_COUNT\s*:', line, re.IGNORECASE):
                continue

            # Skip conversational LLM artifacts
            if re.match(r'^#*\s*(User|Assistant|Human|AI|System|Here is|Below is|I\'ll|Let me|Sure|Certainly)[:.]?\s', line, re.IGNORECASE):
                continue

            # Check for section header - multiple formats supported
            is_section_header = (
                line.startswith("##") or
                line.startswith("**") or
                line.startswith("- ") or
                (len(line) > 0 and line[0].isdigit() and ("." in line[:4] or ")" in line[:4])) or
                (line.endswith(":") and len(line) < 100 and not line.lower().startswith("description"))
            )

            if is_section_header:
                if current_section and current_section["title"]:
                    sections.append(current_section)

                # Extract section title, removing markdown formatting and numbering
                section_title = re.sub(r'^[#\d.\s\-\*\)]+', '', line).strip()
                section_title = section_title.rstrip(':').strip()
                section_title = re.sub(r'\*+', '', section_title).strip()

                # Check if title is generic
                is_generic = any(
                    re.match(pattern, section_title.lower().strip())
                    for pattern in GENERIC_PATTERNS
                )

                if is_generic or not section_title:
                    # Generate a better title based on document topic
                    section_title = f"Key Aspect {len(sections) + 1} of {description[:30].split()[0].title() if description else 'Topic'}"

                current_section = {"title": section_title, "description": ""}

            elif current_section and line:
                # Handle description lines
                if line.lower().startswith("description:"):
                    desc_content = line[12:].strip()
                    if desc_content:
                        current_section["description"] = desc_content + " "
                elif not current_section["description"]:
                    current_section["description"] = line.strip() + " "
                elif current_section["description"] and not current_section["description"].strip():
                    current_section["description"] = line.strip() + " "

        if current_section and current_section["title"]:
            sections.append(current_section)

        # Log parsed sections count
        logger.info(
            "Outline sections parsed from LLM response",
            parsed_sections=len(sections),
            target_sections=target_sections,
            section_titles=[s.get("title", "")[:50] for s in sections[:5]],
        )

        # Ensure all sections have non-empty descriptions
        for section in sections:
            if not section.get("description") or not section["description"].strip():
                section["description"] = f"Content covering {section['title'].lower()}. "

        # Ensure we have the requested number of sections
        # Extract a short topic phrase from the title for topic-aware fallbacks
        topic = title.split(":")[0].strip() if title else "the Topic"
        # Trim to a reasonable length
        if len(topic) > 40:
            topic_words = topic.split()[:5]
            topic = " ".join(topic_words)

        while len(sections) < target_sections:
            section_num = len(sections) + 1
            fallback_titles = [
                (f"Key Findings on {topic}", f"Important findings and insights related to {topic}"),
                (f"Analysis of {topic}", f"Detailed analysis and recommendations for {topic}"),
                (f"Implementation for {topic}", f"How to implement approaches for {topic}"),
                (f"Strategic Considerations", f"Strategic factors and considerations for {topic}"),
                (f"Supporting Details", f"Additional supporting details for {topic}"),
                (f"Additional Context", f"Extra context and background for {topic}"),
            ]
            idx = min(section_num - 1, len(fallback_titles) - 1)
            sections.append({
                "title": fallback_titles[idx][0],
                "description": fallback_titles[idx][1],
            })

        return sections, target_sections

    def _create_fallback_outline(
        self,
        title: str,
        description: str,
        num_sections: Optional[int],
    ) -> "DocumentOutline":
        """Create a fallback outline when LLM generation fails.

        Uses topic-aware titles derived from the document title so that
        fallback outlines are contextually relevant rather than generic.
        """
        from ..models import DocumentOutline

        # Extract a short topic phrase from the title
        topic = title.split(":")[0].strip() if title else "the Topic"
        if len(topic) > 40:
            topic = " ".join(topic.split()[:5])

        fallback_count = num_sections if num_sections is not None else 5
        fallback_section_templates = [
            (f"Introduction to {topic}", f"Overview and background of {title}"),
            (f"Current State of {topic}", f"Analysis of the current landscape for {title}"),
            (f"Key Challenges in {topic}", f"Challenges and issues related to {title}"),
            (f"Strategies for {topic}", f"Recommended approaches for {title}"),
            (f"Conclusion & Next Steps", f"Summary and action items for {title}"),
            (f"Additional Resources", f"Supporting details and references for {title}"),
        ]

        return DocumentOutline(
            title=title,
            description=description,
            sections=[
                {
                    "title": fallback_section_templates[min(i, len(fallback_section_templates) - 1)][0],
                    "description": fallback_section_templates[min(i, len(fallback_section_templates) - 1)][1]
                }
                for i in range(fallback_count)
            ],
        )

    async def generate_with_context(
        self,
        topic: str,
        description: str,
        template_analysis: "TemplateAnalysis",
        num_sections: int = 5,
        target_audience: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> "DocumentOutline":
        """Generate an outline with template context.

        Args:
            topic: The document topic
            description: Description of what the document should cover
            template_analysis: Template analysis for constraints
            num_sections: Target number of sections
            target_audience: Optional target audience
            tone: Optional tone (professional, casual, etc.)

        Returns:
            DocumentOutline that fits template constraints
        """
        from ..models import DocumentOutline
        from ..prompt_builder import PromptBuilder

        # Build prompt with template context
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_outline_prompt(
            topic=topic,
            description=description,
            template_analysis=template_analysis,
            num_sections=num_sections,
            target_audience=target_audience,
            tone=tone,
        )

        llm = await self._get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse the response into a DocumentOutline
        sections, target = self._parse_outline_response(content, topic, description, num_sections)

        return DocumentOutline(
            title=topic,
            description=description,
            sections=sections[:target],
            target_audience=target_audience,
            tone=tone,
        )
