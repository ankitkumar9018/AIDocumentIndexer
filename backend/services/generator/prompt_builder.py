"""
Prompt Builder

Builds template-aware prompts for LLM content generation.
Includes template context (layouts, constraints, theme) in prompts
so the LLM generates content that fits the template.
"""

from typing import Optional, List, Dict, Any
from .template_analyzer import TemplateAnalysis
from .theme import ThemeProfile


class PromptBuilder:
    """
    Builds prompts for LLM content generation with template context.

    The prompts include:
    - Template constraints (char limits, bullet counts)
    - Available layouts
    - Theme style information
    - Content structure requirements
    """

    def __init__(self, analysis: Optional[TemplateAnalysis] = None):
        """
        Initialize with optional template analysis.

        Args:
            analysis: Pre-analyzed template (optional)
        """
        self.analysis = analysis

    def build_presentation_prompt(
        self,
        topic: str,
        num_slides: int = 10,
        style: str = "professional",
        additional_context: str = "",
        outline: Optional[List[str]] = None,
    ) -> str:
        """
        Build a prompt for generating presentation content.

        Args:
            topic: The presentation topic
            num_slides: Target number of slides
            style: Tone/style (professional, casual, academic, etc.)
            additional_context: Extra context from user
            outline: Optional pre-defined outline

        Returns:
            Complete prompt string
        """
        template_context = ""
        if self.analysis:
            template_context = self.analysis.to_llm_context()

        outline_section = ""
        if outline:
            outline_section = f"""
REQUIRED OUTLINE:
{chr(10).join(f'{i+1}. {item}' for i, item in enumerate(outline))}

Follow this outline exactly. Each item should be a separate slide or section.
"""

        prompt = f"""You are generating content for a presentation about: {topic}

{template_context}

GENERATION REQUIREMENTS:
- Generate exactly {num_slides} slides
- Style: {style}
- Each slide must have a clear, concise title (max 60 chars)
- Bullet points must be complete sentences (80-150 chars per bullet)
- Include speaker notes for each slide (2-3 sentences)
- First slide should be a title slide
- Last slide should be a summary/conclusion

{outline_section}

{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

OUTPUT FORMAT:
Generate content as JSON with this structure:
{{
  "title": "Presentation Title",
  "subtitle": "Optional subtitle",
  "slides": [
    {{
      "slide_number": 1,
      "layout": "title_slide",
      "title": "Main Title",
      "subtitle": "Subtitle text",
      "speaker_notes": "Notes for the presenter"
    }},
    {{
      "slide_number": 2,
      "layout": "title_content",
      "title": "Slide Title",
      "bullets": [
        {{"text": "First bullet point", "sub_bullets": []}},
        {{"text": "Second bullet point", "sub_bullets": ["Sub-point 1"]}}
      ],
      "speaker_notes": "Additional context for this slide"
    }}
  ]
}}

Ensure ALL content fits within the specified constraints.
Generate engaging, professional content."""

        return prompt

    def build_document_prompt(
        self,
        topic: str,
        document_type: str = "report",
        num_sections: int = 5,
        style: str = "professional",
        additional_context: str = "",
    ) -> str:
        """
        Build a prompt for generating document content.

        Args:
            topic: The document topic
            document_type: Type (report, proposal, memo, etc.)
            num_sections: Target number of sections
            style: Tone/style
            additional_context: Extra context

        Returns:
            Complete prompt string
        """
        template_context = ""
        if self.analysis:
            template_context = self.analysis.to_llm_context()

        prompt = f"""You are generating content for a {document_type} about: {topic}

{template_context}

GENERATION REQUIREMENTS:
- Generate {num_sections} main sections
- Style: {style}
- Each section should have a clear heading (max 80 chars)
- Paragraphs should be 3-5 sentences each
- Include bullet points where appropriate
- Start with an executive summary
- End with conclusions/recommendations

{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

OUTPUT FORMAT:
Generate content as JSON with this structure:
{{
  "title": "Document Title",
  "subtitle": "Optional subtitle",
  "sections": [
    {{
      "section_number": 1,
      "heading": "Section Heading",
      "heading_level": 1,
      "paragraphs": [
        {{"text": "Paragraph text...", "style": "Normal"}}
      ],
      "bullet_points": ["Point 1", "Point 2"]
    }}
  ]
}}

Ensure content is well-structured and fits constraints."""

        return prompt

    def build_spreadsheet_prompt(
        self,
        topic: str,
        sheet_type: str = "data",
        num_columns: int = 5,
        additional_context: str = "",
    ) -> str:
        """
        Build a prompt for generating spreadsheet content.

        Args:
            topic: The spreadsheet topic
            sheet_type: Type (data, budget, tracker, etc.)
            num_columns: Target number of columns
            additional_context: Extra context

        Returns:
            Complete prompt string
        """
        template_context = ""
        if self.analysis:
            template_context = self.analysis.to_llm_context()

        prompt = f"""You are generating content for a {sheet_type} spreadsheet about: {topic}

{template_context}

GENERATION REQUIREMENTS:
- Generate headers for {num_columns} columns
- Provide 5-10 sample rows of data
- Cell content should be max 256 characters
- Sheet name max 31 characters
- Use appropriate data types (numbers, dates, text)

{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

OUTPUT FORMAT:
Generate content as JSON with this structure:
{{
  "title": "Spreadsheet Title",
  "sheets": [
    {{
      "name": "Sheet1",
      "headers": ["Column1", "Column2", "Column3"],
      "rows": [
        {{"cells": [{{"value": "Data1"}}, {{"value": 100}}, {{"value": "2024-01-01"}}]}}
      ]
    }}
  ]
}}"""

        return prompt

    def build_slide_regeneration_prompt(
        self,
        slide_content: Dict[str, Any],
        feedback: str,
        action: str = "regenerate",
    ) -> str:
        """
        Build a prompt to regenerate a specific slide.

        Args:
            slide_content: Current slide content
            feedback: User feedback/instructions
            action: Type of action (regenerate, enhance, shorten, etc.)

        Returns:
            Prompt for regeneration
        """
        constraints = ""
        if self.analysis:
            constraints = self.analysis.constraints.to_llm_context()

        action_instructions = {
            "regenerate": "Completely regenerate this slide content based on the feedback.",
            "enhance": "Enhance and improve this slide content while keeping the same structure.",
            "shorten": "Make this content more concise. Reduce text length by 30-50%.",
            "expand": "Add more detail and depth to this content.",
            "change_tone": "Adjust the tone of this content as specified in the feedback.",
        }

        instruction = action_instructions.get(action, action_instructions["regenerate"])

        current_content = f"""
Current slide:
- Title: {slide_content.get('title', '')}
- Layout: {slide_content.get('layout', 'title_content')}
- Bullets: {slide_content.get('bullets', [])}
- Speaker notes: {slide_content.get('speaker_notes', '')}
"""

        prompt = f"""You are editing a slide in a presentation.

{constraints}

{current_content}

USER FEEDBACK:
{feedback}

INSTRUCTION:
{instruction}

OUTPUT FORMAT:
Return the updated slide as JSON:
{{
  "title": "Updated title",
  "layout": "layout_type",
  "bullets": [
    {{"text": "Bullet text", "sub_bullets": []}}
  ],
  "speaker_notes": "Updated speaker notes"
}}

Ensure the updated content:
1. Addresses the user feedback
2. Stays within character limits
3. Maintains professional quality"""

        return prompt

    def build_section_regeneration_prompt(
        self,
        section_content: Dict[str, Any],
        feedback: str,
        action: str = "regenerate",
    ) -> str:
        """
        Build a prompt to regenerate a document section.

        Args:
            section_content: Current section content
            feedback: User feedback
            action: Type of action

        Returns:
            Prompt for regeneration
        """
        constraints = ""
        if self.analysis:
            constraints = self.analysis.constraints.to_llm_context()

        current_content = f"""
Current section:
- Heading: {section_content.get('heading', '')}
- Heading level: {section_content.get('heading_level', 1)}
- Paragraphs: {len(section_content.get('paragraphs', []))}
- Bullet points: {section_content.get('bullet_points', [])}
"""

        prompt = f"""You are editing a section in a document.

{constraints}

{current_content}

USER FEEDBACK:
{feedback}

OUTPUT FORMAT:
Return the updated section as JSON:
{{
  "heading": "Updated heading",
  "heading_level": 1,
  "paragraphs": [
    {{"text": "Paragraph text", "style": "Normal"}}
  ],
  "bullet_points": ["Point 1", "Point 2"]
}}

Ensure the update addresses the feedback while maintaining quality."""

        return prompt

    def add_template_context(self, base_prompt: str) -> str:
        """
        Add template context to an existing prompt.

        Args:
            base_prompt: The base prompt

        Returns:
            Prompt with template context added
        """
        if not self.analysis:
            return base_prompt

        context = self.analysis.to_llm_context()
        return f"""{context}

---

{base_prompt}"""

    def get_constraints_summary(self) -> Dict[str, int]:
        """Get a summary of template constraints."""
        if self.analysis:
            return self.analysis.constraints.model_dump()
        return {
            "title_max_chars": 60,
            "subtitle_max_chars": 100,
            "bullet_max_chars": 150,  # PHASE 12: Increased from 120 for more complete points
            "bullets_per_slide": 7,
            "body_max_chars": 500,
            "speaker_notes_max_chars": 300,
            "heading_max_chars": 80,
            "paragraph_max_chars": 1000,
        }
