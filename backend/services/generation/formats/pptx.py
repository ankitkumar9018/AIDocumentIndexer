"""
AIDocumentIndexer - PPTX Generator
===================================

PowerPoint presentation generator.

Note: The full implementation is in the main service module.
This module provides a stub for the modular architecture.
"""

from backend.services.generation.formats.base import BaseFormatGenerator


class PPTXGenerator(BaseFormatGenerator):
    """PowerPoint presentation generator."""

    @property
    def extension(self) -> str:
        return "pptx"

    @property
    def content_type(self) -> str:
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"

    async def generate(self, job, filename: str) -> str:
        """
        Generate PPTX file.

        Note: Currently delegates to the main service implementation.
        Full implementation will be migrated here.
        """
        raise NotImplementedError(
            "PPTX generation is implemented in DocumentGenerationService._generate_pptx"
        )

    def get_format_guidelines(self) -> str:
        return """FORMAT: PPTX (PowerPoint Presentation)

STRUCTURE GUIDELINES:
1. Create 5-10 slides covering the main points
2. Title slide with document title and subtitle
3. One main point per slide for clarity
4. Use bullet points (max 5-6 per slide)
5. Keep bullet text concise (under 80 characters per bullet)
6. Include a summary/conclusion slide
7. Consider adding a "Questions?" slide at the end

CONTENT GUIDELINES:
- Headlines should be impactful and clear (5-8 words)
- Bullet points should be action-oriented when possible
- Use consistent formatting throughout
- Include speaker notes for additional context
- Consider visual hierarchy: title > subtitle > body"""
