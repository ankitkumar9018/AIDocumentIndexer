"""
AIDocumentIndexer - DOCX Generator
===================================

Word document generator.
"""

from backend.services.generation.formats.base import BaseFormatGenerator


class DOCXGenerator(BaseFormatGenerator):
    """Word document generator."""

    @property
    def extension(self) -> str:
        return "docx"

    @property
    def content_type(self) -> str:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    async def generate(self, job, filename: str) -> str:
        """Generate DOCX file."""
        raise NotImplementedError(
            "DOCX generation is implemented in DocumentGenerationService._generate_docx"
        )

    def get_format_guidelines(self) -> str:
        return """FORMAT: DOCX (Word Document)

STRUCTURE GUIDELINES:
1. Use clear headings hierarchy (H1 for title, H2 for sections, H3 for subsections)
2. Include a table of contents if document has 5+ sections
3. Use paragraphs for detailed explanations
4. Include bullet or numbered lists where appropriate
5. Add page breaks between major sections

CONTENT GUIDELINES:
- Write in complete sentences and paragraphs
- Use professional language appropriate for formal documents
- Include section transitions for flow
- Consider including a summary or executive summary
- Add citations and references where applicable"""
