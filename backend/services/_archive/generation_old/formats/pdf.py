"""
AIDocumentIndexer - PDF Generator
==================================

PDF document generator using ReportLab.
"""

from backend.services.generation.formats.base import BaseFormatGenerator


class PDFGenerator(BaseFormatGenerator):
    """PDF document generator."""

    @property
    def extension(self) -> str:
        return "pdf"

    @property
    def content_type(self) -> str:
        return "application/pdf"

    async def generate(self, job, filename: str) -> str:
        """Generate PDF file."""
        raise NotImplementedError(
            "PDF generation is implemented in DocumentGenerationService._generate_pdf"
        )

    def get_format_guidelines(self) -> str:
        return """FORMAT: PDF (Portable Document Format)

STRUCTURE GUIDELINES:
1. Include a title page with document title and date
2. Add a table of contents for longer documents
3. Use clear headings and subheadings
4. Include page numbers
5. Consider adding headers/footers

CONTENT GUIDELINES:
- Use professional formatting
- Ensure good readability with appropriate font sizes
- Include margins for printing
- Add visual breaks between sections
- Consider including charts or diagrams where helpful"""
