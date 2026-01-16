"""
AIDocumentIndexer - XLSX Generator
===================================

Excel spreadsheet generator.
"""

from backend.services.generation.formats.base import BaseFormatGenerator


class XLSXGenerator(BaseFormatGenerator):
    """Excel spreadsheet generator."""

    @property
    def extension(self) -> str:
        return "xlsx"

    @property
    def content_type(self) -> str:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    async def generate(self, job, filename: str) -> str:
        """Generate XLSX file."""
        raise NotImplementedError(
            "XLSX generation is implemented in DocumentGenerationService._generate_xlsx"
        )

    def get_format_guidelines(self) -> str:
        return """FORMAT: XLSX (Excel Spreadsheet)

STRUCTURE GUIDELINES:
1. Create a summary sheet as the first sheet
2. Use separate sheets for different sections/topics
3. Include clear headers in row 1 of each sheet
4. Use consistent column widths
5. Add sheet names that reflect content

CONTENT GUIDELINES:
- Structure data in tabular format when possible
- Use bullet points or numbered steps in cells
- Keep cell content concise
- Add formulas or calculations where relevant
- Consider using conditional formatting for key data"""
