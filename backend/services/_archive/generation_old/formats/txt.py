"""
AIDocumentIndexer - TXT Generator
==================================

Plain text document generator.
"""

from backend.services.generation.formats.base import BaseFormatGenerator


class TXTGenerator(BaseFormatGenerator):
    """Plain text document generator."""

    @property
    def extension(self) -> str:
        return "txt"

    @property
    def content_type(self) -> str:
        return "text/plain"

    async def generate(self, job, filename: str) -> str:
        """Generate TXT file."""
        raise NotImplementedError(
            "TXT generation is implemented in DocumentGenerationService._generate_txt"
        )

    def get_format_guidelines(self) -> str:
        return """FORMAT: TXT (Plain Text)

STRUCTURE GUIDELINES:
1. Use ALL CAPS for main title
2. Use "=" or "-" underlines for section headers
3. Use blank lines to separate sections
4. Use consistent indentation for structure
5. Keep line length under 80 characters for readability

CONTENT GUIDELINES:
- Write in plain, readable text without formatting
- Use ASCII characters for compatibility
- Use numbers or asterisks for lists
- Add clear visual separation between sections
- Consider email/terminal readability"""
