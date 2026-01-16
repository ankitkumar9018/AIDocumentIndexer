"""
AIDocumentIndexer - Markdown Generator
=======================================

Markdown document generator.
"""

from backend.services.generation.formats.base import BaseFormatGenerator


class MarkdownGenerator(BaseFormatGenerator):
    """Markdown document generator."""

    @property
    def extension(self) -> str:
        return "md"

    @property
    def content_type(self) -> str:
        return "text/markdown"

    async def generate(self, job, filename: str) -> str:
        """Generate Markdown file."""
        raise NotImplementedError(
            "Markdown generation is implemented in DocumentGenerationService._generate_markdown"
        )

    def get_format_guidelines(self) -> str:
        return """FORMAT: MARKDOWN

STRUCTURE GUIDELINES:
1. Use # for main title, ## for sections, ### for subsections
2. Use - or * for unordered lists
3. Use 1. 2. 3. for ordered lists
4. Use **bold** and *italic* for emphasis
5. Use --- for horizontal rules between major sections

CONTENT GUIDELINES:
- Write in a format suitable for rendering in any Markdown viewer
- Include code blocks with appropriate language tags if needed
- Use links [text](url) where helpful
- Add blockquotes > for important callouts
- Keep formatting consistent throughout"""
