"""
AIDocumentIndexer - HTML Generator
===================================

HTML document generator.
"""

from backend.services.generation.formats.base import BaseFormatGenerator


class HTMLGenerator(BaseFormatGenerator):
    """HTML document generator."""

    @property
    def extension(self) -> str:
        return "html"

    @property
    def content_type(self) -> str:
        return "text/html"

    async def generate(self, job, filename: str) -> str:
        """Generate HTML file."""
        raise NotImplementedError(
            "HTML generation is implemented in DocumentGenerationService._generate_html"
        )

    def get_format_guidelines(self) -> str:
        return """FORMAT: HTML

STRUCTURE GUIDELINES:
1. Use semantic HTML5 elements (header, nav, main, section, article, footer)
2. Use h1 for main title, h2-h6 for hierarchy
3. Use ul/ol for lists
4. Include proper meta tags for SEO if public-facing
5. Use responsive-friendly structure

CONTENT GUIDELINES:
- Write content that renders well in web browsers
- Use appropriate semantic markup
- Include alt text for any images
- Consider accessibility guidelines
- Use CSS classes for consistent styling"""
