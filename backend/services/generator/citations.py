"""
Citation Support for Document Generation

Functions for generating content with inline citations and formatting
citations for different output formats.
"""

import re
from typing import List, TYPE_CHECKING

import structlog

from .models import CitationMapping, ContentWithCitations

if TYPE_CHECKING:
    from .models import SourceReference, Section

logger = structlog.get_logger(__name__)


# =============================================================================
# Citation Generation
# =============================================================================

async def generate_content_with_citations(
    prompt: str,
    sources: List["SourceReference"],
    citation_style: str = "numbered",
    max_citations: int = 10,
) -> ContentWithCitations:
    """Generate content with inline citation markers.

    Uses LLM to generate content that includes citation markers [1], [2], etc.
    for facts sourced from the provided documents.

    Args:
        prompt: The content generation prompt
        sources: List of SourceReference objects to cite from
        citation_style: Style of citations ("numbered", "superscript", "author_year")
        max_citations: Maximum number of distinct citations to include

    Returns:
        ContentWithCitations with content and citation mapping
    """
    if not sources:
        # No sources to cite - generate without citations
        try:
            from backend.services.llm import EnhancedLLMFactory
            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,
            )
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return ContentWithCitations(
                content=content.strip(),
                citations=[],
                citation_style=citation_style,
            )
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return ContentWithCitations(content="", citations=[], citation_style=citation_style)

    # Prepare sources for citation
    source_context = _format_sources_for_citation(sources[:max_citations])

    # Build citation-aware prompt
    citation_prompt = f"""{prompt}

---CITATION REQUIREMENTS---
You have access to the following source documents. Include inline citations [1], [2], etc. after statements that use information from these sources.

SOURCES:
{source_context}

CITATION RULES:
1. Add citation markers [1], [2], etc. AFTER statements that use specific information from the sources
2. Multiple citations can be combined: [1][2] or [1, 2]
3. Only cite when using specific facts, statistics, or claims from the sources
4. General knowledge does not need citations
5. Each source number corresponds to the numbered sources above
6. Do NOT cite information that isn't in the sources - only cite what you actually use
7. Place citation markers at the end of the sentence, before the period

Example: The company's revenue increased by 45% in Q3 [1]. This growth was driven by new product launches [2].

Generate the content with appropriate citations:"""

    try:
        from backend.services.llm import EnhancedLLMFactory
        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="content_generation",
            user_id=None,
        )
        response = await llm.ainvoke(citation_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        content = content.strip()

        # Extract citation mappings
        citations = _extract_citation_mappings(content, sources[:max_citations])

        return ContentWithCitations(
            content=content,
            citations=citations,
            citation_style=citation_style,
        )

    except Exception as e:
        logger.error(f"Content with citations generation failed: {e}")
        return ContentWithCitations(content="", citations=[], citation_style=citation_style)


def _format_sources_for_citation(sources: List["SourceReference"]) -> str:
    """Format sources for LLM prompt with numbered references."""
    formatted = []
    for i, source in enumerate(sources, 1):
        page_info = f", page {source.page_number}" if source.page_number else ""
        formatted.append(
            f"[{i}] {source.document_name}{page_info}:\n"
            f"    {source.snippet[:500]}..."
        )
    return "\n\n".join(formatted)


def _extract_citation_mappings(
    content: str,
    sources: List["SourceReference"],
) -> List[CitationMapping]:
    """Extract citation markers from content and map to sources."""
    citations = []

    # Find all citation markers in the content
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    found_numbers = set()

    for match in re.finditer(citation_pattern, content):
        # Handle both [1] and [1, 2] formats
        numbers_str = match.group(1)
        numbers = [int(n.strip()) for n in numbers_str.split(',')]
        found_numbers.update(numbers)

    # Create mappings for found citations
    for num in sorted(found_numbers):
        if 1 <= num <= len(sources):
            source = sources[num - 1]
            citations.append(CitationMapping(
                marker=f"[{num}]",
                source_document=source.document_name,
                source_page=source.page_number,
                source_snippet=source.snippet[:200],
                relevance_score=source.relevance_score,
            ))

    return citations


# =============================================================================
# Citation Formatting
# =============================================================================

def format_citations_for_footnotes(citations: List[CitationMapping]) -> str:
    """Format citations as footnotes for DOCX documents."""
    footnotes = []
    for i, citation in enumerate(citations, 1):
        page_ref = f", p. {citation.source_page}" if citation.source_page else ""
        footnotes.append(f"{citation.marker} {citation.source_document}{page_ref}")
    return "\n".join(footnotes)


def format_citations_for_speaker_notes(citations: List[CitationMapping]) -> str:
    """Format citations for PPTX speaker notes."""
    if not citations:
        return ""

    notes = ["Sources referenced in this slide:"]
    for citation in citations:
        page_ref = f" (p. {citation.source_page})" if citation.source_page else ""
        notes.append(f"  {citation.marker} {citation.source_document}{page_ref}")

    return "\n".join(notes)


def strip_citation_markers(content: str) -> str:
    """Remove citation markers from content.

    Useful when outputting to formats that don't support citations
    or when citations should be handled separately.
    """
    return re.sub(r'\s*\[\d+(?:,\s*\d+)*\]\s*', ' ', content).strip()


def convert_citations_to_superscript(content: str) -> str:
    """Convert [1] style citations to superscript numbers.

    Note: This returns Unicode superscript characters.
    For actual superscript formatting, use the document library's
    formatting capabilities.
    """
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    }

    def replace_citation(match):
        numbers = match.group(1)
        superscript = ''.join(superscript_map.get(c, c) for c in numbers if c.isdigit())
        return superscript

    return re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace_citation, content)


async def add_citations_to_section(
    section: "Section",
    include_citations: bool = True,
    citation_style: str = "numbered",
) -> "Section":
    """Add inline citations to a section's content.

    Regenerates content with citation markers if include_citations is True.

    Args:
        section: Section to add citations to
        include_citations: Whether to include inline citations
        citation_style: Style of citations

    Returns:
        Updated section with citations
    """
    if not include_citations or not section.sources:
        return section

    # Generate content with citations
    prompt = f"""Rewrite this content with inline citation markers [1], [2], etc. where appropriate.

Original content:
{section.content}

Add citation markers after statements that use information from the sources.
Keep the content largely the same - just add citation markers."""

    result = await generate_content_with_citations(
        prompt=prompt,
        sources=section.sources,
        citation_style=citation_style,
    )

    if result.content:
        section.content = result.content
        # Store citations in metadata for later use
        if section.metadata is None:
            section.metadata = {}
        section.metadata["citations"] = [
            {
                "marker": c.marker,
                "source_document": c.source_document,
                "source_page": c.source_page,
            }
            for c in result.citations
        ]

    return section
