"""
AIDocumentIndexer - Document Summarization Service
===================================================

Generates summaries for large documents to reduce embedding token costs.
Summaries are stored as special "document_summary" chunks for initial retrieval,
with detailed chunks fetched as needed.

Features:
- Configurable threshold for summarization (page count or KB)
- Uses cost-effective models (gpt-4o-mini by default)
- Creates hierarchical chunk structure (summary -> sections -> details)
- All features are optional via configuration
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import structlog

from langchain_core.messages import HumanMessage, SystemMessage

logger = structlog.get_logger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for document summarization."""

    # Master toggle
    enabled: bool = False

    # Thresholds for when to summarize (either triggers summarization)
    threshold_pages: int = 50
    threshold_kb: int = 100

    # Summary generation
    model: str = "gpt-4o-mini"  # Cost-effective model
    max_summary_tokens: int = 500
    temperature: float = 0.3  # Low temp for consistency

    # Section summarization (for hierarchical chunks)
    enable_section_summaries: bool = False
    sections_per_document: int = 10
    max_section_summary_tokens: int = 200

    # Provider config
    provider: str = "openai"


@dataclass
class SummaryResult:
    """Result of document summarization."""
    document_summary: str
    section_summaries: List[Dict[str, str]] = field(default_factory=list)
    original_length: int = 0
    summary_length: int = 0
    reduction_percent: float = 0.0
    tokens_used: int = 0
    model: str = ""


DOCUMENT_SUMMARY_PROMPT = """You are a document summarization assistant. Your task is to create a concise,
comprehensive summary of the following document that captures its key information, main topics,
and important details.

The summary should:
1. Identify the document type and purpose
2. List the main topics or sections covered
3. Highlight key facts, findings, or conclusions
4. Note any important names, dates, or figures
5. Be written in a neutral, informative tone

Keep the summary under {max_tokens} tokens while preserving the most important information.

Document:
{document_text}

Summary:"""


SECTION_SUMMARY_PROMPT = """Summarize the following section of a larger document in a concise paragraph.
Focus on the main points and key information. Keep it under {max_tokens} tokens.

Section:
{section_text}

Summary:"""


class DocumentSummarizer:
    """
    Document summarization service for large file optimization.

    Generates summaries for documents that exceed configurable thresholds,
    enabling more efficient RAG by searching summaries first.
    """

    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize the summarizer.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or SummarizationConfig()
        self._llm = None

        logger.info(
            "DocumentSummarizer initialized",
            enabled=self.config.enabled,
            threshold_pages=self.config.threshold_pages,
            threshold_kb=self.config.threshold_kb,
            model=self.config.model,
        )

    async def _get_llm(self):
        """Get or create LLM instance for summarization."""
        if self._llm is None:
            try:
                from backend.services.llm import EnhancedLLMFactory

                self._llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                    operation="summarization",
                    track_usage=True,
                )
            except Exception as e:
                logger.warning(
                    "Failed to get LLM from factory, using direct import",
                    error=str(e),
                )
                # Fallback to direct import
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_summary_tokens,
                )

        return self._llm

    def should_summarize(
        self,
        text_length_bytes: int,
        page_count: Optional[int] = None,
    ) -> bool:
        """
        Check if a document should be summarized based on thresholds.

        Args:
            text_length_bytes: Size of document text in bytes
            page_count: Optional page count (if known)

        Returns:
            True if document exceeds threshold and should be summarized
        """
        if not self.config.enabled:
            return False

        text_kb = text_length_bytes / 1024

        # Check KB threshold
        if text_kb >= self.config.threshold_kb:
            logger.debug(
                "Document exceeds KB threshold",
                text_kb=round(text_kb, 2),
                threshold_kb=self.config.threshold_kb,
            )
            return True

        # Check page threshold (if provided)
        if page_count is not None and page_count >= self.config.threshold_pages:
            logger.debug(
                "Document exceeds page threshold",
                page_count=page_count,
                threshold_pages=self.config.threshold_pages,
            )
            return True

        return False

    async def summarize_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SummaryResult:
        """
        Generate a summary for a document.

        Args:
            text: Full document text
            metadata: Optional document metadata (title, type, etc.)

        Returns:
            SummaryResult with document summary and optional section summaries
        """
        if not self.config.enabled:
            return SummaryResult(
                document_summary="",
                original_length=len(text),
                summary_length=0,
                reduction_percent=0.0,
            )

        original_length = len(text)
        logger.info(
            "Summarizing document",
            text_length=original_length,
            metadata=metadata,
        )

        # Truncate if document is very large (prevent excessive token usage)
        # Take beginning, middle, and end for representative sample
        max_chars_for_summary = 50000  # ~12.5k tokens
        if len(text) > max_chars_for_summary:
            chunk_size = max_chars_for_summary // 3
            truncated_text = (
                text[:chunk_size]
                + "\n\n[...middle content...]\n\n"
                + text[len(text) // 2 - chunk_size // 2 : len(text) // 2 + chunk_size // 2]
                + "\n\n[...end content...]\n\n"
                + text[-chunk_size:]
            )
            logger.debug(
                "Truncated document for summarization",
                original_length=original_length,
                truncated_length=len(truncated_text),
            )
        else:
            truncated_text = text

        # Generate document summary
        llm = await self._get_llm()

        prompt = DOCUMENT_SUMMARY_PROMPT.format(
            max_tokens=self.config.max_summary_tokens,
            document_text=truncated_text,
        )

        try:
            messages = [
                SystemMessage(content="You are a helpful document summarization assistant."),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            document_summary = response.content if hasattr(response, 'content') else str(response)

            # Estimate tokens used
            tokens_used = (len(prompt) + len(document_summary)) // 4

            summary_length = len(document_summary)
            reduction = ((original_length - summary_length) / original_length * 100) if original_length > 0 else 0

            logger.info(
                "Document summarized",
                original_length=original_length,
                summary_length=summary_length,
                reduction_percent=round(reduction, 2),
                tokens_used=tokens_used,
            )

            result = SummaryResult(
                document_summary=document_summary,
                original_length=original_length,
                summary_length=summary_length,
                reduction_percent=reduction,
                tokens_used=tokens_used,
                model=self.config.model,
            )

            # Generate section summaries if enabled
            if self.config.enable_section_summaries:
                section_summaries = await self._generate_section_summaries(text)
                result.section_summaries = section_summaries
                result.tokens_used += sum(
                    (len(s.get("summary", "")) + len(s.get("section", ""))) // 4
                    for s in section_summaries
                )

            return result

        except Exception as e:
            logger.error("Failed to summarize document", error=str(e))
            return SummaryResult(
                document_summary="",
                original_length=original_length,
                summary_length=0,
                reduction_percent=0.0,
            )

    async def _generate_section_summaries(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        """
        Generate summaries for document sections.

        Args:
            text: Full document text

        Returns:
            List of section summaries with metadata
        """
        if not self.config.enable_section_summaries:
            return []

        # Split document into sections
        sections = self._split_into_sections(text)

        if len(sections) <= 1:
            return []

        logger.debug(f"Generating summaries for {len(sections)} sections")

        llm = await self._get_llm()
        section_summaries = []

        for i, section in enumerate(sections):
            if len(section.strip()) < 100:  # Skip very short sections
                continue

            prompt = SECTION_SUMMARY_PROMPT.format(
                max_tokens=self.config.max_section_summary_tokens,
                section_text=section[:10000],  # Limit section size
            )

            try:
                messages = [
                    SystemMessage(content="You are a helpful summarization assistant."),
                    HumanMessage(content=prompt),
                ]

                response = await llm.ainvoke(messages)
                summary = response.content if hasattr(response, 'content') else str(response)

                section_summaries.append({
                    "section_index": i,
                    "section_hash": hashlib.md5(section[:500].encode()).hexdigest()[:8],
                    "summary": summary,
                    "original_length": len(section),
                })

            except Exception as e:
                logger.warning(f"Failed to summarize section {i}", error=str(e))

            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.1)

        return section_summaries

    def _split_into_sections(self, text: str) -> List[str]:
        """
        Split document into logical sections.

        Uses various heuristics to identify section boundaries:
        - Double newlines
        - Numbered headings (1., 2., etc.)
        - Common section markers (Chapter, Section, etc.)
        """
        import re

        # Try to split by common section patterns
        section_patterns = [
            r'\n(?=(?:Chapter|Section|Part)\s+\d)',  # Chapter 1, Section 2, etc.
            r'\n(?=\d+\.\s+[A-Z])',  # Numbered headings: 1. Introduction
            r'\n{3,}',  # Multiple blank lines
            r'\n(?=[A-Z][A-Z\s]{10,})\n',  # ALL CAPS HEADINGS
        ]

        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend(p for p in parts if p.strip())
            if len(new_sections) > len(sections):
                sections = new_sections

        # If we got too many sections, merge small ones
        target_sections = self.config.sections_per_document
        if len(sections) > target_sections * 2:
            merged_sections = []
            current_section = ""
            target_length = len(text) // target_sections

            for section in sections:
                current_section += section
                if len(current_section) >= target_length:
                    merged_sections.append(current_section)
                    current_section = ""

            if current_section:
                merged_sections.append(current_section)

            sections = merged_sections

        # If we still have too few sections, split by size
        if len(sections) < 2:
            char_per_section = len(text) // target_sections
            sections = [
                text[i:i + char_per_section]
                for i in range(0, len(text), char_per_section)
            ]

        return sections[:target_sections]

    def create_summary_chunk(
        self,
        document_id: str,
        summary: str,
        chunk_level: int = 2,  # 2 = document level, 1 = section level
        section_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a chunk dictionary for a summary.

        Args:
            document_id: ID of the source document
            summary: The summary text
            chunk_level: Hierarchy level (2=document, 1=section, 0=detail)
            section_index: Optional section index for section summaries

        Returns:
            Chunk dictionary ready for storage
        """
        chunk_hash = hashlib.md5(summary.encode()).hexdigest()

        return {
            "document_id": document_id,
            "content": summary,
            "chunk_index": -(chunk_level * 1000 + (section_index or 0)),  # Negative index for summaries
            "chunk_hash": chunk_hash,
            "is_summary": True,
            "chunk_level": chunk_level,
            "metadata": {
                "type": "document_summary" if chunk_level == 2 else "section_summary",
                "section_index": section_index,
            },
        }


# Singleton instance
_summarizer: Optional[DocumentSummarizer] = None


def get_document_summarizer(config: Optional[SummarizationConfig] = None) -> DocumentSummarizer:
    """
    Get or create the document summarizer singleton.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        DocumentSummarizer singleton instance
    """
    global _summarizer
    if _summarizer is None:
        _summarizer = DocumentSummarizer(config)
    return _summarizer
