"""
AIDocumentIndexer - HyDE (Hypothetical Document Embeddings)
============================================================

Implements HyDE for improved retrieval through query expansion.
Instead of embedding the query directly, HyDE generates a hypothetical
answer/document and uses that for retrieval.

Reference: https://arxiv.org/abs/2212.10496

This bridges the gap between query and document language, improving
recall especially for:
- Short queries
- Queries with different vocabulary than documents
- Complex multi-hop questions
"""

from dataclasses import dataclass
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HyDEResult:
    """Result from HyDE expansion."""
    original_query: str
    hypothetical_document: str
    combined_queries: List[str]  # [original, hypothetical, ...]


class HyDEExpander:
    """
    HyDE query expansion for improved retrieval.

    Generates a hypothetical document/answer that would ideally
    contain the answer to the query, then uses that for retrieval.
    """

    def __init__(
        self,
        include_original: bool = True,
        max_hypothetical_length: int = 500,
    ):
        """
        Initialize HyDE expander.

        Args:
            include_original: Include original query in results
            max_hypothetical_length: Max length of generated document
        """
        self.include_original = include_original
        self.max_hypothetical_length = max_hypothetical_length

    async def expand(
        self,
        query: str,
        llm: object,
        context: Optional[str] = None,
    ) -> HyDEResult:
        """
        Expand query using HyDE.

        Args:
            query: Original user query
            llm: LangChain LLM for generation
            context: Optional context about the document collection

        Returns:
            HyDEResult with hypothetical document
        """
        hypothetical = await self._generate_hypothetical(query, llm, context)

        combined = []
        if self.include_original:
            combined.append(query)
        combined.append(hypothetical)

        logger.info(
            "HyDE expansion complete",
            query_preview=query[:50],
            hypothetical_preview=hypothetical[:100],
        )

        return HyDEResult(
            original_query=query,
            hypothetical_document=hypothetical,
            combined_queries=combined,
        )

    async def _generate_hypothetical(
        self,
        query: str,
        llm: object,
        context: Optional[str] = None,
    ) -> str:
        """Generate hypothetical document/answer."""
        context_hint = ""
        if context:
            context_hint = f"\nContext about the documents: {context}\n"

        prompt = f"""Write a short passage (2-3 sentences) that would be found in a document that answers this question. Write as if you are quoting directly from an authoritative document.
{context_hint}
Question: {query}

Passage:"""

        try:
            from langchain_core.messages import HumanMessage

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            hypothetical = response.content.strip()

            # Truncate if too long
            if len(hypothetical) > self.max_hypothetical_length:
                hypothetical = hypothetical[:self.max_hypothetical_length]

            return hypothetical

        except Exception as e:
            logger.warning("HyDE generation failed, using original query", error=str(e))
            return query

    async def expand_multi(
        self,
        query: str,
        llm: object,
        num_hypotheticals: int = 3,
        context: Optional[str] = None,
    ) -> HyDEResult:
        """
        Generate multiple hypothetical documents for better coverage.

        Args:
            query: Original user query
            llm: LangChain LLM
            num_hypotheticals: Number of hypothetical docs to generate
            context: Optional context

        Returns:
            HyDEResult with multiple hypotheticals
        """
        prompt = f"""Write {num_hypotheticals} different short passages (2-3 sentences each) that might be found in documents answering this question. Each passage should take a slightly different angle or use different terminology.

Question: {query}

Format each passage on a new line, numbered 1., 2., 3., etc."""

        try:
            from langchain_core.messages import HumanMessage

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Parse numbered responses
            hypotheticals = []
            lines = content.split("\n")
            current = []

            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() and (len(line) > 2 and line[1] in ".)")):
                    if current:
                        hypotheticals.append(" ".join(current))
                    current = [line[2:].strip()]
                elif line:
                    current.append(line)

            if current:
                hypotheticals.append(" ".join(current))

            # Limit to requested number
            hypotheticals = hypotheticals[:num_hypotheticals]

            combined = []
            if self.include_original:
                combined.append(query)
            combined.extend(hypotheticals)

            logger.info(
                "HyDE multi-expansion complete",
                query_preview=query[:50],
                num_hypotheticals=len(hypotheticals),
            )

            return HyDEResult(
                original_query=query,
                hypothetical_document=hypotheticals[0] if hypotheticals else query,
                combined_queries=combined,
            )

        except Exception as e:
            logger.warning("HyDE multi-generation failed", error=str(e))
            return await self.expand(query, llm, context)


# Singleton instance
_hyde_instance: Optional[HyDEExpander] = None


def get_hyde_expander() -> HyDEExpander:
    """Get or create the HyDE expander singleton."""
    global _hyde_instance
    if _hyde_instance is None:
        _hyde_instance = HyDEExpander()
    return _hyde_instance
