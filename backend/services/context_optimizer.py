"""
AIDocumentIndexer - Context Optimizer
=====================================

Optimizes context window usage for small LLMs.
Implements smart chunking, relevance scoring, and compression.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import re
import asyncio

import structlog

from backend.services.llm import LLMFactory
from backend.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class ScoredChunk:
    """A chunk with relevance score."""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class OptimizedContext:
    """Optimized context ready for LLM consumption."""
    formatted_context: str
    chunks_included: int
    chunks_total: int
    total_tokens: int
    compression_ratio: float
    relevance_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formatted_context": self.formatted_context,
            "chunks_included": self.chunks_included,
            "chunks_total": self.chunks_total,
            "total_tokens": self.total_tokens,
            "compression_ratio": self.compression_ratio,
            "avg_relevance": sum(self.relevance_scores) / len(self.relevance_scores) if self.relevance_scores else 0,
        }


class ContextOptimizer:
    """
    Optimize context for small LLM context windows.

    Techniques:
    1. Relevance scoring - prioritize most relevant chunks
    2. Smart chunking - optimal chunk sizes with overlap
    3. Compression - summarize when needed
    4. Deduplication - remove redundant content
    5. Structure - add helpful formatting
    """

    # Approximate token-to-character ratios for estimation
    TOKENS_PER_CHAR = 0.25  # Rough estimate: 4 chars per token

    def __init__(
        self,
        provider: str = None,
        model: str = None,
    ):
        """Initialize the context optimizer."""
        self.provider = provider or settings.DEFAULT_LLM_PROVIDER
        self.model = model or settings.DEFAULT_CHAT_MODEL

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.TOKENS_PER_CHAR)

    async def optimize_context(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_tokens: int = 4000,
        strategy: str = "balanced",
    ) -> OptimizedContext:
        """
        Optimize documents to fit within token limit while maximizing relevance.

        Args:
            query: The user's query
            documents: List of document chunks with content/metadata
            max_tokens: Maximum tokens for context
            strategy: Optimization strategy - "balanced", "dense", "sparse"

        Returns:
            OptimizedContext with formatted context
        """
        logger.info(
            "Optimizing context",
            query_length=len(query),
            docs_count=len(documents),
            max_tokens=max_tokens,
            strategy=strategy,
        )

        if not documents:
            return OptimizedContext(
                formatted_context="No relevant documents found.",
                chunks_included=0,
                chunks_total=0,
                total_tokens=10,
                compression_ratio=1.0,
            )

        # Step 1: Score all chunks by relevance
        scored_chunks = await self._score_chunks(query, documents)

        # Step 2: Sort by score
        scored_chunks.sort(key=lambda x: x.score, reverse=True)

        # Step 3: Select chunks within token budget
        selected_chunks = self._select_chunks(scored_chunks, max_tokens, strategy)

        # Step 4: Check if compression needed
        total_tokens = sum(c.token_count for c in selected_chunks)
        original_tokens = sum(c.token_count for c in scored_chunks)

        if total_tokens > max_tokens * 0.9:
            # Need compression
            selected_chunks = await self._compress_chunks(
                selected_chunks, query, max_tokens
            )
            total_tokens = sum(c.token_count for c in selected_chunks)

        # Step 5: Format for LLM
        formatted = self._format_context(selected_chunks, query)

        compression_ratio = original_tokens / max(total_tokens, 1)

        result = OptimizedContext(
            formatted_context=formatted,
            chunks_included=len(selected_chunks),
            chunks_total=len(documents),
            total_tokens=self.estimate_tokens(formatted),
            compression_ratio=compression_ratio,
            relevance_scores=[c.score for c in selected_chunks],
        )

        logger.info(
            "Context optimized",
            chunks_included=result.chunks_included,
            total_tokens=result.total_tokens,
            compression_ratio=f"{compression_ratio:.2f}",
        )

        return result

    async def _score_chunks(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[ScoredChunk]:
        """Score all chunks by relevance to the query."""
        scored = []

        query_words = set(query.lower().split())

        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            if not content:
                continue

            # Calculate relevance score
            score = self._calculate_relevance(query_words, content)

            # Boost score based on metadata
            if doc.get("score"):
                score = (score + float(doc["score"])) / 2

            scored.append(ScoredChunk(
                content=content,
                score=score,
                source=doc.get("document_name", doc.get("source", "Unknown")),
                metadata=doc.get("metadata", {}),
                token_count=self.estimate_tokens(content),
            ))

        return scored

    def _calculate_relevance(self, query_words: set, content: str) -> float:
        """Calculate relevance score based on word overlap and position."""
        content_lower = content.lower()
        content_words = set(content_lower.split())

        # Word overlap score
        overlap = len(query_words & content_words)
        overlap_score = overlap / max(len(query_words), 1)

        # Position score (words appearing early get bonus)
        position_score = 0.0
        for word in query_words:
            pos = content_lower.find(word)
            if pos >= 0:
                # Higher score for words appearing earlier
                position_score += 1.0 / (1 + pos / 100)
        position_score = min(1.0, position_score / max(len(query_words), 1))

        # Combine scores
        return 0.6 * overlap_score + 0.4 * position_score

    def _select_chunks(
        self,
        chunks: List[ScoredChunk],
        max_tokens: int,
        strategy: str,
    ) -> List[ScoredChunk]:
        """Select chunks to fit within token budget."""
        selected = []
        current_tokens = 0

        # Reserve tokens for formatting
        format_overhead = 200
        available_tokens = max_tokens - format_overhead

        # Strategy adjustments
        if strategy == "dense":
            # Take more chunks, smaller portions
            available_tokens = int(available_tokens * 0.8)
        elif strategy == "sparse":
            # Take fewer chunks, full content
            pass

        for chunk in chunks:
            # Skip very low relevance chunks
            if chunk.score < 0.1:
                continue

            chunk_tokens = chunk.token_count

            if current_tokens + chunk_tokens <= available_tokens:
                selected.append(chunk)
                current_tokens += chunk_tokens
            elif strategy == "dense" and current_tokens < available_tokens * 0.5:
                # For dense strategy, try to include partial chunks
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:
                    # Truncate chunk
                    truncated_content = self._truncate_to_tokens(
                        chunk.content, remaining_tokens
                    )
                    truncated_chunk = ScoredChunk(
                        content=truncated_content,
                        score=chunk.score * 0.9,  # Slight penalty for truncation
                        source=chunk.source,
                        metadata=chunk.metadata,
                        token_count=self.estimate_tokens(truncated_content),
                    )
                    selected.append(truncated_chunk)
                    break

        return selected

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        chars_limit = int(max_tokens / self.TOKENS_PER_CHAR)

        if len(text) <= chars_limit:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:chars_limit]
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?'),
        )

        if last_sentence > chars_limit * 0.5:
            return text[:last_sentence + 1] + "..."
        else:
            # Truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > chars_limit * 0.8:
                return text[:last_space] + "..."
            else:
                return truncated + "..."

    async def _compress_chunks(
        self,
        chunks: List[ScoredChunk],
        query: str,
        max_tokens: int,
    ) -> List[ScoredChunk]:
        """Compress chunks using summarization when needed."""
        if not chunks:
            return chunks

        # Calculate how much compression we need
        current_tokens = sum(c.token_count for c in chunks)
        target_tokens = int(max_tokens * 0.85)

        if current_tokens <= target_tokens:
            return chunks

        compression_ratio = target_tokens / current_tokens

        logger.info(
            "Compressing chunks",
            current_tokens=current_tokens,
            target_tokens=target_tokens,
            compression_ratio=f"{compression_ratio:.2f}",
        )

        # Summarize each chunk proportionally
        compressed = []
        for chunk in chunks:
            target_length = int(len(chunk.content) * compression_ratio)

            if target_length < 100:
                # Skip very short summaries
                continue

            summary = await self._summarize_chunk(chunk.content, query, target_length)

            compressed.append(ScoredChunk(
                content=summary,
                score=chunk.score,
                source=chunk.source,
                metadata={**chunk.metadata, "compressed": True},
                token_count=self.estimate_tokens(summary),
            ))

        return compressed

    async def _summarize_chunk(
        self,
        content: str,
        query: str,
        target_length: int,
    ) -> str:
        """Summarize a chunk while preserving query-relevant information."""
        prompt = f"""Summarize the following text, focusing on information relevant to: "{query}"

Text to summarize:
{content}

Requirements:
- Keep the summary under {target_length} characters
- Preserve key facts and figures
- Focus on information that answers the query
- Use concise language

Summary:"""

        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.3,
                max_tokens=target_length // 2,  # Rough estimate
            )

            response = await llm.ainvoke(prompt)
            return response.content.strip()

        except Exception as e:
            logger.warning("Summarization failed, using truncation", error=str(e))
            return self._truncate_to_tokens(content, target_length // 4)

    def _format_context(
        self,
        chunks: List[ScoredChunk],
        query: str,
    ) -> str:
        """Format chunks into a structured context string."""
        if not chunks:
            return "No relevant context available."

        parts = [
            "**Relevant Information:**\n",
        ]

        # Group by source for better organization
        by_source: Dict[str, List[ScoredChunk]] = {}
        for chunk in chunks:
            source = chunk.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)

        for i, (source, source_chunks) in enumerate(by_source.items(), 1):
            parts.append(f"\n**Source {i}: {source}**")

            for j, chunk in enumerate(source_chunks):
                # Add relevance indicator
                if chunk.score >= 0.8:
                    relevance = "⭐ Highly Relevant"
                elif chunk.score >= 0.5:
                    relevance = "📌 Relevant"
                else:
                    relevance = "📎 Background"

                parts.append(f"\n[{relevance}]")
                parts.append(chunk.content)

                if chunk.metadata.get("compressed"):
                    parts.append("_(Summarized)_")

        return "\n".join(parts)

    def deduplicate_chunks(
        self,
        chunks: List[ScoredChunk],
        similarity_threshold: float = 0.8,
    ) -> List[ScoredChunk]:
        """Remove near-duplicate chunks."""
        if len(chunks) <= 1:
            return chunks

        unique = [chunks[0]]

        for chunk in chunks[1:]:
            is_duplicate = False

            for existing in unique:
                similarity = self._text_similarity(chunk.content, existing.content)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    # Keep the one with higher score
                    if chunk.score > existing.score:
                        unique.remove(existing)
                        unique.append(chunk)
                    break

            if not is_duplicate:
                unique.append(chunk)

        return unique

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# Singleton instance
_context_optimizer: Optional[ContextOptimizer] = None


def get_context_optimizer() -> ContextOptimizer:
    """Get or create the context optimizer singleton."""
    global _context_optimizer
    if _context_optimizer is None:
        _context_optimizer = ContextOptimizer()
    return _context_optimizer
