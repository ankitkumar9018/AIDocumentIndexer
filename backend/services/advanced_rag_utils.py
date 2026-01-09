"""
AIDocumentIndexer - Advanced RAG Utilities
===========================================

Implements advanced RAG optimization techniques:
1. RAG-Fusion: Multi-query generation and result fusion
2. Context Compression: Token-efficient context for LLM
3. Lost-in-the-Middle Mitigation: Context reordering
4. Step-Back Prompting: Abstract reasoning for complex queries

These utilities enhance retrieval quality and efficiency for the RAG pipeline.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# RAG-Fusion: Multi-Query Generation and Result Fusion
# =============================================================================

@dataclass
class FusionResult:
    """Result from RAG-Fusion retrieval."""
    original_query: str
    query_variations: List[str]
    fused_results: List[Any]  # SearchResult objects
    individual_results: Dict[str, List[Any]]  # query -> results
    fusion_method: str = "rrf"


class RAGFusion:
    """
    Multi-query generation and Reciprocal Rank Fusion.

    Generates multiple query variations to capture different aspects
    of the user's intent, then fuses results using RRF.

    Reference: RAG-Fusion paper and RRF (Reciprocal Rank Fusion)
    """

    def __init__(
        self,
        num_variations: int = 4,
        rrf_k: int = 60,
        include_original: bool = True,
    ):
        """
        Initialize RAG-Fusion.

        Args:
            num_variations: Number of query variations to generate
            rrf_k: RRF constant (higher = more weight to lower ranks)
            include_original: Include original query in variations
        """
        self.num_variations = num_variations
        self.rrf_k = rrf_k
        self.include_original = include_original

    async def generate_query_variations(
        self,
        original_query: str,
        llm,
        context: Optional[str] = None,
    ) -> List[str]:
        """
        Generate diverse query variations using LLM.

        Args:
            original_query: The user's original query
            llm: LangChain LLM for generation
            context: Optional context about the document collection

        Returns:
            List of query variations
        """
        context_hint = f"\nContext about documents: {context}" if context else ""

        prompt = f"""Generate {self.num_variations} different search queries that could help
answer the following question. Each query should approach the question from a different
angle or use different keywords while seeking the same information.

Make the queries diverse:
- Use synonyms and alternative phrasings
- Focus on different aspects of the question
- Vary the specificity (some broader, some more specific)
{context_hint}

Original question: {original_query}

Generate exactly {self.num_variations} queries, one per line. Output ONLY the queries, no numbering or explanations:"""

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse variations
            variations = [
                line.strip().lstrip('0123456789.-) ')
                for line in content.strip().split('\n')
                if line.strip() and len(line.strip()) > 5
            ]

            # Build result list
            result = []
            if self.include_original:
                result.append(original_query)

            # Add variations up to limit
            for v in variations[:self.num_variations]:
                if v and v not in result:
                    result.append(v)

            logger.debug(
                "Generated query variations",
                original=original_query[:50],
                num_variations=len(result),
            )

            return result

        except Exception as e:
            logger.warning("Query variation generation failed", error=str(e))
            return [original_query]  # Fallback to original

    async def fuse_results(
        self,
        query: str,
        retriever,
        session,
        top_k: int = 10,
        llm=None,
    ) -> FusionResult:
        """
        Retrieve and fuse results from multiple query variations.

        Args:
            query: Original query
            retriever: Retriever with search() method
            session: Database session
            top_k: Number of results to return
            llm: LLM for query variation generation

        Returns:
            FusionResult with fused results
        """
        # Generate variations
        if llm:
            variations = await self.generate_query_variations(query, llm)
        else:
            variations = [query]

        # Retrieve for each variation
        all_results = {}
        individual_results = {}

        for q in variations:
            try:
                results = await retriever.search(
                    query=q,
                    top_k=top_k * 2,  # Get more for fusion
                    search_type="hybrid",
                )
                individual_results[q] = results

                for rank, result in enumerate(results):
                    doc_id = getattr(result, 'chunk_id', str(id(result)))
                    if doc_id not in all_results:
                        all_results[doc_id] = {
                            "result": result,
                            "scores": [],
                        }
                    # RRF score: 1 / (k + rank)
                    all_results[doc_id]["scores"].append(1 / (self.rrf_k + rank + 1))

            except Exception as e:
                logger.warning(f"Retrieval failed for variation: {q[:50]}", error=str(e))

        # Calculate final RRF scores
        fused_results = []
        for doc_id, data in all_results.items():
            result = data["result"]
            rrf_score = sum(data["scores"])
            # Update score on result
            if hasattr(result, 'similarity_score'):
                result.similarity_score = rrf_score
            fused_results.append(result)

        # Sort by fused score descending
        fused_results.sort(
            key=lambda x: getattr(x, 'similarity_score', 0),
            reverse=True,
        )

        logger.info(
            "RAG-Fusion complete",
            num_variations=len(variations),
            total_unique_results=len(fused_results),
            returned=min(top_k, len(fused_results)),
        )

        return FusionResult(
            original_query=query,
            query_variations=variations,
            fused_results=fused_results[:top_k],
            individual_results=individual_results,
            fusion_method="rrf",
        )


# =============================================================================
# Context Compression: Token-Efficient Context for LLM
# =============================================================================

@dataclass
class CompressedContext:
    """Result of context compression."""
    original_contexts: List[str]
    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: str


class ContextCompressor:
    """
    Compress retrieved context for efficient LLM inference.

    Reduces token usage while preserving relevant information,
    addressing both cost and "lost in the middle" issues.

    Reference: ACC-RAG pattern, context compression research
    """

    def __init__(
        self,
        target_tokens: int = 2000,
        min_sentence_length: int = 20,
        use_llm_compression: bool = True,
    ):
        """
        Initialize context compressor.

        Args:
            target_tokens: Target token count for compressed context
            min_sentence_length: Minimum sentence length to keep
            use_llm_compression: Use LLM for semantic compression
        """
        self.target_tokens = target_tokens
        self.min_sentence_length = min_sentence_length
        self.use_llm_compression = use_llm_compression

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~1.3 tokens per word)."""
        return int(len(text.split()) * 1.3)

    async def compress(
        self,
        query: str,
        contexts: List[str],
        llm=None,
    ) -> CompressedContext:
        """
        Compress contexts to target token count.

        Args:
            query: The user's query
            contexts: List of context strings to compress
            llm: Optional LLM for semantic compression

        Returns:
            CompressedContext with compressed text
        """
        original_tokens = sum(self._estimate_tokens(c) for c in contexts)

        # If already under target, just concatenate
        if original_tokens <= self.target_tokens:
            compressed = "\n\n".join(contexts)
            return CompressedContext(
                original_contexts=contexts,
                compressed_context=compressed,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                method="none",
            )

        # Try selective extraction first
        selected = await self._selective_extraction(query, contexts)
        selected_tokens = self._estimate_tokens(selected)

        if selected_tokens <= self.target_tokens:
            return CompressedContext(
                original_contexts=contexts,
                compressed_context=selected,
                original_tokens=original_tokens,
                compressed_tokens=selected_tokens,
                compression_ratio=selected_tokens / original_tokens,
                method="selective",
            )

        # Use LLM compression if available and needed
        if llm and self.use_llm_compression:
            compressed = await self._llm_compress(query, contexts, llm)
            compressed_tokens = self._estimate_tokens(compressed)

            return CompressedContext(
                original_contexts=contexts,
                compressed_context=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compressed_tokens / original_tokens,
                method="llm",
            )

        # Fallback: truncate
        return CompressedContext(
            original_contexts=contexts,
            compressed_context=selected[:self.target_tokens * 4],  # Rough char limit
            original_tokens=original_tokens,
            compressed_tokens=self.target_tokens,
            compression_ratio=self.target_tokens / original_tokens,
            method="truncate",
        )

    async def _selective_extraction(
        self,
        query: str,
        contexts: List[str],
    ) -> str:
        """Extract most relevant sentences based on keyword overlap."""
        query_words = set(query.lower().split())
        scored_sentences = []

        for context in contexts:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', context)

            for sent in sentences:
                if len(sent.strip()) < self.min_sentence_length:
                    continue

                # Score by keyword overlap
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                score = overlap / max(len(query_words), 1)

                scored_sentences.append((sent, score))

        # Sort by relevance
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Build compressed context up to token limit
        selected = []
        token_count = 0

        for sent, score in scored_sentences:
            sent_tokens = self._estimate_tokens(sent)
            if token_count + sent_tokens <= self.target_tokens:
                selected.append(sent)
                token_count += sent_tokens
            else:
                break

        return " ".join(selected)

    async def _llm_compress(
        self,
        query: str,
        contexts: List[str],
        llm,
    ) -> str:
        """Use LLM to extract relevant information."""
        combined = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

        prompt = f"""Given the question and retrieved passages, extract ONLY the
sentences and facts that directly help answer the question. Remove all
irrelevant information, redundant details, and filler text.

Question: {query}

Retrieved Passages:
{combined}

Extract the essential information (target: ~{self.target_tokens // 4} words):"""

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
        except Exception as e:
            logger.warning("LLM compression failed", error=str(e))
            return await self._selective_extraction(query, contexts)


# =============================================================================
# Lost-in-the-Middle Mitigation: Context Reordering
# =============================================================================

class ContextReorderer:
    """
    Reorder context to mitigate the "lost in the middle" effect.

    LLMs pay less attention to information in the middle of long contexts.
    This reorderer places most relevant information at the beginning and end.

    Reference: "Lost in the Middle" paper (Liu et al., 2023)
    """

    @staticmethod
    def reorder(
        results: List[Any],
        strategy: str = "sandwich",
    ) -> List[Any]:
        """
        Reorder results for better LLM attention.

        Args:
            results: List of search results with similarity_score attribute
            strategy: Reordering strategy:
                - "sandwich": Best at start and end, medium in middle
                - "front_loaded": All best at start
                - "alternating": Alternate high/low relevance

        Returns:
            Reordered list
        """
        if len(results) <= 2:
            return results

        # Sort by relevance score descending
        sorted_results = sorted(
            results,
            key=lambda x: getattr(x, 'similarity_score', 0),
            reverse=True,
        )

        if strategy == "sandwich":
            return ContextReorderer._sandwich_order(sorted_results)
        elif strategy == "front_loaded":
            return sorted_results
        elif strategy == "alternating":
            return ContextReorderer._alternating_order(sorted_results)

        return results

    @staticmethod
    def _sandwich_order(sorted_results: List[Any]) -> List[Any]:
        """
        Place best at start, second-best at end, rest in middle.
        Creates a "sandwich" with high relevance at both ends.
        """
        result = []
        left = 0
        right = len(sorted_results) - 1
        toggle = True

        while left <= right:
            if toggle:
                result.append(sorted_results[left])
                left += 1
            else:
                result.insert(len(result) // 2, sorted_results[right])
                right -= 1
            toggle = not toggle

        return result

    @staticmethod
    def _alternating_order(sorted_results: List[Any]) -> List[Any]:
        """
        Alternate between high and lower relevance items.
        Keeps attention refreshed throughout the context.
        """
        result = []
        n = len(sorted_results)

        # Interleave from both ends
        for i in range((n + 1) // 2):
            result.append(sorted_results[i])
            if n - 1 - i > i:
                result.append(sorted_results[n - 1 - i])

        return result

    @staticmethod
    def add_position_markers(
        contexts: List[str],
        marker_format: str = "[Source {i}]",
    ) -> str:
        """
        Add explicit position markers to help LLM track sources.

        Args:
            contexts: List of context strings
            marker_format: Format for markers (use {i} for number)

        Returns:
            Combined context with markers
        """
        marked = []
        for i, ctx in enumerate(contexts):
            marker = marker_format.format(i=i + 1)
            marked.append(f"{marker}: {ctx}")
        return "\n\n".join(marked)


# =============================================================================
# Step-Back Prompting: Abstract Reasoning for Complex Queries
# =============================================================================

@dataclass
class StepBackResult:
    """Result from step-back prompting."""
    original_query: str
    stepback_query: str
    background_context: str
    specific_context: str
    combined_context: str


class StepBackPrompter:
    """
    Generate abstract "step-back" questions for complex queries.

    For complex queries, first retrieves broader context using an abstract
    question, then retrieves specific context. This helps with:
    - Multi-hop reasoning
    - Queries requiring background knowledge
    - Complex cause-effect questions

    Reference: Step-Back Prompting paper (Zheng et al., 2023)
    """

    def __init__(
        self,
        max_background_chunks: int = 3,
    ):
        """
        Initialize step-back prompter.

        Args:
            max_background_chunks: Max chunks to use for background
        """
        self.max_background_chunks = max_background_chunks

    async def generate_stepback_query(
        self,
        original_query: str,
        llm,
    ) -> str:
        """
        Generate a more abstract step-back question.

        Args:
            original_query: The user's specific question
            llm: LangChain LLM for generation

        Returns:
            Abstract step-back question
        """
        prompt = f"""Given the following specific question, generate a more general
"step-back" question that would provide useful background context to help
answer the original question.

The step-back question should be:
- More abstract and general
- Focus on underlying concepts or principles
- Help establish context for the specific question

Original Question: {original_query}

Examples:
- "What caused the 2008 financial crisis?" -> "What factors typically cause financial crises?"
- "How does Python's GIL affect threading?" -> "How do interpreters manage concurrent execution?"
- "Why did Company X's stock drop yesterday?" -> "What factors affect stock price volatility?"

Generate ONE step-back question (just the question, nothing else):"""

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Clean up the response
            stepback = content.strip().strip('"\'')

            # Ensure it ends with question mark
            if not stepback.endswith('?'):
                stepback += '?'

            logger.debug(
                "Generated step-back query",
                original=original_query[:50],
                stepback=stepback[:50],
            )

            return stepback

        except Exception as e:
            logger.warning("Step-back generation failed", error=str(e))
            return original_query  # Fallback to original

    async def retrieve_with_stepback(
        self,
        query: str,
        retriever,
        session,
        llm,
        top_k: int = 10,
    ) -> StepBackResult:
        """
        Retrieve using both original and step-back queries.

        Args:
            query: Original query
            retriever: Retriever with search() method
            session: Database session
            llm: LLM for step-back generation
            top_k: Results per query

        Returns:
            StepBackResult with combined context
        """
        # Generate step-back question
        stepback_query = await self.generate_stepback_query(query, llm)

        # Retrieve for both queries
        try:
            original_results = await retriever.search(
                query=query,
                top_k=top_k,
                search_type="hybrid",
            )
        except Exception as e:
            logger.warning("Original query retrieval failed", error=str(e))
            original_results = []

        try:
            stepback_results = await retriever.search(
                query=stepback_query,
                top_k=self.max_background_chunks,
                search_type="hybrid",
            )
        except Exception as e:
            logger.warning("Step-back query retrieval failed", error=str(e))
            stepback_results = []

        # Build context
        background_context = self._build_context(stepback_results)
        specific_context = self._build_context(original_results)

        combined = self._combine_contexts(background_context, specific_context)

        return StepBackResult(
            original_query=query,
            stepback_query=stepback_query,
            background_context=background_context,
            specific_context=specific_context,
            combined_context=combined,
        )

    def _build_context(self, results: List[Any]) -> str:
        """Build context string from results."""
        if not results:
            return ""

        contexts = []
        for r in results:
            content = getattr(r, 'content', str(r))
            contexts.append(content)

        return "\n\n".join(contexts)

    def _combine_contexts(
        self,
        background: str,
        specific: str,
    ) -> str:
        """Combine background and specific contexts."""
        parts = []

        if background:
            parts.append("BACKGROUND CONTEXT:")
            parts.append(background)
            parts.append("")

        if specific:
            parts.append("SPECIFIC INFORMATION:")
            parts.append(specific)

        return "\n".join(parts)


# =============================================================================
# Convenience Functions
# =============================================================================

async def apply_rag_fusion(
    query: str,
    retriever,
    session,
    llm=None,
    num_variations: int = 4,
    top_k: int = 10,
) -> FusionResult:
    """
    Convenience function for RAG-Fusion.

    Args:
        query: User query
        retriever: Retriever with search() method
        session: Database session
        llm: Optional LLM for query variations
        num_variations: Number of query variations
        top_k: Final results count

    Returns:
        FusionResult
    """
    fusion = RAGFusion(num_variations=num_variations)
    return await fusion.fuse_results(query, retriever, session, top_k, llm)


async def compress_context(
    query: str,
    contexts: List[str],
    target_tokens: int = 2000,
    llm=None,
) -> CompressedContext:
    """
    Convenience function for context compression.

    Args:
        query: User query
        contexts: List of context strings
        target_tokens: Target token count
        llm: Optional LLM for semantic compression

    Returns:
        CompressedContext
    """
    compressor = ContextCompressor(target_tokens=target_tokens)
    return await compressor.compress(query, contexts, llm)


def reorder_for_attention(
    results: List[Any],
    strategy: str = "sandwich",
) -> List[Any]:
    """
    Convenience function for context reordering.

    Args:
        results: Search results with similarity_score
        strategy: "sandwich", "front_loaded", or "alternating"

    Returns:
        Reordered results
    """
    return ContextReorderer.reorder(results, strategy)


async def apply_stepback_prompting(
    query: str,
    retriever,
    session,
    llm,
    top_k: int = 10,
) -> StepBackResult:
    """
    Convenience function for step-back prompting.

    Args:
        query: User query
        retriever: Retriever with search() method
        session: Database session
        llm: LLM for step-back generation
        top_k: Results per query

    Returns:
        StepBackResult
    """
    prompter = StepBackPrompter()
    return await prompter.retrieve_with_stepback(query, retriever, session, llm, top_k)
