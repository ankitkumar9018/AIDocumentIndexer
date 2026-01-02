"""
AIDocumentIndexer - Contextual Chunking Service
================================================

Implements Anthropic's contextual retrieval approach for improved RAG quality.

Research shows 49-67% reduction in failed retrievals when chunks include
contextual information about where they fit in the overall document.

The approach:
1. Take each chunk from a document
2. Use an LLM to generate a brief context (50-100 words)
3. Prepend the context to the chunk before embedding
4. The resulting embedding captures both local and global context

Open-source LLM Support:
- Uses Ollama for context generation when available (cost-free)
- Falls back to OpenAI GPT-4o-mini for speed and quality
- Configurable via settings

Settings-aware: Respects rag.contextual_chunking_enabled setting.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import structlog

from backend.processors.chunker import Chunk

logger = structlog.get_logger(__name__)

# Context generation prompt (following Anthropic's approach)
CONTEXT_PROMPT = """<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context (50-100 words) to situate this chunk within the overall document for the purposes of improving search retrieval. The context should help someone understand what part of the document this chunk comes from and what it relates to.

Answer only with the context, nothing else. Do not use phrases like "This chunk" or "This section". Start directly with the contextual information."""

# Cached settings
_contextual_enabled: Optional[bool] = None
_context_model: Optional[str] = None
_context_provider: Optional[str] = None


async def _get_contextual_settings() -> tuple[bool, str, str]:
    """Get contextual chunking settings from database."""
    global _contextual_enabled, _context_model, _context_provider

    if _contextual_enabled is not None and _context_model is not None:
        return _contextual_enabled, _context_model, _context_provider

    try:
        from backend.services.settings import get_settings_service

        settings = get_settings_service()
        enabled = await settings.get_setting("rag.contextual_chunking_enabled")
        model = await settings.get_setting("rag.context_generation_model")
        provider = await settings.get_setting("rag.context_generation_provider")

        _contextual_enabled = enabled if enabled is not None else False  # Disabled by default
        _context_model = model if model else "llama3.2"  # Default to local Ollama
        _context_provider = provider if provider else "ollama"

        return _contextual_enabled, _context_model, _context_provider
    except Exception as e:
        logger.debug("Could not load contextual chunking settings, using defaults", error=str(e))
        return False, "llama3.2", "ollama"


def invalidate_contextual_settings():
    """Invalidate cached settings (call after settings change)."""
    global _contextual_enabled, _context_model, _context_provider
    _contextual_enabled = None
    _context_model = None
    _context_provider = None


@dataclass
class ContextualChunkResult:
    """Result of contextual enhancement for a chunk."""
    original_content: str
    context: str
    contextualized_content: str
    chunk_index: int
    success: bool
    error: Optional[str] = None


class ContextualChunker:
    """
    Adds contextual summaries to chunks before embedding.

    This implements Anthropic's contextual retrieval approach which achieves
    49-67% reduction in failed retrievals compared to naive chunking.

    The service:
    1. Takes the full document and its chunks
    2. Generates a brief context for each chunk using an LLM
    3. Prepends the context to create enhanced chunks
    4. Returns enhanced chunks ready for embedding
    """

    def __init__(
        self,
        max_document_chars: int = 50000,  # Limit for context window
        max_concurrent_requests: int = 5,  # Rate limiting for API calls
        batch_size: int = 10,  # Chunks per batch
    ):
        """
        Initialize contextual chunker.

        Args:
            max_document_chars: Maximum document length for context (longer docs are truncated)
            max_concurrent_requests: Max concurrent LLM requests
            batch_size: Number of chunks to process in parallel
        """
        self.max_document_chars = max_document_chars
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _get_llm(self):
        """Get LLM instance for context generation."""
        from backend.services.llm import LLMFactory

        _, model, provider = await _get_contextual_settings()

        return LLMFactory.get_chat_model(
            provider=provider,
            model=model,
            temperature=0.0,  # Deterministic for consistency
            max_tokens=200,  # Context should be brief
        )

    async def _generate_context(
        self,
        document_content: str,
        chunk_content: str,
        chunk_index: int,
    ) -> ContextualChunkResult:
        """
        Generate context for a single chunk.

        Args:
            document_content: Full or truncated document content
            chunk_content: The chunk to contextualize
            chunk_index: Index of the chunk

        Returns:
            ContextualChunkResult with context and enhanced content
        """
        async with self._semaphore:
            try:
                llm = await self._get_llm()

                prompt = CONTEXT_PROMPT.format(
                    document_content=document_content,
                    chunk_content=chunk_content,
                )

                # Use ainvoke for async LLM call
                response = await llm.ainvoke(prompt)
                context = response.content.strip()

                # Combine context with original content
                contextualized = f"{context}\n\n---\n\n{chunk_content}"

                return ContextualChunkResult(
                    original_content=chunk_content,
                    context=context,
                    contextualized_content=contextualized,
                    chunk_index=chunk_index,
                    success=True,
                )

            except Exception as e:
                logger.warning(
                    "Context generation failed for chunk",
                    chunk_index=chunk_index,
                    error=str(e),
                )
                # Return original content on failure
                return ContextualChunkResult(
                    original_content=chunk_content,
                    context="",
                    contextualized_content=chunk_content,
                    chunk_index=chunk_index,
                    success=False,
                    error=str(e),
                )

    async def add_context_to_chunks(
        self,
        document_content: str,
        chunks: List[Chunk],
        document_title: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Add contextual information to each chunk.

        Args:
            document_content: Full document text
            chunks: List of chunks to enhance
            document_title: Optional document title for better context

        Returns:
            List of enhanced Chunk objects with contextual content
        """
        # Check if contextual chunking is enabled
        enabled, _, _ = await _get_contextual_settings()
        if not enabled:
            logger.debug("Contextual chunking disabled, returning original chunks")
            return chunks

        if not chunks:
            return chunks

        logger.info(
            "Starting contextual chunking",
            document_title=document_title,
            chunk_count=len(chunks),
        )

        # Truncate document if too long
        truncated_doc = document_content
        if len(document_content) > self.max_document_chars:
            truncated_doc = document_content[:self.max_document_chars]
            logger.debug(
                "Document truncated for context generation",
                original_length=len(document_content),
                truncated_length=len(truncated_doc),
            )

        # Add title to document context if available
        if document_title:
            truncated_doc = f"Document Title: {document_title}\n\n{truncated_doc}"

        # Process chunks in batches
        enhanced_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            # Generate contexts in parallel
            tasks = [
                self._generate_context(
                    document_content=truncated_doc,
                    chunk_content=chunk.content,
                    chunk_index=chunk.chunk_index,
                )
                for chunk in batch_chunks
            ]

            results = await asyncio.gather(*tasks)

            # Update chunks with contextualized content
            for chunk, result in zip(batch_chunks, results):
                if result.success:
                    # Create new chunk with contextualized content
                    enhanced_chunk = Chunk(
                        content=result.contextualized_content,
                        chunk_index=chunk.chunk_index,
                        metadata={
                            **chunk.metadata,
                            "contextual_enhanced": True,
                            "context_prefix": result.context,
                        },
                        document_id=chunk.document_id,
                        page_number=chunk.page_number,
                        slide_number=chunk.slide_number,
                        section=chunk.section,
                        chunk_level=chunk.chunk_level,
                        is_summary=chunk.is_summary,
                        parent_chunk_id=chunk.parent_chunk_id,
                        child_chunk_ids=chunk.child_chunk_ids,
                    )
                    enhanced_chunks.append(enhanced_chunk)
                else:
                    # Keep original chunk on failure
                    chunk.metadata["contextual_enhanced"] = False
                    chunk.metadata["context_error"] = result.error
                    enhanced_chunks.append(chunk)

            logger.debug(
                "Processed contextual chunking batch",
                batch=batch_idx + 1,
                total_batches=total_batches,
            )

        success_count = sum(1 for c in enhanced_chunks if c.metadata.get("contextual_enhanced"))
        logger.info(
            "Contextual chunking completed",
            total_chunks=len(chunks),
            enhanced=success_count,
            failed=len(chunks) - success_count,
        )

        return enhanced_chunks

    async def add_context_to_text(
        self,
        document_content: str,
        chunk_content: str,
        document_title: Optional[str] = None,
    ) -> str:
        """
        Add context to a single piece of text.

        Convenience method for single-chunk contextualization.

        Args:
            document_content: Full document text
            chunk_content: Text to contextualize
            document_title: Optional document title

        Returns:
            Contextualized text string
        """
        # Check if enabled
        enabled, _, _ = await _get_contextual_settings()
        if not enabled:
            return chunk_content

        truncated_doc = document_content[:self.max_document_chars]
        if document_title:
            truncated_doc = f"Document Title: {document_title}\n\n{truncated_doc}"

        result = await self._generate_context(
            document_content=truncated_doc,
            chunk_content=chunk_content,
            chunk_index=0,
        )

        return result.contextualized_content


# Singleton instance
_contextual_chunker: Optional[ContextualChunker] = None


def get_contextual_chunker() -> ContextualChunker:
    """Get or create contextual chunker singleton."""
    global _contextual_chunker
    if _contextual_chunker is None:
        _contextual_chunker = ContextualChunker()
    return _contextual_chunker


async def contextualize_chunks(
    document_content: str,
    chunks: List[Chunk],
    document_title: Optional[str] = None,
) -> List[Chunk]:
    """
    Convenience function to contextualize chunks.

    Args:
        document_content: Full document text
        chunks: List of chunks to enhance
        document_title: Optional document title

    Returns:
        List of enhanced chunks
    """
    chunker = get_contextual_chunker()
    return await chunker.add_context_to_chunks(
        document_content=document_content,
        chunks=chunks,
        document_title=document_title,
    )
