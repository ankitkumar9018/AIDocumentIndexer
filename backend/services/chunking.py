"""
AIDocumentIndexer - High-Performance Chunking Service
======================================================

Ultra-fast chunking using Chonkie library (33x faster than LangChain).

Features:
- Token chunking: 33x faster than LangChain
- Semantic chunking (SDPM): 2.5x faster with better quality
- Late chunking: Embed full document then split for context preservation
- Proposition chunking (Phase 67): Break text into atomic facts for QA
- Automatic strategy selection based on document size
- Integration with existing chunker infrastructure

Research:
- Chonkie: Lightweight, fast chunking (505KB vs 1-12MB for alternatives)
- SDPM (Semantic Double-Pass Merge): Better semantic boundaries
- Late Chunking: Preserves full document context in embeddings
- Proposition Chunking: Dense X Retrieval (2023) - fact-based indexing

Performance benchmarks (vs LangChain/LlamaIndex):
- Token chunking: 33x faster
- Semantic chunking: 2.5x faster
- Memory usage: 10-50x smaller

Chunking strategies:
| Strategy | Use Case | Speed | Quality |
|----------|----------|-------|---------|
| token | Large docs, simple retrieval | Fastest | Basic |
| sentence | General purpose | Fast | Good |
| semantic | Context-aware retrieval | Medium | Better |
| sdpm | High-quality semantic | Slower | Best |
| proposition | QA, fact-heavy docs | Slow (LLM) | Excellent for QA |
| late | Cross-chunk context | Medium | Best context |
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Check for Chonkie availability
try:
    from chonkie import (
        TokenChunker,
        SentenceChunker,
        SemanticChunker as ChonkieSemanticChunker,
        SDPMChunker,
    )
    from chonkie.chunker import Chunk as ChonkieChunk
    HAS_CHONKIE = True
except ImportError:
    HAS_CHONKIE = False
    TokenChunker = None
    SentenceChunker = None
    ChonkieSemanticChunker = None
    SDPMChunker = None
    ChonkieChunk = None


# =============================================================================
# Configuration
# =============================================================================

class FastChunkingStrategy(str, Enum):
    """High-performance chunking strategies."""
    TOKEN = "token"           # Fastest: fixed token count
    SENTENCE = "sentence"     # Fast: sentence boundaries
    SEMANTIC = "semantic"     # Balanced: semantic similarity
    SDPM = "sdpm"            # Best quality: double-pass merge
    LATE = "late"            # Best context: embed then chunk
    PROPOSITION = "proposition"  # Phase 67: factual statements for QA
    AUTO = "auto"            # Automatic selection


@dataclass
class FastChunkingConfig:
    """Configuration for high-performance chunking."""
    # Strategy selection
    strategy: FastChunkingStrategy = FastChunkingStrategy.AUTO

    # Size parameters
    chunk_size: int = 512       # Target tokens per chunk
    chunk_overlap: int = 50     # Overlap in tokens
    min_chunk_size: int = 50    # Minimum chunk size

    # Semantic chunking settings
    similarity_threshold: float = 0.7  # For semantic/SDPM
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast local model

    # Auto strategy thresholds
    auto_semantic_threshold: int = 10000   # Use semantic for docs < 10K chars
    auto_sdpm_threshold: int = 50000       # Use SDPM for docs < 50K chars

    # Late chunking settings
    late_chunking_enabled: bool = False
    late_chunk_positions: bool = True  # Store position info for late chunking


@dataclass(slots=True)
class FastChunk:
    """Memory-efficient chunk representation."""
    content: str
    index: int
    start_pos: int
    end_pos: int
    token_count: int
    chunk_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For late chunking
    embedding_slice: Optional[Tuple[int, int]] = None


# =============================================================================
# High-Performance Chunker
# =============================================================================

class FastChunker:
    """
    High-performance text chunker using Chonkie.

    33x faster than LangChain for token chunking, 2.5x faster for semantic.

    Usage:
        chunker = FastChunker()
        chunks = await chunker.chunk(text)

        # Or with specific strategy
        chunks = await chunker.chunk(text, strategy=FastChunkingStrategy.SDPM)
    """

    def __init__(self, config: Optional[FastChunkingConfig] = None):
        self.config = config or FastChunkingConfig()
        self._chunkers: Dict[FastChunkingStrategy, Any] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

        if not HAS_CHONKIE:
            logger.warning(
                "Chonkie not installed - falling back to standard chunker. "
                "Install with: pip install chonkie"
            )

    async def initialize(self) -> bool:
        """Initialize chunkers (lazy loading)."""
        if self._initialized:
            return True

        if not HAS_CHONKIE:
            return False

        async with self._lock:
            if self._initialized:
                return True

            try:
                # Initialize chunkers in thread pool (some require model loading)
                loop = asyncio.get_running_loop()

                # Token chunker (fastest, no model needed)
                self._chunkers[FastChunkingStrategy.TOKEN] = TokenChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )

                # Sentence chunker (fast)
                self._chunkers[FastChunkingStrategy.SENTENCE] = SentenceChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )

                # Semantic chunker (requires embedding model)
                try:
                    self._chunkers[FastChunkingStrategy.SEMANTIC] = await loop.run_in_executor(
                        None,
                        lambda: ChonkieSemanticChunker(
                            embedding_model=self.config.embedding_model,
                            chunk_size=self.config.chunk_size,
                            similarity_threshold=self.config.similarity_threshold,
                        )
                    )
                except Exception as e:
                    logger.warning("Semantic chunker not available", error=str(e))

                # SDPM chunker (best quality, requires embedding model)
                try:
                    self._chunkers[FastChunkingStrategy.SDPM] = await loop.run_in_executor(
                        None,
                        lambda: SDPMChunker(
                            embedding_model=self.config.embedding_model,
                            chunk_size=self.config.chunk_size,
                            similarity_threshold=self.config.similarity_threshold,
                        )
                    )
                except Exception as e:
                    logger.warning("SDPM chunker not available", error=str(e))

                self._initialized = True
                logger.info(
                    "Fast chunker initialized",
                    available_strategies=list(self._chunkers.keys()),
                )
                return True

            except Exception as e:
                logger.error("Failed to initialize fast chunker", error=str(e))
                return False

    def _select_strategy(self, text: str) -> FastChunkingStrategy:
        """
        Phase 76: Auto-select best strategy based on content analysis.

        Now considers:
        - Text length
        - Content type (code, tables, narrative)
        - Structure (lists, headers, paragraphs)
        """
        if self.config.strategy != FastChunkingStrategy.AUTO:
            return self.config.strategy

        text_len = len(text)
        sample = text[:5000]  # Analyze first 5000 chars

        # Phase 76: Detect content characteristics
        content_type = self._detect_content_type(sample)

        # Code content: use token chunking to preserve syntax structure
        if content_type == "code":
            logger.debug("Auto-selected TOKEN strategy for code content")
            return FastChunkingStrategy.TOKEN

        # Tabular content: use sentence chunking to preserve row structure
        if content_type == "tabular":
            logger.debug("Auto-selected SENTENCE strategy for tabular content")
            return FastChunkingStrategy.SENTENCE

        # For very large documents, use token chunking (fastest)
        if text_len > self.config.auto_sdpm_threshold:
            return FastChunkingStrategy.TOKEN

        # Structured content with clear sections: use semantic
        if content_type == "structured":
            if FastChunkingStrategy.SEMANTIC in self._chunkers:
                logger.debug("Auto-selected SEMANTIC strategy for structured content")
                return FastChunkingStrategy.SEMANTIC

        # For medium documents, use semantic
        if text_len > self.config.auto_semantic_threshold:
            if FastChunkingStrategy.SEMANTIC in self._chunkers:
                return FastChunkingStrategy.SEMANTIC
            return FastChunkingStrategy.SENTENCE

        # Narrative content: use SDPM for best quality
        if content_type == "narrative":
            if FastChunkingStrategy.SDPM in self._chunkers:
                logger.debug("Auto-selected SDPM strategy for narrative content")
                return FastChunkingStrategy.SDPM

        # For small documents, use SDPM for best quality
        if FastChunkingStrategy.SDPM in self._chunkers:
            return FastChunkingStrategy.SDPM
        if FastChunkingStrategy.SEMANTIC in self._chunkers:
            return FastChunkingStrategy.SEMANTIC

        return FastChunkingStrategy.TOKEN

    def _detect_content_type(self, text: str) -> str:
        """
        Phase 76: Detect content type for optimal chunking strategy.

        Returns:
            Content type: "code", "tabular", "structured", "narrative"
        """
        import re

        # Code patterns
        code_patterns = [
            r'\bdef\s+\w+\s*\(',      # Python functions
            r'\bclass\s+\w+',          # Classes
            r'\bfunction\s+\w+\s*\(',  # JavaScript functions
            r'\bimport\s+[\w.]+',      # Import statements
            r'#include\s*<',           # C/C++
            r'\{[\s\n]*[\w]+:',        # JSON-like objects
            r'```[\w]*\n',             # Markdown code blocks
        ]
        code_matches = sum(1 for p in code_patterns if re.search(p, text))
        if code_matches >= 2:
            return "code"

        # Tabular patterns
        tabular_patterns = [
            r'\|[^|]+\|[^|]+\|',       # Markdown tables
            r'\t.*\t.*\t',             # Tab-separated
            r',.*,.*,',                # CSV-like
        ]
        tabular_matches = sum(1 for p in tabular_patterns if re.search(p, text))
        if tabular_matches >= 1:
            # Check for multiple occurrences (actual table)
            table_rows = len(re.findall(r'\|[^|]+\|', text))
            if table_rows > 3:
                return "tabular"

        # Structured content patterns (headers, lists)
        structured_patterns = [
            r'^#+\s+\w',               # Markdown headers
            r'^\d+\.\s+\w',            # Numbered lists
            r'^[-*]\s+\w',             # Bullet lists
            r'^[A-Z][^.!?]*:\s*$',     # Section titles
        ]
        structured_matches = sum(
            len(re.findall(p, text, re.MULTILINE))
            for p in structured_patterns
        )
        if structured_matches >= 5:
            return "structured"

        # Default to narrative
        return "narrative"

    async def chunk(
        self,
        text: str,
        strategy: Optional[FastChunkingStrategy] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[FastChunk]:
        """
        Chunk text using high-performance Chonkie library.

        Args:
            text: Text to chunk
            strategy: Chunking strategy (auto-selected if None)
            document_id: Optional document ID for metadata
            metadata: Optional metadata to include in chunks

        Returns:
            List of FastChunk objects
        """
        if not text or not text.strip():
            return []

        # Phase 67: Handle proposition chunking separately (uses LLM)
        if strategy == FastChunkingStrategy.PROPOSITION:
            prop_chunker = PropositionChunker()
            return await prop_chunker.chunk(text, document_id, metadata)

        # Initialize if needed
        if not await self.initialize():
            # Fall back to simple chunking
            return self._fallback_chunk(text, document_id, metadata)

        # Select strategy
        selected_strategy = strategy or self._select_strategy(text)

        # Get chunker
        chunker = self._chunkers.get(selected_strategy)
        if not chunker:
            # Fall back to token chunking
            chunker = self._chunkers.get(FastChunkingStrategy.TOKEN)
            if not chunker:
                return self._fallback_chunk(text, document_id, metadata)

        try:
            # Run chunking in thread pool
            loop = asyncio.get_running_loop()
            chonkie_chunks = await loop.run_in_executor(
                None,
                lambda: list(chunker.chunk(text))
            )

            # Convert to our format
            chunks = []
            for i, cc in enumerate(chonkie_chunks):
                chunk_text = cc.text if hasattr(cc, 'text') else str(cc)
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]

                chunk = FastChunk(
                    content=chunk_text,
                    index=i,
                    start_pos=cc.start_index if hasattr(cc, 'start_index') else 0,
                    end_pos=cc.end_index if hasattr(cc, 'end_index') else len(chunk_text),
                    token_count=cc.token_count if hasattr(cc, 'token_count') else len(chunk_text.split()),
                    chunk_hash=chunk_hash,
                    metadata={
                        **(metadata or {}),
                        "document_id": document_id,
                        "strategy": selected_strategy.value,
                        "chunker": "chonkie",
                    },
                )
                chunks.append(chunk)

            logger.debug(
                "Text chunked",
                text_length=len(text),
                chunks=len(chunks),
                strategy=selected_strategy.value,
            )

            return chunks

        except Exception as e:
            logger.error("Chunking failed", error=str(e))
            return self._fallback_chunk(text, document_id, metadata)

    def _fallback_chunk(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[FastChunk]:
        """Simple fallback chunking without Chonkie."""
        chunks = []
        chunk_size = self.config.chunk_size * 4  # Approximate chars per chunk
        overlap = self.config.chunk_overlap * 4

        start = 0
        index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                    break_pos = text.rfind(sep, start + chunk_size // 2, end)
                    if break_pos > start:
                        end = break_pos + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                chunks.append(FastChunk(
                    content=chunk_text,
                    index=index,
                    start_pos=start,
                    end_pos=end,
                    token_count=len(chunk_text.split()),
                    chunk_hash=chunk_hash,
                    metadata={
                        **(metadata or {}),
                        "document_id": document_id,
                        "strategy": "fallback",
                        "chunker": "simple",
                    },
                ))
                index += 1

            start = end - overlap if end < len(text) else end

        return chunks

    async def chunk_batch(
        self,
        texts: List[str],
        strategy: Optional[FastChunkingStrategy] = None,
        max_concurrent: int = 4,
    ) -> List[List[FastChunk]]:
        """
        Chunk multiple texts concurrently.

        Args:
            texts: List of texts to chunk
            strategy: Chunking strategy
            max_concurrent: Max concurrent operations

        Returns:
            List of chunk lists, one per input text
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def chunk_one(text: str, idx: int) -> List[FastChunk]:
            async with semaphore:
                return await self.chunk(text, strategy, document_id=f"doc_{idx}")

        tasks = [chunk_one(text, i) for i, text in enumerate(texts)]
        return await asyncio.gather(*tasks)


# =============================================================================
# Phase 67: Proposition Chunking Support
# =============================================================================

class PropositionChunker:
    """
    Proposition Chunking: Break text into atomic factual statements.

    Each chunk is a self-contained proposition/fact that can be independently
    verified and retrieved. Excellent for:
    - Fact-heavy documents (Wikipedia, research papers)
    - Question-answering retrieval
    - Reducing hallucinations

    Research: "Dense X Retrieval" (2023) - proposition-based indexing

    Example:
        Input: "Paris is the capital of France. It has a population of 2.1 million."
        Output: [
            "Paris is the capital of France.",
            "Paris has a population of 2.1 million."
        ]
    """

    PROPOSITION_PROMPT = """Break the following text into atomic factual statements.
Each statement should:
1. Be self-contained and understandable without context
2. Contain exactly one fact
3. Use explicit subjects (replace pronouns with nouns)
4. Be concise but complete

Text:
{text}

Output each proposition on a new line, numbered:"""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        max_propositions_per_chunk: int = 5,
        min_text_length: int = 100,
    ):
        """
        Initialize proposition chunker.

        Args:
            llm_model: LLM to use for proposition extraction
            max_propositions_per_chunk: Group propositions into chunks of this size
            min_text_length: Minimum text length to apply proposition chunking
        """
        self.llm_model = llm_model
        self.max_propositions_per_chunk = max_propositions_per_chunk
        self.min_text_length = min_text_length
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the LLM."""
        if self._initialized:
            return True

        try:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(model=self.llm_model, temperature=0)
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"PropositionChunker init failed: {e}")
            return False

    async def extract_propositions(self, text: str) -> List[str]:
        """
        Extract atomic propositions from text using LLM.

        Args:
            text: Text to decompose into propositions

        Returns:
            List of proposition strings
        """
        if not await self.initialize():
            return [text]  # Fallback: return original text

        if len(text) < self.min_text_length:
            return [text]

        try:
            from langchain_core.messages import HumanMessage

            prompt = self.PROPOSITION_PROMPT.format(text=text[:4000])
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])

            # Parse numbered propositions
            propositions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line:
                    # Remove numbering (1., 2., etc.)
                    import re
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if cleaned and len(cleaned) > 10:
                        propositions.append(cleaned)

            return propositions if propositions else [text]

        except Exception as e:
            logger.warning(f"Proposition extraction failed: {e}")
            return [text]

    async def chunk(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[FastChunk]:
        """
        Chunk text into proposition-based chunks.

        Args:
            text: Text to chunk
            document_id: Optional document ID
            metadata: Optional metadata

        Returns:
            List of FastChunk objects
        """
        # First, split into paragraphs/sections
        paragraphs = text.split('\n\n')
        all_propositions = []

        for para in paragraphs:
            para = para.strip()
            if len(para) > self.min_text_length:
                props = await self.extract_propositions(para)
                all_propositions.extend(props)
            elif para:
                all_propositions.append(para)

        # Group propositions into chunks
        chunks = []
        for i in range(0, len(all_propositions), self.max_propositions_per_chunk):
            batch = all_propositions[i:i + self.max_propositions_per_chunk]
            chunk_text = ' '.join(batch)
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]

            chunks.append(FastChunk(
                content=chunk_text,
                index=len(chunks),
                start_pos=0,  # Not tracking exact positions for propositions
                end_pos=len(chunk_text),
                token_count=len(chunk_text.split()),
                chunk_hash=chunk_hash,
                metadata={
                    **(metadata or {}),
                    "document_id": document_id,
                    "strategy": "proposition",
                    "chunker": "proposition",
                    "num_propositions": len(batch),
                },
            ))

        logger.debug(
            "Proposition chunking complete",
            text_length=len(text),
            propositions=len(all_propositions),
            chunks=len(chunks),
        )

        return chunks


# =============================================================================
# Late Chunking Support
# =============================================================================

class LateChunker:
    """
    Late Chunking: Embed full document, then split.

    Traditional: chunk → embed each chunk
    Late chunking: embed full doc → slice embeddings by chunk positions

    Benefits:
    - Each chunk embedding has full document context
    - Better for cross-chunk queries
    - Same number of embeddings, better quality

    Research: Jina AI (2024) - "Late Chunking"
    """

    def __init__(self, fast_chunker: Optional[FastChunker] = None):
        self.chunker = fast_chunker or FastChunker()

    async def prepare_for_late_chunking(
        self,
        text: str,
        strategy: FastChunkingStrategy = FastChunkingStrategy.TOKEN,
    ) -> Tuple[str, List[FastChunk]]:
        """
        Prepare text for late chunking.

        Returns the full text (for embedding) and chunks with position info.

        Usage:
            full_text, chunks = await late_chunker.prepare_for_late_chunking(doc)

            # Embed full document (preserves cross-chunk context)
            full_embedding = await embed(full_text)  # [seq_len, dim]

            # Slice embeddings by chunk positions
            for chunk in chunks:
                chunk_embedding = full_embedding[chunk.start_pos:chunk.end_pos].mean(axis=0)
        """
        chunks = await self.chunker.chunk(text, strategy)

        # Store token positions for embedding slicing
        for chunk in chunks:
            chunk.embedding_slice = (chunk.start_pos, chunk.end_pos)

        return text, chunks

    def slice_embeddings(
        self,
        full_embeddings: List[List[float]],  # [seq_len, dim]
        chunks: List[FastChunk],
        pooling: str = "mean",
    ) -> List[List[float]]:
        """
        Slice full document embeddings by chunk positions.

        Args:
            full_embeddings: Token-level embeddings [seq_len, dim]
            chunks: Chunks with embedding_slice positions
            pooling: How to pool token embeddings ("mean", "max", "first")

        Returns:
            List of chunk embeddings [num_chunks, dim]
        """
        import numpy as np

        full_emb = np.array(full_embeddings)
        chunk_embeddings = []

        for chunk in chunks:
            if chunk.embedding_slice:
                start, end = chunk.embedding_slice
                # Map character positions to token positions (approximate)
                # In practice, you'd use tokenizer for exact mapping
                start_tok = min(start // 4, len(full_emb) - 1)
                end_tok = min(end // 4, len(full_emb))

                if start_tok < end_tok:
                    slice_emb = full_emb[start_tok:end_tok]

                    if pooling == "mean":
                        chunk_emb = slice_emb.mean(axis=0)
                    elif pooling == "max":
                        chunk_emb = slice_emb.max(axis=0)
                    else:  # first
                        chunk_emb = slice_emb[0]

                    chunk_embeddings.append(chunk_emb.tolist())
                else:
                    # Fallback to full document mean
                    chunk_embeddings.append(full_emb.mean(axis=0).tolist())
            else:
                # No position info, use full document
                chunk_embeddings.append(full_emb.mean(axis=0).tolist())

        return chunk_embeddings


# =============================================================================
# Singleton Management
# =============================================================================

_fast_chunker: Optional[FastChunker] = None
_chunker_lock = asyncio.Lock()


async def get_fast_chunker(
    config: Optional[FastChunkingConfig] = None,
) -> FastChunker:
    """Get or create fast chunker singleton."""
    global _fast_chunker

    if _fast_chunker is not None:
        return _fast_chunker

    async with _chunker_lock:
        if _fast_chunker is not None:
            return _fast_chunker

        _fast_chunker = FastChunker(config)
        return _fast_chunker


def get_fast_chunker_sync(
    config: Optional[FastChunkingConfig] = None,
) -> FastChunker:
    """Get fast chunker synchronously."""
    global _fast_chunker

    if _fast_chunker is None:
        _fast_chunker = FastChunker(config)

    return _fast_chunker


# =============================================================================
# Convenience Functions
# =============================================================================

async def chunk_text(
    text: str,
    strategy: FastChunkingStrategy = FastChunkingStrategy.AUTO,
    chunk_size: int = 512,
) -> List[FastChunk]:
    """
    Convenience function to chunk text.

    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size in tokens

    Returns:
        List of FastChunk objects
    """
    config = FastChunkingConfig(strategy=strategy, chunk_size=chunk_size)
    chunker = FastChunker(config)
    return await chunker.chunk(text)
