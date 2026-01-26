"""
AIDocumentIndexer - Vector Store Service
=========================================

Provides vector storage and similarity search using PostgreSQL + pgvector.
Supports hybrid search (vector + keyword) with access tier filtering.
"""

import asyncio
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import structlog
from sqlalchemy import select, text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context, get_async_session_factory
from backend.db.models import Chunk, Document, AccessTier, HAS_PGVECTOR, ProcessingStatus

logger = structlog.get_logger(__name__)


# =============================================================================
# Query Parser for Search Operators
# =============================================================================

def parse_search_query(query: str) -> Tuple[str, bool]:
    """
    Parse a search query with AND/OR/NOT operators into PostgreSQL tsquery format.

    Supports:
    - AND: term1 AND term2 → term1 & term2
    - OR: term1 OR term2 → term1 | term2
    - NOT: NOT term → !term
    - Phrases: "exact phrase" → preserved with <->
    - Parentheses: (term1 OR term2) AND term3

    Args:
        query: User search query with optional operators

    Returns:
        Tuple of (tsquery_string, has_operators)
        - tsquery_string: PostgreSQL-compatible tsquery
        - has_operators: True if explicit operators were found

    Examples:
        "python AND machine learning" → ("python & machine & learning", True)
        "python OR java" → ("python | java", True)
        "python NOT java" → ("python & !java", True)
        "machine learning" → ("machine & learning", False)
        '"exact phrase"' → ("exact <-> phrase", True)
    """
    if not query or not query.strip():
        return ("", False)

    original_query = query.strip()

    # Check if query has explicit operators
    has_operators = bool(re.search(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE))

    # Extract quoted phrases first and replace with placeholders
    phrases = []
    phrase_pattern = r'"([^"]+)"'

    def replace_phrase(match):
        phrase = match.group(1).strip()
        phrases.append(phrase)
        return f"__PHRASE_{len(phrases) - 1}__"

    query = re.sub(phrase_pattern, replace_phrase, query)

    # If no explicit operators, convert to simple tsquery (plainto_tsquery behavior)
    if not has_operators and not phrases:
        # Just return the original - let PostgreSQL handle it with plainto_tsquery
        return (original_query, False)

    # Normalize operators to uppercase for consistent processing
    query = re.sub(r'\bAND\b', ' & ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bOR\b', ' | ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bNOT\s+', ' !', query, flags=re.IGNORECASE)

    # Handle parentheses - keep them for grouping
    # Convert spaces between words (not operators) to &

    # Split by operators and parentheses while keeping them
    tokens = re.split(r'(\s*[&|!()]\s*)', query)

    result_parts = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Keep operators and parentheses as-is
        if token in ['&', '|', '!', '(', ')']:
            result_parts.append(token)
        elif token.startswith('__PHRASE_') and token.endswith('__'):
            # Restore phrase with phrase search operator
            idx = int(token.replace('__PHRASE_', '').replace('__', ''))
            phrase_words = phrases[idx].split()
            if len(phrase_words) > 1:
                # Use <-> for phrase matching (adjacent words)
                phrase_tsquery = ' <-> '.join(phrase_words)
                result_parts.append(f"({phrase_tsquery})")
            else:
                result_parts.append(phrase_words[0])
        else:
            # Regular term - might contain multiple words
            words = token.split()
            if len(words) > 1:
                # Multiple words without operator - AND them together
                result_parts.append('(' + ' & '.join(words) + ')')
            elif len(words) == 1:
                result_parts.append(words[0])

    # Join result
    result = ' '.join(result_parts)

    # Clean up multiple spaces and fix operator spacing
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*&\s*', ' & ', result)
    result = re.sub(r'\s*\|\s*', ' | ', result)
    result = re.sub(r'\s*!\s*', ' !', result)

    # Remove trailing/leading operators
    result = re.sub(r'^[\s&|]+', '', result)
    result = re.sub(r'[\s&|]+$', '', result)

    # If result is empty after processing, use original
    if not result.strip():
        return (original_query, False)

    return (result.strip(), True)


# Cross-encoder reranking support
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    CrossEncoder = None


# =============================================================================
# Types
# =============================================================================

class SearchType(str, Enum):
    """Search type for retrieval."""
    VECTOR = "vector"           # Pure vector similarity
    KEYWORD = "keyword"         # Full-text search
    HYBRID = "hybrid"           # Combined vector + keyword


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk_id: str
    document_id: str
    content: str
    score: float  # RRF score for ranking in hybrid search
    similarity_score: float = 0.0  # Original vector similarity (0-1) for display/confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Document info
    document_title: Optional[str] = None
    document_filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    collection: Optional[str] = None  # Collection/tag for document grouping

    # Enhanced metadata (from LLM analysis)
    enhanced_summary: Optional[str] = None
    enhanced_keywords: Optional[List[str]] = None

    # Context expansion (surrounding chunks for better context)
    prev_chunk_snippet: Optional[str] = None  # Preview of previous chunk
    next_chunk_snippet: Optional[str] = None  # Preview of next chunk
    chunk_index: Optional[int] = None  # Position in document for navigation


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    # Search settings
    default_top_k: int = 10
    similarity_threshold: float = 0.40  # PHASE 12: Lowered for better semantic recall (was 0.55)
    search_type: SearchType = SearchType.HYBRID

    # Context expansion (surrounding chunks)
    enable_context_expansion: bool = True  # Include prev/next chunk snippets
    context_snippet_length: int = 200  # Max characters for context snippets

    # Hybrid search weights
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    # Re-ranking with cross-encoder
    enable_reranking: bool = True  # Enabled by default for better accuracy
    rerank_top_k: int = 20  # Fetch more candidates for reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, accurate reranker

    # Enhanced metadata search (from LLM document analysis)
    use_enhanced_search: bool = True  # Search summaries, keywords, hypothetical questions
    enhanced_weight: float = 0.3  # Weight for enhanced metadata matches in RRF

    # MMR (Maximal Marginal Relevance) diversity settings
    enable_mmr: bool = True  # Enable diversity in results
    mmr_lambda: float = 0.5  # Balance: 0=max diversity, 1=max relevance
    mmr_fetch_k: int = 20  # Fetch more candidates for MMR selection

    # HNSW Index Optimization (Phase 57 + Phase 65 Scale Enhancement)
    hnsw_ef_search: int = 40  # Default ef_search for normal queries
    hnsw_ef_search_high_precision: int = 100  # ef_search for high-precision queries
    pgvector_iterative_scan: str = "relaxed_order"  # off, strict_order, relaxed_order

    # Phase 68: Allowed values for SQL injection prevention
    ALLOWED_ITERATIVE_SCAN_VALUES = frozenset({"off", "strict_order", "relaxed_order"})

    # Phase 65: Scale-aware HNSW (auto-tuned based on corpus size)
    hnsw_auto_tune: bool = True  # Auto-adjust params based on corpus size
    hnsw_ef_construction: int = 200  # Build-time parameter (higher = better recall)
    hnsw_m: int = 32  # Number of connections per layer (16-64 typical)

    # Phase 65: BM25 Scoring (better than TF-IDF for term saturation)
    enable_bm25: bool = True  # Enable BM25 scoring for keyword search
    bm25_k1: float = 1.5  # Term frequency saturation parameter (1.2-2.0 typical)
    bm25_b: float = 0.75  # Length normalization parameter (0.0-1.0)

    # Phase 65: Field Boosting (search-engine quality ranking)
    enable_field_boosting: bool = True  # Boost matches in titles/sections
    field_weight_section_title: float = 3.0  # Section title matches
    field_weight_document_title: float = 2.5  # Document title matches
    field_weight_enhanced_summary: float = 1.5  # LLM summary matches
    field_weight_content: float = 1.0  # Regular content matches

    # Phase 65: Freshness Boosting
    enable_freshness_boost: bool = False  # Boost recently updated documents
    freshness_decay_rate: float = 0.1  # Decay rate per day

    # Phase 92: Binary Quantization (32x memory reduction for large-scale search)
    enable_binary_quantization: bool = False  # Master toggle (reads from admin setting)
    binary_rerank_factor: int = 10  # Fetch N*rerank_factor candidates for pgvector reranking
    binary_min_corpus_size: int = 1000  # Minimum corpus size before binary search activates
    binary_use_matryoshka: bool = False  # Multi-resolution search (64→256→768 dims)


# =============================================================================
# BM25 Scorer (Phase 65 - Search Engine Quality)
# =============================================================================

class BM25Scorer:
    """
    BM25 scoring implementation for search-engine quality ranking.

    BM25 (Best Matching 25) is superior to TF-IDF because:
    1. Term frequency saturation - frequent terms don't dominate
    2. Document length normalization - fair comparison across lengths
    3. Proven effectiveness in search engines (Elasticsearch, Lucene)

    Formula:
    BM25(d, Q) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d|/avgdl))

    Where:
    - f(qi, d) = frequency of term qi in document d
    - |d| = document length (in terms)
    - avgdl = average document length
    - k1 = term frequency saturation (1.2-2.0, default 1.5)
    - b = length normalization (0.0-1.0, default 0.75)
    - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        avg_doc_length: float = 500.0,
        total_docs: int = 1000,
    ):
        """
        Initialize BM25 scorer.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            avg_doc_length: Average document length in terms
            total_docs: Total number of documents (for IDF calculation)
        """
        self.k1 = k1
        self.b = b
        self.avg_doc_length = avg_doc_length
        self.total_docs = total_docs
        self._doc_freq_cache: Dict[str, int] = {}

    def update_stats(
        self,
        avg_doc_length: float,
        total_docs: int,
        doc_freq: Optional[Dict[str, int]] = None,
    ) -> None:
        """Update corpus statistics for accurate IDF calculation."""
        self.avg_doc_length = avg_doc_length
        self.total_docs = total_docs
        if doc_freq:
            self._doc_freq_cache = doc_freq

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric."""
        import re
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    def _idf(self, term: str, doc_freq: Optional[int] = None) -> float:
        """
        Calculate Inverse Document Frequency for a term.

        Uses BM25's IDF variant which handles edge cases better:
        IDF = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        """
        import math

        if doc_freq is None:
            doc_freq = self._doc_freq_cache.get(term, 1)

        # BM25 IDF formula - always positive
        numerator = self.total_docs - doc_freq + 0.5
        denominator = doc_freq + 0.5
        return math.log(numerator / denominator + 1)

    def score(
        self,
        query: str,
        document: str,
        doc_freq: Optional[Dict[str, int]] = None,
    ) -> float:
        """
        Calculate BM25 score for a query-document pair.

        Args:
            query: Search query string
            document: Document text to score
            doc_freq: Optional dict of term -> document frequency

        Returns:
            BM25 score (higher is better)
        """
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(document)

        if not query_terms or not doc_terms:
            return 0.0

        doc_length = len(doc_terms)
        doc_term_freq: Dict[str, int] = {}
        for term in doc_terms:
            doc_term_freq[term] = doc_term_freq.get(term, 0) + 1

        score = 0.0
        for term in set(query_terms):
            if term not in doc_term_freq:
                continue

            freq = doc_term_freq[term]

            # Get IDF
            if doc_freq and term in doc_freq:
                idf = self._idf(term, doc_freq[term])
            else:
                idf = self._idf(term)

            # BM25 term score
            # (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d|/avgdl))
            numerator = freq * (self.k1 + 1)
            length_norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
            denominator = freq + self.k1 * length_norm

            score += idf * (numerator / denominator)

        return score

    def score_batch(
        self,
        query: str,
        documents: List[str],
        doc_freq: Optional[Dict[str, int]] = None,
    ) -> List[float]:
        """Score multiple documents against a query."""
        return [self.score(query, doc, doc_freq) for doc in documents]


# =============================================================================
# Field Boosting (Phase 65 - Search Engine Quality)
# =============================================================================

FIELD_WEIGHTS = {
    "section_title": 3.0,
    "document_title": 2.5,
    "enhanced_summary": 1.5,
    "content": 1.0,
}


def calculate_field_boost(
    query: str,
    section_title: Optional[str] = None,
    document_title: Optional[str] = None,
    enhanced_summary: Optional[str] = None,
    content: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate field-based boost score for a search result.

    This implements search-engine style field boosting where matches
    in titles/sections are weighted more heavily than content matches.

    Args:
        query: Search query string
        section_title: Section title of the chunk
        document_title: Document title
        enhanced_summary: LLM-generated summary
        content: Chunk content
        weights: Optional custom weights (defaults to FIELD_WEIGHTS)

    Returns:
        Field boost score (multiplier, typically 1.0-3.0)
    """
    weights = weights or FIELD_WEIGHTS
    query_lower = query.lower()
    query_terms = set(query_lower.split())

    boost = 0.0
    match_count = 0

    # Check each field for matches
    fields = [
        ("section_title", section_title),
        ("document_title", document_title),
        ("enhanced_summary", enhanced_summary),
        ("content", content),
    ]

    for field_name, field_value in fields:
        if not field_value:
            continue

        field_lower = field_value.lower()
        weight = weights.get(field_name, 1.0)

        # Exact phrase match (highest boost)
        if query_lower in field_lower:
            boost += weight * 1.0
            match_count += 1
        else:
            # Partial term match
            field_terms = set(field_lower.split())
            overlap = len(query_terms & field_terms)
            if overlap > 0:
                # Scale by proportion of terms matched
                partial_boost = weight * (overlap / len(query_terms)) * 0.5
                boost += partial_boost
                match_count += 1

    # Normalize: if matches found, return boost + 1.0 base
    # If no matches, return 1.0 (neutral)
    if match_count > 0:
        return 1.0 + (boost / match_count)
    return 1.0


# =============================================================================
# Vector Store Service
# =============================================================================

class VectorStore:
    """
    Vector storage and retrieval service using PostgreSQL + pgvector.

    Features:
    - Vector similarity search (cosine, L2, inner product)
    - Hybrid search with full-text keywords
    - Access tier filtering (RLS-compatible)
    - Batch operations for efficiency
    - HNSW index optimization (Phase 57)
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize vector store.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or VectorStoreConfig()
        self._has_pgvector = HAS_PGVECTOR
        self._reranker = None
        self._pgvector_version: Optional[str] = None

        if not self._has_pgvector:
            logger.warning("pgvector not available, vector search will be limited")

        # Initialize reranker if enabled and available
        if self.config.enable_reranking and HAS_CROSS_ENCODER:
            try:
                self._reranker = CrossEncoder(self.config.rerank_model)
                logger.info("Initialized cross-encoder reranker", model=self.config.rerank_model)
            except Exception as e:
                logger.warning("Failed to initialize reranker, disabling", error=str(e))
                self._reranker = None
        elif self.config.enable_reranking:
            logger.warning("Reranking enabled but sentence-transformers not installed")

        # ColBERT reranker (lazy-loaded when setting is enabled)
        self._colbert_reranker = None

        # Phase 92: Binary Quantization (in-memory pre-filter for large corpora)
        self._binary_quantizer = None
        self._binary_index_built: bool = False
        self._binary_chunk_ids: Optional[List[str]] = None
        self._binary_corpus_size: int = 0
        self._binary_index_lock = asyncio.Lock()

        # Phase 65: Initialize BM25 scorer for search-engine quality ranking
        self._bm25_scorer: Optional[BM25Scorer] = None
        self._bm25_stats_loaded: bool = False
        if self.config.enable_bm25:
            self._bm25_scorer = BM25Scorer(
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )
            logger.info(
                "Initialized BM25 scorer",
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )

    # =========================================================================
    # Phase 92: Binary Quantization Pre-Filter
    # =========================================================================

    async def _build_binary_index(
        self,
        force_rebuild: bool = False,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Build the in-memory binary quantization index from pgvector embeddings.

        Loads all chunk embeddings, quantizes to binary for fast Hamming-distance
        pre-filtering. The full-precision vectors remain in pgvector for final ranking.

        Args:
            force_rebuild: Force rebuild even if index exists
            session: Optional database session

        Returns:
            True if index was built successfully
        """
        if self._binary_index_built and not force_rebuild:
            return True

        async with self._binary_index_lock:
            # Double-check after acquiring lock
            if self._binary_index_built and not force_rebuild:
                return True

            async def _load_and_index(db: AsyncSession):
                import numpy as np

                # Count embeddings to check minimum corpus size
                count_result = await db.execute(
                    select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
                )
                embedded_count = count_result.scalar() or 0

                if embedded_count < self.config.binary_min_corpus_size:
                    logger.info(
                        "Corpus too small for binary quantization",
                        embedded_count=embedded_count,
                        min_required=self.config.binary_min_corpus_size,
                    )
                    return False

                # Load all embeddings in batches
                batch_size = 10000
                all_embeddings = []
                all_chunk_ids = []
                offset = 0

                while True:
                    result = await db.execute(
                        select(Chunk.id, Chunk.embedding)
                        .where(Chunk.embedding.isnot(None))
                        .order_by(Chunk.id)
                        .offset(offset)
                        .limit(batch_size)
                    )
                    rows = result.all()
                    if not rows:
                        break

                    for chunk_id, embedding in rows:
                        all_chunk_ids.append(str(chunk_id))
                        all_embeddings.append(embedding)

                    offset += batch_size

                if not all_embeddings:
                    return False

                # Build binary index (store_full=False: pgvector handles reranking)
                from backend.services.binary_quantization import (
                    BinaryQuantizer,
                    BinaryQuantizationConfig,
                )
                corpus = np.array(all_embeddings, dtype=np.float32)

                config = BinaryQuantizationConfig(
                    rerank_factor=self.config.binary_rerank_factor,
                    use_matryoshka=self.config.binary_use_matryoshka,
                )
                self._binary_quantizer = BinaryQuantizer(config)
                index_stats = self._binary_quantizer.index_corpus(corpus, store_full=False)

                self._binary_chunk_ids = all_chunk_ids
                self._binary_corpus_size = len(all_chunk_ids)
                self._binary_index_built = True

                logger.info(
                    "Built binary quantization index",
                    **index_stats,
                )
                return True

            try:
                if session:
                    return await _load_and_index(session)
                else:
                    async with async_session_context() as db:
                        return await _load_and_index(db)
            except Exception as e:
                logger.warning("Failed to build binary index", error=str(e))
                return False

    async def _binary_prefilter(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> Optional[List[str]]:
        """
        Use binary quantization to pre-filter search candidates.

        Returns chunk IDs of top candidates via Hamming-distance search,
        or None if binary index is unavailable (caller falls back to full search).

        Args:
            query_embedding: Full-precision query embedding
            top_k: Number of final results desired

        Returns:
            List of candidate chunk IDs, or None if unavailable
        """
        # Check admin setting
        try:
            from backend.services.settings import get_settings_service
            settings_service = get_settings_service()
            enabled = await settings_service.get_setting("vectorstore.binary_quantization_enabled")
            if not enabled:
                return None
        except Exception:
            if not self.config.enable_binary_quantization:
                return None

        if not self._binary_index_built:
            built = await self._build_binary_index()
            if not built:
                return None

        try:
            import numpy as np

            query = np.array(query_embedding, dtype=np.float32)
            n_candidates = top_k * self.config.binary_rerank_factor

            # Use matryoshka if configured and corpus has full vectors
            if self.config.binary_use_matryoshka and self._binary_quantizer._corpus_full is not None:
                results = await self._binary_quantizer.search_matryoshka(
                    query=query,
                    corpus_full=self._binary_quantizer._corpus_full,
                    top_k=n_candidates,
                )
            else:
                results = await self._binary_quantizer.search_binary(
                    query=query,
                    top_k=n_candidates,
                )

            # Map binary indices back to chunk IDs
            candidate_ids = []
            for result in results:
                if result.index < len(self._binary_chunk_ids):
                    candidate_ids.append(self._binary_chunk_ids[result.index])

            logger.debug(
                "Binary pre-filter completed",
                candidates=len(candidate_ids),
                top_k=top_k,
                corpus_size=self._binary_corpus_size,
            )

            return candidate_ids if candidate_ids else None

        except Exception as e:
            logger.warning("Binary pre-filter failed, falling back to full search", error=str(e))
            return None

    # =========================================================================
    # HNSW Index Optimization (Phase 57)
    # =========================================================================

    # Cache for scale-aware HNSW config (Phase 65)
    _cached_hnsw_config: Optional[Dict[str, Any]] = None
    _hnsw_config_cached_at: Optional[datetime] = None

    async def _apply_hnsw_settings(
        self,
        db: AsyncSession,
        high_precision: bool = False
    ) -> None:
        """
        Apply HNSW index optimization settings before search.

        This sets pgvector session parameters for optimal search performance:
        - hnsw.ef_search: Higher values = better recall, slower speed
        - hnsw.iterative_scan: relaxed_order provides ~9x speedup with 95-99% accuracy

        Phase 65: Now supports scale-aware auto-tuning.

        Args:
            db: Database session
            high_precision: If True, use higher ef_search for better recall
        """
        try:
            # Phase 65: Use scale-aware config if auto-tune enabled
            if self.config.hnsw_auto_tune:
                # Cache config for 5 minutes to avoid repeated DB queries
                cache_valid = (
                    self._cached_hnsw_config is not None
                    and self._hnsw_config_cached_at is not None
                    and (datetime.utcnow() - self._hnsw_config_cached_at).seconds < 300
                )

                if not cache_valid:
                    self._cached_hnsw_config = await self.get_optimal_hnsw_config()
                    self._hnsw_config_cached_at = datetime.utcnow()

                config = self._cached_hnsw_config
                ef_search = (
                    config.get("ef_search_high", self.config.hnsw_ef_search_high_precision)
                    if high_precision
                    else config.get("ef_search", self.config.hnsw_ef_search)
                )
            else:
                ef_search = (
                    self.config.hnsw_ef_search_high_precision
                    if high_precision
                    else self.config.hnsw_ef_search
                )

            # Set ef_search for this session
            # Phase 68: Validate ef_search is a positive integer to prevent SQL injection
            if not isinstance(ef_search, int) or ef_search < 1 or ef_search > 10000:
                logger.warning("Invalid ef_search value, using default", ef_search=ef_search)
                ef_search = 40
            await db.execute(text(f"SET hnsw.ef_search = {ef_search}"))

            # Set iterative scan mode (pgvector 0.8.0+)
            # relaxed_order provides best speed/accuracy tradeoff
            iterative_scan = self.config.pgvector_iterative_scan
            # Phase 68: Validate against allowed values to prevent SQL injection
            if iterative_scan and iterative_scan in VectorStoreConfig.ALLOWED_ITERATIVE_SCAN_VALUES:
                if iterative_scan != "off":
                    try:
                        await db.execute(
                            text(f"SET hnsw.iterative_scan = '{iterative_scan}'")
                        )
                    except Exception:
                        # pgvector < 0.8.0 doesn't support this setting
                        pass
            elif iterative_scan:
                logger.warning("Invalid iterative_scan value ignored", value=iterative_scan)

            logger.debug(
                "Applied HNSW settings",
                ef_search=ef_search,
                iterative_scan=iterative_scan,
                high_precision=high_precision,
                auto_tuned=self.config.hnsw_auto_tune,
            )
        except Exception as e:
            # Don't fail searches if settings can't be applied
            logger.debug("Could not apply HNSW settings", error=str(e))

    # Phase 65: Scale-aware HNSW configuration
    HNSW_SCALE_CONFIGS = {
        "small": {  # <10K vectors
            "ef_construction": 128,
            "m": 16,
            "ef_search": 64,
            "ef_search_high": 100,
        },
        "medium": {  # 10K-100K vectors
            "ef_construction": 200,
            "m": 32,
            "ef_search": 100,
            "ef_search_high": 200,
        },
        "large": {  # 100K-1M vectors
            "ef_construction": 256,
            "m": 48,
            "ef_search": 150,
            "ef_search_high": 300,
        },
        "xlarge": {  # >1M vectors
            "ef_construction": 300,
            "m": 64,
            "ef_search": 200,
            "ef_search_high": 400,
        },
    }

    async def get_optimal_hnsw_config(
        self,
        vector_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get optimal HNSW configuration based on corpus size.

        For 1M+ document scale, higher M and ef values improve recall
        at the cost of memory and build time. This method returns
        recommended parameters based on the current vector count.

        Args:
            vector_count: Optional explicit count, otherwise queries DB

        Returns:
            Dict with optimal HNSW parameters
        """
        if not self.config.hnsw_auto_tune:
            return {
                "ef_construction": self.config.hnsw_ef_construction,
                "m": self.config.hnsw_m,
                "ef_search": self.config.hnsw_ef_search,
                "ef_search_high": self.config.hnsw_ef_search_high_precision,
                "scale": "manual",
            }

        # Get vector count if not provided
        if vector_count is None:
            try:
                stats = await self.get_stats()
                vector_count = stats.get("embedded_chunks", 0)
            except Exception:
                vector_count = 10000  # Default to medium scale

        # Determine scale tier
        if vector_count < 10000:
            scale = "small"
        elif vector_count < 100000:
            scale = "medium"
        elif vector_count < 1000000:
            scale = "large"
        else:
            scale = "xlarge"

        config = self.HNSW_SCALE_CONFIGS[scale].copy()
        config["scale"] = scale
        config["vector_count"] = vector_count

        logger.debug(
            "Determined HNSW scale config",
            scale=scale,
            vector_count=vector_count,
            config=config,
        )

        return config

    async def get_recommended_index_ddl(
        self,
        vector_count: Optional[int] = None,
    ) -> str:
        """
        Get recommended CREATE INDEX DDL for the current scale.

        Returns optimized HNSW index creation SQL based on corpus size.

        Args:
            vector_count: Optional explicit count

        Returns:
            SQL DDL for creating the optimal index
        """
        config = await self.get_optimal_hnsw_config(vector_count)

        ddl = f"""
-- Optimal HNSW index for {config.get('scale', 'unknown')} scale ({config.get('vector_count', 'unknown')} vectors)
-- Run with: SET maintenance_work_mem = '2GB'; SET max_parallel_maintenance_workers = 4;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (
    m = {config['m']},
    ef_construction = {config['ef_construction']}
);

-- After creation, set search parameters:
-- SET hnsw.ef_search = {config['ef_search']};
"""
        return ddl.strip()

    async def get_pgvector_version(self) -> Optional[str]:
        """
        Get the installed pgvector version.

        Returns:
            Version string (e.g., "0.8.0") or None if not available
        """
        if self._pgvector_version:
            return self._pgvector_version

        try:
            async with async_session_context() as db:
                result = await db.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
                row = result.fetchone()
                if row:
                    self._pgvector_version = row[0]
                    return self._pgvector_version
        except Exception as e:
            logger.debug("Could not get pgvector version", error=str(e))

        return None

    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about vector indexes.

        Returns:
            Dictionary with index statistics
        """
        stats = {
            "pgvector_version": await self.get_pgvector_version(),
            "indexes": [],
            "total_vectors": 0,
        }

        try:
            async with async_session_context() as db:
                # Get index information
                result = await db.execute(text("""
                    SELECT
                        i.relname as index_name,
                        t.relname as table_name,
                        pg_size_pretty(pg_relation_size(i.oid)) as index_size,
                        pg_relation_size(i.oid) as index_size_bytes
                    FROM pg_class i
                    JOIN pg_index ix ON i.oid = ix.indexrelid
                    JOIN pg_class t ON t.oid = ix.indrelid
                    WHERE i.relname LIKE '%embedding%hnsw%'
                       OR i.relname LIKE '%vector%'
                    ORDER BY pg_relation_size(i.oid) DESC
                """))
                for row in result:
                    stats["indexes"].append({
                        "name": row[0],
                        "table": row[1],
                        "size": row[2],
                        "size_bytes": row[3],
                    })

                # Get total vector count
                result = await db.execute(
                    text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                )
                row = result.fetchone()
                if row:
                    stats["total_vectors"] = row[0]

        except Exception as e:
            logger.warning("Could not get index stats", error=str(e))

        return stats

    # =========================================================================
    # BM25 Statistics (Phase 65)
    # =========================================================================

    async def _load_bm25_stats(self, db: AsyncSession) -> None:
        """
        Load corpus statistics for BM25 scoring.

        This computes:
        - Average document length (for length normalization)
        - Total document count (for IDF calculation)
        - Term document frequencies (optional, for accurate IDF)
        """
        if not self._bm25_scorer or self._bm25_stats_loaded:
            return

        try:
            # Get average chunk length and count
            result = await db.execute(text("""
                SELECT
                    COUNT(*) as total_docs,
                    AVG(char_count) as avg_length
                FROM chunks
                WHERE content IS NOT NULL
            """))
            row = result.fetchone()

            if row and row[0] > 0:
                total_docs = int(row[0])
                avg_length = float(row[1]) if row[1] else 500.0

                # Convert char count to approximate word count
                # (average English word is ~5 characters + space)
                avg_doc_length = avg_length / 6.0

                self._bm25_scorer.update_stats(
                    avg_doc_length=avg_doc_length,
                    total_docs=total_docs,
                )
                self._bm25_stats_loaded = True

                logger.debug(
                    "Loaded BM25 corpus stats",
                    total_docs=total_docs,
                    avg_doc_length=avg_doc_length,
                )

        except Exception as e:
            logger.warning("Could not load BM25 stats", error=str(e))

    async def calculate_bm25_scores(
        self,
        query: str,
        results: List[SearchResult],
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Calculate BM25 scores for search results.

        This post-processes results with BM25 scoring to get more
        accurate relevance scores than PostgreSQL's ts_rank.

        Args:
            query: Search query
            results: List of results to score
            session: Optional database session

        Returns:
            Results with updated scores based on BM25
        """
        if not self._bm25_scorer or not results:
            return results

        async def _calc(db: AsyncSession) -> List[SearchResult]:
            # Ensure stats are loaded
            await self._load_bm25_stats(db)

            # Calculate BM25 score for each result
            for result in results:
                bm25_score = self._bm25_scorer.score(query, result.content)
                result.metadata["bm25_score"] = bm25_score

                # Blend BM25 with existing score
                # BM25 typically 0-15, normalize to 0-1 range
                normalized_bm25 = min(bm25_score / 10.0, 1.0)

                # Weighted average: 60% BM25, 40% ts_rank
                original_score = result.score
                result.score = 0.6 * normalized_bm25 + 0.4 * original_score

            # Re-sort by new scores
            results.sort(key=lambda r: r.score, reverse=True)

            return results

        if session:
            return await _calc(session)
        else:
            async with async_session_context() as db:
                return await _calc(db)

    async def apply_field_boosting(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Apply field-based boosting to search results.

        Boosts results where query terms match in titles/sections.

        Args:
            query: Search query
            results: List of results to boost

        Returns:
            Results with updated scores based on field matches
        """
        if not self.config.enable_field_boosting or not results:
            return results

        # Build custom weights from config
        weights = {
            "section_title": self.config.field_weight_section_title,
            "document_title": self.config.field_weight_document_title,
            "enhanced_summary": self.config.field_weight_enhanced_summary,
            "content": self.config.field_weight_content,
        }

        for result in results:
            boost = calculate_field_boost(
                query=query,
                section_title=result.section_title,
                document_title=result.document_title,
                enhanced_summary=result.enhanced_summary,
                content=result.content[:500] if result.content else None,  # Sample
                weights=weights,
            )

            # Apply boost as a multiplier
            result.score *= boost
            result.metadata["field_boost"] = boost
            result.metadata["field_boosting_applied"] = True

        # Re-sort by boosted scores
        results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "Applied field boosting",
            result_count=len(results),
            avg_boost=sum(r.metadata.get("field_boost", 1.0) for r in results) / len(results),
        )

        return results

    # Phase 68: Regex pattern for PostgreSQL memory settings (SQL injection prevention)
    _VALID_MEMORY_PATTERN = re.compile(r'^[0-9]+\s*(kB|MB|GB|TB)?$', re.IGNORECASE)

    async def apply_index_build_settings(
        self,
        db: AsyncSession,
        maintenance_work_mem: str = "2GB",
        parallel_workers: int = 4
    ) -> None:
        """
        Apply settings for faster index builds.

        Should be called before CREATE INDEX for optimal performance.
        These settings apply to the current session only.

        Args:
            db: Database session
            maintenance_work_mem: Memory for index builds (e.g., "2GB", "512MB")
            parallel_workers: Number of parallel workers (0-7)
        """
        try:
            # Phase 68: Validate maintenance_work_mem format to prevent SQL injection
            if not maintenance_work_mem or not self._VALID_MEMORY_PATTERN.match(maintenance_work_mem):
                logger.warning(
                    "Invalid maintenance_work_mem format, using default",
                    value=maintenance_work_mem,
                )
                maintenance_work_mem = "2GB"

            # Set maintenance_work_mem for faster index builds
            await db.execute(text(f"SET maintenance_work_mem = '{maintenance_work_mem}'"))

            # Phase 68: Validate parallel_workers is in valid range
            if not isinstance(parallel_workers, int) or parallel_workers < 0:
                parallel_workers = 0

            # Set parallel workers for index creation
            if parallel_workers > 0:
                await db.execute(
                    text(f"SET max_parallel_maintenance_workers = {min(parallel_workers, 7)}")
                )

            logger.info(
                "Applied index build settings",
                maintenance_work_mem=maintenance_work_mem,
                parallel_workers=parallel_workers
            )
        except Exception as e:
            logger.warning("Could not apply index build settings", error=str(e))

    # Phase 68: Regex pattern for valid PostgreSQL identifiers (SQL injection prevention)
    _VALID_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    async def reindex_with_optimization(
        self,
        index_name: str,
        concurrent: bool = True
    ) -> bool:
        """
        Reindex a vector index with optimized settings.

        This applies performance settings before reindexing.

        Args:
            index_name: Name of the index to rebuild (must be valid PostgreSQL identifier)
            concurrent: If True, use REINDEX CONCURRENTLY (non-blocking)

        Returns:
            True if successful
        """
        # Phase 68: Validate index_name to prevent SQL injection
        if not index_name or not self._VALID_IDENTIFIER_PATTERN.match(index_name):
            logger.error(
                "Invalid index name - must be valid PostgreSQL identifier",
                index_name=index_name,
            )
            return False

        # Additional length check (PostgreSQL identifier max length is 63)
        if len(index_name) > 63:
            logger.error("Index name too long", index_name=index_name)
            return False

        try:
            async with async_session_context() as db:
                # Apply build settings
                await self.apply_index_build_settings(db)

                # Reindex with validated identifier
                concurrently = "CONCURRENTLY " if concurrent else ""
                await db.execute(
                    text(f"REINDEX INDEX {concurrently}{index_name}")
                )
                await db.commit()

                logger.info("Reindexed successfully", index_name=index_name)
                return True

        except Exception as e:
            logger.error("Failed to reindex", index_name=index_name, error=str(e))
            return False

    # =========================================================================
    # Storage Operations
    # =========================================================================

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        access_tier_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[str]:
        """
        Add chunks with embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'content', 'embedding', and metadata
            document_id: ID of the parent document
            access_tier_id: Access tier for permission filtering
            session: Optional existing database session

        Returns:
            List of created chunk IDs
        """
        chunk_ids = []

        async def _add_chunks(db: AsyncSession):
            # Build all chunk objects first for batch insertion
            chunk_objects = []
            for i, chunk_data in enumerate(chunks):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=uuid.UUID(document_id),
                    access_tier_id=uuid.UUID(access_tier_id),
                    content=chunk_data["content"],
                    content_hash=chunk_data.get("content_hash", ""),
                    embedding=chunk_data.get("embedding"),
                    chunk_index=chunk_data.get("chunk_index", i),
                    page_number=chunk_data.get("page_number"),
                    section_title=chunk_data.get("section_title"),
                    token_count=chunk_data.get("token_count"),
                    char_count=chunk_data.get("char_count", len(chunk_data["content"])),
                )
                chunk_objects.append(chunk)
                chunk_ids.append(str(chunk.id))

            # Batch add all chunks at once (2-3x faster than individual adds)
            db.add_all(chunk_objects)
            await db.flush()

        if session:
            await _add_chunks(session)
        else:
            async with async_session_context() as db:
                await _add_chunks(db)

        logger.info(
            "Added chunks to vector store",
            document_id=document_id,
            chunk_count=len(chunk_ids),
        )

        # Phase 92: Invalidate binary index (will rebuild on next search)
        if self._binary_index_built:
            self._binary_index_built = False
            logger.debug("Binary index invalidated after adding chunks")

        return chunk_ids

    async def delete_document_chunks(
        self,
        document_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID
            session: Optional existing database session

        Returns:
            Number of chunks deleted
        """
        async def _delete(db: AsyncSession) -> int:
            from sqlalchemy import delete
            # Use bulk delete for better performance (avoids loading all objects)
            stmt = delete(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
            result = await db.execute(stmt)
            return result.rowcount

        if session:
            count = await _delete(session)
        else:
            async with async_session_context() as db:
                count = await _delete(db)

        logger.info("Deleted document chunks", document_id=document_id, count=count)

        # Phase 92: Invalidate binary index (will rebuild on next search)
        if self._binary_index_built:
            self._binary_index_built = False
            logger.debug("Binary index invalidated after deleting chunks")

        return count

    async def update_chunk_embedding(
        self,
        chunk_id: str,
        embedding: List[float],
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Update embedding for a specific chunk.

        Args:
            chunk_id: Chunk ID
            embedding: New embedding vector
            session: Optional existing database session

        Returns:
            True if updated successfully
        """
        async def _update(db: AsyncSession) -> bool:
            result = await db.execute(
                select(Chunk).where(Chunk.id == uuid.UUID(chunk_id))
            )
            chunk = result.scalar_one_or_none()

            if chunk:
                chunk.embedding = embedding
                return True
            return False

        if session:
            return await _update(session)
        else:
            async with async_session_context() as db:
                return await _update(db)

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            similarity_threshold: Minimum similarity score
            session: Optional existing database session
            organization_id: Optional organization ID for multi-tenant isolation
            user_id: Optional user ID for private document access
            is_superadmin: Whether the user is a superadmin (can see all private docs)

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k
        threshold = similarity_threshold or self.config.similarity_threshold

        if not self._has_pgvector:
            logger.warning("pgvector not available, returning empty results")
            return []

        async def _search(db: AsyncSession) -> List[SearchResult]:
            # Apply HNSW index optimization settings (Phase 57)
            await self._apply_hnsw_settings(db, high_precision=False)

            # Build the query using pgvector's cosine similarity
            # 1 - (embedding <=> query) gives similarity (higher is better)
            similarity_expr = 1 - Chunk.embedding.cosine_distance(query_embedding)

            # Base query with similarity
            query = (
                select(
                    Chunk,
                    Document,
                    similarity_expr.label("similarity"),
                )
                .join(Document, Chunk.document_id == Document.id)
                .join(AccessTier, Chunk.access_tier_id == AccessTier.id)
                .where(AccessTier.level <= access_tier_level)
                .where(Chunk.embedding.isnot(None))
                # Exclude soft-deleted documents (status=FAILED with error="Deleted by user")
                .where(
                    ~(
                        (Document.processing_status == ProcessingStatus.FAILED)
                        & (Document.processing_error == "Deleted by user")
                    )
                )
            )

            # Filter by organization for multi-tenant isolation
            # PHASE 12 FIX: Include docs from user's org AND docs without org (legacy/shared)
            if organization_id and not is_superadmin:
                org_uuid = uuid.UUID(organization_id)
                query = query.where(
                    or_(
                        Chunk.organization_id == org_uuid,
                        Chunk.organization_id.is_(None),  # Include docs without org
                    )
                )

            # Filter private documents (only owner or superadmin can access)
            # Superadmins can see all private documents
            # Regular users can only see their own private documents
            if not is_superadmin:
                if user_id:
                    user_uuid = uuid.UUID(user_id)
                    # Allow: public documents OR private docs owned by this user
                    query = query.where(
                        or_(
                            Document.is_private == False,
                            and_(
                                Document.is_private == True,
                                Document.uploaded_by_id == user_uuid
                            )
                        )
                    )
                else:
                    # No user ID - only show public documents
                    query = query.where(Document.is_private == False)

            # Filter by document IDs if provided
            if document_ids:
                doc_uuids = [uuid.UUID(d) for d in document_ids]
                query = query.where(Document.id.in_(doc_uuids))

            # Phase 92: Binary quantization pre-filter (narrow candidates before pgvector)
            binary_candidate_ids = await self._binary_prefilter(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            if binary_candidate_ids:
                candidate_uuids = [uuid.UUID(cid) for cid in binary_candidate_ids]
                query = query.where(Chunk.id.in_(candidate_uuids))

            # Apply similarity threshold and ordering
            query = (
                query
                .where(similarity_expr >= threshold)
                .order_by(similarity_expr.desc())
                .limit(top_k)
            )

            result = await db.execute(query)
            rows = result.all()

            results = []
            for chunk, doc, similarity in rows:
                sim_float = float(similarity)
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=sim_float,
                    similarity_score=sim_float,  # Store original similarity
                    metadata={
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                    },
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                ))

            return results

        if session:
            return await _search(session)
        else:
            async with async_session_context() as db:
                return await _search(db)

    async def keyword_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[SearchResult]:
        """
        Perform full-text keyword search with operator support.

        Supports search operators:
        - AND: term1 AND term2 (both must match)
        - OR: term1 OR term2 (either must match)
        - NOT: NOT term (exclude matches)
        - Phrases: "exact phrase" (words must appear adjacent)
        - Parentheses: (term1 OR term2) AND term3 (grouping)

        Args:
            query: Search query string (may contain operators)
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            session: Optional existing database session
            organization_id: Optional organization ID for multi-tenant isolation
            user_id: Optional user ID for private document access
            is_superadmin: Whether the user is a superadmin (can see all private docs)

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k

        # Parse query for operators (AND, OR, NOT, phrases)
        parsed_query, has_operators = parse_search_query(query)

        async def _search(db: AsyncSession) -> List[SearchResult]:
            # PostgreSQL full-text search
            # Use to_tsquery for operator support, plainto_tsquery for simple queries
            if has_operators:
                # Use to_tsquery for advanced queries with operators
                ts_query = func.to_tsquery('english', parsed_query)
            else:
                # Use plainto_tsquery for simple queries (handles stemming better)
                ts_query = func.plainto_tsquery('english', query)
            ts_rank = func.ts_rank(
                func.to_tsvector('english', Chunk.content),
                ts_query
            )

            base_query = (
                select(
                    Chunk,
                    Document,
                    ts_rank.label("rank"),
                )
                .join(Document, Chunk.document_id == Document.id)
                .join(AccessTier, Chunk.access_tier_id == AccessTier.id)
                .where(AccessTier.level <= access_tier_level)
                .where(
                    func.to_tsvector('english', Chunk.content).op('@@')(ts_query)
                )
                # Exclude soft-deleted documents (status=FAILED with error="Deleted by user")
                .where(
                    ~(
                        (Document.processing_status == ProcessingStatus.FAILED)
                        & (Document.processing_error == "Deleted by user")
                    )
                )
            )

            # Filter by organization for multi-tenant isolation
            # PHASE 12 FIX: Include docs from user's org AND docs without org (legacy/shared)
            if organization_id and not is_superadmin:
                org_uuid = uuid.UUID(organization_id)
                base_query = base_query.where(
                    or_(
                        Chunk.organization_id == org_uuid,
                        Chunk.organization_id.is_(None),  # Include docs without org
                    )
                )

            # Filter private documents (only owner or superadmin can access)
            if not is_superadmin:
                if user_id:
                    user_uuid = uuid.UUID(user_id)
                    base_query = base_query.where(
                        or_(
                            Document.is_private == False,
                            and_(
                                Document.is_private == True,
                                Document.uploaded_by_id == user_uuid
                            )
                        )
                    )
                else:
                    base_query = base_query.where(Document.is_private == False)

            # Filter by document IDs if provided
            if document_ids:
                doc_uuids = [uuid.UUID(d) for d in document_ids]
                base_query = base_query.where(Document.id.in_(doc_uuids))

            # Order by rank
            base_query = (
                base_query
                .order_by(ts_rank.desc())
                .limit(top_k)
            )

            result = await db.execute(base_query)
            rows = result.all()

            results = []
            for chunk, doc, rank in rows:
                # Normalize rank to 0-1 range for display (keyword ranks are typically 1-N)
                # Use a reasonable default similarity for keyword matches
                normalized_score = min(1.0, 1.0 / (float(rank) + 1)) if rank > 0 else 0.5
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=float(rank),
                    similarity_score=normalized_score,  # Keyword match score for display
                    metadata={
                        "chunk_index": chunk.chunk_index,
                        "search_type": "keyword",
                        "used_operators": has_operators,
                        "parsed_query": parsed_query if has_operators else None,
                    },
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                ))

            return results

        # Execute search
        if session:
            results = await _search(session)
        else:
            async with async_session_context() as db:
                results = await _search(db)

        # Phase 65: Apply BM25 scoring for better ranking
        if self.config.enable_bm25 and self._bm25_scorer and results:
            results = await self.calculate_bm25_scores(
                query=query,
                results=results,
                session=session,
            )

        return results

    async def enhanced_metadata_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[SearchResult]:
        """
        Search documents using enhanced metadata (summaries, keywords, questions).

        This searches the LLM-extracted metadata stored in Document.enhanced_metadata
        including summaries, keywords, topics, and hypothetical questions.

        Args:
            query: Search query string
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            session: Optional existing database session
            organization_id: Optional organization ID for multi-tenant isolation
            user_id: Optional user ID for private document access
            is_superadmin: Whether the user is a superadmin (can see all private docs)

        Returns:
            List of SearchResult objects (one per matching document)
        """
        top_k = top_k or self.config.default_top_k

        async def _search(db: AsyncSession) -> List[SearchResult]:
            # Get documents with enhanced metadata
            # Only include completed documents (excludes soft-deleted which are FAILED)
            base_query = (
                select(Document)
                .where(Document.enhanced_metadata.isnot(None))
                .where(Document.processing_status == ProcessingStatus.COMPLETED)
            )

            # Filter by organization for multi-tenant isolation
            # PHASE 12 FIX: Include docs from user's org AND docs without org (legacy/shared)
            if organization_id and not is_superadmin:
                org_uuid = uuid.UUID(organization_id)
                base_query = base_query.where(
                    or_(
                        Document.organization_id == org_uuid,
                        Document.organization_id.is_(None),  # Include docs without org
                    )
                )

            # Filter private documents (only owner or superadmin can access)
            if not is_superadmin:
                if user_id:
                    user_uuid = uuid.UUID(user_id)
                    base_query = base_query.where(
                        or_(
                            Document.is_private == False,
                            and_(
                                Document.is_private == True,
                                Document.uploaded_by_id == user_uuid
                            )
                        )
                    )
                else:
                    base_query = base_query.where(Document.is_private == False)

            # Filter by document IDs if provided
            if document_ids:
                doc_uuids = [uuid.UUID(d) for d in document_ids]
                base_query = base_query.where(Document.id.in_(doc_uuids))

            result = await db.execute(base_query)
            documents = result.scalars().all()

            if not documents:
                return []

            # Score each document based on query match to enhanced metadata
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            scored_docs = []

            for doc in documents:
                metadata = doc.enhanced_metadata or {}
                score = 0.0

                # Search in summary
                summary_short = (metadata.get("summary_short") or "").lower()
                summary_detailed = (metadata.get("summary_detailed") or "").lower()
                if query_lower in summary_short:
                    score += 0.5
                if query_lower in summary_detailed:
                    score += 0.3

                # Search in keywords
                keywords = [k.lower() for k in metadata.get("keywords", [])]
                keyword_matches = sum(1 for term in query_terms if any(term in k for k in keywords))
                score += keyword_matches * 0.2

                # Search in topics
                topics = [t.lower() for t in metadata.get("topics", [])]
                topic_matches = sum(1 for term in query_terms if any(term in t for t in topics))
                score += topic_matches * 0.15

                # Search in hypothetical questions
                questions = [q.lower() for q in metadata.get("hypothetical_questions", [])]
                for question in questions:
                    # Check for query term overlap with questions
                    question_terms = set(question.split())
                    overlap = len(query_terms & question_terms)
                    if overlap > 0:
                        score += overlap * 0.1

                # Search in entities
                entities = metadata.get("entities", {})
                for entity_type, entity_list in entities.items():
                    entity_values = [e.lower() for e in entity_list]
                    entity_matches = sum(1 for term in query_terms if any(term in e for e in entity_values))
                    score += entity_matches * 0.1

                if score > 0:
                    scored_docs.append((doc, score, metadata))

            # Sort by score and take top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = scored_docs[:top_k]

            # Convert to SearchResult (using first chunk as representative)
            results = []
            for doc, score, metadata in top_docs:
                # Get first chunk for content preview
                chunk_result = await db.execute(
                    select(Chunk)
                    .where(Chunk.document_id == doc.id)
                    .order_by(Chunk.chunk_index)
                    .limit(1)
                )
                first_chunk = chunk_result.scalar_one_or_none()

                summary = metadata.get("summary_detailed") or metadata.get("summary_short") or ""
                content = summary if summary else (first_chunk.content if first_chunk else "")

                results.append(SearchResult(
                    chunk_id=str(first_chunk.id) if first_chunk else "",
                    document_id=str(doc.id),
                    content=content,
                    score=score,
                    similarity_score=min(score, 1.0),  # Metadata match score for display
                    metadata={
                        "search_type": "enhanced",
                        "topics": metadata.get("topics", []),
                        "document_type": metadata.get("document_type"),
                    },
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    enhanced_summary=metadata.get("summary_short"),
                    enhanced_keywords=metadata.get("keywords", []),
                ))

            return results

        if session:
            return await _search(session)
        else:
            async with async_session_context() as db:
                return await _search(db)

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        use_enhanced: Optional[bool] = None,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity, keyword matching,
        and optionally enhanced metadata search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from all sources.

        Args:
            query: Search query string
            query_embedding: Query embedding vector
            top_k: Number of final results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            vector_weight: Weight for vector results (0-1)
            keyword_weight: Weight for keyword results (0-1)
            use_enhanced: Whether to include enhanced metadata search
            session: Optional existing database session
            organization_id: Optional organization ID for multi-tenant isolation
            user_id: Optional user ID for private document access
            is_superadmin: Whether the user is a superadmin (can see all private docs)

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k
        vec_weight = vector_weight or self.config.vector_weight
        kw_weight = keyword_weight or self.config.keyword_weight
        enhanced_weight = self.config.enhanced_weight
        use_enhanced = use_enhanced if use_enhanced is not None else self.config.use_enhanced_search

        # Get more results from each method for better fusion
        fetch_k = self.config.rerank_top_k if self.config.enable_reranking else top_k * 2

        # Run searches in parallel conceptually (await each)
        vector_results = await self.similarity_search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
            session=session,
            organization_id=organization_id,
            user_id=user_id,
            is_superadmin=is_superadmin,
        )

        keyword_results = await self.keyword_search(
            query=query,
            top_k=fetch_k,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
            session=session,
            organization_id=organization_id,
            user_id=user_id,
            is_superadmin=is_superadmin,
        )

        # Optionally search enhanced metadata
        enhanced_results = []
        if use_enhanced:
            enhanced_results = await self.enhanced_metadata_search(
                query=query,
                top_k=fetch_k,
                organization_id=organization_id,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
                session=session,
                user_id=user_id,
                is_superadmin=is_superadmin,
            )

        # Reciprocal Rank Fusion
        # RRF(d) = Σ 1/(k + rank(d)) where k is a constant (typically 60)
        k = 60
        scores: Dict[str, Tuple[float, SearchResult]] = {}

        # Process vector results - preserve original similarity scores
        for rank, result in enumerate(vector_results):
            rrf_score = vec_weight * (1.0 / (k + rank + 1))
            original_similarity = result.similarity_score  # Preserve before RRF overwrites
            if result.chunk_id in scores:
                existing_rrf, existing_result = scores[result.chunk_id]
                # Keep max similarity from vector search
                existing_result.similarity_score = max(existing_result.similarity_score, original_similarity)
                scores[result.chunk_id] = (existing_rrf + rrf_score, existing_result)
            else:
                # Ensure similarity_score is preserved (already set from similarity_search)
                scores[result.chunk_id] = (rrf_score, result)

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            rrf_score = kw_weight * (1.0 / (k + rank + 1))
            if result.chunk_id in scores:
                existing_rrf, existing_result = scores[result.chunk_id]
                scores[result.chunk_id] = (existing_rrf + rrf_score, existing_result)
            else:
                # Keyword results don't have similarity score - leave as 0
                scores[result.chunk_id] = (rrf_score, result)

        # Process enhanced metadata results
        # Enhanced results are document-level, so we boost all chunks from matching docs
        enhanced_doc_scores: Dict[str, float] = {}
        for rank, result in enumerate(enhanced_results):
            rrf_score = enhanced_weight * (1.0 / (k + rank + 1))
            enhanced_doc_scores[result.document_id] = rrf_score
            # Also add the enhanced result itself if it has a chunk
            if result.chunk_id:
                if result.chunk_id in scores:
                    existing_score, existing_result = scores[result.chunk_id]
                    # Merge enhanced metadata into existing result
                    existing_result.enhanced_summary = result.enhanced_summary
                    existing_result.enhanced_keywords = result.enhanced_keywords
                    scores[result.chunk_id] = (existing_score + rrf_score, existing_result)
                else:
                    scores[result.chunk_id] = (rrf_score, result)

        # Boost existing chunk scores for documents with enhanced metadata matches
        for chunk_id, (current_score, result) in list(scores.items()):
            if result.document_id in enhanced_doc_scores:
                boost = enhanced_doc_scores[result.document_id] * 0.5  # 50% of doc-level score
                scores[chunk_id] = (current_score + boost, result)

        # Sort by combined score and return top_k
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        # Update scores in results
        final_results = []
        for score, result in sorted_results:
            result.score = score
            result.metadata["search_type"] = "hybrid"
            if use_enhanced and result.document_id in enhanced_doc_scores:
                result.metadata["enhanced_boost"] = True
            final_results.append(result)

        logger.debug(
            "Hybrid search completed",
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            enhanced_count=len(enhanced_results),
            final_count=len(final_results),
        )

        # Phase 65: Apply BM25 scoring for search-engine quality
        if self.config.enable_bm25 and self._bm25_scorer:
            final_results = await self.calculate_bm25_scores(
                query=query,
                results=final_results,
                session=session,
            )
            logger.debug("Applied BM25 scoring", result_count=len(final_results))

        # Phase 65: Apply field boosting for title/section matches
        if self.config.enable_field_boosting:
            final_results = await self.apply_field_boosting(
                query=query,
                results=final_results,
            )
            logger.debug("Applied field boosting", result_count=len(final_results))

        # Apply reranking if enabled and available
        if self.config.enable_reranking and query:
            final_results = await self._rerank_results_async(query, final_results, top_k)

        # Expand context with surrounding chunks
        if self.config.enable_context_expansion:
            final_results = await self.expand_context(final_results, session=session)

        return final_results

    async def _rerank_results_async(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank search results using ColBERT or cross-encoder based on settings.

        Checks rag.use_colbert_reranker setting to decide which reranker to use.
        ColBERT provides 10-20% better precision with similar speed.

        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return results

        # Check if ColBERT reranking is enabled
        try:
            from backend.services.settings import get_settings_service
            settings = get_settings_service()
            use_colbert = await settings.get_setting("rag.use_colbert_reranker")

            if use_colbert:
                return await self._rerank_with_colbert(query, results, top_k)
        except Exception as e:
            logger.debug("Could not check ColBERT setting, using cross-encoder", error=str(e))

        # Fall back to cross-encoder
        return self._rerank_results(query, results, top_k)

    async def _rerank_with_colbert(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank results using ColBERT reranker.

        ColBERT uses late interaction (MaxSim) scoring for better precision.
        """
        try:
            # Lazy-load ColBERT reranker
            if self._colbert_reranker is None:
                from backend.services.colbert_reranker import ColBERTReranker
                self._colbert_reranker = ColBERTReranker()
                logger.info("Initialized ColBERT reranker")

            # Prepare documents for ColBERT (expects list of dicts)
            documents = [
                {
                    "content": result.content,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in results
            ]

            # Rerank using ColBERT
            reranked_results = await self._colbert_reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )

            # Build lookup for original results by chunk_id
            result_lookup = {r.chunk_id: r for r in results}

            # Map reranked results back to SearchResult objects
            reranked = []
            for rr in reranked_results:
                if rr.chunk_id in result_lookup:
                    result = result_lookup[rr.chunk_id]
                    result.score = rr.rerank_score
                    result.metadata["reranked"] = True
                    result.metadata["rerank_score"] = rr.rerank_score
                    result.metadata["reranker"] = "colbert"
                    reranked.append(result)

            logger.debug(
                "ColBERT reranked results",
                original_count=len(results),
                reranked_count=len(reranked),
            )

            return reranked

        except Exception as e:
            logger.warning("ColBERT reranking failed, falling back to cross-encoder", error=str(e))
            return self._rerank_results(query, results, top_k)

    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder model.

        Cross-encoders provide more accurate relevance scores by jointly
        encoding the query and document, at the cost of being slower.

        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        if not results or self._reranker is None:
            return results

        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [(query, result.content) for result in results]

            # Get reranking scores
            scores = self._reranker.predict(pairs)

            # Pair results with their rerank scores
            scored_results = list(zip(results, scores))

            # Sort by rerank score (higher is better)
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Update scores and return top_k
            reranked = []
            for result, score in scored_results[:top_k]:
                result.score = float(score)  # Use rerank score
                result.metadata["reranked"] = True
                result.metadata["rerank_score"] = float(score)
                result.metadata["reranker"] = "cross-encoder"
                reranked.append(result)

            logger.debug(
                "Cross-encoder reranked results",
                original_count=len(results),
                reranked_count=len(reranked),
            )

            return reranked

        except Exception as e:
            logger.warning("Reranking failed, returning original results", error=str(e))
            return results[:top_k]

    def _apply_mmr(
        self,
        results: List[SearchResult],
        query_embedding: List[float],
        lambda_param: Optional[float] = None,
        k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Apply Maximal Marginal Relevance for diversity in results.

        MMR balances relevance to the query with diversity among results.
        It iteratively selects results that are both relevant and different
        from already-selected results.

        Formula: MMR = λ * Sim(doc, query) - (1-λ) * max(Sim(doc, selected_docs))

        Args:
            results: List of search results (should have more than k items)
            query_embedding: Query embedding vector
            lambda_param: Balance parameter (0=diversity, 1=relevance). Default from config.
            k: Number of results to return. Default from config.

        Returns:
            Diversified list of SearchResult objects
        """
        if not results or not query_embedding:
            return results

        lambda_param = lambda_param if lambda_param is not None else self.config.mmr_lambda
        k = k if k is not None else self.config.default_top_k

        if len(results) <= k:
            return results

        # Helper function for cosine similarity
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            if not vec1 or not vec2:
                return 0.0
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        # Get embeddings from results (if available)
        result_embeddings = []
        for r in results:
            emb = r.metadata.get("embedding") or r.embedding if hasattr(r, 'embedding') else None
            result_embeddings.append(emb)

        # If embeddings not available, return original results
        if all(e is None for e in result_embeddings):
            logger.debug("No embeddings available for MMR, returning original results")
            return results[:k]

        # Calculate relevance scores for all results
        relevance_scores = []
        for i, result in enumerate(results):
            if result_embeddings[i]:
                rel_score = cosine_similarity(query_embedding, result_embeddings[i])
            else:
                # Fall back to existing score
                rel_score = result.similarity_score or result.score or 0.0
            relevance_scores.append(rel_score)

        # MMR selection
        selected_indices = []
        candidate_indices = list(range(len(results)))

        while len(selected_indices) < k and candidate_indices:
            best_score = -float('inf')
            best_idx = None

            for idx in candidate_indices:
                # Relevance to query
                relevance = relevance_scores[idx]

                # Maximum similarity to already selected
                max_sim_to_selected = 0.0
                if selected_indices and result_embeddings[idx]:
                    for sel_idx in selected_indices:
                        if result_embeddings[sel_idx]:
                            sim = cosine_similarity(
                                result_embeddings[idx],
                                result_embeddings[sel_idx]
                            )
                            max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            else:
                break

        # Build diversified results
        diversified = []
        for idx in selected_indices:
            result = results[idx]
            result.metadata["mmr_applied"] = True
            result.metadata["mmr_lambda"] = lambda_param
            diversified.append(result)

        logger.debug(
            "Applied MMR diversity",
            original_count=len(results),
            diversified_count=len(diversified),
            lambda_param=lambda_param,
        )

        return diversified

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        search_type: Optional[SearchType] = None,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[SearchResult]:
        """
        Unified search interface.

        Args:
            query: Search query string
            query_embedding: Optional query embedding (required for vector/hybrid)
            search_type: Type of search to perform
            top_k: Number of results
            access_tier_level: Maximum access tier level
            document_ids: Optional document filter
            session: Optional database session
            vector_weight: Dynamic weight for vector results (0-1), overrides config
            keyword_weight: Dynamic weight for keyword results (0-1), overrides config
            organization_id: Optional organization ID for multi-tenant isolation
            user_id: Optional user ID for private document access
            is_superadmin: Whether the user is a superadmin (can see all private docs)

        Returns:
            List of SearchResult objects
        """
        search_type = search_type or self.config.search_type

        if search_type == SearchType.VECTOR:
            if not query_embedding:
                raise ValueError("query_embedding required for vector search")
            return await self.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
                session=session,
                organization_id=organization_id,
                user_id=user_id,
                is_superadmin=is_superadmin,
            )

        elif search_type == SearchType.KEYWORD:
            return await self.keyword_search(
                query=query,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
                session=session,
                organization_id=organization_id,
                user_id=user_id,
                is_superadmin=is_superadmin,
            )

        else:  # HYBRID
            if not query_embedding:
                # Fall back to keyword search if no embedding
                logger.warning("No embedding provided for hybrid search, using keyword only")
                return await self.keyword_search(
                    query=query,
                    top_k=top_k,
                    access_tier_level=access_tier_level,
                    document_ids=document_ids,
                    session=session,
                    organization_id=organization_id,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                )

            return await self.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
                session=session,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                organization_id=organization_id,
                user_id=user_id,
                is_superadmin=is_superadmin,
            )

    # =========================================================================
    # Context Expansion
    # =========================================================================

    async def expand_context(
        self,
        results: List[SearchResult],
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Expand search results with surrounding chunk context.

        For each result, fetches snippets from the previous and next chunks
        in the same document to provide better context for the LLM.

        Args:
            results: List of search results to expand
            session: Optional existing database session

        Returns:
            The same results with prev_chunk_snippet and next_chunk_snippet populated
        """
        if not self.config.enable_context_expansion or not results:
            return results

        async def _expand(db: AsyncSession) -> List[SearchResult]:
            # Group results by document for efficient querying
            doc_chunks: Dict[str, List[Tuple[int, SearchResult]]] = {}
            for result in results:
                chunk_index = result.metadata.get("chunk_index")
                if chunk_index is not None:
                    if result.document_id not in doc_chunks:
                        doc_chunks[result.document_id] = []
                    doc_chunks[result.document_id].append((chunk_index, result))
                    result.chunk_index = chunk_index

            # For each document, fetch surrounding chunks
            snippet_length = self.config.context_snippet_length

            for doc_id, chunks in doc_chunks.items():
                # Get all relevant chunk indices
                indices = set()
                for chunk_index, _ in chunks:
                    if chunk_index > 0:
                        indices.add(chunk_index - 1)  # Previous
                    indices.add(chunk_index + 1)  # Next

                if not indices:
                    continue

                # Fetch surrounding chunks in one query
                try:
                    surrounding = await db.execute(
                        select(Chunk.chunk_index, Chunk.content)
                        .where(Chunk.document_id == uuid.UUID(doc_id))
                        .where(Chunk.chunk_index.in_(list(indices)))
                    )
                    surrounding_map = {row[0]: row[1] for row in surrounding.fetchall()}

                    # Populate prev/next snippets
                    for chunk_index, result in chunks:
                        # Previous chunk snippet
                        if chunk_index > 0 and (chunk_index - 1) in surrounding_map:
                            prev_content = surrounding_map[chunk_index - 1]
                            # Take last snippet_length characters
                            if len(prev_content) > snippet_length:
                                result.prev_chunk_snippet = "..." + prev_content[-snippet_length:]
                            else:
                                result.prev_chunk_snippet = prev_content

                        # Next chunk snippet
                        if (chunk_index + 1) in surrounding_map:
                            next_content = surrounding_map[chunk_index + 1]
                            # Take first snippet_length characters
                            if len(next_content) > snippet_length:
                                result.next_chunk_snippet = next_content[:snippet_length] + "..."
                            else:
                                result.next_chunk_snippet = next_content

                except Exception as e:
                    logger.warning(
                        "Failed to fetch surrounding chunks",
                        document_id=doc_id,
                        error=str(e),
                    )

            return results

        if session:
            return await _expand(session)
        else:
            async with async_session_context() as db:
                return await _expand(db)

    # =========================================================================
    # Utility Operations
    # =========================================================================

    async def get_chunk_by_id(
        self,
        chunk_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        async def _get(db: AsyncSession) -> Optional[SearchResult]:
            result = await db.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.id == uuid.UUID(chunk_id))
            )
            row = result.first()

            if row:
                chunk, doc = row
                return SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=1.0,
                    metadata={"chunk_index": chunk.chunk_index},
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                )
            return None

        if session:
            return await _get(session)
        else:
            async with async_session_context() as db:
                return await _get(db)

    async def get_document_chunks(
        self,
        document_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """Get all chunks for a document."""
        async def _get(db: AsyncSession) -> List[SearchResult]:
            result = await db.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.document_id == uuid.UUID(document_id))
                .order_by(Chunk.chunk_index)
            )
            rows = result.all()

            results = []
            for chunk, doc in rows:
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=1.0,
                    similarity_score=1.0,  # Direct document access = 100% match
                    metadata={"chunk_index": chunk.chunk_index},
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                ))

            return results

        if session:
            return await _get(session)
        else:
            async with async_session_context() as db:
                return await _get(db)

    async def get_stats(
        self,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """Get vector store statistics."""
        async def _stats(db: AsyncSession) -> Dict[str, Any]:
            # Total chunks
            chunk_count = await db.scalar(select(func.count(Chunk.id)))

            # Chunks with embeddings
            embedded_count = await db.scalar(
                select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
            )

            # Total documents
            doc_count = await db.scalar(select(func.count(Document.id)))

            return {
                "total_chunks": chunk_count or 0,
                "embedded_chunks": embedded_count or 0,
                "total_documents": doc_count or 0,
                "embedding_coverage": (
                    (embedded_count / chunk_count * 100) if chunk_count else 0
                ),
                "has_pgvector": self._has_pgvector,
                # Phase 92: Binary quantization stats
                "binary_quantization_enabled": self.config.enable_binary_quantization,
                "binary_index_built": self._binary_index_built,
                "binary_index_size": self._binary_corpus_size,
            }

        if session:
            return await _stats(session)
        else:
            async with async_session_context() as db:
                return await _stats(db)


# =============================================================================
# Factory Function
# =============================================================================

_vector_store: Optional[VectorStore] = None


def get_vector_store(
    config: Optional[VectorStoreConfig] = None,
    backend: Optional[str] = None,
) -> VectorStore:
    """
    Get or create vector store instance.

    Supports multiple backends:
    - "pgvector" (default): PostgreSQL + pgvector (recommended for production)
    - "chroma": ChromaDB (local, no server required, good for development)
    - "auto": Auto-detect based on DATABASE_URL and configuration

    Args:
        config: Optional configuration
        backend: Backend type ("pgvector", "chroma", "auto", or None for default)

    Returns:
        VectorStore instance (either VectorStore or ChromaVectorStore)
    """
    import os
    global _vector_store

    # Determine backend
    if backend is None:
        backend = os.getenv("VECTOR_STORE_BACKEND", "auto")

    if backend == "auto":
        # Auto-detect: use ChromaDB if SQLite or explicitly configured
        database_url = os.getenv("DATABASE_URL", "")
        if "sqlite" in database_url or not HAS_PGVECTOR:
            backend = "chroma"
        else:
            backend = "pgvector"

    if backend == "chroma":
        # Use ChromaDB local vector store
        from backend.services.vectorstore_local import (
            get_chroma_vector_store,
            ChromaVectorStore,
        )
        return get_chroma_vector_store(config=config)

    # Default: PostgreSQL + pgvector
    if _vector_store is None or config is not None:
        _vector_store = VectorStore(config=config)

    return _vector_store


def get_vector_store_backend() -> str:
    """
    Get the current vector store backend type.

    Returns:
        "pgvector" or "chroma"
    """
    import os
    backend = os.getenv("VECTOR_STORE_BACKEND", "auto")

    if backend == "auto":
        database_url = os.getenv("DATABASE_URL", "")
        if "sqlite" in database_url or not HAS_PGVECTOR:
            return "chroma"
        return "pgvector"

    return backend
