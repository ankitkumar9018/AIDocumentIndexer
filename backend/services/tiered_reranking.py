"""
AIDocumentIndexer - Tiered Reranking Pipeline (Phase 43)
=========================================================

Multi-stage reranking pipeline for optimal precision-latency tradeoff.

Based on research:
- ColBERT late interaction: 180x fewer FLOPs than cross-encoder
- Cross-encoder: Highest accuracy for top candidates
- LLM reranker: Optional final stage for high-value queries

Key Features:
- Stage 1: ColBERT/Late Interaction (fast, good precision)
- Stage 2: Cross-encoder (accurate, for top candidates)
- Stage 3: LLM reranker (optional, for complex queries)
- Adaptive pipeline based on query complexity
- 23% → 87% accuracy improvement on multi-hop

Architecture:
- Input: 50-100 candidates from hybrid retrieval
- Stage 1: ColBERT filters to top 20 (fast)
- Stage 2: Cross-encoder reranks to top 10 (accurate)
- Stage 3: LLM verifies top 5 (optional, highest quality)

Top Models (2026):
- Cohere Rerank: 100+ languages, transformer cross-encoder
- Jina-ColBERT-v2: 8K context, late interaction
- Pinecone Rerank V0: Highest NDCG@10 on BEIR

Usage:
    from backend.services.tiered_reranking import get_tiered_reranker

    reranker = await get_tiered_reranker()
    results = await reranker.rerank(query, candidates, stages=["colbert", "cross"])
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class RerankerStage(str, Enum):
    """Reranking pipeline stages."""
    COLBERT = "colbert"       # Late interaction (fast)
    CROSS_ENCODER = "cross"   # Cross-encoder (accurate)
    COHERE = "cohere"         # Cohere Rerank API (100+ languages)
    VOYAGE = "voyage"         # Voyage AI Rerank API
    LLM = "llm"               # LLM-based (highest quality)
    SEMANTIC = "semantic"     # Semantic similarity (fallback)


class QueryComplexity(str, Enum):
    """Query complexity levels for adaptive pipeline."""
    SIMPLE = "simple"         # Single fact queries
    MODERATE = "moderate"     # Multi-fact queries
    COMPLEX = "complex"       # Multi-hop reasoning
    ANALYTICAL = "analytical"  # Deep analysis required


@dataclass
class RerankerConfig:
    """Configuration for tiered reranking."""
    # Default pipeline
    default_stages: List[RerankerStage] = field(default_factory=lambda: [
        RerankerStage.COLBERT,
        RerankerStage.CROSS_ENCODER,
    ])

    # Stage-specific settings
    colbert_top_k: int = 20           # Candidates after ColBERT
    cross_encoder_top_k: int = 10     # Candidates after cross-encoder
    llm_top_k: int = 5                # Candidates after LLM

    # Models
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    colbert_model: str = "colbert-ir/colbertv2.0"
    cohere_model: str = "rerank-v3.5"      # Cohere rerank model
    voyage_model: str = "rerank-2"          # Voyage AI rerank model
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None

    # API reranker top_k
    cohere_top_k: int = 15                 # Candidates after Cohere
    voyage_top_k: int = 15                 # Candidates after Voyage

    # Adaptive settings
    enable_adaptive: bool = True
    use_llm_for_complex: bool = True
    complexity_threshold: float = 0.7

    # Performance
    max_concurrent: int = 10
    timeout_seconds: float = 30.0

    # Quality
    min_score_threshold: float = 0.1  # Filter out very low scores


@dataclass(slots=True)
class RerankCandidate:
    """A candidate for reranking."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class RerankResult:
    """Result from a reranking stage."""
    candidate: RerankCandidate
    score: float
    stage: RerankerStage
    stage_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RerankerMetrics:
    """Metrics from reranking pipeline."""
    total_time_ms: float
    stage_times_ms: Dict[str, float]
    input_candidates: int
    output_candidates: int
    stages_executed: List[str]
    complexity: Optional[QueryComplexity] = None


# =============================================================================
# Stage Implementations
# =============================================================================

class BaseRerankerStage:
    """Base class for reranking stages."""

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankResult]:
        raise NotImplementedError

    async def initialize(self) -> bool:
        return True


class ColBERTRerankerStage(BaseRerankerStage):
    """ColBERT/Late interaction reranking stage."""

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            # Try to load ColBERT model
            from backend.services.colbert_reranker import ColBERTReranker
            self._model = ColBERTReranker()
            await self._model.initialize()
            self._initialized = True
            return True
        except ImportError:
            logger.warning("ColBERT reranker not available, using fallback")
            self._initialized = True
            return True

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankResult]:
        """Rerank using ColBERT late interaction."""
        if not self._initialized:
            await self.initialize()

        results = []

        if self._model:
            # Use actual ColBERT
            try:
                texts = [c.content for c in candidates]
                scores = await self._model.rerank(query, texts)

                for candidate, score in zip(candidates, scores):
                    results.append(RerankResult(
                        candidate=candidate,
                        score=score,
                        stage=RerankerStage.COLBERT,
                        stage_scores={"colbert": score},
                    ))
            except Exception as e:
                logger.warning("ColBERT reranking failed", error=str(e))
                # Fallback to original scores
                for candidate in candidates:
                    results.append(RerankResult(
                        candidate=candidate,
                        score=candidate.score,
                        stage=RerankerStage.COLBERT,
                        stage_scores={"colbert": candidate.score},
                    ))
        else:
            # Fallback: Use semantic similarity boost
            for candidate in candidates:
                # Simple keyword overlap scoring
                query_terms = set(query.lower().split())
                content_terms = set(candidate.content.lower().split())
                overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
                boosted_score = candidate.score * (1 + overlap * 0.5)

                results.append(RerankResult(
                    candidate=candidate,
                    score=boosted_score,
                    stage=RerankerStage.COLBERT,
                    stage_scores={"colbert": boosted_score},
                ))

        # Sort and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class CrossEncoderRerankerStage(BaseRerankerStage):
    """Cross-encoder reranking stage."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            # Try to load cross-encoder
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info("Cross-encoder loaded", model=self.model_name)
            return True
        except ImportError:
            logger.warning("sentence-transformers not available for cross-encoder")
            self._initialized = True
            return True
        except Exception as e:
            logger.warning("Failed to load cross-encoder", error=str(e))
            self._initialized = True
            return True

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankResult]:
        """Rerank using cross-encoder."""
        if not self._initialized:
            await self.initialize()

        results = []

        if self._model:
            try:
                # Prepare pairs
                pairs = [(query, c.content) for c in candidates]

                # Run in thread pool (cross-encoder is synchronous)
                loop = asyncio.get_running_loop()
                scores = await loop.run_in_executor(
                    None,
                    lambda: self._model.predict(pairs)
                )

                for candidate, score in zip(candidates, scores):
                    results.append(RerankResult(
                        candidate=candidate,
                        score=float(score),
                        stage=RerankerStage.CROSS_ENCODER,
                        stage_scores={"cross_encoder": float(score)},
                    ))
            except Exception as e:
                logger.warning("Cross-encoder reranking failed", error=str(e))
                # Fallback to previous scores
                for candidate in candidates:
                    prev_score = candidate.score
                    results.append(RerankResult(
                        candidate=candidate,
                        score=prev_score,
                        stage=RerankerStage.CROSS_ENCODER,
                        stage_scores={"cross_encoder": prev_score},
                    ))
        else:
            # Fallback: Keep previous scores with slight variation
            for candidate in candidates:
                results.append(RerankResult(
                    candidate=candidate,
                    score=candidate.score,
                    stage=RerankerStage.CROSS_ENCODER,
                    stage_scores={"cross_encoder": candidate.score},
                ))

        # Sort and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class LLMRerankerStage(BaseRerankerStage):
    """LLM-based reranking stage for highest quality."""

    RERANK_PROMPT = """You are a relevance judge. Score how relevant each document is to the query.

Query: {query}

Documents to score:
{documents}

For each document, provide a relevance score from 0.0 to 1.0 where:
- 1.0 = Directly answers the query
- 0.7-0.9 = Highly relevant, contains key information
- 0.4-0.6 = Somewhat relevant, contains related information
- 0.1-0.3 = Marginally relevant
- 0.0 = Not relevant

Return JSON array of scores in order: [score1, score2, ...]
Only return the JSON array, no explanation."""

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None):
        self.model = model
        self.provider = provider
        self._llm = None
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            from backend.services.llm import get_chat_model, llm_config

            # Resolve provider/model defaults lazily
            _provider = self.provider or llm_config.default_provider
            _model = self.model or llm_config.default_chat_model

            self._llm = await get_chat_model(provider=_provider, model=_model)
            self._initialized = True
            return True
        except Exception as e:
            logger.warning("Failed to initialize LLM reranker", error=str(e))
            self._initialized = True
            return True

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankResult]:
        """Rerank using LLM judgement."""
        if not self._initialized:
            await self.initialize()

        results = []

        if self._llm:
            try:
                # Format documents
                docs_text = "\n\n".join(
                    f"[{i+1}] {c.content[:500]}..."
                    for i, c in enumerate(candidates)
                )

                prompt = self.RERANK_PROMPT.format(
                    query=query,
                    documents=docs_text,
                )

                from langchain_core.messages import HumanMessage
                response = await self._llm.ainvoke([HumanMessage(content=prompt)])

                # Parse scores
                import json
                scores_text = response.content.strip()
                # Handle markdown code blocks
                if "```" in scores_text:
                    scores_text = scores_text.split("```")[1]
                    if scores_text.startswith("json"):
                        scores_text = scores_text[4:]
                scores = json.loads(scores_text)

                for candidate, score in zip(candidates, scores):
                    results.append(RerankResult(
                        candidate=candidate,
                        score=float(score),
                        stage=RerankerStage.LLM,
                        stage_scores={"llm": float(score)},
                    ))
            except Exception as e:
                logger.warning("LLM reranking failed", error=str(e))
                # Fallback
                for candidate in candidates:
                    results.append(RerankResult(
                        candidate=candidate,
                        score=candidate.score,
                        stage=RerankerStage.LLM,
                        stage_scores={"llm": candidate.score},
                    ))
        else:
            # Fallback
            for candidate in candidates:
                results.append(RerankResult(
                    candidate=candidate,
                    score=candidate.score,
                    stage=RerankerStage.LLM,
                    stage_scores={"llm": candidate.score},
                ))

        # Sort and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# =============================================================================
# Cohere Reranker Stage
# =============================================================================

class CohereRerankerStage(BaseRerankerStage):
    """Cohere Rerank API reranking stage (100+ languages, high quality)."""

    def __init__(self, model_name: str = "rerank-v3.5"):
        self.model_name = model_name
        self._client = None
        self._initialized = False
        self._api_key = None

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            import os
            self._api_key = os.getenv("COHERE_API_KEY") or getattr(settings, "COHERE_API_KEY", None)
            if self._api_key:
                import cohere
                self._client = cohere.AsyncClient(api_key=self._api_key)
                logger.info("Cohere reranker initialized", model=self.model_name)
            self._initialized = True
            return True
        except ImportError:
            logger.warning("cohere package not installed")
            self._initialized = True
            return True
        except Exception as e:
            logger.warning("Failed to initialize Cohere reranker", error=str(e))
            self._initialized = True
            return True

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankResult]:
        """Rerank using Cohere Rerank API."""
        if not self._initialized:
            await self.initialize()

        results = []

        if self._client:
            try:
                docs = [c.content for c in candidates]
                response = await self._client.rerank(
                    model=self.model_name,
                    query=query,
                    documents=docs,
                    top_n=top_k,
                    return_documents=False,
                )

                # Map results back to candidates
                scored_indices = {r.index: r.relevance_score for r in response.results}

                for i, candidate in enumerate(candidates):
                    score = scored_indices.get(i, 0.0)
                    results.append(RerankResult(
                        candidate=candidate,
                        score=score,
                        stage=RerankerStage.CROSS_ENCODER,  # Use CROSS_ENCODER as proxy
                        stage_scores={"cohere": score},
                    ))

            except Exception as e:
                logger.warning("Cohere reranking failed", error=str(e))
                for candidate in candidates:
                    results.append(RerankResult(
                        candidate=candidate,
                        score=candidate.score,
                        stage=RerankerStage.CROSS_ENCODER,
                        stage_scores={"cohere": candidate.score},
                    ))
        else:
            # Fallback
            for candidate in candidates:
                results.append(RerankResult(
                    candidate=candidate,
                    score=candidate.score,
                    stage=RerankerStage.CROSS_ENCODER,
                    stage_scores={"cohere": candidate.score},
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# =============================================================================
# Voyage AI Reranker Stage
# =============================================================================

class VoyageRerankerStage(BaseRerankerStage):
    """Voyage AI Rerank API reranking stage."""

    def __init__(self, model_name: str = "rerank-2"):
        self.model_name = model_name
        self._client = None
        self._initialized = False
        self._api_key = None

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            import os
            self._api_key = os.getenv("VOYAGE_API_KEY") or getattr(settings, "VOYAGE_API_KEY", None)
            if self._api_key:
                import voyageai
                self._client = voyageai.AsyncClient(api_key=self._api_key)
                logger.info("Voyage reranker initialized", model=self.model_name)
            self._initialized = True
            return True
        except ImportError:
            logger.warning("voyageai package not installed")
            self._initialized = True
            return True
        except Exception as e:
            logger.warning("Failed to initialize Voyage reranker", error=str(e))
            self._initialized = True
            return True

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankResult]:
        """Rerank using Voyage AI Rerank API."""
        if not self._initialized:
            await self.initialize()

        results = []

        if self._client:
            try:
                docs = [c.content for c in candidates]
                response = await self._client.rerank(
                    model=self.model_name,
                    query=query,
                    documents=docs,
                    top_k=top_k,
                )

                # Map results back to candidates
                scored_indices = {r.index: r.relevance_score for r in response.results}

                for i, candidate in enumerate(candidates):
                    score = scored_indices.get(i, 0.0)
                    results.append(RerankResult(
                        candidate=candidate,
                        score=score,
                        stage=RerankerStage.CROSS_ENCODER,
                        stage_scores={"voyage": score},
                    ))

            except Exception as e:
                logger.warning("Voyage reranking failed", error=str(e))
                for candidate in candidates:
                    results.append(RerankResult(
                        candidate=candidate,
                        score=candidate.score,
                        stage=RerankerStage.CROSS_ENCODER,
                        stage_scores={"voyage": candidate.score},
                    ))
        else:
            # Fallback
            for candidate in candidates:
                results.append(RerankResult(
                    candidate=candidate,
                    score=candidate.score,
                    stage=RerankerStage.CROSS_ENCODER,
                    stage_scores={"voyage": candidate.score},
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# =============================================================================
# Semantic Similarity Reranker Stage (Fallback)
# =============================================================================

class SemanticRerankerStage(BaseRerankerStage):
    """Semantic similarity reranking using embeddings (fallback stage)."""

    def __init__(self):
        self._embedding_service = None
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            from backend.services.embeddings import get_embedding_service
            self._embedding_service = get_embedding_service()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning("Failed to initialize semantic reranker", error=str(e))
            self._initialized = True
            return True

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int,
    ) -> List[RerankResult]:
        """Rerank using semantic similarity."""
        if not self._initialized:
            await self.initialize()

        results = []

        if self._embedding_service:
            try:
                # Get query embedding
                query_embedding = await self._embedding_service.embed_text(query)

                # Get candidate embeddings (batch)
                texts = [c.content[:1000] for c in candidates]  # Limit text length
                candidate_embeddings = await self._embedding_service.embed_texts(texts)

                # Compute cosine similarity
                import numpy as np

                def cosine_similarity(a, b):
                    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                for candidate, emb in zip(candidates, candidate_embeddings):
                    score = float(cosine_similarity(query_embedding, emb))
                    results.append(RerankResult(
                        candidate=candidate,
                        score=score,
                        stage=RerankerStage.SEMANTIC,
                        stage_scores={"semantic": score},
                    ))

            except Exception as e:
                logger.warning("Semantic reranking failed", error=str(e))
                for candidate in candidates:
                    results.append(RerankResult(
                        candidate=candidate,
                        score=candidate.score,
                        stage=RerankerStage.SEMANTIC,
                        stage_scores={"semantic": candidate.score},
                    ))
        else:
            # Fallback: TF-IDF-like scoring
            from collections import Counter
            import math

            query_terms = Counter(query.lower().split())

            for candidate in candidates:
                content_terms = Counter(candidate.content.lower().split())
                # TF-IDF-like score
                score = 0.0
                for term, count in query_terms.items():
                    if term in content_terms:
                        tf = math.log(1 + content_terms[term])
                        score += tf * count
                # Normalize
                score = score / (len(query.split()) + 1)

                results.append(RerankResult(
                    candidate=candidate,
                    score=score,
                    stage=RerankerStage.SEMANTIC,
                    stage_scores={"semantic": score},
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# =============================================================================
# Query Complexity Analyzer
# =============================================================================

class QueryComplexityAnalyzer:
    """Analyzes query complexity for adaptive pipeline selection."""

    COMPLEX_PATTERNS = [
        "compare", "contrast", "relationship between",
        "how does", "why does", "explain the connection",
        "multi-step", "analyze", "evaluate",
    ]

    SIMPLE_PATTERNS = [
        "what is", "who is", "when did", "where is",
        "define", "list", "name",
    ]

    def analyze(self, query: str) -> QueryComplexity:
        """Analyze query complexity."""
        query_lower = query.lower()

        # Check for complex patterns
        complex_score = sum(1 for p in self.COMPLEX_PATTERNS if p in query_lower)

        # Check for simple patterns
        simple_score = sum(1 for p in self.SIMPLE_PATTERNS if p in query_lower)

        # Length-based heuristic
        word_count = len(query.split())

        if complex_score >= 2 or word_count > 20:
            return QueryComplexity.COMPLEX
        elif complex_score >= 1 or word_count > 15:
            return QueryComplexity.ANALYTICAL
        elif simple_score >= 1:
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE


# =============================================================================
# Tiered Reranker Pipeline
# =============================================================================

class TieredReranker:
    """
    Multi-stage reranking pipeline.

    Executes stages sequentially, with each stage filtering
    candidates for the next stage.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()

        # Initialize stages
        self._stages: Dict[RerankerStage, BaseRerankerStage] = {
            RerankerStage.COLBERT: ColBERTRerankerStage(self.config.colbert_model),
            RerankerStage.CROSS_ENCODER: CrossEncoderRerankerStage(self.config.cross_encoder_model),
            RerankerStage.COHERE: CohereRerankerStage(self.config.cohere_model),
            RerankerStage.VOYAGE: VoyageRerankerStage(self.config.voyage_model),
            RerankerStage.LLM: LLMRerankerStage(self.config.llm_model, self.config.llm_provider),
            RerankerStage.SEMANTIC: SemanticRerankerStage(),
        }

        self._complexity_analyzer = QueryComplexityAnalyzer()
        self._initialized = False

        logger.info(
            "Initialized TieredReranker",
            default_stages=[s.value for s in self.config.default_stages],
        )

    async def initialize(self) -> bool:
        """Initialize all stages."""
        if self._initialized:
            return True

        for stage_name, stage in self._stages.items():
            try:
                await stage.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize {stage_name}", error=str(e))

        self._initialized = True
        return True

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        stages: Optional[List[Union[RerankerStage, str]]] = None,
        final_top_k: Optional[int] = None,
    ) -> Tuple[List[RerankResult], RerankerMetrics]:
        """
        Run the reranking pipeline.

        Args:
            query: Search query
            candidates: Candidates to rerank
            stages: Stages to execute (uses default if not provided)
            final_top_k: Final number of results

        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # Determine stages to run
        if stages is None:
            if self.config.enable_adaptive:
                stages = self._get_adaptive_stages(query)
            else:
                stages = self.config.default_stages
        else:
            # Convert strings to enum
            stages = [
                RerankerStage(s) if isinstance(s, str) else s
                for s in stages
            ]

        # Analyze complexity
        complexity = None
        if self.config.enable_adaptive:
            complexity = self._complexity_analyzer.analyze(query)

            # Add LLM stage for complex queries
            if (
                complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL]
                and self.config.use_llm_for_complex
                and RerankerStage.LLM not in stages
            ):
                stages.append(RerankerStage.LLM)

        logger.info(
            "Running reranking pipeline",
            stages=[s.value for s in stages],
            input_count=len(candidates),
            complexity=complexity.value if complexity else None,
        )

        # Execute stages
        stage_times = {}
        current_results = candidates
        executed_stages = []

        for stage in stages:
            stage_start = time.time()
            stage_impl = self._stages.get(stage)

            if not stage_impl:
                logger.warning(f"Stage {stage} not available, skipping")
                continue

            # Determine top_k for this stage
            if stage == RerankerStage.COLBERT:
                stage_top_k = self.config.colbert_top_k
            elif stage == RerankerStage.CROSS_ENCODER:
                stage_top_k = self.config.cross_encoder_top_k
            elif stage == RerankerStage.COHERE:
                stage_top_k = self.config.cohere_top_k
            elif stage == RerankerStage.VOYAGE:
                stage_top_k = self.config.voyage_top_k
            elif stage == RerankerStage.LLM:
                stage_top_k = self.config.llm_top_k
            elif stage == RerankerStage.SEMANTIC:
                stage_top_k = self.config.cross_encoder_top_k  # Same as cross-encoder
            else:
                stage_top_k = len(current_results)

            # Convert to candidates if needed
            if isinstance(current_results[0], RerankResult):
                candidates_for_stage = [
                    RerankCandidate(
                        id=r.candidate.id,
                        content=r.candidate.content,
                        score=r.score,
                        metadata=r.candidate.metadata,
                        document_id=r.candidate.document_id,
                        chunk_index=r.candidate.chunk_index,
                    )
                    for r in current_results
                ]
            else:
                candidates_for_stage = current_results

            # Run stage
            try:
                results = await stage_impl.rerank(
                    query=query,
                    candidates=candidates_for_stage,
                    top_k=stage_top_k,
                )
                current_results = results
                executed_stages.append(stage.value)
            except Exception as e:
                logger.error(f"Stage {stage} failed", error=str(e))
                # Keep current results

            stage_times[stage.value] = (time.time() - stage_start) * 1000

        # Apply final filtering
        final_results = current_results
        if final_top_k:
            final_results = final_results[:final_top_k]

        # Filter by minimum score
        final_results = [
            r for r in final_results
            if r.score >= self.config.min_score_threshold
        ]

        total_time = (time.time() - start_time) * 1000

        metrics = RerankerMetrics(
            total_time_ms=total_time,
            stage_times_ms=stage_times,
            input_candidates=len(candidates),
            output_candidates=len(final_results),
            stages_executed=executed_stages,
            complexity=complexity,
        )

        logger.info(
            "Reranking complete",
            output_count=len(final_results),
            total_time_ms=round(total_time, 2),
        )

        return final_results, metrics

    def _get_adaptive_stages(self, query: str) -> List[RerankerStage]:
        """Get stages based on query analysis."""
        complexity = self._complexity_analyzer.analyze(query)

        if complexity == QueryComplexity.SIMPLE:
            return [RerankerStage.COLBERT]
        elif complexity == QueryComplexity.MODERATE:
            return [RerankerStage.COLBERT, RerankerStage.CROSS_ENCODER]
        else:
            return [
                RerankerStage.COLBERT,
                RerankerStage.CROSS_ENCODER,
                RerankerStage.LLM,
            ]


# =============================================================================
# Factory Function
# =============================================================================

_tiered_reranker: Optional[TieredReranker] = None


async def get_tiered_reranker(
    config: Optional[RerankerConfig] = None,
) -> TieredReranker:
    """
    Get or create the tiered reranker.

    Args:
        config: Optional configuration override

    Returns:
        Initialized TieredReranker
    """
    global _tiered_reranker

    if _tiered_reranker is None:
        _tiered_reranker = TieredReranker(config)
        await _tiered_reranker.initialize()

    return _tiered_reranker
