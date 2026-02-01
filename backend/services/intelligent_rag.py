"""
AIDocumentIndexer - Intelligent RAG Service
============================================

Wrapper around the base RAG service that integrates:
1. Chain-of-Thought reasoning
2. Self-verification
3. Context optimization
4. Extended thinking
5. Parallel knowledge enhancement
6. Session compaction
7. Tool augmentation (calculator, code executor, fact checker)
8. Query analysis for auto-detection of needed features

This makes small LLMs perform at Claude-level quality.
The LLM intelligently decides what features to use based on query analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import asyncio
import time

import structlog

from backend.services.rag import RAGService, RAGResponse, get_rag_service
from backend.services.cot_engine import ChainOfThoughtEngine, get_cot_engine, ReasoningResult
from backend.services.self_verification import SelfVerificationService, get_verification_service, VerifiedAnswer
from backend.services.context_optimizer import ContextOptimizer, get_context_optimizer, OptimizedContext
from backend.services.extended_thinking import ExtendedThinkingService, get_extended_thinking_service, ThinkingLevel, ThinkingResult
from backend.services.parallel_knowledge import ParallelKnowledgeService, OutputMode, MergeStrategy, ParallelKnowledgeResult
from backend.services.session_compactor import SessionCompactor, get_session_compactor, Message, CompactedSession
from backend.services.tool_augmentation import ToolAugmentationService, get_tool_augmentation_service
from backend.services.query_analyzer import QueryAnalyzerService, get_query_analyzer, QueryAnalysis
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class IntelligenceLevel(str, Enum):
    """Intelligence enhancement level."""
    BASIC = "basic"         # Just RAG, no enhancements
    STANDARD = "standard"   # CoT + basic verification
    ENHANCED = "enhanced"   # CoT + verification + context optimization
    MAXIMUM = "maximum"     # All features enabled


@dataclass
class IntelligentRAGConfig:
    """Configuration for intelligent RAG."""
    intelligence_level: IntelligenceLevel = IntelligenceLevel.ENHANCED
    auto_detect_level: bool = True  # Let LLM auto-detect optimal level
    enable_cot: bool = True
    enable_verification: bool = True
    enable_context_optimization: bool = True
    enable_extended_thinking: bool = False
    thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM
    enable_parallel_knowledge: bool = False
    parallel_output_mode: OutputMode = OutputMode.MERGED
    enable_session_compaction: bool = True
    enable_tool_augmentation: bool = True  # Calculator, code executor, fact checker
    auto_detect_tools: bool = True  # Let LLM decide which tools to use
    max_verification_rounds: int = 2
    context_max_tokens: int = 4000
    # Enhanced prompting options (for small LLMs)
    use_enhanced_prompts: bool = True  # Use XML structure, few-shot, etc.
    use_xml_structure: bool = True  # XML tags for reliable parsing
    use_few_shot_examples: bool = True  # Include examples in prompts
    use_recursive_improvement: bool = False  # Multi-pass self-improvement


@dataclass
class IntelligentRAGResponse:
    """Response from intelligent RAG."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: float
    reasoning_steps: List[str] = field(default_factory=list)
    verification_status: str = ""
    corrections_made: List[str] = field(default_factory=list)
    thinking_summary: str = ""
    parallel_answers: Optional[Dict[str, str]] = None
    tool_results: Optional[Dict[str, Any]] = None  # Calculator, code executor, fact checker results
    query_analysis: Optional[Dict[str, Any]] = None  # Auto-detected query features
    intelligence_level: str = ""
    processing_time_ms: int = 0
    model_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "verification_status": self.verification_status,
            "corrections_made": self.corrections_made,
            "thinking_summary": self.thinking_summary,
            "parallel_answers": self.parallel_answers,
            "tool_results": self.tool_results,
            "query_analysis": self.query_analysis,
            "intelligence_level": self.intelligence_level,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


class IntelligentRAGService:
    """
    Intelligent RAG service that wraps the base RAG with
    Claude-level intelligence capabilities.

    Applies multiple intelligence enhancement layers:
    1. Context Optimization - Maximize relevance in limited context
    2. Chain-of-Thought - Step-by-step reasoning
    3. Extended Thinking - Deep analysis for complex queries
    4. Self-Verification - Check and correct answers
    5. Parallel Knowledge - Combine RAG + general knowledge
    6. Session Compaction - Handle long conversations
    7. Tool Augmentation - Calculator, code executor, fact checker
    """

    def __init__(
        self,
        config: Optional[IntelligentRAGConfig] = None,
        rag_service: Optional[RAGService] = None,
    ):
        """Initialize the intelligent RAG service."""
        self.config = config or IntelligentRAGConfig()
        self._rag_service = rag_service

        # Lazy-loaded services
        self._cot_engine: Optional[ChainOfThoughtEngine] = None
        self._verification_service: Optional[SelfVerificationService] = None
        self._context_optimizer: Optional[ContextOptimizer] = None
        self._extended_thinking: Optional[ExtendedThinkingService] = None
        self._parallel_knowledge: Optional[ParallelKnowledgeService] = None
        self._session_compactor: Optional[SessionCompactor] = None
        self._tool_augmentation: Optional[ToolAugmentationService] = None
        self._query_analyzer: Optional[QueryAnalyzerService] = None

    @property
    def rag_service(self) -> RAGService:
        """Get the underlying RAG service."""
        if self._rag_service is None:
            self._rag_service = get_rag_service()
        return self._rag_service

    @property
    def cot_engine(self) -> ChainOfThoughtEngine:
        """Get the CoT engine."""
        if self._cot_engine is None:
            self._cot_engine = get_cot_engine()
        return self._cot_engine

    @property
    def verification_service(self) -> SelfVerificationService:
        """Get the verification service."""
        if self._verification_service is None:
            self._verification_service = get_verification_service()
        return self._verification_service

    @property
    def context_optimizer(self) -> ContextOptimizer:
        """Get the context optimizer."""
        if self._context_optimizer is None:
            self._context_optimizer = get_context_optimizer()
        return self._context_optimizer

    @property
    def extended_thinking(self) -> ExtendedThinkingService:
        """Get the extended thinking service."""
        if self._extended_thinking is None:
            self._extended_thinking = get_extended_thinking_service()
        return self._extended_thinking

    @property
    def session_compactor(self) -> SessionCompactor:
        """Get the session compactor."""
        if self._session_compactor is None:
            self._session_compactor = get_session_compactor()
        return self._session_compactor

    @property
    def tool_augmentation(self) -> ToolAugmentationService:
        """Get the tool augmentation service."""
        if self._tool_augmentation is None:
            self._tool_augmentation = get_tool_augmentation_service()
        return self._tool_augmentation

    @property
    def query_analyzer(self) -> QueryAnalyzerService:
        """Get the query analyzer service."""
        if self._query_analyzer is None:
            self._query_analyzer = get_query_analyzer()
        return self._query_analyzer

    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
        top_k: int = 5,
        intelligence_level: Optional[IntelligenceLevel] = None,
        enable_parallel: bool = False,
        parallel_model: Optional[str] = None,
        **kwargs,
    ) -> IntelligentRAGResponse:
        """
        Query with intelligent enhancements.

        Args:
            question: User's question
            session_id: Session ID for memory
            collection_filter: Filter by collection
            access_tier: User's access tier
            user_id: User ID
            top_k: Documents to retrieve
            intelligence_level: Override intelligence level
            enable_parallel: Enable parallel RAG + non-RAG
            parallel_model: Model for parallel non-RAG query
            **kwargs: Additional args for base RAG

        Returns:
            IntelligentRAGResponse with enhanced answer
        """
        start_time = time.time()

        # Step 0: Auto-detect optimal settings if enabled
        query_analysis: Optional[QueryAnalysis] = None
        if self.config.auto_detect_level or self.config.auto_detect_tools:
            try:
                query_analysis = await self.query_analyzer.analyze(
                    query=question,
                    use_llm=True,
                )

                # Auto-adjust intelligence level based on analysis
                if self.config.auto_detect_level and intelligence_level is None:
                    level_map = {
                        "basic": IntelligenceLevel.BASIC,
                        "standard": IntelligenceLevel.STANDARD,
                        "enhanced": IntelligenceLevel.ENHANCED,
                        "maximum": IntelligenceLevel.MAXIMUM,
                    }
                    detected_level = level_map.get(
                        query_analysis.recommended_intelligence_level,
                        IntelligenceLevel.ENHANCED
                    )
                    intelligence_level = detected_level

                    # Auto-enable features based on analysis
                    if query_analysis.enable_extended_thinking:
                        self.config.enable_extended_thinking = True
                    if query_analysis.enable_parallel_knowledge:
                        enable_parallel = True

                logger.info(
                    "Query auto-analyzed",
                    complexity=query_analysis.complexity.value,
                    detected_level=query_analysis.recommended_intelligence_level,
                    recommended_tools=query_analysis.recommended_tools,
                )

            except Exception as e:
                logger.warning("Query analysis failed, using defaults", error=str(e))

        level = intelligence_level or self.config.intelligence_level

        logger.info(
            "Starting intelligent RAG query",
            intelligence_level=level.value,
            enable_parallel=enable_parallel,
            question_length=len(question),
            auto_detected=query_analysis is not None,
        )

        # Step 1: Get base RAG response
        rag_response = await self.rag_service.query(
            question=question,
            session_id=session_id,
            collection_filter=collection_filter,
            access_tier=access_tier,
            user_id=user_id,
            top_k=top_k,
            **kwargs,
        )

        # For BASIC level, just return the RAG response
        if level == IntelligenceLevel.BASIC:
            return IntelligentRAGResponse(
                answer=rag_response.content,
                sources=rag_response.sources,
                query=question,
                confidence=rag_response.confidence_score or 0.7,
                intelligence_level=level.value,
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used=rag_response.model,
            )

        # Build context from sources
        context = self._build_context_from_sources(rag_response.sources)

        # Step 2: Tool Augmentation (if enabled or auto-detected)
        tool_results: Optional[Dict[str, Any]] = None
        should_use_tools = (
            self.config.enable_tool_augmentation and
            level in [IntelligenceLevel.ENHANCED, IntelligenceLevel.MAXIMUM]
        ) or (
            query_analysis is not None and
            len(query_analysis.recommended_tools) > 0
        )

        if should_use_tools:
            try:
                # Use auto-detected tools if available
                auto_detect = self.config.auto_detect_tools
                if query_analysis and query_analysis.recommended_tools:
                    # Pass specific tools from analysis
                    logger.info(
                        "Using auto-detected tools",
                        tools=query_analysis.recommended_tools,
                    )

                augmented = await self.tool_augmentation.augment_query(
                    query=question,
                    context=context,
                    sources=rag_response.sources,
                    auto_detect_tools=True,
                )
                if augmented.get("tools_used"):
                    tool_results = augmented.get("tool_results", {})
                    # Enhance context with tool results
                    context = augmented.get("enhanced_context", context)
                    logger.info(
                        "Tool augmentation applied",
                        tools_used=augmented.get("tools_used", []),
                    )
            except Exception as e:
                logger.warning("Tool augmentation failed", error=str(e))

        # Step 3: Context Optimization (if enabled)
        if level in [IntelligenceLevel.ENHANCED, IntelligenceLevel.MAXIMUM]:
            if self.config.enable_context_optimization:
                optimized = await self.context_optimizer.optimize_context(
                    query=question,
                    documents=rag_response.sources,
                    max_tokens=self.config.context_max_tokens,
                )
                context = optimized.formatted_context

        # Step 4: Extended Thinking (if enabled)
        thinking_result: Optional[ThinkingResult] = None
        if level == IntelligenceLevel.MAXIMUM and self.config.enable_extended_thinking:
            thinking_result = await self.extended_thinking.think(
                query=question,
                context=context,
                level=self.config.thinking_level,
            )

        # Step 5: Chain-of-Thought Reasoning
        reasoning_result: Optional[ReasoningResult] = None
        if self.config.enable_cot and level != IntelligenceLevel.BASIC:
            # Use enhanced reasoning for small LLMs or when configured
            if self.config.use_enhanced_prompts:
                reasoning_result = await self.cot_engine.reason_enhanced(
                    question=question,
                    context=context,
                    use_few_shot=self.config.use_few_shot_examples,
                    use_xml=self.config.use_xml_structure,
                )
            else:
                reasoning_result = await self.cot_engine.reason(
                    question=question,
                    context=context,
                )

        # Step 6: Parallel Knowledge (if enabled)
        parallel_result: Optional[ParallelKnowledgeResult] = None
        parallel_answers: Optional[Dict[str, str]] = None
        if enable_parallel and level in [IntelligenceLevel.ENHANCED, IntelligenceLevel.MAXIMUM]:
            from backend.services.parallel_knowledge import get_parallel_knowledge_service
            parallel_service = get_parallel_knowledge_service(rag_service=self.rag_service)

            parallel_result = await parallel_service.query(
                query=question,
                general_model=parallel_model,
                output_mode=self.config.parallel_output_mode,
                merge_strategy=MergeStrategy.SYNTHESIS,
            )

            if parallel_result.rag_result and parallel_result.general_result:
                parallel_answers = {
                    "rag": parallel_result.rag_result.answer,
                    "general": parallel_result.general_result.answer,
                    "merged": parallel_result.merged_result,
                }

        # Step 7: Determine the best answer
        if parallel_result and parallel_result.merged_result:
            best_answer = parallel_result.merged_result
        elif reasoning_result:
            best_answer = reasoning_result.final_answer
        elif thinking_result:
            best_answer = thinking_result.final_answer
        else:
            best_answer = rag_response.content

        # Step 8: Self-Verification
        verification_result: Optional[VerifiedAnswer] = None
        if self.config.enable_verification and level != IntelligenceLevel.BASIC:
            # Use enhanced verification for small LLMs or when configured
            if self.config.use_enhanced_prompts:
                verification_result = await self.verification_service.verify_enhanced(
                    question=question,
                    answer=best_answer,
                    sources=rag_response.sources,
                    use_xml=self.config.use_xml_structure,
                    use_few_shot=self.config.use_few_shot_examples,
                    use_recursive=self.config.use_recursive_improvement,
                    max_correction_rounds=self.config.max_verification_rounds,
                )
            else:
                verification_result = await self.verification_service.verify(
                    question=question,
                    answer=best_answer,
                    sources=rag_response.sources,
                    max_correction_rounds=self.config.max_verification_rounds,
                )

            # Use verified/corrected answer
            if verification_result:
                best_answer = verification_result.verified_answer

        # Build response
        reasoning_steps = []
        if reasoning_result:
            reasoning_steps = reasoning_result.thinking_steps
        elif thinking_result:
            reasoning_steps = [s.content for s in thinking_result.thinking_steps]

        confidence = 0.7
        if verification_result:
            confidence = verification_result.confidence
        elif reasoning_result:
            confidence = reasoning_result.confidence
        elif thinking_result:
            confidence = thinking_result.confidence

        processing_time = int((time.time() - start_time) * 1000)

        response = IntelligentRAGResponse(
            answer=best_answer,
            sources=rag_response.sources,
            query=question,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_status=verification_result.status.value if verification_result else "",
            corrections_made=verification_result.corrections_made if verification_result else [],
            thinking_summary=thinking_result.thinking_summary if thinking_result else "",
            parallel_answers=parallel_answers,
            tool_results=tool_results,
            query_analysis=query_analysis.to_dict() if query_analysis else None,
            intelligence_level=level.value,
            processing_time_ms=processing_time,
            model_used=rag_response.model,
            metadata={
                "rag_sources_count": len(rag_response.sources),
                "cot_used": reasoning_result is not None,
                "verification_used": verification_result is not None,
                "extended_thinking_used": thinking_result is not None,
                "parallel_knowledge_used": parallel_result is not None,
                "tool_augmentation_used": tool_results is not None,
                "auto_detected": query_analysis is not None,
                "detected_complexity": query_analysis.complexity.value if query_analysis else None,
            },
        )

        logger.info(
            "Intelligent RAG query complete",
            intelligence_level=level.value,
            confidence=confidence,
            processing_time_ms=processing_time,
            verification_status=response.verification_status,
        )

        return response

    def _build_context_from_sources(
        self,
        sources: List[Dict[str, Any]],
    ) -> str:
        """Build context string from RAG sources."""
        if not sources:
            return ""

        parts = []
        for i, source in enumerate(sources[:5], 1):
            content = source.get("content", source.get("text", ""))
            title = source.get("document_name", source.get("title", f"Source {i}"))
            parts.append(f"[{title}]\n{content}")

        return "\n\n---\n\n".join(parts)

    async def compact_session(
        self,
        messages: List[Dict[str, str]],
        target_tokens: int = 4000,
    ) -> CompactedSession:
        """
        Compact a conversation session.

        Args:
            messages: List of messages with 'role' and 'content'
            target_tokens: Target token count

        Returns:
            CompactedSession with summary and recent messages
        """
        # Convert to Message objects
        msg_objects = [
            Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )
            for m in messages
        ]

        return await self.session_compactor.compact(
            messages=msg_objects,
            target_tokens=target_tokens,
        )


# Singleton instance
_intelligent_rag: Optional[IntelligentRAGService] = None


def get_intelligent_rag_service(
    config: Optional[IntelligentRAGConfig] = None,
) -> IntelligentRAGService:
    """Get or create the intelligent RAG service."""
    global _intelligent_rag
    if _intelligent_rag is None:
        _intelligent_rag = IntelligentRAGService(config=config)
    return _intelligent_rag
