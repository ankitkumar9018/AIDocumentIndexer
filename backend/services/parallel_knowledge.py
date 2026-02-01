"""
AIDocumentIndexer - Parallel Knowledge Enhancement
===================================================

Runs RAG and non-RAG queries in parallel, allowing users to:
1. See both outputs separately
2. Merge outputs via an agent
3. Choose display mode at runtime

This enhances answers with both document knowledge AND general LLM knowledge.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import asyncio
import time

import structlog

from backend.services.llm import LLMFactory
from backend.services.rag import RAGService
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class OutputMode(str, Enum):
    """How to display the parallel outputs."""
    SEPARATE = "separate"       # Show both outputs side by side
    MERGED = "merged"           # Merge via synthesis agent
    RAG_ONLY = "rag_only"       # Only show RAG output
    GENERAL_ONLY = "general"    # Only show general LLM output
    TOGGLE = "toggle"           # User can toggle between views


class MergeStrategy(str, Enum):
    """How to merge the two outputs."""
    SYNTHESIS = "synthesis"     # LLM synthesizes both into one
    WEIGHTED = "weighted"       # Combine based on confidence
    RAG_PRIMARY = "rag_primary" # RAG as main, general supplements
    GENERAL_PRIMARY = "general_primary"  # General as main, RAG validates


@dataclass
class KnowledgeSource:
    """A single knowledge source result."""
    source_type: str  # "rag" or "general"
    answer: str
    confidence: float
    model_used: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    thinking_trace: str = ""
    latency_ms: int = 0


@dataclass
class ParallelKnowledgeResult:
    """Result from parallel knowledge enhancement."""
    query: str
    rag_result: Optional[KnowledgeSource] = None
    general_result: Optional[KnowledgeSource] = None
    merged_result: Optional[str] = None
    output_mode: OutputMode = OutputMode.SEPARATE
    merge_strategy: Optional[MergeStrategy] = None
    total_latency_ms: int = 0
    models_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rag_result": {
                "answer": self.rag_result.answer,
                "confidence": self.rag_result.confidence,
                "model": self.rag_result.model_used,
                "sources_count": len(self.rag_result.sources),
                "latency_ms": self.rag_result.latency_ms,
            } if self.rag_result else None,
            "general_result": {
                "answer": self.general_result.answer,
                "confidence": self.general_result.confidence,
                "model": self.general_result.model_used,
                "latency_ms": self.general_result.latency_ms,
            } if self.general_result else None,
            "merged_result": self.merged_result,
            "output_mode": self.output_mode.value,
            "merge_strategy": self.merge_strategy.value if self.merge_strategy else None,
            "total_latency_ms": self.total_latency_ms,
            "models_used": self.models_used,
        }

    def get_display_answer(self) -> str:
        """Get the answer based on output mode."""
        if self.output_mode == OutputMode.MERGED and self.merged_result:
            return self.merged_result
        elif self.output_mode == OutputMode.RAG_ONLY and self.rag_result:
            return self.rag_result.answer
        elif self.output_mode == OutputMode.GENERAL_ONLY and self.general_result:
            return self.general_result.answer
        elif self.output_mode == OutputMode.SEPARATE:
            # Return structured format for UI to display
            parts = []
            if self.rag_result:
                parts.append(f"**📚 Document-Based Answer:**\n{self.rag_result.answer}")
            if self.general_result:
                parts.append(f"**🌐 General Knowledge:**\n{self.general_result.answer}")
            return "\n\n---\n\n".join(parts)
        else:
            return self.merged_result or (self.rag_result.answer if self.rag_result else "")


class ParallelKnowledgeService:
    """
    Run RAG and non-RAG queries in parallel for enhanced answers.

    This service:
    1. Sends the query to RAG pipeline (document-grounded)
    2. Sends the query to a general LLM (without document context)
    3. Optionally merges the results via a synthesis agent
    4. Allows user to choose how to display the results
    """

    MERGE_PROMPT = """You are an expert at synthesizing information from multiple sources.

**Original Question:** {query}

**Answer 1 - Based on Documents (RAG):**
{rag_answer}
Sources used: {sources_summary}
Confidence: {rag_confidence}%

**Answer 2 - General Knowledge (No Documents):**
{general_answer}
Confidence: {general_confidence}%

**Your Task:**
Create a single, comprehensive answer that:
1. Prioritizes information from documents (Answer 1) when available
2. Supplements with general knowledge (Answer 2) for context and completeness
3. Clearly indicates which parts come from documents vs. general knowledge
4. Notes any conflicts between the two sources
5. Provides a balanced, well-rounded response

**Merged Answer:**"""

    VALIDATION_PROMPT = """Compare these two answers to the same question:

**Question:** {query}

**Document-Based Answer:**
{rag_answer}

**General Knowledge Answer:**
{general_answer}

Analyze:
1. Do they agree or conflict?
2. Which claims are well-supported?
3. What unique value does each provide?

Response (JSON):
{{
    "agreement_level": "agree|partial|conflict",
    "conflicts": ["list of conflicting points"],
    "rag_unique_value": "what documents add",
    "general_unique_value": "what general knowledge adds",
    "recommended_merge": "how to best combine"
}}"""

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
    ):
        """Initialize the parallel knowledge service."""
        self.rag_service = rag_service

    async def query(
        self,
        query: str,
        rag_provider: str = None,
        rag_model: str = None,
        general_provider: str = None,
        general_model: str = None,
        output_mode: OutputMode = OutputMode.SEPARATE,
        merge_strategy: MergeStrategy = MergeStrategy.SYNTHESIS,
        rag_top_k: int = 5,
    ) -> ParallelKnowledgeResult:
        """
        Execute parallel RAG and non-RAG queries.

        Args:
            query: The user's question
            rag_provider: Provider for RAG query (uses default if None)
            rag_model: Model for RAG query
            general_provider: Provider for general query (can be different)
            general_model: Model for general query
            output_mode: How to display results
            merge_strategy: How to merge if merging
            rag_top_k: Number of documents to retrieve

        Returns:
            ParallelKnowledgeResult with both answers and optional merge
        """
        start_time = time.time()

        logger.info(
            "Starting parallel knowledge query",
            query_length=len(query),
            output_mode=output_mode.value,
            rag_model=rag_model,
            general_model=general_model,
        )

        # Run both queries in parallel
        rag_task = self._run_rag_query(
            query, rag_provider, rag_model, rag_top_k
        )
        general_task = self._run_general_query(
            query, general_provider, general_model
        )

        rag_result, general_result = await asyncio.gather(
            rag_task, general_task,
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(rag_result, Exception):
            logger.error("RAG query failed", error=str(rag_result))
            rag_result = None
        if isinstance(general_result, Exception):
            logger.error("General query failed", error=str(general_result))
            general_result = None

        # Collect models used
        models_used = []
        if rag_result:
            models_used.append(rag_result.model_used)
        if general_result:
            models_used.append(general_result.model_used)

        # Create result
        result = ParallelKnowledgeResult(
            query=query,
            rag_result=rag_result,
            general_result=general_result,
            output_mode=output_mode,
            merge_strategy=merge_strategy if output_mode == OutputMode.MERGED else None,
            models_used=models_used,
        )

        # Merge if requested
        if output_mode == OutputMode.MERGED and rag_result and general_result:
            result.merged_result = await self._merge_results(
                query, rag_result, general_result, merge_strategy
            )

        result.total_latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Parallel knowledge query complete",
            output_mode=output_mode.value,
            has_rag=rag_result is not None,
            has_general=general_result is not None,
            has_merged=result.merged_result is not None,
            latency_ms=result.total_latency_ms,
        )

        return result

    async def _run_rag_query(
        self,
        query: str,
        provider: str,
        model: str,
        top_k: int,
    ) -> KnowledgeSource:
        """Run the RAG-enhanced query."""
        start_time = time.time()

        provider = provider or settings.DEFAULT_LLM_PROVIDER
        model = model or settings.DEFAULT_CHAT_MODEL

        try:
            # Get RAG service
            if self.rag_service:
                rag_result = await self.rag_service.query(
                    query=query,
                    top_k=top_k,
                    provider=provider,
                    model=model,
                )

                answer = rag_result.get("answer", "")
                sources = rag_result.get("sources", [])
                confidence = rag_result.get("confidence", 0.7)
            else:
                # Fallback: just use LLM without RAG
                answer = "RAG service not available"
                sources = []
                confidence = 0.5

            latency = int((time.time() - start_time) * 1000)

            return KnowledgeSource(
                source_type="rag",
                answer=answer,
                confidence=confidence,
                model_used=f"{provider}/{model}",
                sources=sources,
                latency_ms=latency,
            )

        except Exception as e:
            logger.error("RAG query error", error=str(e))
            raise

    async def _run_general_query(
        self,
        query: str,
        provider: str,
        model: str,
    ) -> KnowledgeSource:
        """Run the general (non-RAG) query."""
        start_time = time.time()

        provider = provider or settings.DEFAULT_LLM_PROVIDER
        model = model or settings.DEFAULT_CHAT_MODEL

        try:
            llm = LLMFactory.get_chat_model(
                provider=provider,
                model=model,
                temperature=0.3,
                max_tokens=2048,
            )

            # Prompt for general knowledge answer
            prompt = f"""Answer this question using your general knowledge.
Do not assume access to any specific documents or databases.

Question: {query}

Provide a comprehensive answer based on your training knowledge.
If you're uncertain about something, indicate your confidence level.

Answer:"""

            response = await llm.ainvoke(prompt)
            answer = response.content

            latency = int((time.time() - start_time) * 1000)

            # Estimate confidence based on answer hedging
            confidence = self._estimate_confidence(answer)

            return KnowledgeSource(
                source_type="general",
                answer=answer,
                confidence=confidence,
                model_used=f"{provider}/{model}",
                sources=[],
                latency_ms=latency,
            )

        except Exception as e:
            logger.error("General query error", error=str(e))
            raise

    def _estimate_confidence(self, answer: str) -> float:
        """Estimate confidence based on language hedging."""
        answer_lower = answer.lower()

        # Words that indicate uncertainty
        uncertain_words = [
            "might", "maybe", "perhaps", "possibly", "could be",
            "not sure", "uncertain", "i think", "it seems",
            "approximately", "roughly", "about",
        ]

        # Words that indicate confidence
        confident_words = [
            "definitely", "certainly", "clearly", "obviously",
            "is", "are", "was", "were", "will",
        ]

        uncertain_count = sum(1 for word in uncertain_words if word in answer_lower)
        confident_count = sum(1 for word in confident_words if word in answer_lower)

        # Base confidence
        confidence = 0.7

        # Adjust based on hedging
        confidence -= uncertain_count * 0.05
        confidence += confident_count * 0.02

        return max(0.3, min(0.95, confidence))

    async def _merge_results(
        self,
        query: str,
        rag_result: KnowledgeSource,
        general_result: KnowledgeSource,
        strategy: MergeStrategy,
    ) -> str:
        """Merge the RAG and general results."""
        logger.info("Merging parallel results", strategy=strategy.value)

        if strategy == MergeStrategy.RAG_PRIMARY:
            # RAG as main, supplement with general
            return self._merge_rag_primary(rag_result, general_result)

        elif strategy == MergeStrategy.GENERAL_PRIMARY:
            # General as main, validate with RAG
            return self._merge_general_primary(rag_result, general_result)

        elif strategy == MergeStrategy.WEIGHTED:
            # Weight by confidence
            if rag_result.confidence >= general_result.confidence:
                return rag_result.answer
            else:
                return general_result.answer

        else:  # SYNTHESIS
            # Full synthesis via LLM
            return await self._synthesize_merge(query, rag_result, general_result)

    def _merge_rag_primary(
        self,
        rag_result: KnowledgeSource,
        general_result: KnowledgeSource,
    ) -> str:
        """Merge with RAG as primary source."""
        main_answer = rag_result.answer

        # Add supplementary info if general has something different
        if len(general_result.answer) > 100:
            supplement = f"\n\n**Additional Context (General Knowledge):**\n{general_result.answer[:500]}..."
            return main_answer + supplement

        return main_answer

    def _merge_general_primary(
        self,
        rag_result: KnowledgeSource,
        general_result: KnowledgeSource,
    ) -> str:
        """Merge with general as primary, RAG validates."""
        main_answer = general_result.answer

        # Add document validation
        if rag_result.sources:
            sources_summary = ", ".join([
                s.get("document_name", "doc")
                for s in rag_result.sources[:3]
            ])
            validation = f"\n\n**📚 Verified by Documents:**\n{rag_result.answer[:300]}...\n_Sources: {sources_summary}_"
            return main_answer + validation

        return main_answer

    async def _synthesize_merge(
        self,
        query: str,
        rag_result: KnowledgeSource,
        general_result: KnowledgeSource,
    ) -> str:
        """Full synthesis merge via LLM."""
        sources_summary = ", ".join([
            s.get("document_name", "doc")[:30]
            for s in rag_result.sources[:5]
        ]) or "No specific sources"

        prompt = self.MERGE_PROMPT.format(
            query=query,
            rag_answer=rag_result.answer,
            sources_summary=sources_summary,
            rag_confidence=int(rag_result.confidence * 100),
            general_answer=general_result.answer,
            general_confidence=int(general_result.confidence * 100),
        )

        try:
            llm = LLMFactory.get_chat_model(
                provider=settings.DEFAULT_LLM_PROVIDER,
                model=settings.DEFAULT_CHAT_MODEL,
                temperature=0.3,
                max_tokens=2048,
            )

            response = await llm.ainvoke(prompt)
            return response.content

        except Exception as e:
            logger.error("Synthesis merge failed", error=str(e))
            # Fallback to simple concatenation
            return self._merge_rag_primary(rag_result, general_result)

    async def validate_consistency(
        self,
        query: str,
        rag_result: KnowledgeSource,
        general_result: KnowledgeSource,
    ) -> Dict[str, Any]:
        """Validate consistency between RAG and general answers."""
        prompt = self.VALIDATION_PROMPT.format(
            query=query,
            rag_answer=rag_result.answer,
            general_answer=general_result.answer,
        )

        try:
            llm = LLMFactory.get_chat_model(
                provider=settings.DEFAULT_LLM_PROVIDER,
                model=settings.DEFAULT_CHAT_MODEL,
                temperature=0.2,
                max_tokens=1024,
            )

            response = await llm.ainvoke(prompt)

            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.error("Validation failed", error=str(e))

        return {
            "agreement_level": "unknown",
            "conflicts": [],
            "rag_unique_value": "",
            "general_unique_value": "",
        }


# Factory function
def get_parallel_knowledge_service(
    rag_service: Optional[RAGService] = None,
) -> ParallelKnowledgeService:
    """Create a parallel knowledge service."""
    return ParallelKnowledgeService(rag_service=rag_service)
