"""
AIDocumentIndexer - Specialized Sub-Agents
==========================================

Phase 23B: Implements specialized agents for Agentic RAG pipeline:

1. PlannerAgent - Query decomposition and strategy planning
2. RetrieverAgent - Hybrid search with reranking
3. ValidatorAgent - Quality check and hallucination filtering
4. GeneratorAgent - Response synthesis with memory

Architecture:
    User Query → [Planner] → [Retriever] → [Validator] → [Generator] → Response
                    ↓             ↓              ↓
               Decomposition  Hybrid Search   Quality Check
               Tool Selection   Reranking     Hallucination Filter

Based on:
- Agentic RAG patterns (2025 research)
- ReAct loop (Reason → Act → Observe)
- Self-RAG and CRAG architectures
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import uuid

import structlog

from backend.services.agent_memory import AgentMemory, MemoryType

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class AgentRole(str, Enum):
    PLANNER = "planner"
    RETRIEVER = "retriever"
    VALIDATOR = "validator"
    GENERATOR = "generator"
    COORDINATOR = "coordinator"


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    id: str
    query: str
    intent: str  # "factual", "comparative", "procedural", "exploratory"
    priority: int = 1  # Higher = more important
    dependencies: List[str] = field(default_factory=list)  # IDs of queries this depends on
    answered: bool = False
    answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "intent": self.intent,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "answered": self.answered,
            "answer": self.answer,
        }


@dataclass
class RetrievalResult:
    """Result from retrieval agent."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    source: str  # "dense", "sparse", "colbert", "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from validation agent."""
    is_valid: bool
    confidence: float  # 0-1
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    filtered_content: Optional[str] = None


@dataclass
class AgentAction:
    """An action taken by an agent."""
    agent: AgentRole
    action_type: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentState:
    """State passed between agents."""
    query: str
    sub_queries: List[SubQuery] = field(default_factory=list)
    retrieved_docs: List[RetrievalResult] = field(default_factory=list)
    validated_docs: List[RetrievalResult] = field(default_factory=list)
    intermediate_answers: Dict[str, str] = field(default_factory=dict)
    final_answer: Optional[str] = None
    actions: List[AgentAction] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 5
    confidence: float = 0.0


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """Base class for specialized agents."""

    def __init__(
        self,
        role: AgentRole,
        llm_service=None,
        memory: Optional[AgentMemory] = None,
    ):
        self.role = role
        self.llm = llm_service
        self.memory = memory
        self._action_history: List[AgentAction] = []

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute agent logic and update state."""
        pass

    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """Call LLM with prompt."""
        if not self.llm:
            raise ValueError(f"{self.role} agent requires LLM service")

        try:
            response = await self.llm.generate(prompt, **kwargs)
            return response
        except Exception as e:
            logger.error(f"{self.role} LLM call failed", error=str(e))
            raise

    def _record_action(
        self,
        action_type: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> AgentAction:
        """Record an action for tracing."""
        action = AgentAction(
            agent=self.role,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self._action_history.append(action)
        return action


# =============================================================================
# Planner Agent
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    Decomposes complex queries into sub-queries and plans retrieval strategy.

    Responsibilities:
    - Analyze query complexity
    - Break down into atomic sub-questions
    - Identify query intent (factual, comparative, etc.)
    - Select appropriate retrieval tools
    - Order sub-queries by dependencies
    """

    DECOMPOSITION_PROMPT = """You are a query planning expert. Analyze the following query and break it down into simpler sub-questions that can be answered independently.

Query: {query}

Instructions:
1. If the query is simple and can be answered directly, return it as a single sub-query
2. For complex queries, decompose into 2-5 atomic sub-questions
3. Identify the intent of each sub-query: factual, comparative, procedural, or exploratory
4. Note any dependencies between sub-queries

Return your analysis as JSON:
{{
    "complexity": "simple" | "moderate" | "complex",
    "sub_queries": [
        {{
            "query": "the sub-question",
            "intent": "factual" | "comparative" | "procedural" | "exploratory",
            "priority": 1-5,
            "depends_on": []  // indices of queries this depends on
        }}
    ],
    "strategy": "description of retrieval strategy"
}}"""

    def __init__(self, llm_service=None, memory: Optional[AgentMemory] = None):
        super().__init__(AgentRole.PLANNER, llm_service, memory)

    async def execute(self, state: AgentState) -> AgentState:
        """Decompose query and plan retrieval strategy."""
        start_time = datetime.utcnow()

        try:
            # Get decomposition from LLM
            prompt = self.DECOMPOSITION_PROMPT.format(query=state.query)
            response = await self._call_llm(prompt)

            # Parse response
            plan = self._parse_plan(response)

            # Create sub-queries
            state.sub_queries = []
            for i, sq in enumerate(plan.get("sub_queries", [])):
                sub_query = SubQuery(
                    id=f"sq_{i}",
                    query=sq["query"],
                    intent=sq.get("intent", "factual"),
                    priority=sq.get("priority", 1),
                    dependencies=[f"sq_{d}" for d in sq.get("depends_on", [])],
                )
                state.sub_queries.append(sub_query)

            # If no sub-queries, use original query
            if not state.sub_queries:
                state.sub_queries = [
                    SubQuery(
                        id="sq_0",
                        query=state.query,
                        intent="factual",
                        priority=1,
                    )
                ]

            # Sort by priority and dependencies
            state.sub_queries.sort(
                key=lambda x: (len(x.dependencies), -x.priority)
            )

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            action = self._record_action(
                "decompose",
                {"query": state.query},
                {
                    "sub_queries": [sq.to_dict() for sq in state.sub_queries],
                    "complexity": plan.get("complexity", "unknown"),
                },
                duration_ms=duration,
            )
            state.actions.append(action)

            logger.info(
                "Query decomposed",
                query=state.query[:50],
                sub_query_count=len(state.sub_queries),
            )

        except Exception as e:
            logger.error("Planner failed", error=str(e))
            # Fallback to original query
            state.sub_queries = [
                SubQuery(id="sq_0", query=state.query, intent="factual")
            ]
            self._record_action(
                "decompose",
                {"query": state.query},
                error=str(e),
                success=False,
            )

        return state

    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into plan dict."""
        # Try to extract JSON
        try:
            # Look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: simple parsing
        return {
            "complexity": "simple",
            "sub_queries": [{"query": response.strip(), "intent": "factual"}],
        }


# =============================================================================
# Retriever Agent
# =============================================================================

class RetrieverAgent(BaseAgent):
    """
    Performs hybrid retrieval with multiple strategies.

    Responsibilities:
    - Dense vector search
    - Sparse BM25 search
    - ColBERT late interaction (if available)
    - Fusion and reranking
    - Knowledge graph augmentation
    """

    def __init__(
        self,
        llm_service=None,
        memory: Optional[AgentMemory] = None,
        rag_service=None,
        kg_service=None,
        reranker=None,
    ):
        super().__init__(AgentRole.RETRIEVER, llm_service, memory)
        self.rag = rag_service
        self.kg = kg_service
        self.reranker = reranker

    async def execute(self, state: AgentState) -> AgentState:
        """Retrieve documents for all sub-queries."""
        start_time = datetime.utcnow()

        all_results = []

        for sub_query in state.sub_queries:
            if sub_query.answered:
                continue

            # Check dependencies
            deps_satisfied = all(
                state.intermediate_answers.get(dep)
                for dep in sub_query.dependencies
            )
            if not deps_satisfied:
                continue

            # Perform retrieval
            results = await self._retrieve_for_query(sub_query.query, state)
            all_results.extend(results)

        # Deduplicate by chunk_id
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                unique_results.append(r)

        # Rerank if available
        if self.reranker and unique_results:
            unique_results = await self._rerank(state.query, unique_results)

        state.retrieved_docs = unique_results[:20]  # Keep top 20

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        action = self._record_action(
            "retrieve",
            {"sub_queries": [sq.query for sq in state.sub_queries]},
            {"doc_count": len(state.retrieved_docs)},
            duration_ms=duration,
        )
        state.actions.append(action)

        logger.info(
            "Documents retrieved",
            query=state.query[:50],
            doc_count=len(state.retrieved_docs),
        )

        return state

    async def _retrieve_for_query(
        self,
        query: str,
        state: AgentState,
    ) -> List[RetrievalResult]:
        """Retrieve documents for a single query."""
        results = []

        # Use RAG service if available
        if self.rag:
            try:
                rag_results = await self.rag.retrieve(
                    query=query,
                    top_k=10,
                    use_hybrid=True,
                )

                for r in rag_results:
                    results.append(RetrievalResult(
                        chunk_id=r.get("chunk_id", str(uuid.uuid4())),
                        document_id=r.get("document_id", ""),
                        content=r.get("content", r.get("text", "")),
                        score=r.get("score", 0.5),
                        source=r.get("source", "hybrid"),
                        metadata=r.get("metadata", {}),
                    ))
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Augment with knowledge graph
        if self.kg:
            try:
                kg_results = await self.kg.query_related(query, limit=5)
                for entity in kg_results:
                    results.append(RetrievalResult(
                        chunk_id=f"kg_{entity.get('id', '')}",
                        document_id="knowledge_graph",
                        content=entity.get("description", str(entity)),
                        score=entity.get("relevance", 0.3),
                        source="knowledge_graph",
                        metadata={"entity": entity},
                    ))
            except Exception as e:
                logger.debug(f"KG augmentation failed: {e}")

        return results

    async def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder."""
        try:
            # Prepare for reranking
            passages = [r.content for r in results]

            # Call reranker
            scores = await self.reranker.rerank(query, passages)

            # Update scores
            for result, score in zip(results, scores):
                result.score = score

            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")

        return results


# =============================================================================
# Validator Agent
# =============================================================================

class ValidatorAgent(BaseAgent):
    """
    Validates retrieved content and filters hallucinations.

    Responsibilities:
    - Check relevance to query
    - Verify factual consistency
    - Filter contradictory information
    - Assess source quality
    - Detect potential hallucinations
    """

    VALIDATION_PROMPT = """You are a content validator. Analyze if the following content is relevant and accurate for answering the query.

Query: {query}

Content to validate:
{content}

Evaluate:
1. Relevance: Does this content help answer the query? (0-1)
2. Consistency: Is the information internally consistent? (0-1)
3. Quality: Is this from a reliable source with clear information? (0-1)

Return JSON:
{{
    "is_valid": true/false,
    "relevance": 0.0-1.0,
    "consistency": 0.0-1.0,
    "quality": 0.0-1.0,
    "issues": ["list of any issues found"],
    "key_facts": ["list of key facts extracted"]
}}"""

    def __init__(
        self,
        llm_service=None,
        memory: Optional[AgentMemory] = None,
        sufficiency_service=None,
    ):
        super().__init__(AgentRole.VALIDATOR, llm_service, memory)
        self.sufficiency = sufficiency_service

    async def execute(self, state: AgentState) -> AgentState:
        """Validate retrieved documents."""
        start_time = datetime.utcnow()

        validated = []
        issues_found = []

        # Check overall sufficiency first
        if self.sufficiency:
            try:
                context = "\n\n".join([r.content for r in state.retrieved_docs[:5]])
                is_sufficient = await self.sufficiency.check(
                    query=state.query,
                    context=context,
                )
                if not is_sufficient.get("sufficient", True):
                    issues_found.append("Context may be insufficient to answer query")
                    state.confidence = is_sufficient.get("confidence", 0.5)
            except Exception as e:
                logger.debug(f"Sufficiency check failed: {e}")

        # Validate each document
        for doc in state.retrieved_docs:
            validation = await self._validate_document(doc, state.query)

            if validation.is_valid:
                doc.score = doc.score * validation.confidence
                validated.append(doc)
            else:
                issues_found.extend(validation.issues)

        # Sort validated docs by adjusted score
        validated.sort(key=lambda x: x.score, reverse=True)
        state.validated_docs = validated

        # Update confidence based on validation
        if validated:
            avg_score = sum(d.score for d in validated) / len(validated)
            state.confidence = max(state.confidence, avg_score)

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        action = self._record_action(
            "validate",
            {"doc_count": len(state.retrieved_docs)},
            {
                "validated_count": len(validated),
                "issues": issues_found[:5],
                "confidence": state.confidence,
            },
            duration_ms=duration,
        )
        state.actions.append(action)

        logger.info(
            "Documents validated",
            original=len(state.retrieved_docs),
            validated=len(validated),
            confidence=state.confidence,
        )

        return state

    async def _validate_document(
        self,
        doc: RetrievalResult,
        query: str,
    ) -> ValidationResult:
        """Validate a single document."""
        # Quick heuristic checks
        if len(doc.content.strip()) < 20:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["Content too short"],
            )

        # LLM-based validation for important docs
        if self.llm and doc.score > 0.5:
            try:
                prompt = self.VALIDATION_PROMPT.format(
                    query=query,
                    content=doc.content[:1000],
                )
                response = await self._call_llm(prompt)
                result = self._parse_validation(response)
                return result
            except Exception as e:
                logger.debug(f"LLM validation failed: {e}")

        # Default: accept with moderate confidence
        return ValidationResult(
            is_valid=True,
            confidence=doc.score * 0.8,
        )

    def _parse_validation(self, response: str) -> ValidationResult:
        """Parse LLM validation response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                # Calculate overall confidence
                relevance = data.get("relevance", 0.5)
                consistency = data.get("consistency", 0.5)
                quality = data.get("quality", 0.5)
                confidence = (relevance + consistency + quality) / 3

                return ValidationResult(
                    is_valid=data.get("is_valid", confidence > 0.4),
                    confidence=confidence,
                    issues=data.get("issues", []),
                    suggestions=data.get("key_facts", []),
                )
        except Exception:
            pass

        return ValidationResult(is_valid=True, confidence=0.5)


# =============================================================================
# Generator Agent
# =============================================================================

class GeneratorAgent(BaseAgent):
    """
    Synthesizes final response from validated content.

    Responsibilities:
    - Combine information from multiple sources
    - Maintain coherent narrative
    - Include citations
    - Use memory for personalization
    - Self-verify generated content
    """

    GENERATION_PROMPT = """Based on the following context and conversation, answer the user's question.

{memory_context}

Context from documents:
{context}

User question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Include specific details and cite sources when possible
4. Be concise but comprehensive
5. If there are conflicting pieces of information, acknowledge them

Answer:"""

    SELF_VERIFY_PROMPT = """Review the following answer for accuracy and completeness.

Question: {query}
Context: {context}
Generated Answer: {answer}

Check:
1. Is every claim supported by the context?
2. Are there any hallucinations (information not in context)?
3. Is anything important missing?

If issues found, provide a corrected answer. If the answer is good, confirm it.

Response:"""

    def __init__(
        self,
        llm_service=None,
        memory: Optional[AgentMemory] = None,
        use_self_verify: bool = True,
    ):
        super().__init__(AgentRole.GENERATOR, llm_service, memory)
        self.use_self_verify = use_self_verify

    async def execute(self, state: AgentState) -> AgentState:
        """Generate final answer."""
        start_time = datetime.utcnow()

        try:
            # Build context from validated docs
            context = self._build_context(state.validated_docs)

            # Get memory context if available
            memory_context = ""
            if self.memory:
                mem_ctx = await self.memory.get_context_for_prompt(max_tokens=1000)
                memory_context = mem_ctx.get("system_prefix", "")

            # Generate initial answer
            prompt = self.GENERATION_PROMPT.format(
                memory_context=memory_context,
                context=context,
                query=state.query,
            )

            answer = await self._call_llm(prompt)

            # Self-verification
            if self.use_self_verify and state.confidence < 0.8:
                answer = await self._self_verify(state.query, context, answer)

            state.final_answer = answer
            state.confidence = min(state.confidence + 0.1, 1.0)

            # Update memory with interaction
            if self.memory:
                await self.memory.add_message("user", state.query)
                await self.memory.add_message("assistant", answer)

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            action = self._record_action(
                "generate",
                {"query": state.query, "context_docs": len(state.validated_docs)},
                {"answer_length": len(answer), "confidence": state.confidence},
                duration_ms=duration,
            )
            state.actions.append(action)

            logger.info(
                "Answer generated",
                query=state.query[:50],
                answer_length=len(answer),
                confidence=state.confidence,
            )

        except Exception as e:
            logger.error("Generation failed", error=str(e))
            state.final_answer = f"I apologize, but I encountered an error while generating the response: {str(e)}"
            self._record_action(
                "generate",
                {"query": state.query},
                error=str(e),
                success=False,
            )

        return state

    def _build_context(self, docs: List[RetrievalResult]) -> str:
        """Build context string from validated documents."""
        if not docs:
            return "No relevant context available."

        parts = []
        for i, doc in enumerate(docs[:10], 1):
            source = doc.metadata.get("source", doc.document_id or "Unknown")
            parts.append(f"[{i}] (Source: {source})\n{doc.content}")

        return "\n\n".join(parts)

    async def _self_verify(self, query: str, context: str, answer: str) -> str:
        """Self-verify and potentially correct the answer."""
        try:
            prompt = self.SELF_VERIFY_PROMPT.format(
                query=query,
                context=context[:2000],
                answer=answer,
            )

            verification = await self._call_llm(prompt)

            # Check if verification suggests corrections
            lower_verification = verification.lower()
            if any(word in lower_verification for word in ["corrected", "revised", "fixed"]):
                # Extract corrected answer
                if "corrected answer:" in lower_verification:
                    return verification.split("corrected answer:", 1)[1].strip()
                return verification

            return answer

        except Exception as e:
            logger.warning(f"Self-verification failed: {e}")
            return answer


# =============================================================================
# Agent Coordinator
# =============================================================================

class AgentCoordinator:
    """
    Coordinates the multi-agent RAG pipeline.

    Orchestrates: Planner → Retriever → Validator → Generator
    with iterative refinement if needed.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        retriever: RetrieverAgent,
        validator: ValidatorAgent,
        generator: GeneratorAgent,
        max_iterations: int = 3,
    ):
        self.planner = planner
        self.retriever = retriever
        self.validator = validator
        self.generator = generator
        self.max_iterations = max_iterations

    async def process(
        self,
        query: str,
        memory: Optional[AgentMemory] = None,
    ) -> AgentState:
        """
        Process a query through the multi-agent pipeline.

        Args:
            query: User query
            memory: Optional agent memory

        Returns:
            Final agent state with answer
        """
        state = AgentState(
            query=query,
            max_iterations=self.max_iterations,
        )

        logger.info("Starting multi-agent processing", query=query[:50])

        # Set memory on agents
        for agent in [self.planner, self.retriever, self.validator, self.generator]:
            if memory:
                agent.memory = memory

        # 1. Planning
        state = await self.planner.execute(state)

        # 2. Iterative retrieval and validation
        while state.iteration < state.max_iterations:
            state.iteration += 1

            # Retrieve
            state = await self.retriever.execute(state)

            # Validate
            state = await self.validator.execute(state)

            # Check if we have enough validated content
            if state.validated_docs and state.confidence >= 0.6:
                break

            # If not enough docs, refine search
            if not state.validated_docs:
                logger.info("No validated docs, refining search", iteration=state.iteration)
                # Could add query expansion here
                continue

        # 3. Generation
        state = await self.generator.execute(state)

        logger.info(
            "Multi-agent processing complete",
            query=query[:50],
            iterations=state.iteration,
            confidence=state.confidence,
            actions=len(state.actions),
        )

        return state


# =============================================================================
# Factory Functions
# =============================================================================

async def create_agent_pipeline(
    llm_service=None,
    rag_service=None,
    kg_service=None,
    reranker=None,
    sufficiency_service=None,
    memory: Optional[AgentMemory] = None,
) -> AgentCoordinator:
    """
    Create a complete agent pipeline.

    Args:
        llm_service: LLM for agents
        rag_service: RAG retrieval service
        kg_service: Knowledge graph service
        reranker: Reranking service
        sufficiency_service: Sufficiency detection service
        memory: Agent memory

    Returns:
        Configured AgentCoordinator
    """
    planner = PlannerAgent(llm_service=llm_service, memory=memory)
    retriever = RetrieverAgent(
        llm_service=llm_service,
        memory=memory,
        rag_service=rag_service,
        kg_service=kg_service,
        reranker=reranker,
    )
    validator = ValidatorAgent(
        llm_service=llm_service,
        memory=memory,
        sufficiency_service=sufficiency_service,
    )
    generator = GeneratorAgent(llm_service=llm_service, memory=memory)

    return AgentCoordinator(
        planner=planner,
        retriever=retriever,
        validator=validator,
        generator=generator,
    )
