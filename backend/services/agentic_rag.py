"""
AIDocumentIndexer - Agentic RAG Service
=======================================

Implements Agentic RAG for complex multi-step queries using:
- Query decomposition into sub-questions
- ReAct loop (Reason → Act → Observe → Iterate)
- Dynamic retrieval based on intermediate results
- Self-verification and correction

Features:
- Handles complex queries requiring multiple retrieval steps
- Breaks down questions into atomic sub-queries
- Iteratively refines search based on findings
- Synthesizes final answer from multiple sources
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
import uuid

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class AgentAction(str, Enum):
    """Actions the agent can take."""
    SEARCH = "search"           # Vector search in documents
    GRAPH_SEARCH = "graph"      # Knowledge graph search
    LOOKUP = "lookup"           # Look up specific document
    CALCULATE = "calculate"     # Perform calculation
    COMPARE = "compare"         # Compare multiple items
    SUMMARIZE = "summarize"     # Summarize findings
    ANSWER = "answer"           # Provide final answer
    CLARIFY = "clarify"         # Ask for clarification


@dataclass
class SubQuery:
    """A decomposed sub-question."""
    query: str
    purpose: str  # Why this sub-query is needed
    depends_on: List[int] = field(default_factory=list)  # Indices of dependencies
    priority: int = 0
    completed: bool = False
    result: Optional[str] = None


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    step_number: int
    thought: str           # Agent's reasoning
    action: AgentAction    # Action to take
    action_input: str      # Input for the action
    observation: Optional[str] = None  # Result of action
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgenticRAGResult:
    """Result from agentic RAG processing."""
    query: str
    final_answer: str
    sub_queries: List[SubQuery]
    react_steps: List[ReActStep]
    sources_used: List[Dict[str, Any]]
    confidence: float
    processing_time_ms: float
    iterations: int


# =============================================================================
# Prompts
# =============================================================================

QUERY_DECOMPOSITION_PROMPT = """Analyze the following complex question and break it down into simpler sub-questions that can be answered independently.

Original Question: {query}

Consider:
1. What information is needed to fully answer this question?
2. Are there multiple aspects or entities being asked about?
3. Does answering require comparing or combining information?
4. Are there temporal or conditional dependencies?

Return JSON in this exact format:
{{
  "is_complex": true/false,
  "sub_queries": [
    {{
      "query": "specific sub-question",
      "purpose": "why this is needed",
      "depends_on": [],
      "priority": 1
    }}
  ],
  "synthesis_approach": "how to combine sub-answers into final answer"
}}

If the question is simple and doesn't need decomposition, set is_complex to false and return empty sub_queries."""


REACT_PROMPT = """You are an intelligent document assistant using the ReAct framework.
You have access to a document archive and must answer questions by thinking step-by-step.

Available Actions:
- search(query): Search documents for information
- graph(query): Search knowledge graph for entity relationships
- lookup(doc_id): Look up a specific document
- calculate(expression): Perform a calculation
- compare(items): Compare multiple items
- summarize(text): Summarize information
- answer(response): Provide the final answer

Current Question: {query}

Previous Steps:
{previous_steps}

Sub-Questions to Address:
{sub_queries}

Current Knowledge:
{current_knowledge}

Based on your analysis, what should you do next?

Think step by step:
1. What do I know so far?
2. What do I still need to find out?
3. What action will help me most?

Respond in this format:
Thought: [your reasoning]
Action: [action_name]
Action Input: [input for the action]

Or if you have enough information:
Thought: [your reasoning]
Action: answer
Action Input: [your final answer]"""


SYNTHESIS_PROMPT = """Synthesize the following information into a comprehensive answer.

Original Question: {query}

Sub-Questions and Answers:
{sub_answers}

Knowledge Graph Context:
{graph_context}

Retrieved Information:
{retrieved_context}

Provide a well-structured answer that:
1. Directly addresses the original question
2. Integrates information from all sources
3. Notes any uncertainties or gaps
4. Cites sources where appropriate

Answer:"""


# =============================================================================
# Agentic RAG Service
# =============================================================================

class AgenticRAGService:
    """
    Agentic RAG service for complex query handling.

    Uses a ReAct loop to iteratively:
    1. Reason about what information is needed
    2. Take actions to retrieve information
    3. Observe results and update understanding
    4. Repeat until answer is found or max iterations
    """

    def __init__(
        self,
        rag_service,
        knowledge_graph_service=None,
        llm_service=None,
        max_iterations: int = 5,
        max_sub_queries: int = 5,
    ):
        self.rag = rag_service
        self.graph = knowledge_graph_service
        self.llm = llm_service
        self.max_iterations = max_iterations
        self.max_sub_queries = max_sub_queries

    async def _get_llm(self):
        """Get LLM for generation."""
        if self.llm:
            return self.llm
        # Fallback to RAG service's LLM
        return await self.rag.get_llm_for_session(None, None)

    # -------------------------------------------------------------------------
    # Query Decomposition
    # -------------------------------------------------------------------------

    async def decompose_query(self, query: str) -> Tuple[bool, List[SubQuery], str]:
        """
        Decompose a complex query into sub-questions.

        Args:
            query: Original user query

        Returns:
            Tuple of (is_complex, sub_queries, synthesis_approach)
        """
        llm, _ = await self._get_llm()

        prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return False, [], "direct"

            data = json.loads(json_match.group())

            is_complex = data.get("is_complex", False)
            synthesis = data.get("synthesis_approach", "direct")

            sub_queries = []
            for sq in data.get("sub_queries", [])[:self.max_sub_queries]:
                sub_queries.append(SubQuery(
                    query=sq.get("query", ""),
                    purpose=sq.get("purpose", ""),
                    depends_on=sq.get("depends_on", []),
                    priority=sq.get("priority", 0),
                ))

            logger.info(
                "Query decomposition complete",
                is_complex=is_complex,
                sub_query_count=len(sub_queries),
            )

            return is_complex, sub_queries, synthesis

        except Exception as e:
            logger.error("Query decomposition failed", error=str(e))
            return False, [], "direct"

    # -------------------------------------------------------------------------
    # ReAct Loop
    # -------------------------------------------------------------------------

    async def run_react_loop(
        self,
        query: str,
        sub_queries: List[SubQuery],
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
    ) -> Tuple[List[ReActStep], str, List[Dict[str, Any]]]:
        """
        Run the ReAct loop to iteratively find an answer.

        Args:
            query: Original query
            sub_queries: Decomposed sub-queries
            collection_filter: Collection to search
            access_tier: User's access tier

        Returns:
            Tuple of (steps, final_answer, sources)
        """
        steps: List[ReActStep] = []
        knowledge: Dict[str, str] = {}
        sources: List[Dict[str, Any]] = []
        final_answer = ""

        for iteration in range(self.max_iterations):
            # Build context from previous steps
            previous_steps_text = self._format_previous_steps(steps)
            sub_queries_text = self._format_sub_queries(sub_queries, knowledge)
            knowledge_text = self._format_knowledge(knowledge)

            # Get next action from LLM
            llm, _ = await self._get_llm()
            prompt = REACT_PROMPT.format(
                query=query,
                previous_steps=previous_steps_text or "None yet.",
                sub_queries=sub_queries_text or "No specific sub-questions.",
                current_knowledge=knowledge_text or "No information gathered yet.",
            )

            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            thought, action, action_input = self._parse_react_response(content)

            step = ReActStep(
                step_number=len(steps) + 1,
                thought=thought,
                action=action,
                action_input=action_input,
            )

            # Execute action
            observation, action_sources = await self._execute_action(
                action,
                action_input,
                collection_filter=collection_filter,
                access_tier=access_tier,
            )

            step.observation = observation
            steps.append(step)
            sources.extend(action_sources)

            # Update knowledge
            if observation and action != AgentAction.ANSWER:
                knowledge[f"step_{len(steps)}"] = observation

            # Mark relevant sub-queries as completed
            self._update_sub_query_completion(sub_queries, action_input, observation)

            # Check if we have final answer
            if action == AgentAction.ANSWER:
                final_answer = action_input
                break

            logger.debug(
                "ReAct step completed",
                iteration=iteration,
                action=action.value,
                has_observation=bool(observation),
            )

        # If no answer yet, synthesize from gathered knowledge
        if not final_answer and knowledge:
            final_answer = await self._synthesize_answer(
                query, sub_queries, knowledge, sources
            )

        return steps, final_answer, sources

    def _format_previous_steps(self, steps: List[ReActStep]) -> str:
        """Format previous steps for prompt."""
        if not steps:
            return ""

        lines = []
        for step in steps[-5:]:  # Last 5 steps
            lines.append(f"Step {step.step_number}:")
            lines.append(f"  Thought: {step.thought}")
            lines.append(f"  Action: {step.action.value}({step.action_input[:100]}...)")
            if step.observation:
                lines.append(f"  Observation: {step.observation[:200]}...")
        return "\n".join(lines)

    def _format_sub_queries(
        self,
        sub_queries: List[SubQuery],
        knowledge: Dict[str, str],
    ) -> str:
        """Format sub-queries with completion status."""
        if not sub_queries:
            return ""

        lines = []
        for i, sq in enumerate(sub_queries):
            status = "✓" if sq.completed else "○"
            lines.append(f"{status} {i+1}. {sq.query}")
            if sq.result:
                lines.append(f"   Answer: {sq.result[:100]}...")
        return "\n".join(lines)

    def _format_knowledge(self, knowledge: Dict[str, str]) -> str:
        """Format gathered knowledge."""
        if not knowledge:
            return ""

        lines = []
        for key, value in list(knowledge.items())[-5:]:
            lines.append(f"- {value[:200]}...")
        return "\n".join(lines)

    def _parse_react_response(
        self,
        content: str,
    ) -> Tuple[str, AgentAction, str]:
        """Parse LLM response into thought, action, and input."""
        thought = ""
        action = AgentAction.SEARCH
        action_input = ""

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', content, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r'Action:\s*(\w+)', content)
        if action_match:
            action_str = action_match.group(1).lower()
            try:
                action = AgentAction(action_str)
            except ValueError:
                action = AgentAction.SEARCH

        # Extract action input
        input_match = re.search(r'Action Input:\s*(.+?)(?=$)', content, re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()

        return thought, action, action_input

    async def _execute_action(
        self,
        action: AgentAction,
        action_input: str,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Execute an action and return observation.

        Args:
            action: Action to take
            action_input: Input for action
            collection_filter: Collection filter
            access_tier: Access tier

        Returns:
            Tuple of (observation, sources)
        """
        sources = []

        try:
            if action == AgentAction.SEARCH:
                # Use RAG service to search
                result = await self.rag.query(
                    question=action_input,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                )

                observation = result.content
                sources = [
                    {
                        "document_id": str(s.document_id),
                        "document_name": s.document_name,
                        "snippet": s.snippet,
                        "relevance_score": s.relevance_score,
                    }
                    for s in result.sources
                ]

            elif action == AgentAction.GRAPH_SEARCH:
                # Use knowledge graph if available
                if self.graph:
                    context = await self.graph.graph_search(action_input)
                    observation = context.graph_summary
                else:
                    observation = "Knowledge graph not available."

            elif action == AgentAction.SUMMARIZE:
                # Summarize provided text
                llm, _ = await self._get_llm()
                prompt = f"Summarize the following in 2-3 sentences:\n\n{action_input}"
                response = await llm.ainvoke(prompt)
                observation = response.content if hasattr(response, 'content') else str(response)

            elif action == AgentAction.COMPARE:
                # Compare items
                llm, _ = await self._get_llm()
                prompt = f"Compare and contrast the following:\n\n{action_input}"
                response = await llm.ainvoke(prompt)
                observation = response.content if hasattr(response, 'content') else str(response)

            elif action == AgentAction.ANSWER:
                # Final answer - just return the input
                observation = action_input

            else:
                observation = f"Action {action.value} not implemented."

        except Exception as e:
            logger.error("Action execution failed", action=action.value, error=str(e))
            observation = f"Error executing action: {str(e)}"

        return observation, sources

    def _update_sub_query_completion(
        self,
        sub_queries: List[SubQuery],
        action_input: str,
        observation: Optional[str],
    ):
        """Mark sub-queries as completed based on action results."""
        if not observation:
            return

        action_lower = action_input.lower()
        for sq in sub_queries:
            if not sq.completed:
                # Simple heuristic: if action_input contains key words from sub-query
                query_words = set(sq.query.lower().split())
                input_words = set(action_lower.split())
                overlap = query_words & input_words

                if len(overlap) >= 2:  # At least 2 words match
                    sq.completed = True
                    sq.result = observation[:500]

    async def _synthesize_answer(
        self,
        query: str,
        sub_queries: List[SubQuery],
        knowledge: Dict[str, str],
        sources: List[Dict[str, Any]],
    ) -> str:
        """Synthesize final answer from gathered information."""
        llm, _ = await self._get_llm()

        # Format sub-answers
        sub_answers = []
        for sq in sub_queries:
            if sq.result:
                sub_answers.append(f"Q: {sq.query}\nA: {sq.result}")

        # Format retrieved context
        retrieved_context = "\n".join(
            f"- {s.get('document_name', 'Unknown')}: {s.get('snippet', '')[:200]}..."
            for s in sources[:10]
        )

        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            sub_answers="\n\n".join(sub_answers) or "No sub-questions answered.",
            graph_context="",  # TODO: Add graph context
            retrieved_context=retrieved_context or "No additional context.",
        )

        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    async def process_query(
        self,
        query: str,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
    ) -> AgenticRAGResult:
        """
        Process a query using agentic RAG.

        Args:
            query: User query
            collection_filter: Collection to search
            access_tier: User's access tier
            user_id: User ID for tracking

        Returns:
            AgenticRAGResult with answer and metadata
        """
        import time
        start_time = time.time()

        # Step 1: Decompose query
        is_complex, sub_queries, synthesis = await self.decompose_query(query)

        if not is_complex:
            # Simple query - use standard RAG
            result = await self.rag.query(
                question=query,
                collection_filter=collection_filter,
                access_tier=access_tier,
                user_id=user_id,
            )

            return AgenticRAGResult(
                query=query,
                final_answer=result.content,
                sub_queries=[],
                react_steps=[],
                sources_used=[
                    {"document_id": str(s.document_id), "name": s.document_name}
                    for s in result.sources
                ],
                confidence=result.confidence_score or 0.8,
                processing_time_ms=(time.time() - start_time) * 1000,
                iterations=1,
            )

        # Step 2: Run ReAct loop for complex queries
        steps, final_answer, sources = await self.run_react_loop(
            query,
            sub_queries,
            collection_filter=collection_filter,
            access_tier=access_tier,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Calculate confidence based on completed sub-queries
        completed = sum(1 for sq in sub_queries if sq.completed)
        confidence = completed / max(len(sub_queries), 1)

        logger.info(
            "Agentic RAG complete",
            query_length=len(query),
            sub_queries=len(sub_queries),
            steps=len(steps),
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )

        return AgenticRAGResult(
            query=query,
            final_answer=final_answer,
            sub_queries=sub_queries,
            react_steps=steps,
            sources_used=sources,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            iterations=len(steps),
        )

    async def process_query_stream(
        self,
        query: str,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query with streaming updates.

        Yields progress updates during processing.
        """
        yield {"type": "start", "message": "Analyzing query complexity..."}

        # Decompose
        is_complex, sub_queries, synthesis = await self.decompose_query(query)

        if is_complex:
            yield {
                "type": "decomposition",
                "is_complex": True,
                "sub_queries": [sq.query for sq in sub_queries],
            }
        else:
            yield {"type": "decomposition", "is_complex": False}

        # Process
        result = await self.process_query(
            query,
            collection_filter=collection_filter,
            access_tier=access_tier,
            user_id=user_id,
        )

        # Stream steps
        for step in result.react_steps:
            yield {
                "type": "step",
                "step_number": step.step_number,
                "thought": step.thought,
                "action": step.action.value,
            }

        # Final result
        yield {
            "type": "complete",
            "answer": result.final_answer,
            "confidence": result.confidence,
            "sources_count": len(result.sources_used),
        }


# =============================================================================
# Factory Function
# =============================================================================

def get_agentic_rag_service(
    rag_service,
    knowledge_graph_service=None,
    llm_service=None,
) -> AgenticRAGService:
    """Create configured agentic RAG service."""
    return AgenticRAGService(
        rag_service=rag_service,
        knowledge_graph_service=knowledge_graph_service,
        llm_service=llm_service,
    )
