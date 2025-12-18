"""
AIDocumentIndexer - Worker Agents
==================================

Specialized worker agents for different task types:

1. GeneratorAgent - Content creation, drafting, outlining
2. CriticAgent - Quality evaluation using LLM-as-judge pattern
3. ResearchAgent - Information retrieval from documents and web
4. ToolExecutionAgent - File operations and document generation

Each worker has structured prompts and validation for its domain.
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.agents.agent_base import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
    TaskStatus,
    PromptTemplate,
    ValidationResult,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Generator Agent
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are a Content Generation Agent specialized in creating high-quality written content.

Your capabilities:
- Draft documents, reports, and summaries
- Create outlines and structured content
- Write in various styles and formats
- Incorporate context and research findings

Guidelines:
- Always structure your output clearly
- Use appropriate formatting (headers, lists, paragraphs)
- Be comprehensive but concise
- Cite sources when provided in context
- Match the requested tone and style"""

GENERATOR_TASK_PROMPT = """{{task}}

{{context}}

Output Format: {{format}}

Please generate the requested content:"""


class GeneratorAgent(BaseAgent):
    """
    Content generation agent.

    Creates documents, summaries, outlines, and other written content.
    Reuses patterns from CollaborationService generator role.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm=None,
        prompt_template: Optional[PromptTemplate] = None,
        trajectory_collector=None,
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                id="generator_default",
                version=1,
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                task_prompt_template=GENERATOR_TASK_PROMPT,
            )

        super().__init__(
            config=config,
            llm=llm,
            prompt_template=prompt_template,
            trajectory_collector=trajectory_collector,
        )

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Generate content based on task.

        Args:
            task: Generation task
            context: Context including prior step results, format spec

        Returns:
            AgentResult with generated content
        """
        self.clear_trajectory()
        start_time = time.time()

        # Validate inputs
        validation = self.validate_inputs(task, context)
        if not validation.valid:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=f"Invalid inputs: {validation.errors}",
            )

        # Build context string from dependencies
        context_parts = []

        # Add dependency results - especially research findings
        if context.get("dependency_results"):
            context_parts.append("Information from previous steps:\n")
            for step_id, result in context["dependency_results"].items():
                if isinstance(result, str):
                    # Use more content for better context (up to 2000 chars)
                    context_parts.append(f"{result[:2000]}")
                elif isinstance(result, dict):
                    # Research agent returns findings dict - extract the actual findings
                    if "findings" in result:
                        findings = result["findings"]
                        context_parts.append(f"Research Findings:\n{findings[:2000]}")
                    else:
                        context_parts.append(f"{json.dumps(result, indent=2)[:2000]}")

        # Add document context
        if context.get("documents"):
            context_parts.append(f"\nRelevant documents: {context['documents']}")

        # Add search results
        if context.get("search_results"):
            context_parts.append(f"\nSearch results: {context['search_results']}")

        context_str = "\n".join(context_parts) if context_parts else "No additional context provided."

        # Build messages
        # expected_outputs might be a dict or str, handle both
        default_format = "text"
        if isinstance(task.expected_outputs, dict):
            default_format = task.expected_outputs.get("format", "text")
        output_format = context.get("output_format", default_format)
        messages = self.prompt_template.build_messages(
            task=task.description,
            context=context_str,
            format=output_format,
        )

        try:
            response_text, input_tokens, output_tokens = await self.invoke_llm(
                messages, record=True
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Validate output
            output_validation = self.validate_outputs(task, response_text)

            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output=response_text,
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                confidence_score=output_validation.score,
                trajectory_steps=self._current_trajectory,
            )

        except Exception as e:
            logger.error(
                "GeneratorAgent execution failed",
                error=str(e),
                task_id=task.id,
                agent_id=self.agent_id,
                exc_info=True,
            )
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(e),
                confidence_score=0.0,  # Failed tasks should have 0 confidence
                trajectory_steps=self._current_trajectory,
            )


# =============================================================================
# Critic Agent
# =============================================================================

CRITIC_SYSTEM_PROMPT = """You are a Quality Evaluation Agent that assesses content using the LLM-as-judge pattern.

Your role is to evaluate content against specific criteria and provide:
1. Scores for each criterion (1-5 scale)
2. Specific feedback for improvement
3. Overall quality assessment

Evaluation Criteria:
- Factual Accuracy (30%): Are claims correct and verifiable?
- Completeness (25%): Does it fully address the request?
- Clarity (20%): Is it well-written and easy to understand?
- Relevance (15%): Is everything pertinent to the topic?
- Coherence (10%): Does it flow logically?

Be constructive but honest in your evaluation. Flag any issues clearly."""

CRITIC_TASK_PROMPT = """Evaluate the following content:

Original Request: {{request}}

Content to Evaluate:
{{content}}

{{context}}

Provide your evaluation as JSON:
{
    "scores": {
        "factual_accuracy": 1-5,
        "completeness": 1-5,
        "clarity": 1-5,
        "relevance": 1-5,
        "coherence": 1-5
    },
    "overall_score": 1-5,
    "feedback": [
        "Specific feedback point 1",
        "Specific feedback point 2"
    ],
    "improvements_needed": ["List of suggested improvements"],
    "passed": true/false
}"""


@dataclass
class EvaluationResult:
    """Result of content evaluation."""
    scores: Dict[str, float]
    overall_score: float
    feedback: List[str]
    improvements_needed: List[str]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": self.scores,
            "overall_score": self.overall_score,
            "feedback": self.feedback,
            "improvements_needed": self.improvements_needed,
            "passed": self.passed,
        }


class CriticAgent(BaseAgent):
    """
    Quality evaluation agent using LLM-as-judge pattern.

    Evaluates content against rubric with weighted criteria.
    """

    # Evaluation rubric with weights
    EVALUATION_RUBRIC = {
        "factual_accuracy": {"weight": 0.30, "description": "Correctness of claims"},
        "completeness": {"weight": 0.25, "description": "Fully addresses request"},
        "clarity": {"weight": 0.20, "description": "Well-written and clear"},
        "relevance": {"weight": 0.15, "description": "On-topic content"},
        "coherence": {"weight": 0.10, "description": "Logical flow"},
    }

    PASSING_THRESHOLD = 3.5  # Minimum overall score to pass

    def __init__(
        self,
        config: AgentConfig,
        llm=None,
        prompt_template: Optional[PromptTemplate] = None,
        trajectory_collector=None,
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                id="critic_default",
                version=1,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                task_prompt_template=CRITIC_TASK_PROMPT,
            )

        super().__init__(
            config=config,
            llm=llm,
            prompt_template=prompt_template,
            trajectory_collector=trajectory_collector,
        )

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Evaluate content quality.

        Args:
            task: Evaluation task
            context: Context with content to evaluate and original request

        Returns:
            AgentResult with EvaluationResult
        """
        self.clear_trajectory()
        start_time = time.time()

        # Get content to evaluate from dependencies or context
        content_to_evaluate = None
        if context.get("dependency_results"):
            # Get the last dependency's output
            for result in context["dependency_results"].values():
                content_to_evaluate = result

        if not content_to_evaluate:
            content_to_evaluate = context.get("content", "")

        if not content_to_evaluate:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message="No content provided for evaluation",
            )

        original_request = context.get("original_request", task.description)

        # Build messages
        messages = self.prompt_template.build_messages(
            task="",  # Not used, template uses specific variables
            context="",
            request=original_request,
            content=content_to_evaluate if isinstance(content_to_evaluate, str) else json.dumps(content_to_evaluate),
        )

        try:
            response_text, input_tokens, output_tokens = await self.invoke_llm(
                messages, record=True
            )

            # Parse evaluation result
            evaluation = self._parse_evaluation(response_text)

            duration_ms = int((time.time() - start_time) * 1000)

            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output=evaluation.to_dict(),
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                confidence_score=evaluation.overall_score / 5.0,
                trajectory_steps=self._current_trajectory,
                metadata={"passed": evaluation.passed},
            )

        except Exception as e:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(e),
                trajectory_steps=self._current_trajectory,
            )

    def _parse_evaluation(self, response: str) -> EvaluationResult:
        """Parse evaluation response into structured result."""
        try:
            # Extract JSON
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response:
                json_start = response.index("{")
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(response[json_start:], json_start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                json_str = response[json_start:json_end]
            else:
                json_str = response

            data = json.loads(json_str)

            scores = data.get("scores", {})
            overall = data.get("overall_score", 3.0)

            # Calculate weighted overall if not provided
            if not overall and scores:
                weighted_sum = sum(
                    scores.get(criterion, 3) * info["weight"]
                    for criterion, info in self.EVALUATION_RUBRIC.items()
                )
                overall = weighted_sum

            return EvaluationResult(
                scores=scores,
                overall_score=overall,
                feedback=data.get("feedback", []),
                improvements_needed=data.get("improvements_needed", []),
                passed=data.get("passed", overall >= self.PASSING_THRESHOLD),
            )

        except (json.JSONDecodeError, ValueError):
            # Fallback: extract score from text
            logger.warning("Failed to parse evaluation JSON, using fallback")
            return EvaluationResult(
                scores={},
                overall_score=3.0,
                feedback=["Unable to parse detailed evaluation"],
                improvements_needed=[],
                passed=True,
            )

    async def evaluate(
        self,
        content: str,
        original_request: str,
        criteria: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Convenience method for direct evaluation.

        Args:
            content: Content to evaluate
            original_request: Original request for context
            criteria: Optional specific criteria to focus on

        Returns:
            EvaluationResult
        """
        task = AgentTask(
            id=str(uuid.uuid4()),
            type="evaluation",
            name="Content Evaluation",
            description="Evaluate content quality",
        )

        result = await self.execute(
            task,
            {
                "content": content,
                "original_request": original_request,
                "criteria": criteria,
            }
        )

        if result.is_success and result.output:
            return EvaluationResult(**result.output)
        else:
            raise ValueError(result.error_message or "Evaluation failed")


# =============================================================================
# Research Agent
# =============================================================================

RESEARCH_SYSTEM_PROMPT = """You are a Research Agent specialized in information retrieval and synthesis.

Your capabilities:
- Search and retrieve relevant information from documents
- Synthesize information from multiple sources
- Identify key facts and insights
- Provide citations and source references

Guidelines:
- Always cite your sources
- Be thorough but relevant
- Organize findings clearly
- Flag any conflicting information
- Note when information might be outdated or incomplete"""

RESEARCH_TASK_PROMPT = """Research Task: {{task}}

{{context}}

Available Sources:
{{sources}}

Please conduct research and provide your findings:"""


class ResearchAgent(BaseAgent):
    """
    Information retrieval agent.

    Searches documents via RAGService and optionally web via ScraperService.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm=None,
        prompt_template: Optional[PromptTemplate] = None,
        trajectory_collector=None,
        rag_service=None,
        scraper_service=None,
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                id="research_default",
                version=1,
                system_prompt=RESEARCH_SYSTEM_PROMPT,
                task_prompt_template=RESEARCH_TASK_PROMPT,
            )

        super().__init__(
            config=config,
            llm=llm,
            prompt_template=prompt_template,
            trajectory_collector=trajectory_collector,
        )

        self.rag_service = rag_service
        self.scraper_service = scraper_service

    def set_services(
        self,
        rag_service=None,
        scraper_service=None
    ) -> None:
        """Set service instances."""
        if rag_service:
            self.rag_service = rag_service
        if scraper_service:
            self.scraper_service = scraper_service

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Research information for task.

        Args:
            task: Research task
            context: Context with search parameters

        Returns:
            AgentResult with research findings
        """
        self.clear_trajectory()
        start_time = time.time()

        # Extract search query
        query = task.description or context.get("query", "")

        # Record research start
        self.record_step(
            action_type="research_start",
            input_data={"query": query},
            output_data={},
        )

        sources = []
        search_results = []

        # 1. Search documents via RAG
        if self.rag_service:
            try:
                rag_results = await self._search_documents(query)
                if rag_results:
                    search_results.extend(rag_results)
                    sources.append("Document search")
            except Exception as e:
                logger.warning(f"Document search failed: {e}")

        # 2. Search web if enabled and needed
        if self.scraper_service and context.get("include_web", False):
            try:
                web_results = await self._search_web(query)
                if web_results:
                    search_results.extend(web_results)
                    sources.append("Web search")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        # 3. Synthesize findings using LLM
        if search_results:
            sources_text = "\n".join(
                f"- {r.get('source', 'Unknown')}: {r.get('content', '')[:300]}"
                for r in search_results
            )
        else:
            sources_text = "No search results available. Use your knowledge to provide relevant information."

        messages = self.prompt_template.build_messages(
            task=task.description,
            context=context.get("additional_context", ""),
            sources=sources_text,
        )

        try:
            response_text, input_tokens, output_tokens = await self.invoke_llm(
                messages, record=True
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output={
                    "findings": response_text,
                    "sources": sources,
                    "result_count": len(search_results),
                },
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                confidence_score=0.9 if search_results else 0.6,
                trajectory_steps=self._current_trajectory,
            )

        except Exception as e:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(e),
                trajectory_steps=self._current_trajectory,
            )

    async def _search_documents(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search documents using RAGService."""
        if not self.rag_service:
            logger.warning("No RAG service available for document search")
            return []

        try:
            # Use RAGService search method
            results = await self.rag_service.search(
                query=query,
                limit=limit,
            )

            self.record_step(
                action_type="document_search",
                input_data={"query": query, "limit": limit},
                output_data={"result_count": len(results)},
            )

            logger.info(
                "Document search completed",
                query=query[:50],
                result_count=len(results),
            )

            return [
                {
                    "source": r.get("document_name", "Document"),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Document search failed: {e}", exc_info=True)
            return []

    async def _search_web(
        self,
        query: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Search web using ScraperService."""
        if not self.scraper_service:
            return []

        try:
            results = await self.scraper_service.scrape_and_query(
                urls=[],  # Let scraper determine URLs
                query=query,
                max_results=limit,
            )

            self.record_step(
                action_type="web_search",
                input_data={"query": query},
                output_data={"result_count": len(results)},
            )

            return [
                {
                    "source": r.get("url", "Web"),
                    "content": r.get("content", ""),
                }
                for r in results
            ]

        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return []


# =============================================================================
# Tool Execution Agent
# =============================================================================

TOOL_SYSTEM_PROMPT = """You are a Tool Execution Agent that handles file operations and document generation.

Your capabilities:
- Generate documents in various formats (PPTX, DOCX, PDF)
- Export content to files
- Handle format conversions
- Execute file operations

When generating documents, provide clear structure and formatting instructions."""

TOOL_TASK_PROMPT = """Tool Task: {{task}}

Content to process:
{{content}}

Output format: {{format}}

Please provide instructions for document generation:"""


class ToolExecutionAgent(BaseAgent):
    """
    Tool execution agent for file operations.

    Handles document generation (PPTX, DOCX, PDF) by preparing
    content and delegating to GeneratorService.
    """

    AVAILABLE_TOOLS = [
        "generate_pptx",
        "generate_docx",
        "generate_pdf",
        "export_markdown",
    ]

    def __init__(
        self,
        config: AgentConfig,
        llm=None,
        prompt_template: Optional[PromptTemplate] = None,
        trajectory_collector=None,
        generator_service=None,
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                id="tool_default",
                version=1,
                system_prompt=TOOL_SYSTEM_PROMPT,
                task_prompt_template=TOOL_TASK_PROMPT,
            )

        super().__init__(
            config=config,
            llm=llm,
            prompt_template=prompt_template,
            trajectory_collector=trajectory_collector,
        )

        self.generator_service = generator_service

    def set_services(self, generator_service=None) -> None:
        """Set service instances."""
        if generator_service:
            self.generator_service = generator_service

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute tool operation.

        Args:
            task: Tool task
            context: Context with content and format spec

        Returns:
            AgentResult with file path or content
        """
        self.clear_trajectory()
        start_time = time.time()

        # Get tool/format from task
        tool = task.expected_outputs.get("tool", context.get("tool", "generate_docx"))
        output_format = context.get("output_format", "docx")

        # Get content from dependencies or context
        content = None
        if context.get("dependency_results"):
            for result in context["dependency_results"].values():
                if isinstance(result, str):
                    content = result
                elif isinstance(result, dict) and "findings" in result:
                    content = result["findings"]
                elif isinstance(result, dict) and "content" in result:
                    content = result["content"]

        if not content:
            content = context.get("content", task.description)

        self.record_step(
            action_type="tool_start",
            input_data={"tool": tool, "format": output_format},
            output_data={},
        )

        try:
            if tool in ("generate_pptx", "generate_docx", "generate_pdf"):
                result = await self._generate_document(
                    content=content,
                    output_format=output_format,
                    context=context,
                )
            elif tool == "export_markdown":
                result = await self._export_markdown(content, context)
            else:
                result = {"content": content, "format": "text"}

            duration_ms = int((time.time() - start_time) * 1000)

            self.record_step(
                action_type="tool_complete",
                input_data={},
                output_data={"result": str(result)[:200]},
                duration_ms=duration_ms,
            )

            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output=result,
                duration_ms=duration_ms,
                trajectory_steps=self._current_trajectory,
            )

        except Exception as e:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(e),
                trajectory_steps=self._current_trajectory,
            )

    async def _generate_document(
        self,
        content: str,
        output_format: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate document using GeneratorService."""
        if not self.generator_service:
            # Return content as-is if no generator service
            return {
                "type": "content",
                "content": content,
                "format": output_format,
                "note": "GeneratorService not available, returning raw content",
            }

        try:
            # Prepare content structure
            # Use LLM to format content for document generation
            structure_prompt = f"""Convert the following content into a structured format suitable for {output_format} generation.

Content:
{content}

Provide the structure as JSON with title, sections, and content."""

            messages = [
                SystemMessage(content="You are formatting content for document generation."),
                HumanMessage(content=structure_prompt),
            ]

            structure_response, _, _ = await self.invoke_llm(messages, record=True)

            # Generate document
            result = await self.generator_service.generate(
                content=structure_response,
                output_format=output_format,
                **context.get("generation_options", {}),
            )

            return {
                "type": "file",
                "file_path": result.get("file_path"),
                "format": output_format,
            }

        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            return {
                "type": "content",
                "content": content,
                "format": "text",
                "error": str(e),
            }

    async def _export_markdown(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Export content as markdown."""
        return {
            "type": "content",
            "content": content,
            "format": "markdown",
        }


# =============================================================================
# Factory
# =============================================================================

def create_worker_agents(
    rag_service=None,
    scraper_service=None,
    generator_service=None,
    trajectory_collector=None,
) -> Dict[str, BaseAgent]:
    """
    Factory function to create all worker agents.

    Args:
        rag_service: RAGService instance
        scraper_service: ScraperService instance
        generator_service: GeneratorService instance
        trajectory_collector: TrajectoryCollector instance

    Returns:
        Dict mapping agent_type to agent instance
    """
    workers = {}

    # Generator Agent
    generator_config = AgentConfig(
        agent_id=str(uuid.uuid4()),
        name="Generator Agent",
        description="Content generation and drafting",
    )
    workers["generator"] = GeneratorAgent(
        config=generator_config,
        trajectory_collector=trajectory_collector,
    )

    # Critic Agent
    critic_config = AgentConfig(
        agent_id=str(uuid.uuid4()),
        name="Critic Agent",
        description="Quality evaluation and feedback",
    )
    workers["critic"] = CriticAgent(
        config=critic_config,
        trajectory_collector=trajectory_collector,
    )

    # Research Agent
    research_config = AgentConfig(
        agent_id=str(uuid.uuid4()),
        name="Research Agent",
        description="Information retrieval and synthesis",
    )
    research_agent = ResearchAgent(
        config=research_config,
        trajectory_collector=trajectory_collector,
    )
    if rag_service:
        research_agent.set_services(rag_service=rag_service)
    if scraper_service:
        research_agent.set_services(scraper_service=scraper_service)
    workers["research"] = research_agent

    # Tool Execution Agent
    tool_config = AgentConfig(
        agent_id=str(uuid.uuid4()),
        name="Tool Execution Agent",
        description="File operations and document generation",
    )
    tool_agent = ToolExecutionAgent(
        config=tool_config,
        trajectory_collector=trajectory_collector,
    )
    if generator_service:
        tool_agent.set_services(generator_service=generator_service)
    workers["tool"] = tool_agent

    logger.info(f"Created {len(workers)} worker agents")

    return workers
