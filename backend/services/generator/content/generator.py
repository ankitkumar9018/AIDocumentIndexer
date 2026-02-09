"""
Content Generator

Generates section content using LLM with template context and constraints.
Migrated from generator.py for modularity.
"""

import uuid
import re
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import structlog

from ..config import LANGUAGE_NAMES
from ..utils import (
    filter_llm_metatext,
    filter_title_echo,
    validate_language_purity,
    filter_incomplete_sentences,
)

if TYPE_CHECKING:
    from ..models import GenerationJob, Section, SourceReference, OutputFormat
    from ..template_analyzer import TemplateAnalysis

logger = structlog.get_logger(__name__)


class ContentGenerator:
    """Generates section content using LLM.

    This class provides content generation with optional template context,
    ensuring the generated content fits the template constraints.
    """

    def __init__(self, config=None):
        self._llm = None
        self.config = config

    async def _get_llm(self, job: "GenerationJob" = None):
        """Get the LLM instance for content generation.

        If the job has provider_id/model overrides in metadata, creates a
        dedicated LLM for that job instead of using the cached default.
        """
        provider_override = job.metadata.get("provider_id") if job and job.metadata else None
        model_override = job.metadata.get("model") if job and job.metadata else None

        if provider_override or model_override:
            from backend.services.llm import LLMFactory, llm_config
            from backend.services.llm_provider import LLMProviderService
            provider_type = llm_config.default_provider
            model_name = model_override

            if provider_override:
                try:
                    from backend.db.database import async_session_context
                    async with async_session_context() as session:
                        provider = await LLMProviderService.get_provider(session, provider_override)
                        if provider:
                            provider_type = provider.provider_type
                            if not model_name:
                                model_name = provider.default_chat_model
                except Exception as e:
                    logger.warning(f"Could not resolve provider override {provider_override}: {e}")

            logger.info(
                "Using per-job LLM override for content generation",
                provider=provider_type,
                model=model_name,
            )
            return LLMFactory.get_chat_model(
                provider=provider_type,
                model=model_name,
            )

        if self._llm is None:
            from backend.services.llm import EnhancedLLMFactory
            self._llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="content_generation",
                user_id=None,
            )
        return self._llm

    async def generate_section(
        self,
        job: "GenerationJob",
        section_title: str,
        section_description: str,
        order: int,
        existing_section_id: Optional[str] = None,
        sources: Optional[List["SourceReference"]] = None,
        template_analysis: Optional["TemplateAnalysis"] = None,
        slide_constraints: Optional[Dict[str, Any]] = None,
        used_chunk_ids: Optional[set] = None,
    ) -> "Section":
        """Generate content for a single section.

        Args:
            job: The generation job
            section_title: Title of the section
            section_description: Description of what to generate
            order: Order/index of the section
            existing_section_id: Optional existing section ID to update
            sources: Optional source references for RAG
            template_analysis: Optional template analysis for constraints
            slide_constraints: Optional per-slide constraints from TemplateLayoutLearner
                Contains: title_constraints, content_constraints, layout, avoid_zones

        Returns:
            Section with generated content
        """
        from ..models import Section, OutputFormat

        section_id = existing_section_id or str(uuid.uuid4())

        # Search for relevant sources if not provided
        if sources is None:
            sources = await self._search_sources_for_section(
                job, section_title, section_description,
                exclude_chunk_ids=used_chunk_ids,
            )

        # Log sources found for debugging
        logger.info(
            "Section sources search completed",
            section_title=section_title[:50],
            sources_found=len(sources) if sources else 0,
            source_names=[s.document_name for s in sources[:3]] if sources else [],
        )

        # Build context from sources
        context = self._build_source_context(sources)

        # Calculate total sections for position-aware instructions
        total_sections = len(job.outline.sections) if job.outline else 5

        # Get format-specific instructions (with template constraints and position awareness)
        format_instructions = self._get_format_instructions(
            job.output_format,
            template_analysis,
            section_order=order,
            total_sections=total_sections,
            slide_constraints=slide_constraints,
        )

        # Calculate section position context
        position_context = self._get_position_context(job, order)

        # Build style context if available
        style_context = self._build_style_context(job)

        # Build language instruction
        output_language = job.metadata.get("output_language", "en") if job.metadata else "en"
        language_instruction = self._build_language_instruction(output_language, job)

        # Generate content
        prompt = self._build_content_prompt(
            job=job,
            section_title=section_title,
            section_description=section_description,
            context=context,
            format_instructions=format_instructions,
            position_context=position_context,
            style_context=style_context,
            language_instruction=language_instruction,
        )

        try:
            llm = await self._get_llm(job)

            # PHASE 15: Apply model-specific enhancements for small models
            # Extract model name and enhance prompt if using small model
            model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
            enhanced_prompt = prompt
            if model_name:
                from backend.services.rag_module.prompts import enhance_agent_system_prompt
                # If prompt is a string, enhance it directly
                if isinstance(prompt, str):
                    enhanced_prompt = enhance_agent_system_prompt(prompt, model_name)
                # If prompt is a list of messages, enhance the SystemMessage
                elif isinstance(prompt, list):
                    from langchain_core.messages import SystemMessage
                    enhanced_prompt = []
                    for msg in prompt:
                        if isinstance(msg, SystemMessage):
                            enhanced_content = enhance_agent_system_prompt(msg.content, model_name)
                            enhanced_prompt.append(SystemMessage(content=enhanced_content))
                        else:
                            enhanced_prompt.append(msg)

            # Phase 15 LLM Optimization - apply temperature override if specified
            invoke_kwargs = {}
            if "temperature_override" in job.metadata:
                invoke_kwargs["temperature"] = job.metadata["temperature_override"]

            response = await llm.ainvoke(enhanced_prompt, **invoke_kwargs)
            content = response.content

            # Filter out LLM conversational artifacts (meta-text)
            content = filter_llm_metatext(content)

            # Filter out bullets that just echo the section title
            content = filter_title_echo(content, section_title)

            # Validate language purity - remove foreign language words
            # This addresses issue of LLM code-switching from multilingual sources
            content, removed_foreign_words = validate_language_purity(
                content, target_language=output_language
            )
            if removed_foreign_words:
                logger.warning(
                    "Removed foreign words from generated content",
                    section_title=section_title[:50],
                    foreign_words=removed_foreign_words[:5],
                )

            # Filter out incomplete sentences (truncated bullets)
            content = filter_incomplete_sentences(content)

            # Dual Mode: Combine RAG content with general AI knowledge
            dual_mode = job.metadata.get("dual_mode", False) if job.metadata else False
            if dual_mode and content and not content.startswith("[Content for"):
                content = await self._apply_dual_mode(
                    rag_content=content,
                    job=job,
                    section_title=section_title,
                    section_description=section_description,
                    llm=llm,
                    invoke_kwargs=invoke_kwargs,
                )

        except Exception as e:
            logger.error("Failed to generate section", error=str(e))
            content = f"[Content for {section_title} - generation failed]"

        # Quality scoring and optional auto-regeneration
        quality_report = None
        section_metadata = {}

        # Check if quality review is enabled
        enable_quality = False
        if self.config:
            enable_quality = job.metadata.get("enable_quality_review", self.config.enable_quality_review)

        if enable_quality and content and not content.startswith("[Content for"):
            content, quality_report, section_metadata = await self._run_quality_review(
                content=content,
                section_title=section_title,
                job=job,
                sources=sources,
                order=order,
                format_instructions=format_instructions,
            )

        # CriticAgent proofreading - runs after basic quality scoring
        enable_critic = job.metadata.get("enable_critic_review", False) if job.metadata else False
        if enable_critic and content and not content.startswith("[Content for"):
            content, critic_metadata = await self._review_with_critic(
                content=content,
                section_title=section_title,
                job=job,
                format_instructions=format_instructions,
                output_language=output_language,
            )
            if critic_metadata:
                section_metadata.update(critic_metadata)

        # Fact-checking - verify claims against sources (Phase 2 enhancement)
        # Reduces hallucinations by 25-40% according to research
        fact_check_level = job.metadata.get("fact_check_level", "off") if job.metadata else "off"
        fact_check_report = None
        if fact_check_level != "off" and content and sources and not content.startswith("[Content for"):
            content, fact_check_report = await self._run_fact_check(
                content=content,
                section_title=section_title,
                sources=sources,
                fact_check_level=fact_check_level,
            )
            if fact_check_report:
                section_metadata["fact_check"] = {
                    "verification_rate": fact_check_report.verification_rate,
                    "verified_claims": fact_check_report.verified_claims,
                    "total_claims": fact_check_report.total_claims,
                    "overall_confidence": fact_check_report.overall_confidence,
                }
                if fact_check_report.revision_suggestions:
                    section_metadata["fact_check"]["suggestions"] = fact_check_report.revision_suggestions[:3]

        # Store quality report in section metadata
        if quality_report:
            section_metadata["quality_score"] = quality_report.overall_score
            section_metadata["quality_summary"] = quality_report.summary
            section_metadata["needs_revision"] = quality_report.needs_revision

        require_approval = False
        if self.config:
            require_approval = self.config.require_section_approval

        return Section(
            id=section_id,
            title=section_title,
            content=content,
            order=order,
            sources=sources or [],
            approved=not require_approval,
            metadata=section_metadata if section_metadata else None,
        )

    async def _apply_dual_mode(
        self,
        rag_content: str,
        job: "GenerationJob",
        section_title: str,
        section_description: str,
        llm,
        invoke_kwargs: dict,
    ) -> str:
        """Combine RAG-based content with general AI knowledge.

        Generates a second version of the content without document context,
        then merges the two based on the blend strategy.
        """
        blend = job.metadata.get("dual_mode_blend", "merged") if job.metadata else "merged"

        try:
            # Generate general knowledge version (no RAG context)
            general_prompt = (
                f"Write content for a section titled \"{section_title}\".\n"
                f"Description: {section_description}\n"
                f"Document title: {job.title}\n\n"
                f"Use your general knowledge to write comprehensive, informative content. "
                f"Do not mention that you are an AI. Write in a professional tone."
            )
            general_response = await llm.ainvoke(general_prompt, **invoke_kwargs)
            general_content = general_response.content

            if not general_content or not general_content.strip():
                return rag_content

            # Merge based on blend strategy
            if blend == "docs_first":
                merge_prompt = (
                    f"You have two versions of content for the section \"{section_title}\".\n\n"
                    f"PRIMARY (from documents - prioritize this):\n{rag_content}\n\n"
                    f"SUPPLEMENTARY (general knowledge - use to fill gaps):\n{general_content}\n\n"
                    f"Create a final version that uses the document-based content as the foundation, "
                    f"supplementing with general knowledge only where the documents lack detail. "
                    f"Maintain the same format and style as the primary version. "
                    f"Output ONLY the merged content, nothing else."
                )
            else:  # "merged"
                merge_prompt = (
                    f"You have two versions of content for the section \"{section_title}\".\n\n"
                    f"VERSION A (from documents):\n{rag_content}\n\n"
                    f"VERSION B (general knowledge):\n{general_content}\n\n"
                    f"Synthesize both into one comprehensive section that combines the best of both. "
                    f"Prioritize document-sourced facts but enrich with general knowledge for depth. "
                    f"Maintain a consistent tone and format. "
                    f"Output ONLY the merged content, nothing else."
                )

            merge_response = await llm.ainvoke(merge_prompt, **invoke_kwargs)
            merged = merge_response.content

            if merged and merged.strip():
                logger.info(
                    "Dual mode content merged",
                    section=section_title[:50],
                    blend=blend,
                )
                return filter_llm_metatext(merged)

            return rag_content

        except Exception as e:
            logger.warning(
                "Dual mode merge failed, using RAG content only",
                section=section_title[:50],
                error=str(e),
            )
            return rag_content

    async def generate_sections_parallel(
        self,
        job: "GenerationJob",
        sections_info: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List["Section"]:
        """Generate multiple sections in parallel for faster document creation.

        This method processes multiple sections concurrently using asyncio,
        significantly reducing total generation time for multi-section documents.

        Args:
            job: The generation job
            sections_info: List of dicts with keys:
                - section_title: str
                - section_description: str
                - order: int
                - existing_section_id: Optional[str]
                - sources: Optional[List[SourceReference]]
                - template_analysis: Optional[TemplateAnalysis]
            max_concurrent: Maximum concurrent section generations (default 3)

        Returns:
            List of generated Section objects, ordered by their 'order' field
        """
        import asyncio
        from ..models import Section

        if not sections_info:
            return []

        logger.info(
            "Starting parallel section generation",
            total_sections=len(sections_info),
            max_concurrent=max_concurrent,
            job_id=str(job.id) if job.id else None,
        )

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(info: Dict[str, Any]) -> "Section":
            """Generate a section with concurrency limiting."""
            async with semaphore:
                logger.debug(
                    "Generating section",
                    title=info.get("section_title", "")[:30],
                    order=info.get("order", 0),
                )
                return await self.generate_section(
                    job=job,
                    section_title=info["section_title"],
                    section_description=info.get("section_description", ""),
                    order=info.get("order", 0),
                    existing_section_id=info.get("existing_section_id"),
                    sources=info.get("sources"),
                    template_analysis=info.get("template_analysis"),
                )

        # Create tasks for all sections
        tasks = [generate_with_limit(info) for info in sections_info]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, filtering out exceptions
        sections = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Section generation failed",
                    section_title=sections_info[i].get("section_title", "unknown"),
                    error=str(result),
                )
                # Create placeholder section for failed generations
                sections.append(Section(
                    id=sections_info[i].get("existing_section_id") or str(uuid.uuid4()),
                    title=sections_info[i].get("section_title", "Unknown"),
                    content=f"[Generation failed: {str(result)[:100]}]",
                    order=sections_info[i].get("order", i),
                    sources=[],
                    approved=False,
                    metadata={"error": str(result)},
                ))
            else:
                sections.append(result)

        # Sort by order
        sections.sort(key=lambda s: s.order)

        logger.info(
            "Parallel section generation completed",
            total_sections=len(sections_info),
            successful=sum(1 for s in sections if not s.metadata or "error" not in s.metadata),
            failed=sum(1 for s in sections if s.metadata and "error" in s.metadata),
        )

        return sections

    async def _search_sources_for_section(
        self,
        job: "GenerationJob",
        section_title: str,
        section_description: str,
        exclude_chunk_ids: Optional[set] = None,
    ) -> List["SourceReference"]:
        """Search for relevant sources for a section.

        Uses LLM-based smart filtering to select only truly relevant documents
        rather than a fixed number. Returns SourceReference objects with
        usage_type=CONTENT since these are used for content generation.
        """
        from ..models import SourceReference, SourceUsageType

        # Include job title/description for better context
        section_query = f"{job.title} - {section_title}"
        if section_description:
            section_query += f": {section_description}"

        # Try to get RAG service
        try:
            from backend.services.rag import get_rag_service
            rag = get_rag_service()

            collection_filter = job.metadata.get("collection_filter") if job.metadata else None

            # Check if user has set a specific source limit, otherwise use LLM filtering
            user_source_limit = job.metadata.get("max_sources_per_section") if job.metadata else None
            use_smart_filter = job.metadata.get("smart_source_filter", True) if job.metadata else True
            use_knowledge_graph = job.metadata.get("use_knowledge_graph", True) if job.metadata else True

            # Retrieve more candidates for LLM to filter (unless user set a limit)
            initial_limit = user_source_limit if user_source_limit else 15

            # Try to enhance search with knowledge graph for entity-based retrieval
            results = []
            if use_knowledge_graph:
                try:
                    graph_results = await self._search_with_knowledge_graph(
                        section_query, collection_filter, initial_limit
                    )
                    if graph_results:
                        results.extend(graph_results)
                        logger.info(
                            "Knowledge graph enhanced search",
                            graph_results=len(graph_results),
                            section=section_title[:30],
                        )
                except Exception as kg_err:
                    logger.debug(f"Knowledge graph search not available: {kg_err}")

            # Also do standard vector search
            logger.info(
                "Searching sources for section",
                section_title=section_title[:50],
                query=section_query[:100],
                collection_filter=collection_filter,
                limit=initial_limit,
            )
            vector_results = await rag.search(
                query=section_query,
                limit=initial_limit,
                collection_filter=collection_filter,
                exclude_chunk_ids=exclude_chunk_ids,
            )
            logger.info(
                "Vector search completed",
                section_title=section_title[:50],
                vector_results_count=len(vector_results),
                kg_results_count=len(results),
            )

            # Merge and deduplicate results
            seen_chunks = {r.get("chunk_id") for r in results if r.get("chunk_id")}
            for vr in vector_results:
                chunk_id = vr.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    results.append(vr)
                    seen_chunks.add(chunk_id)
                elif not chunk_id:
                    results.append(vr)

            if not results:
                # If collection filter was set but no results, try without filter
                if collection_filter:
                    logger.info(
                        "No sources found with collection filter, trying without filter",
                        section_title=section_title[:50],
                        collection_filter=collection_filter,
                    )
                    fallback_results = await rag.search(
                        query=section_query,
                        limit=initial_limit,
                        collection_filter=None,  # No filter
                    )
                    if fallback_results:
                        logger.info(
                            "Fallback search found sources",
                            count=len(fallback_results),
                            section_title=section_title[:50],
                        )
                        results = fallback_results

                if not results:
                    logger.warning(
                        "No sources found for section (even without collection filter)",
                        section_title=section_title[:50],
                        query=section_query[:100],
                        collection_filter=collection_filter,
                    )
                    return []

            # If smart filtering is enabled and no user limit, let LLM decide relevance
            if use_smart_filter and not user_source_limit and len(results) > 3:
                results = await self._filter_sources_with_llm(
                    results, section_title, section_description, job.title
                )

            # Convert RAG search results to SourceReference objects with CONTENT usage type
            source_refs = []
            for result in results:
                metadata = result.get("metadata", {})

                # Determine page/slide number
                page_num = metadata.get("page_number") or metadata.get("slide_number")

                # Get document path/URL for hyperlinks
                doc_path = metadata.get("source") or metadata.get("file_path")
                doc_url = metadata.get("url") or metadata.get("document_url")

                # Get LLM-assigned usage description if available
                usage_desc = result.get("usage_description") or f"Content for: {section_title[:50]}"

                source_ref = SourceReference(
                    document_id=metadata.get("document_id", result.get("chunk_id", "")),
                    document_name=result.get("document_name", "Unknown"),
                    chunk_id=result.get("chunk_id"),
                    page_number=page_num,
                    relevance_score=result.get("score", 0.0),
                    snippet=result.get("content", "")[:500],  # Truncate long snippets
                    usage_type=SourceUsageType.CONTENT,
                    usage_description=usage_desc,
                    document_path=doc_path,
                    document_url=doc_url,
                )
                source_refs.append(source_ref)

            return source_refs

        except Exception as e:
            logger.warning(f"Could not search for sources: {e}")
            return []

    async def _filter_sources_with_llm(
        self,
        candidates: List[dict],
        section_title: str,
        section_description: str,
        doc_title: str,
    ) -> List[dict]:
        """Use LLM to filter and select only truly relevant source documents.

        This allows the LLM to decide which documents are actually useful rather
        than just taking a fixed number based on vector similarity.
        """
        try:
            llm = await self._get_llm()

            # Build candidate list for LLM
            candidate_summaries = []
            for i, result in enumerate(candidates):
                doc_name = result.get("document_name", "Unknown")
                snippet = result.get("content", "")[:300]
                score = result.get("score", 0.0)
                candidate_summaries.append(
                    f"[{i}] {doc_name} (score: {score:.3f})\n   Preview: {snippet}..."
                )

            candidates_text = "\n\n".join(candidate_summaries)

            prompt = f"""You are evaluating source documents for content generation.

DOCUMENT BEING CREATED: {doc_title}
SECTION: {section_title}
SECTION DESCRIPTION: {section_description or 'N/A'}

CANDIDATE SOURCE DOCUMENTS:
{candidates_text}

TASK: Select ONLY the documents that are TRULY RELEVANT and USEFUL for writing this section.
- Do NOT select documents just because they have high similarity scores
- Select documents that provide actual facts, data, or insights for the topic
- Skip documents that are tangentially related or won't add value
- It's better to have fewer highly-relevant sources than many loosely-related ones
- You may select 0 documents if none are truly relevant

For each selected document, briefly explain how it will be used.

Return ONLY valid JSON (no markdown code blocks):
{{
    "selected_indices": [0, 2, 5],
    "reasoning": {{
        "0": "Provides key statistics on X",
        "2": "Contains methodology details",
        "5": "Includes case study relevant to Y"
    }},
    "rejected_reason": "Documents 1,3,4,6,7,8,9 were not selected because..."
}}

If no documents are relevant: {{"selected_indices": [], "reasoning": {{}}, "rejected_reason": "None of the candidates directly address the topic"}}
"""

            response = await llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse LLM response
            import json
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                selected_indices = result.get("selected_indices", [])
                reasoning = result.get("reasoning", {})

                # Filter candidates based on LLM selection
                filtered = []
                for idx in selected_indices:
                    if 0 <= idx < len(candidates):
                        candidate = candidates[idx].copy()
                        # Add LLM's usage description
                        if str(idx) in reasoning:
                            candidate["usage_description"] = reasoning[str(idx)]
                        filtered.append(candidate)

                logger.info(
                    "LLM source filtering complete",
                    candidates_count=len(candidates),
                    selected_count=len(filtered),
                    selected_indices=selected_indices,
                )
                return filtered

            # Fallback to top 5 if parsing fails
            logger.warning("Could not parse LLM filter response, using top 5 candidates")
            return candidates[:5]

        except Exception as e:
            logger.warning(f"LLM source filtering failed: {e}, using top 5 candidates")
            return candidates[:5]

    async def _search_with_knowledge_graph(
        self,
        query: str,
        collection_filter: Optional[str],
        limit: int,
    ) -> List[dict]:
        """Search using knowledge graph for entity-based retrieval.

        This finds documents through entity relationships that might not
        have high vector similarity but are semantically related through
        entities mentioned in the query.
        """
        try:
            from backend.services.knowledge_graph import get_knowledge_graph_service
            from backend.db.database import async_session_context

            kg_service = get_knowledge_graph_service()

            async with async_session_context() as session:
                # Get graph-enhanced context
                graph_context = await kg_service.get_graph_rag_context(
                    session=session,
                    query=query,
                    top_k=limit,
                )

                if not graph_context or not graph_context.chunks:
                    return []

                # Convert chunks to search result format
                results = []
                for chunk in graph_context.chunks:
                    # Get document info
                    doc_name = "Unknown"
                    if chunk.document:
                        doc_name = chunk.document.filename or chunk.document.title or "Unknown"

                    results.append({
                        "content": chunk.content,
                        "document_name": doc_name,
                        "source": doc_name,
                        "chunk_id": str(chunk.id),
                        "score": 0.8,  # Graph-based results get good score
                        "metadata": {
                            "document_id": str(chunk.document_id) if chunk.document_id else None,
                            "page_number": chunk.page_number,
                            "source": doc_name,
                            # Mark as graph-enhanced
                            "from_knowledge_graph": True,
                        },
                    })

                return results

        except ImportError:
            logger.debug("Knowledge graph service not available")
            return []
        except Exception as e:
            logger.debug(f"Knowledge graph search failed: {e}")
            return []

    def _build_source_context(self, sources: Optional[List["SourceReference"]]) -> str:
        """Build context from sources."""
        if not sources:
            return ""

        context = "Use the following information:\n\n"
        for source in sources:
            context += f"- {source.snippet}\n\n"
        return context

    def _get_format_instructions(
        self,
        output_format: "OutputFormat",
        template_analysis: Optional["TemplateAnalysis"] = None,
        section_order: int = 0,
        total_sections: int = 5,
        slide_constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get format-specific generation instructions.

        Args:
            output_format: The output format (PPTX, DOCX, etc.)
            template_analysis: Optional template analysis with constraints
            section_order: The position of this section (0-indexed)
            total_sections: Total number of sections in the document
            slide_constraints: Optional per-slide constraints from TemplateLayoutLearner
        """
        from ..models import OutputFormat

        if output_format == OutputFormat.PPTX:
            return self._get_pptx_instructions(template_analysis, section_order, total_sections, slide_constraints)

        elif output_format in (OutputFormat.DOCX, OutputFormat.PDF):
            return self._get_docx_pdf_instructions(template_analysis, section_order, total_sections)

        elif output_format == OutputFormat.XLSX:
            return self._get_xlsx_instructions(template_analysis)

        else:
            return self._get_default_instructions(template_analysis)

    def _get_pptx_instructions(
        self,
        template_analysis: Optional["TemplateAnalysis"],
        section_order: int,
        total_sections: int,
        slide_constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get PPTX-specific instructions with slide-aware constraints.

        Args:
            template_analysis: Optional template analysis with general constraints
            section_order: Position of this section (0-indexed)
            total_sections: Total number of sections
            slide_constraints: Optional per-slide constraints from TemplateLayoutLearner
                Contains: title_constraints, content_constraints, layout, avoid_zones
        """
        # Priority: slide_constraints > template_analysis > defaults
        max_bullets = 7
        max_bullet_chars = 120  # PHASE 11: Increased from 70 to allow complete sentences
        max_title_chars = 50
        layout_type = "content"

        # First, check for per-slide constraints from TemplateLayoutLearner
        if slide_constraints:
            title_constraints = slide_constraints.get('title_constraints', {})
            content_constraints = slide_constraints.get('content_constraints', {})
            layout = slide_constraints.get('layout')

            max_title_chars = title_constraints.get('max_chars', max_title_chars)
            max_bullets = content_constraints.get('max_bullets', max_bullets)
            max_bullet_chars = content_constraints.get('max_bullet_chars', max_bullet_chars)

            if layout:
                layout_type = getattr(layout, 'layout_type', 'content')

            logger.debug(
                "Using per-slide constraints",
                section_order=section_order,
                max_title_chars=max_title_chars,
                max_bullets=max_bullets,
                max_bullet_chars=max_bullet_chars,
                layout_type=layout_type,
            )

        # Fallback to template_analysis constraints
        elif template_analysis and template_analysis.constraints:
            max_bullets = template_analysis.constraints.bullets_per_slide
            max_bullet_chars = template_analysis.constraints.bullet_max_chars
            max_title_chars = template_analysis.constraints.title_max_chars

        # Determine slide type based on layout_type from template learner OR position
        is_intro = section_order == 0
        is_conclusion = section_order >= total_sections - 1
        is_middle = not is_intro and not is_conclusion

        # Adjust constraints based on slide type and layout_type
        slide_type_guidance = ""

        # Use layout_type from template learner if available
        if layout_type == 'title':
            slide_type_guidance = """
SLIDE TYPE: TITLE SLIDE
- This is a TITLE SLIDE - keep it impactful and minimal
- Use a powerful, concise headline
- Optional subtitle (keep short)
- NO bullet points on title slides - just the title"""
            max_bullets = 0  # No bullets for title slides
        elif layout_type == 'section_header':
            slide_type_guidance = """
SLIDE TYPE: SECTION HEADER
- This marks a new section transition
- Clear section title that introduces what follows
- Brief description (1-2 lines maximum)
- Keep it simple to transition smoothly"""
            max_bullets = min(max_bullets, 3)
        elif layout_type == 'two_column':
            slide_type_guidance = """
SLIDE TYPE: TWO-COLUMN LAYOUT
- Content should be split into two balanced columns
- Mark columns with LEFT: and RIGHT: prefixes
- Each column should have similar content volume
- 3-4 bullets per column works best"""
            max_bullets = min(max_bullets, 8)  # 4 per column
        elif layout_type == 'image_text':
            slide_type_guidance = """
SLIDE TYPE: IMAGE + TEXT LAYOUT
- Image will be placed on one side automatically
- Keep text concise - bullets should complement the visual
- Fewer bullets (3-5) work best with images
- Focus on key points that the image supports"""
            max_bullets = min(max_bullets, 5)
        elif is_intro:
            slide_type_guidance = """
SLIDE TYPE: INTRODUCTION/OVERVIEW
- This is an opening section - set context and introduce key themes
- Keep points high-level and engaging
- 4-6 bullet points is ideal for introduction slides
- Focus on "why this matters" rather than details"""
            max_bullets = min(max_bullets, 6)  # Fewer bullets for intro
        elif is_conclusion:
            slide_type_guidance = """
SLIDE TYPE: CONCLUSION/SUMMARY
- This is a closing section - summarize and provide takeaways
- Focus on key insights and actionable recommendations
- 4-6 bullet points summarizing main points
- End with a strong call-to-action or next steps"""
            max_bullets = min(max_bullets, 6)  # Fewer bullets for conclusion
        else:
            slide_type_guidance = """
SLIDE TYPE: CONTENT/DETAIL
- This is a detail section - provide substantive information
- Include specific data, examples, or analysis
- 5-7 bullet points with supporting details"""

        # Add template theme context if available
        theme_context = ""
        if template_analysis and template_analysis.theme_profile:
            theme_context = f"""
TEMPLATE THEME CONTEXT:
- This presentation uses a professional template
- Primary brand color: {template_analysis.theme_profile.primary or 'Blue'}
- Style: {template_analysis.theme_profile.description or 'Corporate/Professional'}
- Match the tone and style expected for this template
"""

        return f"""FORMAT REQUIREMENTS FOR PRESENTATION SLIDES:
{slide_type_guidance}

BULLET POINT REQUIREMENTS:
- Write {max_bullets-2} to {max_bullets} bullet points
- Each bullet MUST be a COMPLETE, grammatically correct sentence
- Typical length: 60-{max_bullet_chars} characters per bullet
- NEVER truncate or cut off a sentence mid-thought
- If a point needs more than {max_bullet_chars} chars, SPLIT into two complete bullets
- Start each bullet with an action verb or key noun
- Use "• " for main points, "  ◦ " (2-space indent) for sub-points
- NO markdown formatting (no **, no ##, no _)
{theme_context}

CONTENT QUALITY REQUIREMENTS (CRITICAL):
- Include SPECIFIC details: names, numbers, percentages, dates from sources
- Avoid vague phrases like "enhances experience" or "improves performance"
- Be CONCRETE: "Real Madrid partnership reached 2M fans in 2025"
- Each bullet = one complete idea that stands alone
- Use data from the provided context whenever possible

CRITICAL OUTPUT RULES:
1. Start DIRECTLY with the first bullet point - NO introductory text
2. Do NOT include phrases like "Here are the bullet points" or "Let me provide"
3. Do NOT include closing remarks
4. ONLY output the bullet points themselves

CONSTRAINTS:
- Section titles: max {max_title_chars} characters
- Bullet points: 60-{max_bullet_chars} characters EACH (complete sentences only)
- Total bullets: {max_bullets-2} to {max_bullets} per slide

Example of GOOD bullets (complete, specific):
• Real Madrid partnership drove 25% increase in brand awareness among fans.
  ◦ Social media engagement grew 150% during campaign period.
• Stadium activation at Santiago Bernabéu reached 80,000 attendees.
• Product sampling achieved 35% conversion rate at match day events.

Example of BAD bullets (incomplete, vague - DO NOT WRITE LIKE THIS):
• Partner with popular football (INCOMPLETE - cut off)
• Campaign objective is (INCOMPLETE - cut off)
• Enhances brand experience (VAGUE - no specifics)"""

    def _get_docx_pdf_instructions(
        self,
        template_analysis: Optional["TemplateAnalysis"],
        section_order: int,
        total_sections: int,
    ) -> str:
        """Get DOCX/PDF-specific instructions with template awareness."""
        # Get constraints from template analysis or use defaults
        max_heading_chars = 80
        max_paragraph_chars = 1000
        target_words = "200-400"

        if template_analysis and template_analysis.constraints:
            max_heading_chars = template_analysis.constraints.heading_max_chars
            max_paragraph_chars = template_analysis.constraints.paragraph_max_chars

        # Adjust based on section position
        section_guidance = ""
        if section_order == 0:
            section_guidance = """
SECTION TYPE: INTRODUCTION
- Set the context and introduce the topic
- Keep paragraphs focused and engaging
- 2-3 paragraphs is ideal for introductions"""
            target_words = "150-250"
        elif section_order >= total_sections - 1:
            section_guidance = """
SECTION TYPE: CONCLUSION
- Summarize key findings and provide recommendations
- Focus on actionable takeaways
- 2-3 concise paragraphs"""
            target_words = "150-250"
        else:
            section_guidance = """
SECTION TYPE: BODY CONTENT
- Provide detailed analysis and information
- Include relevant examples and data
- 3-5 well-structured paragraphs"""
            target_words = "250-400"

        # Add template theme context if available
        theme_context = ""
        if template_analysis and template_analysis.theme_profile:
            theme_context = f"""
DOCUMENT STYLE:
- Tone: {template_analysis.theme_profile.description or 'Professional'}
- Match the formality expected for this document type
"""

        return f"""FORMAT REQUIREMENTS FOR DOCUMENT:
{section_guidance}
{theme_context}
- Write 3-5 well-structured paragraphs
- Use clear topic sentences for each paragraph
- Include relevant details and examples
- Maintain professional, formal tone
- Use **bold** for key terms (will be styled)
- Use _italic_ for emphasis (will be styled)
- Avoid excessive bullet points - prefer prose
- Target {target_words} words per section

CONSTRAINTS:
- Heading: max {max_heading_chars} characters
- Paragraphs: max {max_paragraph_chars} characters each"""

    def _get_xlsx_instructions(
        self,
        template_analysis: Optional["TemplateAnalysis"],
    ) -> str:
        """Get XLSX-specific instructions with template awareness."""
        max_cell_chars = 256

        if template_analysis and template_analysis.constraints:
            max_cell_chars = getattr(template_analysis.constraints, 'cell_max_chars', 256)

        return f"""FORMAT REQUIREMENTS FOR SPREADSHEET:
- Write concise, data-oriented content
- Use short sentences that fit in cells (max {max_cell_chars} chars per cell)
- NO markdown formatting (no **, no ##, no _)
- Focus on quantifiable information
- Use clear, structured points
- Each point on a new line
- Target 5-10 key points
- Avoid long paragraphs
- Format data for easy column/row organization"""

    def _get_default_instructions(
        self,
        template_analysis: Optional["TemplateAnalysis"],
    ) -> str:
        """Get default instructions for other formats."""
        theme_context = ""
        if template_analysis and template_analysis.theme_profile:
            theme_context = f"\nStyle: {template_analysis.theme_profile.description or 'Professional'}\n"

        return f"""Write clear, well-structured content.
{theme_context}Include relevant details and maintain a professional tone."""

    def _get_position_context(self, job: "GenerationJob", order: int) -> str:
        """Get context based on section position in document."""
        total_sections = len(job.outline.sections) if job.outline else 5
        current_section_num = order + 1  # order is 0-indexed

        if current_section_num <= 2:
            return "This is an EARLY section - set the stage and introduce key concepts."
        elif current_section_num >= total_sections - 1:
            return "This is a CONCLUDING section - summarize key points and provide actionable takeaways."
        else:
            return "This is a MIDDLE section - dive into details, analysis, and supporting information."

    def _build_style_context(self, job: "GenerationJob") -> str:
        """Build style context from job metadata.

        Args:
            job: The generation job containing metadata with optional style_guide.
                 style_guide can be either a StyleProfile dataclass or a dict.
        """
        style_guide = job.metadata.get("style_guide") if job.metadata else None
        if not style_guide:
            return ""

        # Handle both StyleProfile dataclass and dict
        if hasattr(style_guide, 'tone'):
            # It's a StyleProfile dataclass
            tone = getattr(style_guide, 'tone', 'professional')
            vocabulary = getattr(style_guide, 'vocabulary_level', 'moderate')
            structure = getattr(style_guide, 'structure_pattern', 'mixed')
            sentence_style = getattr(style_guide, 'sentence_style', 'medium')
            recommended_approach = getattr(style_guide, 'recommended_approach', None)
        elif isinstance(style_guide, dict):
            # It's a dict
            tone = style_guide.get('tone', 'professional')
            vocabulary = style_guide.get('vocabulary_level', 'moderate')
            structure = style_guide.get('structure_pattern', 'mixed')
            sentence_style = style_guide.get('sentence_style', 'medium')
            recommended_approach = style_guide.get('recommended_approach')
        else:
            return ""

        approach_line = f"Approach: {recommended_approach}" if recommended_approach else ""

        return f"""
---INTERNAL STYLE GUIDANCE (follow these rules but do NOT include them in your output)---
Tone: {tone}
Vocabulary: {vocabulary}
Structure: {structure}
Sentence style: {sentence_style}
{approach_line}
Match the style and tone of existing documents. Do NOT output these instructions as content.
---END INTERNAL GUIDANCE---
"""

    def _build_language_instruction(self, output_language: str, job: "GenerationJob") -> str:
        """Build language instruction for content generation."""
        if output_language == "auto":
            return self._build_auto_language_instruction(job)
        elif output_language == "en":
            return """
---LANGUAGE REQUIREMENT---
OUTPUT LANGUAGE: English

IMPORTANT: The user has explicitly selected ENGLISH as the output language.
- Generate ALL content in English.
- Even if the title/topic contains words from other languages (Hindi, German, etc.), output must be in English.
- Translate any non-English terms to English.
- Do NOT use Hinglish, Hindi, or any other language - only English.
---END LANGUAGE REQUIREMENT---
"""
        else:
            language_name = LANGUAGE_NAMES.get(output_language, output_language.upper())
            return f"""
---LANGUAGE REQUIREMENT---
OUTPUT LANGUAGE: {language_name}

IMPORTANT: The user has explicitly selected {language_name.upper()} as the output language.
- Generate ALL content in {language_name}.
- Even if the title/topic contains words from other languages, output must be in {language_name}.
- Translate any terms from other languages to {language_name}.
- Do NOT use any other language - only {language_name}.
- Technical terms may remain in English if commonly used that way in {language_name}.
---END LANGUAGE REQUIREMENT---
"""

    def _build_auto_language_instruction(self, job: "GenerationJob") -> str:
        """Build auto-detect language instruction.

        Priority for language detection:
        1. User's prompt/description (what they typed)
        2. Document title
        3. Default to English if unclear

        This ensures the output matches what the user wrote, not the source documents.
        """
        detected_lang = "en"
        detected_lang_name = "English"
        detection_confidence = 0.0

        try:
            from langdetect import detect_langs, DetectorFactory
            DetectorFactory.seed = 0  # Make deterministic

            # PRIORITY: Detect from user's prompt/description FIRST
            # This is what the user actually typed, so it should determine output language
            user_prompt = job.description or ''

            # Also check title, but give less weight
            title_text = job.title or ''

            # Combine with description weighted more heavily (repeat it for weight)
            # The user's prompt should be the primary signal
            text_to_detect = f"{user_prompt} {user_prompt} {title_text}"

            # If no description, fall back to title only
            if not user_prompt.strip():
                text_to_detect = title_text

            detected_results = detect_langs(text_to_detect)

            if detected_results:
                top_result = detected_results[0]
                detected_lang = top_result.lang
                detection_confidence = top_result.prob

                lang_map = {
                    "en": "English", "de": "German", "es": "Spanish", "fr": "French",
                    "it": "Italian", "pt": "Portuguese", "zh-cn": "Chinese", "zh-tw": "Chinese",
                    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
                    "ru": "Russian", "nl": "Dutch", "pl": "Polish", "tr": "Turkish"
                }
                detected_lang_name = lang_map.get(detected_lang, detected_lang.upper())

                logger.info(
                    "Auto-detected language from title",
                    detected_lang=detected_lang,
                    detected_lang_name=detected_lang_name,
                    confidence=f"{detection_confidence:.2%}",
                    text_sample=text_to_detect[:100]
                )

                # If confidence is low (< 80%), fall back to English
                if detection_confidence < 0.80:
                    logger.warning(
                        "Low confidence language detection, defaulting to English",
                        detected_lang=detected_lang,
                        confidence=f"{detection_confidence:.2%}"
                    )
                    detected_lang = "en"
                    detected_lang_name = "English"

                # Additional safeguard for ASCII text detected as non-Latin
                ascii_ratio = sum(1 for c in text_to_detect if c.isascii()) / max(len(text_to_detect), 1)
                non_latin_langs = {"hi", "ar", "zh", "zh-cn", "zh-tw", "ja", "ko", "ru"}
                if ascii_ratio > 0.90 and detected_lang in non_latin_langs:
                    logger.warning(
                        "Text appears to be primarily ASCII but detected as non-Latin language, defaulting to English",
                        detected_lang=detected_lang,
                        ascii_ratio=f"{ascii_ratio:.2%}",
                    )
                    detected_lang = "en"
                    detected_lang_name = "English"

        except Exception as e:
            logger.warning(f"Language detection failed, using English: {e}")

        # Check for Hinglish
        title_lower = job.title.lower()
        hinglish_markers = ["ke", "ka", "ki", "hai", "ko", "se", "mein", "par", "aur", "liye", "kaise"]
        hinglish_word_count = sum(1 for marker in hinglish_markers if marker in title_lower.split())
        has_hinglish = hinglish_word_count >= 2

        if has_hinglish and detected_lang == "hi" and detection_confidence >= 0.70:
            return f"""
---CRITICAL LANGUAGE REQUIREMENT---
DETECTED LANGUAGE: Hinglish (Hindi/English mix)
The document title "{job.title}" is in Hinglish.

YOU MUST:
1. Write ALL content in Hinglish (Hindi words written in Roman script mixed with English)
2. Example: "Marketing strategy ko implement karna" not "Implementing marketing strategy"
3. DO NOT write in pure English!
4. DO NOT write in Devanagari script - use Roman letters only
5. If source documents are in German/English/other, TRANSLATE to Hinglish
6. Every slide/section MUST be in Hinglish - no exceptions!
---END LANGUAGE REQUIREMENT---
"""
        else:
            if detected_lang == "en":
                return """
---LANGUAGE REQUIREMENT---
OUTPUT LANGUAGE: English

Auto-detection defaulted to ENGLISH. Generate ALL content in English.
- Do NOT use Hinglish, Hindi, or any other language
- Even if the title contains words from other languages, output must be in English
- Translate any non-English terms to English concepts
- Technical terms may remain as-is if commonly used in English
---END LANGUAGE REQUIREMENT---
"""
            else:
                return f"""
---LANGUAGE REQUIREMENT---
OUTPUT LANGUAGE: {detected_lang_name}

Auto-detected language: {detected_lang_name}. Generate ALL content in {detected_lang_name}.
If source documents are in a different language, translate them to {detected_lang_name}.
Keep technical terms and proper nouns in their original form if commonly used that way.
---END LANGUAGE REQUIREMENT---
"""

    def _build_content_prompt(
        self,
        job: "GenerationJob",
        section_title: str,
        section_description: str,
        context: str,
        format_instructions: str,
        position_context: str,
        style_context: str,
        language_instruction: str,
    ) -> str:
        """Build the full content generation prompt."""
        return f"""Write content for the following section:

Document Title: {job.title}
Section Title: {section_title}
Description: {section_description}

{position_context}

{context}

{format_instructions}
{style_context}
{language_instruction}"""

    async def _run_quality_review(
        self,
        content: str,
        section_title: str,
        job: "GenerationJob",
        sources: Optional[List["SourceReference"]],
        order: int,
        format_instructions: str,
    ) -> tuple:
        """Run quality review and optional auto-regeneration."""
        from backend.services.content_quality import ContentQualityScorer

        quality_report = None
        section_metadata = {}
        regeneration_attempts = 0

        min_quality_score = 0.7
        auto_regenerate = False
        max_regeneration_attempts = 1

        if self.config:
            min_quality_score = self.config.min_quality_score
            auto_regenerate = getattr(self.config, 'auto_regenerate_low_quality', False)
            max_regeneration_attempts = getattr(self.config, 'max_regeneration_attempts', 1)

        quality_scorer = ContentQualityScorer(min_score=min_quality_score)

        # Get other sections' content for consistency check
        other_sections_content = []
        for section in getattr(job, 'sections', []):
            if section.order != order and section.content:
                other_sections_content.append(section.content)

        # Build context for quality scoring
        quality_context = {
            "title": job.title,
            "description": job.description,
            "output_format": job.output_format.value if hasattr(job.output_format, 'value') else str(job.output_format),
        }

        # Convert sources to dict format for quality scorer
        sources_dicts = [{"snippet": s.snippet} for s in sources] if sources else None

        # Score the content
        quality_report = await quality_scorer.score_section(
            content=content,
            title=section_title,
            sources=sources_dicts,
            context=quality_context,
            other_sections=other_sections_content if other_sections_content else None,
        )

        logger.info(
            "Section quality scored",
            section_title=section_title,
            score=quality_report.overall_score,
            needs_revision=quality_report.needs_revision,
        )

        # Auto-regenerate if quality is too low
        if (quality_report.needs_revision and
            auto_regenerate and
            regeneration_attempts < max_regeneration_attempts):

            logger.info(
                "Auto-regenerating low quality section",
                section_title=section_title,
                score=quality_report.overall_score,
                issues=quality_report.critical_issues[:3],
            )

            # Create feedback from quality report
            feedback_items = quality_report.critical_issues + quality_report.improvements[:3]
            quality_feedback = "Quality issues to address:\n" + "\n".join(f"- {item}" for item in feedback_items)

            # Build language instruction for regeneration
            output_language = job.metadata.get("output_language", "en") if job.metadata else "en"
            language_instruction = self._build_language_instruction(output_language, job)

            regen_prompt = f"""Revise this content to improve quality:
{language_instruction}

ORIGINAL CONTENT:
{content}

QUALITY FEEDBACK:
{quality_feedback}

REQUIREMENTS:
- Address all the quality issues listed above
- Keep the same topic and key information
- Maintain the required format ({format_instructions[:200]}...)
- CRITICAL: Keep the content in the SAME LANGUAGE as the original!

Write the improved content:"""

            try:
                regeneration_attempts += 1
                llm = await self._get_llm()
                response = await llm.ainvoke(regen_prompt)
                improved_content = response.content

                # Re-score the improved content
                new_report = await quality_scorer.score_section(
                    content=improved_content,
                    title=section_title,
                    sources=sources_dicts,
                    context=quality_context,
                    other_sections=other_sections_content if other_sections_content else None,
                )

                if new_report.overall_score > quality_report.overall_score:
                    content = improved_content
                    quality_report = new_report
                    logger.info(
                        "Section quality improved after regeneration",
                        section_title=section_title,
                        new_score=new_report.overall_score,
                    )

            except Exception as e:
                logger.warning("Failed to regenerate section", error=str(e))

        return content, quality_report, section_metadata

    async def _review_with_critic(
        self,
        content: str,
        section_title: str,
        job: "GenerationJob",
        format_instructions: str,
        output_language: str = "en",
    ) -> tuple:
        """Use CriticAgent to review and auto-fix content issues."""
        import uuid as uuid_module

        metadata = {
            "critic_reviewed": True,
            "critic_score": None,
            "critic_feedback": None,
            "was_revised_by_critic": False,
        }

        quality_threshold = job.metadata.get("quality_threshold", 0.7) if job.metadata else 0.7
        fix_styling = job.metadata.get("fix_styling", True) if job.metadata else True
        fix_incomplete = job.metadata.get("fix_incomplete", True) if job.metadata else True

        try:
            from backend.services.agents.worker_agents import CriticAgent
            from backend.services.agents.agent_base import AgentConfig

            critic_config = AgentConfig(
                agent_id=str(uuid_module.uuid4()),
                name="Document Critic",
                description="Reviews generated content for quality, styling, and completeness",
            )
            critic = CriticAgent(critic_config)

            # Build evaluation criteria based on settings
            criteria = ["accuracy", "clarity", "relevance"]
            if fix_styling:
                criteria.append("formatting")
            if fix_incomplete:
                criteria.append("completeness")

            evaluation = await critic.evaluate(
                content=content,
                original_request=f"Section '{section_title}' for document '{job.title}': {job.description}",
                criteria=criteria,
            )

            # Normalize score to 0-1 range (critic returns 1-5)
            normalized_score = evaluation.overall_score / 5.0
            metadata["critic_score"] = normalized_score
            metadata["critic_feedback"] = evaluation.feedback

            logger.info(
                "CriticAgent reviewed section",
                section_title=section_title,
                score=normalized_score,
                passed=evaluation.passed,
                threshold=quality_threshold,
            )

            # Auto-fix if below threshold
            if normalized_score < quality_threshold and evaluation.improvements_needed:
                logger.info(
                    "CriticAgent auto-fixing low quality section",
                    section_title=section_title,
                    score=normalized_score,
                    improvements=evaluation.improvements_needed[:3],
                )

                feedback_text = "\n".join(f"- {item}" for item in evaluation.improvements_needed[:5])

                # Build language instruction for critic fix
                critic_language_instruction = self._build_critic_language_instruction(
                    output_language, job
                )

                fix_prompt = f"""Improve this content based on the following feedback:
{critic_language_instruction}

ORIGINAL CONTENT:
{content}

QUALITY ISSUES TO ADDRESS:
{feedback_text}

{"FORMATTING ISSUES: Fix any styling or formatting problems" if fix_styling else ""}
{"COMPLETENESS ISSUES: Complete any incomplete sentences or bullet points" if fix_incomplete else ""}

REQUIREMENTS:
- Address all the quality issues listed above
- Maintain the same topic and key information
- Follow format requirements: {format_instructions[:300]}
- CRITICAL: Keep the content in the SAME LANGUAGE as the original!

Write the improved content:"""

                try:
                    llm = await self._get_llm()
                    response = await llm.ainvoke(fix_prompt)
                    improved_content = response.content

                    # Re-evaluate to confirm improvement
                    new_evaluation = await critic.evaluate(
                        content=improved_content,
                        original_request=f"Section '{section_title}' for document '{job.title}'",
                        criteria=criteria,
                    )
                    new_score = new_evaluation.overall_score / 5.0

                    if new_score > normalized_score:
                        content = improved_content
                        metadata["critic_score"] = new_score
                        metadata["was_revised_by_critic"] = True
                        logger.info(
                            "CriticAgent improved section quality",
                            section_title=section_title,
                            old_score=normalized_score,
                            new_score=new_score,
                        )

                except Exception as e:
                    logger.warning("CriticAgent failed to fix content", error=str(e))

        except Exception as e:
            logger.warning("CriticAgent review failed", error=str(e))
            metadata["critic_reviewed"] = False

        return content, metadata

    async def _run_fact_check(
        self,
        content: str,
        section_title: str,
        sources: Optional[List["SourceReference"]],
        fact_check_level: str = "standard",
    ) -> tuple:
        """
        Run fact-checking on generated content.

        Phase 2 enhancement: Reduces hallucinations by 25-40% by verifying
        claims against source documents.

        Args:
            content: Generated content to verify
            section_title: Title of the section
            sources: Source references used for generation
            fact_check_level: "off", "standard", or "strict"

        Returns:
            Tuple of (potentially revised content, FactCheckReport)
        """
        from .fact_checker import get_fact_checker, FactCheckReport

        try:
            fact_checker = get_fact_checker()

            # Convert sources to dict format for fact checker
            source_dicts = []
            if sources:
                for s in sources:
                    source_dicts.append({
                        "snippet": s.snippet,
                        "content": getattr(s, 'content', '') or s.snippet,
                        "document_name": s.document_name,
                    })

            # Run enhanced fact check with evidence aggregation and hallucination detection
            report = await fact_checker.enhanced_check_facts(
                content=content,
                sources=source_dicts,
                context={"section_title": section_title},
            )

            logger.info(
                "Fact-check completed for section",
                section_title=section_title,
                total_claims=report.total_claims,
                verified=report.verified_claims,
                verification_rate=f"{report.verification_rate:.1%}",
                needs_revision=report.needs_revision,
            )

            # In strict mode, attempt to revise unverified claims
            if fact_check_level == "strict" and report.needs_revision and report.revision_suggestions:
                logger.info(
                    "Strict fact-check: attempting to revise unverified claims",
                    section_title=section_title,
                    unverified_count=report.unverified_claims,
                )

                content = await self._revise_for_fact_accuracy(
                    content=content,
                    fact_report=report,
                    sources=source_dicts,
                )

                # Re-run enhanced fact check on revised content
                revised_report = await fact_checker.enhanced_check_facts(
                    content=content,
                    sources=source_dicts,
                    context={"section_title": section_title},
                )

                if revised_report.verification_rate > report.verification_rate:
                    logger.info(
                        "Fact-check revision improved accuracy",
                        original_rate=f"{report.verification_rate:.1%}",
                        new_rate=f"{revised_report.verification_rate:.1%}",
                    )
                    report = revised_report

            return content, report

        except Exception as e:
            logger.warning("Fact-checking failed", error=str(e), section_title=section_title)
            return content, None

    async def _revise_for_fact_accuracy(
        self,
        content: str,
        fact_report: "FactCheckReport",
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        Revise content to improve factual accuracy.

        Args:
            content: Original content
            fact_report: Fact-check report with unverified claims
            sources: Source documents for grounding

        Returns:
            Revised content with improved accuracy
        """
        try:
            llm = await self._get_llm()

            # Build revision prompt
            unverified_claims = [
                v.claim for v in fact_report.verifications if not v.verified
            ][:5]  # Top 5 unverified

            source_context = "\n\n".join(
                s.get("snippet", s.get("content", ""))[:500]
                for s in sources[:5]
            )

            revision_prompt = f"""Revise the following content to improve factual accuracy.

ORIGINAL CONTENT:
{content}

UNVERIFIED CLAIMS (please revise or remove these):
{chr(10).join(f"- {claim}" for claim in unverified_claims)}

VERIFIED SOURCE INFORMATION:
{source_context}

INSTRUCTIONS:
1. Revise any claims that cannot be supported by the source documents
2. Replace unverified statistics/numbers with information from sources
3. If a claim cannot be verified, either rephrase it more generally or remove it
4. Keep the overall structure and flow of the content
5. Maintain the same tone and style

Write the revised content:"""

            response = await llm.ainvoke(revision_prompt)
            return response.content

        except Exception as e:
            logger.warning("Content revision for accuracy failed", error=str(e))
            return content

    def _build_critic_language_instruction(self, output_language: str, job: "GenerationJob") -> str:
        """Build language instruction for critic fixes."""
        if output_language == "auto":
            title_lower = job.title.lower()
            hinglish_markers = ["ke", "ka", "ki", "hai", "ko", "se", "mein", "par", "aur", "liye", "kaise"]
            has_hinglish = any(marker in title_lower.split() for marker in hinglish_markers)

            if has_hinglish:
                return f"""
---CRITICAL LANGUAGE REQUIREMENT---
The document title "{job.title}" is in Hinglish.
YOU MUST keep the improved content in Hinglish (Hindi+English mix, Roman script).
DO NOT translate to pure English or German!
Every sentence must be in Hinglish - no exceptions!
---END LANGUAGE REQUIREMENT---
"""
            else:
                return f"""
---CRITICAL LANGUAGE REQUIREMENT---
The document title is: "{job.title}"
Keep the improved content in the EXACT SAME LANGUAGE as the original content.
DO NOT translate to English or any other language!
Maintain language consistency throughout.
---END LANGUAGE REQUIREMENT---
"""
        elif output_language != "en":
            language_name = LANGUAGE_NAMES.get(output_language, "English")
            return f"""
---CRITICAL LANGUAGE REQUIREMENT---
Keep ALL content in {language_name}.
DO NOT translate to English - maintain {language_name} throughout.
Every sentence must be in {language_name} - no exceptions!
---END LANGUAGE REQUIREMENT---
"""
        return ""

    async def generate_all(
        self,
        job: "GenerationJob",
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> List["Section"]:
        """Generate content for all sections in a job.

        Tracks chunk IDs across sections to avoid repeating the same
        source material in different sections, ensuring each section
        draws from distinct content.

        Args:
            job: The generation job with sections to generate
            template_analysis: Optional template analysis for constraints

        Returns:
            List of sections with generated content
        """
        generated_sections = []
        used_chunk_ids: set = set()

        for i, section_info in enumerate(job.outline.sections if job.outline else []):
            if isinstance(section_info, dict):
                title = section_info.get("title", f"Section {i+1}")
                description = section_info.get("description", "")
            else:
                title = getattr(section_info, 'title', f"Section {i+1}")
                description = getattr(section_info, 'description', "")

            section = await self.generate_section(
                job=job,
                section_title=title,
                section_description=description,
                order=i,
                template_analysis=template_analysis,
                used_chunk_ids=used_chunk_ids,
            )
            generated_sections.append(section)

            # Track chunks used by this section so subsequent sections get fresh content
            for source in (section.sources or []):
                if source.chunk_id:
                    used_chunk_ids.add(source.chunk_id)

        return generated_sections

    async def regenerate_with_feedback(
        self,
        section: "Section",
        job: "GenerationJob",
        feedback: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> "Section":
        """Regenerate section content with user feedback.

        Args:
            section: The section to regenerate
            job: The generation job
            feedback: User feedback for regeneration
            template_analysis: Optional template analysis for constraints

        Returns:
            Updated section with regenerated content
        """
        from ..models import Section

        # Build cross-section context for consistency
        other_sections_context = self._build_cross_section_context(job, section)

        # Build source material context for accuracy
        source_context = self._build_source_material_context(section)

        # Determine format-specific guidelines
        format_guidelines = self._get_format_guidelines(job.output_format)

        # Build enhanced revision prompt
        prompt = f"""You are revising a section of a {job.output_format.value.upper() if hasattr(job.output_format, 'value') else str(job.output_format).upper()} document.

# DOCUMENT CONTEXT
Title: {job.title}
Description: {job.description}
Target Audience: {job.outline.target_audience if job.outline else 'General professional audience'}
Tone: {job.outline.tone if job.outline else 'Professional'}
Output Format: {job.output_format.value.upper() if hasattr(job.output_format, 'value') else str(job.output_format).upper()}

# CURRENT SECTION TO REVISE
Section Title: {section.title}
Section Order: {section.order + 1} of {len(job.sections)}

Current Content:
---
{section.content}
---

# USER FEEDBACK
{feedback}

{other_sections_context}

{source_context}

# FORMAT-SPECIFIC GUIDELINES
{format_guidelines}

# QUALITY REQUIREMENTS
Please revise the content ensuring:
1. **Address the feedback directly** - Make specific changes requested
2. **Maintain consistency** - Use similar terminology and tone as other sections
3. **Preserve accuracy** - Keep source-backed claims accurate
4. **Improve clarity** - Ensure content is clear and well-structured
5. **Keep appropriate length** - Match the original section length unless feedback requests changes
6. **Smooth transitions** - Ensure logical flow from previous section

# OUTPUT
Provide ONLY the revised section content. Do not include the section title or any meta-commentary.
Write the content ready for direct use in the document."""

        try:
            llm = await self._get_llm()
            response = await llm.ainvoke(prompt)

            return Section(
                id=section.id,
                title=section.title,
                content=response.content,
                order=section.order,
                sources=section.sources,
                approved=False,
            )

        except Exception as e:
            logger.error("Failed to revise section", error=str(e))
            return section

    def _build_cross_section_context(
        self,
        job: "GenerationJob",
        current_section: "Section",
    ) -> str:
        """Build context from other sections for consistency."""
        context_parts = []

        # Get previous and next sections for flow
        prev_section = None
        next_section = None
        for i, s in enumerate(job.sections):
            if s.id == current_section.id:
                if i > 0:
                    prev_section = job.sections[i - 1]
                if i < len(job.sections) - 1:
                    next_section = job.sections[i + 1]
                break

        if prev_section or next_section:
            context_parts.append("# ADJACENT SECTIONS (for context and consistency)")

            if prev_section:
                prev_content = prev_section.revised_content or prev_section.content
                preview = prev_content[-300:] if len(prev_content) > 300 else prev_content
                context_parts.append(f"Previous Section ({prev_section.title}) ends with:")
                context_parts.append(f"...{preview}")

            if next_section:
                next_content = next_section.revised_content or next_section.content
                preview = next_content[:300] if len(next_content) > 300 else next_content
                context_parts.append(f"\nNext Section ({next_section.title}) begins with:")
                context_parts.append(f"{preview}...")

        # Extract key terminology from other sections
        all_content = []
        for s in job.sections:
            if s.id != current_section.id:
                content = s.revised_content or s.content
                all_content.append(content)

        if all_content:
            combined = " ".join(all_content)
            context_parts.append(f"\n# Document has {len(job.sections)} total sections with ~{len(combined.split())} words overall.")

        return "\n".join(context_parts) if context_parts else ""

    def _build_source_material_context(self, section: "Section") -> str:
        """Build source material context for accuracy checking."""
        if not section.sources:
            return ""

        context_parts = ["# SOURCE MATERIAL (maintain accuracy with these)"]

        for i, source in enumerate(section.sources[:5], 1):
            snippet = source.snippet[:200] if len(source.snippet) > 200 else source.snippet
            context_parts.append(f"[{i}] {source.document_name}: {snippet}")

        return "\n".join(context_parts)

    def _get_format_guidelines(self, output_format: "OutputFormat") -> str:
        """Get format-specific writing guidelines."""
        from ..models import OutputFormat

        guidelines = {
            OutputFormat.PPTX: """For PowerPoint slides:
- Keep bullet points concise (max 8-10 words each)
- Use 3-6 bullet points per slide typically
- Avoid long paragraphs
- Use action-oriented language
- Include speaker notes context if appropriate""",

            OutputFormat.DOCX: """For Word documents:
- Use clear paragraph structure
- Include smooth transitions between ideas
- Maintain consistent heading hierarchy
- Balance detail with readability""",

            OutputFormat.PDF: """For PDF reports:
- Structure content clearly with logical flow
- Use professional language
- Include appropriate detail for formal documents
- Consider visual hierarchy in text structure""",

            OutputFormat.MARKDOWN: """For Markdown:
- Use proper markdown formatting (headers, lists, code blocks)
- Keep structure clean and readable
- Use bullet points and numbered lists appropriately""",

            OutputFormat.HTML: """For HTML:
- Structure content semantically
- Use appropriate heading levels
- Keep paragraphs well-organized""",

            OutputFormat.XLSX: """For Excel:
- Structure data clearly in rows/columns
- Include headers and labels
- Keep text concise for cell content""",

            OutputFormat.TXT: """For plain text:
- Use clear structure and spacing
- Avoid relying on formatting
- Keep content well-organized without markup""",
        }

        return guidelines.get(output_format, "Write clear, professional content.")

    async def enhance_content(
        self,
        content: str,
        enhancement_type: str,
        template_analysis: Optional["TemplateAnalysis"] = None,
    ) -> str:
        """Enhance existing content.

        Args:
            content: The content to enhance
            enhancement_type: Type of enhancement (expand, shorten, formal, casual)
            template_analysis: Optional template analysis for constraints

        Returns:
            Enhanced content
        """
        llm = await self._get_llm()

        prompts = {
            "expand": f"Expand this content with more details and examples:\n\n{content}",
            "shorten": f"Shorten this content while keeping key points:\n\n{content}",
            "formal": f"Rewrite this content in a more formal tone:\n\n{content}",
            "casual": f"Rewrite this content in a more casual, conversational tone:\n\n{content}",
            "simplify": f"Simplify this content for a general audience:\n\n{content}",
            "technical": f"Add more technical depth to this content:\n\n{content}",
        }

        prompt = prompts.get(enhancement_type, prompts["expand"])

        # Add template constraints if available
        if template_analysis:
            constraints = template_analysis.constraints
            prompt += f"\n\nConstraints:\n- Max {constraints.body_max_chars} characters"

        response = await llm.ainvoke(prompt)
        enhanced = response.content if hasattr(response, 'content') else str(response)

        return enhanced.strip()
