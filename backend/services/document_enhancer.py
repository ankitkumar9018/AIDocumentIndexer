"""
AIDocumentIndexer - Document Enhancer Service
==============================================

LLM-based document context extraction for improved RAG search.
Analyzes documents to extract summaries, keywords, topics, entities,
and hypothetical questions for better semantic retrieval.
"""

import asyncio
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from uuid import UUID

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Document, Chunk, ProcessingStatus
from backend.db.database import async_session_context

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class EnhancedMetadata(BaseModel):
    """Enhanced metadata extracted from document via LLM analysis."""
    summary_short: str = Field(..., description="1-2 sentence summary")
    summary_detailed: str = Field(..., description="1 paragraph detailed summary")
    keywords: List[str] = Field(default_factory=list, description="5-10 key terms")
    topics: List[str] = Field(default_factory=list, description="2-5 main topics")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted entities: people, organizations, dates, locations"
    )
    hypothetical_questions: List[str] = Field(
        default_factory=list,
        description="3-5 questions this document answers"
    )
    language: str = Field(default="en", description="Detected language")
    document_type: str = Field(default="unknown", description="Document type classification")
    enhanced_at: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(default="gpt-4o-mini")


class EnhancementResult(BaseModel):
    """Result of enhancing a single document."""
    document_id: str
    success: bool
    metadata: Optional[EnhancedMetadata] = None
    error: Optional[str] = None
    tokens_used: int = 0


class BatchEnhancementResult(BaseModel):
    """Result of batch enhancement operation."""
    total: int
    successful: int
    failed: int
    results: List[EnhancementResult]
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class EnhancementJobStatus(BaseModel):
    """Status of an ongoing enhancement job."""
    job_id: str
    status: str  # pending, running, completed, failed
    total_documents: int
    processed_documents: int
    successful: int
    failed: int
    progress_percent: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_cost_usd: float = 0.0


# =============================================================================
# Enhancement Prompts
# =============================================================================

ENHANCEMENT_SYSTEM_PROMPT = """You are a document analysis assistant. Analyze the provided document content and extract structured metadata to improve search and retrieval.

Your task is to:
1. Create a short summary (1-2 sentences) capturing the main point
2. Create a detailed summary (1 paragraph) with key details
3. Extract 5-10 important keywords/terms
4. Identify 2-5 main topics covered
5. Extract named entities (people, organizations, dates, locations)
6. Generate 3-5 questions this document could answer
7. Classify the document type (report, article, manual, email, presentation, etc.)
8. Detect the primary language

Respond ONLY with valid JSON matching this schema:
{
    "summary_short": "string",
    "summary_detailed": "string",
    "keywords": ["string"],
    "topics": ["string"],
    "entities": {
        "people": ["string"],
        "organizations": ["string"],
        "dates": ["string"],
        "locations": ["string"]
    },
    "hypothetical_questions": ["string"],
    "language": "string (ISO code)",
    "document_type": "string"
}"""


# =============================================================================
# Document Enhancer Service
# =============================================================================

class DocumentEnhancer:
    """
    Service for extracting enhanced metadata from documents using LLM.

    Provides methods for:
    - Single document enhancement
    - Batch processing with progress tracking
    - Processing all unenhanced documents
    """

    def __init__(
        self,
        max_chunk_tokens: int = 4000,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize the document enhancer.

        LLM provider and model are configured via Admin UI (Operation-Level Config).
        Configure the "document_enhancement" operation in Admin > Settings > LLM Configuration.

        Args:
            max_chunk_tokens: Maximum tokens to send for analysis
            rate_limit_delay: Delay between API calls in seconds
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.rate_limit_delay = rate_limit_delay
        self.model = "unknown"  # Will be set when LLM is retrieved
        self.provider = "unknown"  # Will be set when LLM is retrieved

        # Token costs per 1M tokens (input/output) - for cost estimation
        self.token_costs = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            # Ollama models are free (local)
            "llama3.2": {"input": 0.0, "output": 0.0},
            "llama3.1": {"input": 0.0, "output": 0.0},
            "mistral": {"input": 0.0, "output": 0.0},
            "qwen2.5": {"input": 0.0, "output": 0.0},
            "deepseek-r1:8b": {"input": 0.0, "output": 0.0},
        }

    async def _get_llm(self):
        """Get LLM instance using database-driven configuration.

        The LLM provider/model is configured via Admin UI:
        Admin > Settings > LLM Configuration > Operation-Level Config > Document Enhancement
        """
        from backend.services.llm import EnhancedLLMFactory

        llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="document_enhancement",
            user_id=None,  # System-level operation
        )

        # Track which model is being used for logging and cost estimation
        self.model = config.model if config else "unknown"
        self.provider = config.provider_type if config else "unknown"

        logger.info(
            "Document enhancer using LLM",
            provider=self.provider,
            model=self.model,
        )

        return llm

    async def _get_document_content(
        self,
        db: AsyncSession,
        document_id: str,
    ) -> tuple[str, int]:
        """
        Get document content from chunks.

        Returns:
            Tuple of (content_text, approximate_token_count)
        """
        # Get chunks for this document, ordered by chunk index
        result = await db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        chunks = result.scalars().all()

        if not chunks:
            return "", 0

        # Combine chunk content, respecting token limit
        content_parts = []
        total_chars = 0
        max_chars = self.max_chunk_tokens * 4  # Rough char to token ratio

        for chunk in chunks:
            if total_chars + len(chunk.content) > max_chars:
                # Add truncation indicator
                remaining = max_chars - total_chars
                if remaining > 100:
                    content_parts.append(chunk.content[:remaining])
                    content_parts.append("\n[Content truncated...]")
                break
            content_parts.append(chunk.content)
            total_chars += len(chunk.content)

        content = "\n\n".join(content_parts)
        approx_tokens = len(content) // 4

        return content, approx_tokens

    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract JSON from LLM response, handling various formats.

        Handles:
        - Plain JSON
        - JSON in ```json ... ``` blocks
        - JSON in ``` ... ``` blocks
        - Multiple code blocks (takes first valid JSON)
        """
        if not response_text or not response_text.strip():
            raise ValueError("Empty response from LLM")

        response_text = response_text.strip()

        # Try to extract JSON from ```json ... ``` blocks first
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()

        # Try to extract from ``` ... ``` blocks
        code_block_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # If starts with ```, try line-by-line extraction
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Skip first line (```json or ```) and last line (```)
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()

        # Return as-is if no code blocks found
        return response_text

    async def _extract_metadata(
        self,
        content: str,
        document_name: str,
    ) -> tuple[EnhancedMetadata, int]:
        """
        Extract metadata from content using LLM.

        Returns:
            Tuple of (metadata, tokens_used)
        """
        import json

        llm = await self._get_llm()

        # Prepare the prompt
        user_prompt = f"""Document: {document_name}

Content:
{content}

Analyze this document and provide structured metadata as JSON."""

        # Call LLM
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=ENHANCEMENT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Retry logic for empty responses
        max_retries = 3
        response_text = ""

        for attempt in range(max_retries):
            response = await llm.ainvoke(messages)

            # Check if response has content
            if response.content and response.content.strip():
                response_text = response.content.strip()
                logger.debug(
                    "LLM response received",
                    document=document_name,
                    attempt=attempt + 1,
                    response_length=len(response_text),
                )
                break
            else:
                logger.warning(
                    "Empty LLM response, retrying",
                    document=document_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    raw_response=repr(response.content) if response.content else "None",
                )
                # Small delay before retry
                await asyncio.sleep(1.0)
        else:
            # All retries failed
            logger.error(
                "LLM returned empty response after all retries",
                document=document_name,
                retries=max_retries,
            )
            raise ValueError(f"LLM returned empty response after {max_retries} retries")

        # Parse response
        try:
            # Extract JSON from response (handles markdown blocks, etc.)
            json_text = self._extract_json_from_response(response_text)

            logger.debug(
                "Parsing JSON response",
                document=document_name,
                json_length=len(json_text),
                first_100_chars=json_text[:100] if json_text else "",
            )

            data = json.loads(json_text)

            metadata = EnhancedMetadata(
                summary_short=data.get("summary_short", ""),
                summary_detailed=data.get("summary_detailed", ""),
                keywords=data.get("keywords", []),
                topics=data.get("topics", []),
                entities=data.get("entities", {}),
                hypothetical_questions=data.get("hypothetical_questions", []),
                language=data.get("language", "en"),
                document_type=data.get("document_type", "unknown"),
                enhanced_at=datetime.now(),
                model_used=self.model,
            )

            # Estimate tokens used
            input_tokens = len(ENHANCEMENT_SYSTEM_PROMPT + user_prompt) // 4
            output_tokens = len(response_text) // 4
            tokens_used = input_tokens + output_tokens

            return metadata, tokens_used

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse LLM response as JSON",
                error=str(e),
                document=document_name,
                response_preview=response_text[:500] if response_text else "",
            )
            raise ValueError(f"Invalid JSON response from LLM: {e}")

    async def enhance_document(
        self,
        document_id: str,
        force: bool = False,
    ) -> EnhancementResult:
        """
        Enhance a single document with LLM-extracted metadata.

        Args:
            document_id: UUID of the document to enhance
            force: If True, re-enhance even if already enhanced

        Returns:
            EnhancementResult with success status and metadata
        """
        logger.info("Enhancing document", document_id=document_id, force=force)

        try:
            async with async_session_context() as db:
                # Get document
                result = await db.execute(
                    select(Document).where(Document.id == document_id)
                )
                document = result.scalar_one_or_none()

                if not document:
                    return EnhancementResult(
                        document_id=document_id,
                        success=False,
                        error="Document not found",
                    )

                # Check if already enhanced
                if document.enhanced_metadata and not force:
                    return EnhancementResult(
                        document_id=document_id,
                        success=True,
                        metadata=EnhancedMetadata(**document.enhanced_metadata),
                        error="Already enhanced (use force=True to re-enhance)",
                    )

                # Get document content
                content, input_tokens = await self._get_document_content(db, document_id)

                if not content:
                    return EnhancementResult(
                        document_id=document_id,
                        success=False,
                        error="No content found in document chunks",
                    )

                # Extract metadata
                metadata, tokens_used = await self._extract_metadata(
                    content,
                    document.filename or document.original_filename,
                )

                # Save to document (use mode="json" for JSON-serializable output)
                document.enhanced_metadata = metadata.model_dump(mode="json")
                await db.commit()

                logger.info(
                    "Document enhanced successfully",
                    document_id=document_id,
                    tokens_used=tokens_used,
                )

                return EnhancementResult(
                    document_id=document_id,
                    success=True,
                    metadata=metadata,
                    tokens_used=tokens_used,
                )

        except Exception as e:
            logger.error("Document enhancement failed", document_id=document_id, error=str(e))
            return EnhancementResult(
                document_id=document_id,
                success=False,
                error=str(e),
            )

    async def enhance_batch(
        self,
        document_ids: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force: bool = False,
    ) -> BatchEnhancementResult:
        """
        Enhance multiple documents with progress tracking.

        Args:
            document_ids: List of document UUIDs to enhance
            progress_callback: Callback function(processed, total) for progress updates
            force: If True, re-enhance even if already enhanced

        Returns:
            BatchEnhancementResult with aggregate statistics
        """
        logger.info("Starting batch enhancement", count=len(document_ids))

        results = []
        total_tokens = 0
        successful = 0
        failed = 0

        for i, doc_id in enumerate(document_ids):
            result = await self.enhance_document(doc_id, force=force)
            results.append(result)

            if result.success:
                successful += 1
                total_tokens += result.tokens_used
            else:
                failed += 1

            if progress_callback:
                progress_callback(i + 1, len(document_ids))

            # Rate limiting
            if i < len(document_ids) - 1:
                await asyncio.sleep(self.rate_limit_delay)

        # Calculate estimated cost
        cost_per_1m = self.token_costs.get(self.model, {"input": 0.15, "output": 0.60})
        estimated_cost = (total_tokens / 1_000_000) * (cost_per_1m["input"] + cost_per_1m["output"])

        return BatchEnhancementResult(
            total=len(document_ids),
            successful=successful,
            failed=failed,
            results=results,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

    async def enhance_all_unprocessed(
        self,
        collection: Optional[str] = None,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchEnhancementResult:
        """
        Find and enhance all documents without enhanced metadata.

        Args:
            collection: Optional collection filter (first tag)
            limit: Maximum number of documents to process
            progress_callback: Callback for progress updates

        Returns:
            BatchEnhancementResult
        """
        logger.info("Finding unprocessed documents", collection=collection, limit=limit)

        async with async_session_context() as db:
            # Build query for documents without enhanced metadata
            query = select(Document.id).where(
                Document.enhanced_metadata.is_(None),
                Document.processing_status == ProcessingStatus.COMPLETED,
            )

            # Filter by collection if specified
            if collection:
                # Collection is stored as the first tag
                query = query.where(Document.tags[0] == collection)

            if limit:
                query = query.limit(limit)

            result = await db.execute(query)
            document_ids = [str(row[0]) for row in result.all()]

        if not document_ids:
            return BatchEnhancementResult(
                total=0,
                successful=0,
                failed=0,
                results=[],
            )

        logger.info("Found unprocessed documents", count=len(document_ids))

        return await self.enhance_batch(
            document_ids,
            progress_callback=progress_callback,
            force=False,
        )

    async def estimate_cost(
        self,
        document_ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Estimate cost for enhancing documents.

        Args:
            document_ids: Specific documents, or None for all unprocessed
            collection: Filter by collection (if document_ids is None)

        Returns:
            Cost estimation dict
        """
        async with async_session_context() as db:
            if document_ids:
                count = len(document_ids)

                # Get average content size
                result = await db.execute(
                    select(func.avg(func.length(Chunk.content)))
                    .join(Document)
                    .where(Document.id.in_(document_ids))
                )
                avg_content_size = result.scalar() or 2000
            else:
                # Count unprocessed documents
                query = select(func.count(Document.id)).where(
                    Document.enhanced_metadata.is_(None),
                    Document.processing_status == ProcessingStatus.COMPLETED,
                )
                if collection:
                    query = query.where(Document.tags[0] == collection)

                result = await db.execute(query)
                count = result.scalar() or 0
                avg_content_size = 2000  # Default estimate

        # Estimate tokens per document
        prompt_tokens = 500  # System prompt + structure
        content_tokens = min(avg_content_size // 4, self.max_chunk_tokens)
        output_tokens = 400  # Typical response size
        tokens_per_doc = prompt_tokens + content_tokens + output_tokens

        total_tokens = count * tokens_per_doc

        # Calculate cost
        cost_per_1m = self.token_costs.get(self.model, {"input": 0.15, "output": 0.60})
        input_cost = (tokens_per_doc - output_tokens) * count / 1_000_000 * cost_per_1m["input"]
        output_cost = output_tokens * count / 1_000_000 * cost_per_1m["output"]
        total_cost = input_cost + output_cost

        return {
            "document_count": count,
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(total_cost, 4),
            "model": self.model,
            "avg_tokens_per_doc": tokens_per_doc,
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_enhancer_instance: Optional[DocumentEnhancer] = None


def get_document_enhancer() -> DocumentEnhancer:
    """Get or create the document enhancer singleton."""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = DocumentEnhancer()
    return _enhancer_instance
