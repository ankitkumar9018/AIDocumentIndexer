"""
AIDocumentIndexer - Document Processing Pipeline
=================================================

End-to-end document processing pipeline that orchestrates:
1. File validation and deduplication
2. Content extraction (text, images, metadata)
3. Chunking for RAG
4. Embedding generation
5. Vector store indexing

Supports both synchronous and Ray-parallel processing.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union, Awaitable, Tuple
from pathlib import Path
import inspect
from datetime import datetime
from enum import Enum
import hashlib
import asyncio
import uuid
import os
import time
import structlog

import ray
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.processors.universal import UniversalProcessor, ExtractedContent, ExtractedImage
from backend.db.models import ProcessingMode, Document as DocumentModel, AccessTier, ProcessingStatus as DBProcessingStatus, StorageMode
from backend.services.multimodal_rag import MultimodalRAGService, get_multimodal_rag_service
from backend.db.database import async_session_context
from backend.processors.chunker import DocumentChunker, ChunkingConfig, ChunkingStrategy, Chunk

# Phase 59: Import semantic chunker for contextual chunking
try:
    from backend.services.semantic_chunker import (
        SemanticChunker,
        SemanticChunkingConfig,
        ContextualChunkingMode,
    )
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False
from backend.services.embeddings import EmbeddingService, RayEmbeddingService, EmbeddingResult
from backend.services.vectorstore import VectorStore, get_vector_store
from backend.services.text_preprocessor import TextPreprocessor, PreprocessingConfig, get_text_preprocessor
from backend.services.summarizer import DocumentSummarizer, SummarizationConfig, get_document_summarizer
from backend.services.contextual_chunking import contextualize_chunks

logger = structlog.get_logger(__name__)

# Access tier cache to avoid repeated DB lookups
# Maps level -> (tier_id, cached_at_timestamp)
_access_tier_cache: Dict[int, Tuple[str, float]] = {}
_ACCESS_TIER_CACHE_TTL = 300.0  # 5 minutes


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of processing a single document."""
    document_id: str
    file_path: str
    file_hash: str
    status: ProcessingStatus

    # Extracted content stats
    page_count: int = 0
    word_count: int = 0
    chunk_count: int = 0

    # Processing metadata
    file_type: str = ""
    file_size: int = 0
    processing_time_ms: float = 0

    # Results
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: List[EmbeddingResult] = field(default_factory=list)

    # Error info
    error_message: Optional[str] = None

    # Metadata extracted from document
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProcessingResult:
    """Result of batch processing multiple documents."""
    total_documents: int
    successful: int
    failed: int
    results: List[ProcessingResult]
    processing_time_ms: float


class PipelineConfig:
    """Configuration for the processing pipeline."""

    def __init__(
        self,
        # Processing mode
        processing_mode: ProcessingMode = ProcessingMode.FULL,
        use_ray: bool = True,

        # Chunking settings
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,

        # Phase 59: Semantic chunking with contextual headers
        use_semantic_chunker: bool = None,  # Read from ENABLE_SEMANTIC_CHUNKER env var
        semantic_chunking_mode: str = "section_headers",  # none, title_only, section_headers, full_context

        # Embedding settings (uses env var fallback, can be overridden via Admin UI)
        embedding_provider: str = None,  # Will use EMBEDDING_PROVIDER env var (independent of chat LLM)
        embedding_model: Optional[str] = None,  # Will use OLLAMA_EMBEDDING_MODEL env var
        embedding_batch_size: int = 100,

        # Deduplication
        check_duplicates: bool = True,

        # Text preprocessing (reduces token costs)
        enable_preprocessing: bool = True,  # ON by default (lightweight)
        preprocessing_config: Optional[PreprocessingConfig] = None,

        # Document summarization (for large files) - reads from env if not set
        enable_summarization: bool = None,  # Read from ENABLE_SUMMARIZATION env var
        summarization_config: Optional[SummarizationConfig] = None,

        # Multimodal processing (image understanding)
        enable_multimodal: bool = None,  # Read from settings if not set
        caption_images: bool = True,  # Generate captions for extracted images

        # Progress callbacks (can be sync or async)
        on_status_change: Optional[Union[Callable[[str, ProcessingStatus], None], Callable[[str, ProcessingStatus], Awaitable[None]]]] = None,
        on_progress: Optional[Union[Callable[[str, int, int], None], Callable[[str, int, int], Awaitable[None]]]] = None,
    ):
        self.processing_mode = processing_mode
        self.use_ray = use_ray
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

        # Phase 59: Semantic chunker with contextual headers
        if use_semantic_chunker is None:
            self.use_semantic_chunker = os.getenv("ENABLE_SEMANTIC_CHUNKER", "false").lower() == "true"
        else:
            self.use_semantic_chunker = use_semantic_chunker
        self.semantic_chunking_mode = semantic_chunking_mode
        # Use provided value, or fall back to env var, or default to "ollama"
        # IMPORTANT: Use EMBEDDING_PROVIDER, NOT DEFAULT_LLM_PROVIDER
        # Embeddings must be independent of chat LLM to allow switching LLMs without re-indexing
        self.embedding_provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", "ollama")
        # Use provided model, or fall back to provider-specific env var
        if embedding_model:
            self.embedding_model = embedding_model
        elif self.embedding_provider == "ollama":
            self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        else:
            self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_batch_size = embedding_batch_size
        self.check_duplicates = check_duplicates
        self.enable_preprocessing = enable_preprocessing
        self.preprocessing_config = preprocessing_config
        # Read summarization from env if not explicitly set
        if enable_summarization is None:
            self.enable_summarization = os.getenv("ENABLE_SUMMARIZATION", "true").lower() == "true"
        else:
            self.enable_summarization = enable_summarization
        self.summarization_config = summarization_config
        # Multimodal is disabled by default (requires vision model)
        if enable_multimodal is None:
            self.enable_multimodal = os.getenv("ENABLE_MULTIMODAL", "false").lower() == "true"
        else:
            self.enable_multimodal = enable_multimodal
        self.caption_images = caption_images
        self.on_status_change = on_status_change
        self.on_progress = on_progress


class DocumentPipeline:
    """
    Document processing pipeline.

    Orchestrates the full document processing workflow:
    1. Validate file and compute hash
    2. Extract content using UniversalProcessor
    3. Chunk content using DocumentChunker
    4. Generate embeddings using EmbeddingService
    5. (Optionally) Index in vector store
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        vector_store: Optional[Any] = None,
        use_custom_vectorstore: bool = True,
    ):
        """
        Initialize the processing pipeline.

        Args:
            config: Pipeline configuration
            vector_store: Vector store for indexing (optional LangChain compatible)
            use_custom_vectorstore: Use our custom VectorStore service (default: True)
        """
        self.config = config or PipelineConfig()
        self.vector_store = vector_store
        self._custom_vectorstore: Optional[VectorStore] = None
        self._use_custom_vectorstore = use_custom_vectorstore

        # Initialize custom vector store if enabled and no LangChain store provided
        if use_custom_vectorstore and vector_store is None:
            self._custom_vectorstore = get_vector_store()

        # Initialize components
        self._processor = UniversalProcessor()

        # Phase 59: Use semantic chunker with contextual headers if enabled
        self._semantic_chunker = None
        if self.config.use_semantic_chunker and SEMANTIC_CHUNKER_AVAILABLE:
            mode_map = {
                "none": ContextualChunkingMode.NONE,
                "title_only": ContextualChunkingMode.TITLE_ONLY,
                "section_headers": ContextualChunkingMode.SECTION_HEADERS,
                "full_context": ContextualChunkingMode.FULL_CONTEXT,
            }
            semantic_mode = mode_map.get(
                self.config.semantic_chunking_mode,
                ContextualChunkingMode.SECTION_HEADERS
            )

            semantic_config = SemanticChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                mode=semantic_mode,
                include_document_title=True,
                include_section_path=True,
            )
            self._semantic_chunker = SemanticChunker(semantic_config)
            logger.info(
                "Using semantic chunker with contextual headers",
                mode=self.config.semantic_chunking_mode,
            )

        # Standard chunker as fallback
        self._chunker = DocumentChunker(ChunkingConfig(
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        ))

        # Embedding service (Ray or standard)
        if self.config.use_ray:
            self._embeddings = RayEmbeddingService(
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
            )
        else:
            self._embeddings = EmbeddingService(
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
            )

        # Text preprocessor (optional, reduces token costs)
        self._preprocessor: Optional[TextPreprocessor] = None
        if self.config.enable_preprocessing:
            preproc_config = self.config.preprocessing_config or PreprocessingConfig()
            self._preprocessor = TextPreprocessor(preproc_config)

        # Document summarizer (optional, for large files)
        self._summarizer: Optional[DocumentSummarizer] = None
        if self.config.enable_summarization:
            summ_config = self.config.summarization_config or SummarizationConfig(enabled=True)
            self._summarizer = DocumentSummarizer(summ_config)

        # Multimodal service (optional, for image understanding)
        self._multimodal: Optional[MultimodalRAGService] = None
        if self.config.enable_multimodal:
            self._multimodal = get_multimodal_rag_service()
            logger.info("Multimodal processing enabled for image understanding")

        # Track processed file hashes for deduplication
        # Phase 98: Use OrderedDict with max size to prevent memory leaks
        self._processed_hashes: OrderedDict[str, str] = OrderedDict()
        self._max_hash_cache_size: int = int(os.getenv("PIPELINE_MAX_HASH_CACHE", "10000"))

        logger.info(
            "Initialized document pipeline",
            use_ray=self.config.use_ray,
            chunking_strategy=self.config.chunking_strategy.value,
            embedding_provider=self.config.embedding_provider,
            use_custom_vectorstore=use_custom_vectorstore,
            preprocessing_enabled=self.config.enable_preprocessing,
            summarization_enabled=self.config.enable_summarization,
            multimodal_enabled=self.config.enable_multimodal,
        )

    def _safe_uuid(self, value: Optional[str]) -> Optional[uuid.UUID]:
        """
        Safely convert a string to UUID.

        Returns None if value is None, empty, or not a valid UUID.
        This handles cases like "anonymous" user_id that shouldn't be converted to UUID.
        """
        if not value:
            return None
        try:
            return uuid.UUID(value)
        except (ValueError, TypeError):
            # Not a valid UUID (e.g., "anonymous")
            return None

    def _update_status(self, doc_id: str, status: ProcessingStatus):
        """Update processing status via callback (sync or async)."""
        if self.config.on_status_change:
            result = self.config.on_status_change(doc_id, status)
            # If it's a coroutine, schedule it
            if inspect.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(result)
                except RuntimeError:
                    # No event loop running, run in new loop
                    asyncio.run(result)

    def _update_progress(self, doc_id: str, current: int, total: int):
        """Update progress via callback (sync or async)."""
        if self.config.on_progress:
            result = self.config.on_progress(doc_id, current, total)
            # If it's a coroutine, schedule it
            if inspect.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(result)
                except RuntimeError:
                    # No event loop running, run in new loop
                    asyncio.run(result)

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file for deduplication (sync version)."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def compute_file_hash_async(self, file_path: str) -> str:
        """Compute SHA-256 hash of file for deduplication (async version).

        Runs file I/O in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.compute_file_hash, file_path)

    def is_duplicate(self, file_hash: str) -> Optional[str]:
        """
        Check if file hash is already processed.

        Returns:
            Document ID if duplicate, None otherwise
        """
        return self._processed_hashes.get(file_hash)

    async def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        access_tier: int = 10,
        collection: Optional[str] = None,
        folder_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        uploaded_by_id: Optional[str] = None,
        is_private: bool = False,
    ) -> ProcessingResult:
        """
        Process a single document through the full pipeline.

        Args:
            file_path: Path to the document file
            document_id: Optional document ID (generated if not provided)
            metadata: Additional metadata to attach
            access_tier: Access tier for the document
            collection: Collection name for organization
            folder_id: Optional folder ID to place the document in
            organization_id: Organization ID for multi-tenant isolation
            uploaded_by_id: User ID who uploaded the document
            is_private: Whether this is a private document (only visible to uploader)

        Returns:
            ProcessingResult with all extracted data
        """
        start_time = datetime.now()
        path = Path(file_path)

        # Generate document ID if not provided
        if not document_id:
            document_id = hashlib.md5(f"{file_path}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        logger.info(
            "Processing document",
            document_id=document_id,
            file_path=file_path,
        )

        result = ProcessingResult(
            document_id=document_id,
            file_path=file_path,
            file_hash="",
            status=ProcessingStatus.PENDING,
            file_type=path.suffix.lower().lstrip("."),
            file_size=path.stat().st_size if path.exists() else 0,
        )

        try:
            # Step 1: Validate and compute hash
            self._update_status(document_id, ProcessingStatus.VALIDATING)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            result.file_hash = await self.compute_file_hash_async(file_path)

            # Check if document with this hash already exists (for re-uploads)
            # Use existing document_id to keep chunks linked correctly
            existing_doc_id = await self._get_existing_document_id(result.file_hash)
            if existing_doc_id:
                logger.info(
                    "Re-processing existing document",
                    new_document_id=document_id,
                    existing_document_id=existing_doc_id,
                )
                # Delete old chunks before re-processing
                if self._custom_vectorstore:
                    await self._custom_vectorstore.delete_document_chunks(existing_doc_id)
                # Use the existing document ID so chunks stay linked
                document_id = existing_doc_id
                result.document_id = document_id

            # Check for duplicates (only if no existing document found)
            elif self.config.check_duplicates:
                existing_id = self.is_duplicate(result.file_hash)
                if existing_id:
                    logger.info(
                        "Duplicate file detected",
                        document_id=document_id,
                        existing_id=existing_id,
                    )
                    result.status = ProcessingStatus.COMPLETED
                    result.metadata["duplicate_of"] = existing_id
                    return result

            # Step 2: Extract content
            self._update_status(document_id, ProcessingStatus.EXTRACTING)
            self._update_progress(document_id, 1, 5)

            extracted: ExtractedContent = self._processor.process(
                file_path,
                processing_mode=self.config.processing_mode.value,
            )

            # OCR auto-routing: if extraction yielded empty/minimal text on a
            # PDF and we weren't already using OCR, retry with OCR mode.
            file_suffix = Path(file_path).suffix.lower()
            _min_ocr_chars = 50  # threshold for "essentially empty"
            if (
                file_suffix == ".pdf"
                and self.config.processing_mode == ProcessingMode.BASIC
                and len((extracted.text or "").strip()) < _min_ocr_chars
            ):
                logger.info(
                    "Text extraction yielded near-empty result, retrying with OCR",
                    document_id=document_id,
                    chars_extracted=len((extracted.text or "").strip()),
                )
                try:
                    extracted = self._processor.process(
                        file_path,
                        processing_mode=ProcessingMode.OCR_ENABLED.value,
                    )
                    result.metadata["ocr_auto_routed"] = True
                    logger.info(
                        "OCR auto-routing succeeded",
                        document_id=document_id,
                        chars_after_ocr=len((extracted.text or "").strip()),
                    )
                except Exception as ocr_err:
                    logger.warning(
                        "OCR auto-routing failed, continuing with original extraction",
                        document_id=document_id,
                        error=str(ocr_err),
                    )

            # Phase 63: Enhanced document parsing with Docling (97.9% table accuracy)
            # Use runtime settings for hot-reload (no server restart needed)
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            docling_enabled = await settings_svc.get_setting("processing.docling_parser_enabled") or False

            if docling_enabled and file_suffix in ['.pdf', '.docx', '.doc']:
                try:
                    from backend.services.document_parser import DocumentParser
                    docling_parser = DocumentParser()
                    docling_result = await docling_parser.parse(file_path)

                    # Enhance extracted content with Docling's superior table extraction.
                    # Only append tables whose content isn't already in the extracted text
                    # to avoid duplication (standard extraction may have captured them too).
                    if docling_result.tables:
                        existing_text_lower = (extracted.text or "").lower()
                        new_tables = []
                        for t in docling_result.tables:
                            table_str = t.to_markdown() if hasattr(t, 'to_markdown') else str(t)
                            # Check if the first data row is already in the text
                            first_line = table_str.strip().split("\n")[0] if table_str.strip() else ""
                            if first_line and first_line.lower() not in existing_text_lower:
                                new_tables.append(table_str)

                        if new_tables:
                            table_markdown = "\n\n".join(new_tables)
                            extracted.text = extracted.text + "\n\n--- Extracted Tables ---\n" + table_markdown

                        result.metadata["docling_tables_extracted"] = len(docling_result.tables)
                        result.metadata["docling_tables_appended"] = len(new_tables)
                        logger.info(
                            "Enhanced with Docling table extraction",
                            document_id=document_id,
                            tables_found=len(docling_result.tables),
                            tables_appended=len(new_tables),
                        )
                except ImportError:
                    logger.warning("Docling not available, using standard extraction")
                except Exception as e:
                    logger.warning("Docling parsing failed, using standard extraction", error=str(e))

            result.page_count = extracted.page_count
            result.word_count = extracted.word_count
            result.metadata = {
                **(metadata or {}),
                **extracted.metadata,
                "access_tier": access_tier,
                "collection": collection,
                "folder_id": folder_id,
                "language": extracted.language,
            }

            # Step 2.25: Document quality assessment
            # Detect garbled encoding, extremely low information density, or corruption
            # before spending resources on chunking/embedding.
            _quality_text = extracted.text or ""
            if _quality_text:
                _ascii_ratio = sum(1 for c in _quality_text if c.isascii()) / max(len(_quality_text), 1)
                _alpha_ratio = sum(1 for c in _quality_text if c.isalpha()) / max(len(_quality_text), 1)
                _avg_word_len = (
                    sum(len(w) for w in _quality_text.split()) / max(len(_quality_text.split()), 1)
                )
                quality_score = "good"
                quality_warnings = []
                # Garbled encoding: very few alphabetic characters
                if _alpha_ratio < 0.3 and len(_quality_text) > 100:
                    quality_warnings.append("low_alpha_ratio")
                    quality_score = "poor"
                # Abnormal word length (may indicate binary/corrupted content)
                if _avg_word_len > 20 and len(_quality_text) > 100:
                    quality_warnings.append("abnormal_word_length")
                    quality_score = "poor"
                # High control character density
                _control_ratio = sum(1 for c in _quality_text if ord(c) < 32 and c not in '\n\r\t') / max(len(_quality_text), 1)
                if _control_ratio > 0.1:
                    quality_warnings.append("high_control_chars")
                    quality_score = "poor"

                result.metadata["document_quality"] = quality_score
                if quality_warnings:
                    result.metadata["quality_warnings"] = quality_warnings
                    logger.warning(
                        "Document quality issues detected",
                        document_id=document_id,
                        quality=quality_score,
                        warnings=quality_warnings,
                        alpha_ratio=round(_alpha_ratio, 3),
                        avg_word_len=round(_avg_word_len, 1),
                    )

            # Step 2.5: Preprocess text (optional, reduces token costs)
            preprocessing_stats = {}
            if self._preprocessor:
                logger.debug("Running text preprocessing", document_id=document_id)

                if extracted.pages:
                    # Preprocess each page
                    for page in extracted.pages:
                        if "text" in page:
                            preproc_result = self._preprocessor.preprocess(page["text"])
                            page["text"] = preproc_result.processed_text
                            preprocessing_stats["total_reduction"] = preprocessing_stats.get("total_reduction", 0) + (
                                preproc_result.original_length - preproc_result.processed_length
                            )
                else:
                    # Preprocess full text
                    preproc_result = self._preprocessor.preprocess(extracted.text)
                    extracted.text = preproc_result.processed_text
                    preprocessing_stats = preproc_result.stats

                if preprocessing_stats:
                    result.metadata["preprocessing"] = preprocessing_stats
                    logger.info(
                        "Text preprocessing complete",
                        document_id=document_id,
                        chars_reduced=preprocessing_stats.get("total_reduction", 0),
                    )

            # Step 2.75: Generate document summary (optional, for large files)
            summary_chunks = []
            if self._summarizer:
                text_for_summary = extracted.text
                if not text_for_summary and extracted.pages:
                    text_for_summary = "\n\n".join(p.get("text", "") for p in extracted.pages)

                if self._summarizer.should_summarize(
                    text_length_bytes=len(text_for_summary.encode('utf-8')),
                    page_count=extracted.page_count,
                ):
                    logger.info(
                        "Generating document summary",
                        document_id=document_id,
                        page_count=extracted.page_count,
                    )

                    summary_result = await self._summarizer.summarize_document(
                        text=text_for_summary,
                        metadata=result.metadata,
                    )

                    if summary_result.document_summary:
                        # Create summary chunk for indexing
                        summary_chunk_data = self._summarizer.create_summary_chunk(
                            document_id=document_id,
                            summary=summary_result.document_summary,
                            chunk_level=2,  # Document level
                        )
                        # Convert to Chunk object
                        summary_chunk = Chunk(
                            content=summary_chunk_data["content"],
                            chunk_index=summary_chunk_data["chunk_index"],
                            chunk_hash=summary_chunk_data["chunk_hash"],
                            document_id=document_id,
                            metadata={
                                **result.metadata,
                                **summary_chunk_data["metadata"],
                            },
                        )
                        summary_chunks.append(summary_chunk)

                        result.metadata["summarization"] = {
                            "summary_length": summary_result.summary_length,
                            "original_length": summary_result.original_length,
                            "reduction_percent": round(summary_result.reduction_percent, 2),
                            "tokens_used": summary_result.tokens_used,
                        }

                        logger.info(
                            "Document summary generated",
                            document_id=document_id,
                            summary_length=summary_result.summary_length,
                            reduction_percent=round(summary_result.reduction_percent, 2),
                        )

            # Step 2.8: Process images for multimodal understanding with deduplication
            image_chunks = []
            image_stats = {}
            if self._multimodal and extracted.extracted_images:
                logger.info(
                    "Processing images for multimodal understanding",
                    document_id=document_id,
                    image_count=len(extracted.extracted_images),
                )

                image_captions, image_stats = await self._process_document_images(
                    document_id=document_id,
                    images=extracted.extracted_images,
                    skip_duplicates=True,
                )

                if image_captions:
                    # Create chunks for image descriptions
                    # Limit to processed images count
                    from backend.services.settings import get_settings_service
                    settings_svc = get_settings_service()
                    max_images = await settings_svc.get_setting("rag.max_images_per_document") or 50
                    images_to_chunk = extracted.extracted_images[:max_images] if max_images > 0 else extracted.extracted_images

                    for i, (img, caption) in enumerate(zip(images_to_chunk, image_captions)):
                        if caption and caption.strip():
                            page_info = f" on page {img.page_number}" if img.page_number else ""
                            image_content = f"[IMAGE{page_info}]: {caption}"

                            image_chunk = Chunk(
                                content=image_content,
                                chunk_index=-100 - i,  # Negative index for special chunks
                                chunk_hash=hashlib.md5(image_content.encode()).hexdigest()[:16],
                                document_id=document_id,
                                page_number=img.page_number,
                                metadata={
                                    **result.metadata,
                                    "chunk_type": "image_description",
                                    "image_index": img.image_index,
                                    "image_extension": img.extension,
                                },
                            )
                            image_chunks.append(image_chunk)

                    result.metadata["multimodal"] = {
                        "images_found": image_stats.get("images_found", len(extracted.extracted_images)),
                        "images_analyzed": image_stats.get("images_analyzed", 0),
                        "images_cached": image_stats.get("cached_used", 0),
                        "images_skipped_small": image_stats.get("images_skipped_small", 0),
                        "images_skipped_duplicate": image_stats.get("images_skipped_duplicate", 0),
                        "images_failed": image_stats.get("images_failed", 0),
                        "captions_generated": len([c for c in image_captions if c]),
                    }

                    logger.info(
                        "Image processing complete with deduplication",
                        document_id=document_id,
                        **image_stats,
                    )

                # Update document with image tracking fields
                try:
                    async with async_session_context() as db:
                        doc_result = await db.execute(
                            select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
                        )
                        doc = doc_result.scalar_one_or_none()
                        if doc:
                            doc.images_extracted_count = len(extracted.extracted_images)
                            doc.images_analyzed_count = image_stats.get("images_analyzed", 0) + image_stats.get("cached_used", 0)
                            doc.image_analysis_status = "completed"
                            doc.image_analysis_completed_at = datetime.utcnow()
                            await db.commit()
                except Exception as e:
                    logger.warning(
                        "Failed to update document image tracking",
                        document_id=document_id,
                        error=str(e),
                    )
            else:
                # No images in document - mark as not applicable
                try:
                    async with async_session_context() as db:
                        doc_result = await db.execute(
                            select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
                        )
                        doc = doc_result.scalar_one_or_none()
                        if doc:
                            doc.images_extracted_count = 0
                            doc.images_analyzed_count = 0
                            doc.image_analysis_status = "not_applicable"
                            await db.commit()
                except Exception as e:
                    logger.debug(
                        "Failed to update document image status",
                        document_id=document_id,
                        error=str(e),
                    )

            # Step 3: Chunk content
            self._update_status(document_id, ProcessingStatus.CHUNKING)
            self._update_progress(document_id, 2, 5)

            # Initialize chunks list (will be populated by one of the chunking methods)
            chunks: List[Chunk] = []

            # Phase 63: Fast chunking with Chonkie (33x faster than LangChain)
            # Use runtime settings for hot-reload (no server restart needed)
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            # Phase 70: Fast chunking enabled by default (33x faster than LangChain)
            fast_chunking_setting = await settings_svc.get_setting("processing.fast_chunking_enabled")
            fast_chunking_enabled = fast_chunking_setting if fast_chunking_setting is not None else True
            fast_chunking_strategy = await settings_svc.get_setting("processing.fast_chunking_strategy") or "auto"

            if fast_chunking_enabled:
                try:
                    from backend.services.chunking import FastChunker, FastChunkingStrategy
                    fast_chunker = FastChunker()
                    text_to_chunk = "\n\n".join([
                        p.get("text", "") for p in extracted.pages
                    ]) if extracted.pages else extracted.text

                    # Map strategy string to enum
                    strategy_map = {
                        "auto": FastChunkingStrategy.AUTO,
                        "token": FastChunkingStrategy.TOKEN,
                        "sentence": FastChunkingStrategy.SENTENCE,
                        "semantic": FastChunkingStrategy.SEMANTIC,
                        "sdpm": FastChunkingStrategy.SDPM,
                    }
                    strategy = strategy_map.get(fast_chunking_strategy, FastChunkingStrategy.SEMANTIC)

                    fast_chunks = await fast_chunker.chunk(
                        text=text_to_chunk,
                        strategy=strategy,
                    )
                    # Convert to standard Chunk format
                    chunks = [
                        Chunk(
                            text=fc.text,
                            metadata={**result.metadata, "chunk_index": i, "fast_chunked": True},
                            document_id=document_id,
                        )
                        for i, fc in enumerate(fast_chunks)
                    ]
                    logger.info(
                        "Used Chonkie fast chunker",
                        document_id=document_id,
                        chunk_count=len(chunks),
                    )
                except Exception as e:
                    logger.warning("Fast chunking failed, falling back to standard", error=str(e))
                    chunks = []  # Reset to trigger fallback

            # Phase 59: Use semantic chunker with contextual headers if available
            # Also used as fallback when fast chunking fails (chunks is empty)
            if not chunks and self._semantic_chunker is not None:
                # Use semantic chunker with section detection
                text_to_chunk = "\n\n".join([
                    p.get("text", "") for p in extracted.pages
                ]) if extracted.pages else extracted.text

                chunks = self._semantic_chunker.chunk_with_context(
                    text=text_to_chunk,
                    document_title=result.file_name,
                    document_id=document_id,
                    metadata=result.metadata,
                )
                logger.info(
                    "Used semantic chunker with contextual headers",
                    document_id=document_id,
                    chunk_count=len(chunks),
                )
            elif extracted.pages:
                # Chunk with page preservation (standard chunker)
                chunks = self._chunker.chunk_with_pages(
                    pages=[{"content": p.get("text", ""), "page_number": p.get("page", i+1)}
                           for i, p in enumerate(extracted.pages)],
                    metadata=result.metadata,
                    document_id=document_id,
                )
            else:
                # Chunk full text (standard chunker)
                chunks = self._chunker.chunk(
                    text=extracted.text,
                    metadata=result.metadata,
                    document_id=document_id,
                )

            # Add summary chunks to beginning (for hierarchical retrieval)
            if summary_chunks:
                chunks = summary_chunks + chunks
                logger.debug(
                    "Added summary chunks",
                    document_id=document_id,
                    summary_count=len(summary_chunks),
                )

            # Add image description chunks (for multimodal retrieval)
            if image_chunks:
                chunks = image_chunks + chunks
                logger.debug(
                    "Added image description chunks",
                    document_id=document_id,
                    image_chunk_count=len(image_chunks),
                )

            # Step 3.5: Apply contextual chunking (if enabled in settings)
            # This adds contextual information to each chunk for better retrieval
            # See: Anthropic's contextual retrieval approach (49-67% reduction in failed retrievals)
            # Ensure chunks is defined (defensive check for edge cases)
            try:
                _ = chunks
            except (NameError, UnboundLocalError):
                logger.warning("chunks not defined, initializing empty list", document_id=document_id)
                chunks = []
            try:
                full_text = extracted.text
                if not full_text and extracted.pages:
                    full_text = "\n\n".join(p.get("text", "") for p in extracted.pages)

                document_title = result.metadata.get("title") or result.metadata.get("filename")

                chunks = await contextualize_chunks(
                    document_content=full_text,
                    chunks=chunks,
                    document_title=document_title,
                )

                # Track if contextual chunking was applied
                enhanced_count = sum(1 for c in chunks if c.metadata.get("contextual_enhanced"))
                if enhanced_count > 0:
                    result.metadata["contextual_chunking"] = {
                        "enabled": True,
                        "chunks_enhanced": enhanced_count,
                        "total_chunks": len(chunks),
                    }
                    logger.info(
                        "Contextual chunking applied",
                        document_id=document_id,
                        enhanced_count=enhanced_count,
                    )
            except Exception as e:
                logger.warning(
                    "Contextual chunking failed, continuing with regular chunks",
                    document_id=document_id,
                    error=str(e),
                )

            result.chunks = chunks
            result.chunk_count = len(chunks)

            # Phase 98: Inject document tags into chunk metadata for tag-augmented embeddings
            # Tags are prepended to chunk text during embedding generation (embeddings.py)
            doc_tags = (metadata or {}).get("tags", [])
            if isinstance(doc_tags, str):
                doc_tags = [t.strip() for t in doc_tags.split(",") if t.strip()]
            if doc_tags:
                for chunk in chunks:
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata["document_tags"] = doc_tags
                logger.debug(
                    "Injected document tags into chunk metadata",
                    document_id=document_id,
                    num_chunks=len(chunks),
                    tags=doc_tags,
                )

            logger.debug(
                "Document chunked",
                document_id=document_id,
                num_chunks=len(chunks),
            )

            # Validate: Fail if no extractable content (prevents zombie documents)
            if result.chunk_count == 0 and result.word_count == 0:
                logger.warning(
                    "Document has no extractable content - marking as failed",
                    document_id=document_id,
                    page_count=result.page_count,
                    file_path=file_path,
                )
                result.status = ProcessingStatus.FAILED
                result.error_message = "No extractable text content found. The document may be image-based and require OCR, or it may be empty/corrupted."
                self._update_status(document_id, ProcessingStatus.FAILED)

                # Still create document record but with FAILED status so user can see it
                await self._create_document_record(
                    document_id=document_id,
                    file_path=file_path,
                    file_hash=result.file_hash,
                    metadata=metadata or {},
                    access_tier=access_tier,
                    collection=collection,
                    processing_result=result,
                    organization_id=organization_id,
                    uploaded_by_id=uploaded_by_id,
                    is_private=is_private,
                    folder_id=folder_id,
                )
                return result

            # Step 4: Generate embeddings (with retry for transient failures)
            self._update_status(document_id, ProcessingStatus.EMBEDDING)
            self._update_progress(document_id, 3, 5)

            if chunks:
                _embed_max_retries = 3
                for _attempt in range(1, _embed_max_retries + 1):
                    try:
                        if isinstance(self._embeddings, RayEmbeddingService):
                            embeddings = self._embeddings.embed_chunks_parallel(chunks)
                        else:
                            embeddings = self._embeddings.embed_chunks(
                                chunks,
                                batch_size=self.config.embedding_batch_size,
                            )
                        result.embeddings = embeddings
                        break
                    except Exception as embed_err:
                        if _attempt < _embed_max_retries:
                            _backoff = 2 ** (_attempt - 1)  # 1s, 2s
                            logger.warning(
                                "Embedding generation failed, retrying",
                                document_id=document_id,
                                attempt=_attempt,
                                max_retries=_embed_max_retries,
                                backoff_seconds=_backoff,
                                error=str(embed_err),
                            )
                            await asyncio.sleep(_backoff)
                        else:
                            logger.error(
                                "Embedding generation failed after all retries",
                                document_id=document_id,
                                attempts=_embed_max_retries,
                                error=str(embed_err),
                            )
                            raise

            # Step 5: Index in vector store (if available)
            self._update_status(document_id, ProcessingStatus.INDEXING)
            self._update_progress(document_id, 4, 5)

            # Phase 98: Filter garbage chunks before indexing
            # Content types like "image_credit" and "copyright" should not be indexed
            SKIP_CONTENT_TYPES = {"image_credit", "copyright"}
            if chunks:
                original_chunk_count = len(chunks)
                # Get indices of chunks to keep
                keep_indices = [
                    i for i, c in enumerate(chunks)
                    if c.metadata.get("content_type") not in SKIP_CONTENT_TYPES
                ]
                skipped_count = original_chunk_count - len(keep_indices)

                if skipped_count > 0:
                    # Filter chunks
                    chunks = [chunks[i] for i in keep_indices]

                    # Filter corresponding embeddings (they're 1-to-1 with chunks)
                    if result.embeddings and len(result.embeddings) == original_chunk_count:
                        result.embeddings = [result.embeddings[i] for i in keep_indices]

                    logger.info(
                        "Filtered garbage chunks before indexing",
                        document_id=document_id,
                        skipped=skipped_count,
                        remaining=len(chunks),
                        skipped_types=list(SKIP_CONTENT_TYPES),
                    )
                    result.metadata = result.metadata or {}
                    result.metadata["garbage_chunks_skipped"] = skipped_count

            # Warn if embeddings are missing (will affect searchability)
            if chunks and not result.embeddings:
                logger.warning(
                    "Document processed but NO EMBEDDINGS generated - will not be searchable!",
                    document_id=document_id,
                    chunk_count=len(chunks),
                    file_path=file_path,
                )

            # Index using custom vector store
            if self._custom_vectorstore and chunks and result.embeddings:
                # Look up actual access tier UUID by level
                access_tier_id = await self._get_access_tier_id(access_tier)
                if access_tier_id:
                    # Use original_filename from metadata if available, fallback to path.name
                    original_filename = (metadata or {}).get("original_filename") or path.name
                    # Phase 98: Extract document tags for tag-augmented embeddings
                    document_tags = (metadata or {}).get("tags", [])
                    if isinstance(document_tags, str):
                        document_tags = [t.strip() for t in document_tags.split(",") if t.strip()]
                    await self._index_with_custom_store(
                        document_id=document_id,
                        chunks=chunks,
                        embeddings=result.embeddings,
                        access_tier_id=access_tier_id,
                        filename=original_filename,
                        collection=collection,
                        organization_id=organization_id,
                        uploaded_by_id=uploaded_by_id,
                        is_private=is_private,
                        document_tags=document_tags if document_tags else None,
                    )
                else:
                    logger.warning(
                        "Could not find access tier, skipping chunk indexing",
                        document_id=document_id,
                        access_tier=access_tier,
                    )
            # Fallback to LangChain vector store
            elif self.vector_store and chunks:
                await self._index_document(document_id, chunks, result.embeddings)

            # Mark as completed BEFORE creating/updating db record
            # This ensures the document is saved with COMPLETED status
            result.status = ProcessingStatus.COMPLETED
            self._update_status(document_id, ProcessingStatus.COMPLETED)
            self._update_progress(document_id, 5, 5)

            # Step 6: Create Document record in database (CRITICAL FIX)
            # This was the missing step - documents were processed but never recorded
            await self._create_document_record(
                document_id=document_id,
                file_path=file_path,
                file_hash=result.file_hash,
                metadata=metadata or {},
                access_tier=access_tier,
                collection=collection,
                processing_result=result,
                organization_id=organization_id,
                uploaded_by_id=uploaded_by_id,
                is_private=is_private,
                folder_id=folder_id,
            )

            # Track hash for deduplication
            # Phase 98: Evict oldest entries to prevent unbounded memory growth
            self._processed_hashes[result.file_hash] = document_id
            while len(self._processed_hashes) > self._max_hash_cache_size:
                self._processed_hashes.popitem(last=False)  # Remove oldest (FIFO)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time

            logger.info(
                "Document processing complete",
                document_id=document_id,
                chunks=len(chunks),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(
                "Document processing failed",
                document_id=document_id,
                error=str(e),
            )
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            self._update_status(document_id, ProcessingStatus.FAILED)

        return result

    async def _process_document_images(
        self,
        document_id: str,
        images: List[ExtractedImage],
        skip_duplicates: bool = True,
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Process extracted images to generate captions/descriptions with deduplication.

        Phase 71: Parallel processing with semaphore for 50-100x speedup on image-heavy docs.
        Phase 95: Image deduplication - skip identical images using hash-based cache.

        Args:
            document_id: Document ID for logging
            images: List of extracted images
            skip_duplicates: If True, use cached captions for identical images

        Returns:
            Tuple of (captions list, stats dict)
        """
        stats = {
            "images_found": len(images),
            "images_analyzed": 0,
            "images_skipped_small": 0,
            "images_skipped_duplicate": 0,
            "images_failed": 0,
            "cached_used": 0,
            "newly_analyzed": 0,
        }

        if not self._multimodal or not images:
            return [], stats

        # Get settings for image analysis
        from backend.services.settings import get_settings_service
        from backend.services.image_analysis import get_image_analysis_service

        settings = get_settings_service()
        image_service = get_image_analysis_service()

        # Load settings (with defaults)
        max_images = await settings.get_setting("rag.max_images_per_document") or 50
        min_size_kb = await settings.get_setting("rag.min_image_size_kb") or 5
        enable_dedup = await settings.get_setting("rag.image_duplicate_detection")
        if enable_dedup is None:
            enable_dedup = True

        # Limit images to process
        images_to_process = images[:max_images] if max_images > 0 else images

        # Phase 71: Parallel image captioning (configurable concurrency)
        # Check settings service first, fall back to env var, default to 8 (was 4)
        max_concurrent = await settings.get_setting("processing.max_concurrent_image_captions")
        if max_concurrent is None:
            max_concurrent = int(os.getenv("PIPELINE_MAX_CONCURRENT_CAPTIONS", "8"))
        semaphore = asyncio.Semaphore(max_concurrent)

        async def caption_with_limit(idx: int, img: ExtractedImage) -> Tuple[int, str, str]:
            """Caption a single image with concurrency control and deduplication."""
            async with semaphore:
                image_data = img.image_bytes
                caption_source = "new"

                # Skip small images
                if len(image_data) < min_size_kb * 1024:
                    return (idx, "", "skipped_small")

                # Compute hash for deduplication
                image_hash = image_service.compute_image_hash(image_data)

                # Check cache if deduplication enabled
                if skip_duplicates and enable_dedup:
                    try:
                        async with async_session_context() as db:
                            cached = await image_service.find_cached_caption(db, image_hash)
                            if cached:
                                return (idx, cached, "cached")
                    except Exception as e:
                        logger.debug(
                            "Cache lookup failed, will analyze",
                            document_id=document_id,
                            error=str(e),
                        )

                # Generate caption with vision model
                try:
                    caption = await self._multimodal.caption_image(
                        image_data=image_data,
                        image_format=img.extension,
                    )

                    # Cache the result
                    if caption and enable_dedup:
                        try:
                            async with async_session_context() as db:
                                await image_service.cache_caption(
                                    db=db,
                                    image_hash=image_hash,
                                    caption=caption,
                                    element_type="image",
                                    provider=None,  # Will be set by the service
                                    model=None,
                                    document_id=uuid.UUID(document_id) if document_id else None,
                                )
                        except Exception as e:
                            logger.debug(
                                "Failed to cache caption",
                                document_id=document_id,
                                error=str(e),
                            )

                    return (idx, caption, "new")
                except Exception as e:
                    logger.warning(
                        "Failed to caption image",
                        document_id=document_id,
                        image_index=img.image_index,
                        error=str(e),
                    )
                    return (idx, "", "failed")

        if len(images_to_process) > 1:
            logger.info(
                "Processing images in parallel with deduplication",
                document_id=document_id,
                image_count=len(images_to_process),
                max_concurrent=max_concurrent,
                skip_duplicates=skip_duplicates,
            )
            # Process all images in parallel
            results = await asyncio.gather(*[
                caption_with_limit(i, img)
                for i, img in enumerate(images_to_process)
            ])
            # Sort by index to maintain order
            results.sort(key=lambda x: x[0])

            # Collect stats and captions
            captions = []
            for idx, caption, source in results:
                captions.append(caption)
                if source == "cached":
                    stats["cached_used"] += 1
                    stats["images_skipped_duplicate"] += 1
                elif source == "new":
                    stats["newly_analyzed"] += 1
                    stats["images_analyzed"] += 1
                elif source == "skipped_small":
                    stats["images_skipped_small"] += 1
                elif source == "failed":
                    stats["images_failed"] += 1

            return captions, stats

        elif len(images_to_process) == 1:
            # Single image - process directly
            idx, caption, source = await caption_with_limit(0, images_to_process[0])
            if source == "cached":
                stats["cached_used"] = 1
                stats["images_skipped_duplicate"] = 1
            elif source == "new":
                stats["newly_analyzed"] = 1
                stats["images_analyzed"] = 1
            elif source == "skipped_small":
                stats["images_skipped_small"] = 1
            elif source == "failed":
                stats["images_failed"] = 1
            return [caption], stats
        else:
            return [], stats

    async def _get_existing_document_id(self, file_hash: str) -> Optional[str]:
        """
        Check if a document with the given file_hash already exists.

        Excludes soft-deleted documents (those marked as FAILED with "Deleted by user" error)
        so they don't get reactivated when a new file with the same hash is uploaded.

        Args:
            file_hash: SHA-256 hash of the file

        Returns:
            Existing document ID string if found, None otherwise
        """
        try:
            async with async_session_context() as db:
                # Exclude soft-deleted documents from hash lookup
                query = (
                    select(DocumentModel)
                    .where(DocumentModel.file_hash == file_hash)
                    .where(
                        ~(
                            (DocumentModel.processing_status == DBProcessingStatus.FAILED)
                            & (DocumentModel.processing_error == "Deleted by user")
                        )
                    )
                )
                result = await db.execute(query)
                existing_doc = result.scalar_one_or_none()
                if existing_doc:
                    return str(existing_doc.id)
                return None
        except Exception as e:
            logger.warning("Failed to check for existing document", file_hash=file_hash, error=str(e))
            return None

    async def _get_access_tier_id(self, access_tier_level: int) -> Optional[str]:
        """
        Look up access tier UUID by level with caching.

        Args:
            access_tier_level: Integer level of the access tier

        Returns:
            UUID string of the access tier, or None if not found
        """
        # Check cache first
        if access_tier_level in _access_tier_cache:
            tier_id, cached_at = _access_tier_cache[access_tier_level]
            if time.time() - cached_at < _ACCESS_TIER_CACHE_TTL:
                logger.debug("Access tier cache hit", level=access_tier_level)
                return tier_id

        try:
            async with async_session_context() as db:
                # Use .limit(1) to handle case where multiple tiers have same level
                tier_query = select(AccessTier).where(AccessTier.level == access_tier_level).limit(1)
                result = await db.execute(tier_query)
                access_tier_obj = result.scalar_one_or_none()

                if access_tier_obj:
                    tier_id = str(access_tier_obj.id)
                    _access_tier_cache[access_tier_level] = (tier_id, time.time())
                    return tier_id

                # Try to find a suitable tier by closest level
                all_tiers_query = select(AccessTier).order_by(AccessTier.level)
                all_result = await db.execute(all_tiers_query)
                all_tiers = all_result.scalars().all()

                if all_tiers:
                    # Find closest tier at or below requested level
                    for tier in reversed(all_tiers):
                        if tier.level <= access_tier_level:
                            tier_id = str(tier.id)
                            _access_tier_cache[access_tier_level] = (tier_id, time.time())
                            logger.info(
                                "Using closest access tier",
                                requested_level=access_tier_level,
                                using_tier=tier.name,
                                using_level=tier.level,
                            )
                            return tier_id

                    # If all tiers are above requested level, use the lowest one
                    tier_id = str(all_tiers[0].id)
                    _access_tier_cache[access_tier_level] = (tier_id, time.time())
                    logger.info(
                        "Using lowest available access tier",
                        requested_level=access_tier_level,
                        using_tier=all_tiers[0].name,
                    )
                    return tier_id

                return None
        except Exception as e:
            logger.error("Failed to look up access tier", error=str(e))
            return None

    async def _index_with_custom_store(
        self,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[EmbeddingResult],
        access_tier_id: str,
        filename: Optional[str] = None,
        collection: Optional[str] = None,
        organization_id: Optional[str] = None,
        uploaded_by_id: Optional[str] = None,
        is_private: bool = False,
        document_tags: Optional[List[str]] = None,
    ):
        """Index document chunks using our custom VectorStore."""
        if not self._custom_vectorstore:
            return

        try:
            # Prepare chunk data with embeddings
            chunk_data = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_data.append({
                    "content": chunk.content,
                    "content_hash": chunk.chunk_hash,
                    "embedding": embedding.embedding,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section_title": chunk.metadata.get("section_title"),
                    # Estimate token count: ~4 chars per token for English text
                    "token_count": getattr(embedding, 'token_count', None) or len(chunk.content) // 4,
                    "char_count": len(chunk.content),
                })

            # Add to vector store with multi-tenant metadata
            chunk_ids = await self._custom_vectorstore.add_chunks(
                chunks=chunk_data,
                document_id=document_id,
                access_tier_id=access_tier_id,
                document_filename=filename,
                collection=collection,
                organization_id=organization_id,
                uploaded_by_id=uploaded_by_id,
                is_private=is_private,
                document_tags=document_tags,
            )

            logger.debug(
                "Indexed document with custom vector store",
                document_id=document_id,
                num_chunks=len(chunk_ids),
            )

        except Exception as e:
            logger.error(
                "Custom vector store indexing failed",
                document_id=document_id,
                error=str(e),
            )
            # Re-raise to fail the document processing
            raise

    async def _index_document(
        self,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[EmbeddingResult],
    ):
        """Index document chunks in LangChain vector store."""
        if not self.vector_store:
            return

        # Prepare documents for vector store
        from langchain_core.documents import Document

        docs = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "document_id": document_id,
                    "chunk_id": chunk.chunk_hash,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                },
            )
            docs.append(doc)

        # Add to vector store
        try:
            await self.vector_store.aadd_documents(docs)
            logger.debug(
                "Indexed document in LangChain vector store",
                document_id=document_id,
                num_chunks=len(docs),
            )
        except Exception as e:
            logger.error(
                "LangChain vector store indexing failed",
                document_id=document_id,
                error=str(e),
            )

    async def _create_document_record(
        self,
        document_id: str,
        file_path: str,
        file_hash: str,
        metadata: Dict[str, Any],
        access_tier: int,
        collection: Optional[str],
        processing_result: "ProcessingResult",
        organization_id: Optional[str] = None,
        uploaded_by_id: Optional[str] = None,
        is_private: bool = False,
        folder_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> None:
        """
        Create or update Document record in database after successful processing.

        This handles both new documents and re-uploads of previously deleted files.
        If a document with the same file_hash exists (even if soft-deleted), we
        update it instead of creating a new one to avoid UNIQUE constraint violations.

        Phase 68: Added optional session parameter to support transactional consistency
        with chunk indexing operations.
        """
        async def _create_record(db: AsyncSession) -> None:
            # Find or create access tier by level (limit 1 in case of duplicate levels)
            tier_query = select(AccessTier).where(AccessTier.level == access_tier).limit(1)
            result = await db.execute(tier_query)
            access_tier_obj = result.scalar_one_or_none()

            if not access_tier_obj:
                # Create default tier if not exists
                access_tier_obj = AccessTier(
                    name=f"Tier {access_tier}",
                    level=access_tier,
                    description=f"Auto-created tier for level {access_tier}",
                )
                db.add(access_tier_obj)
                await db.flush()

            path = Path(file_path)

            # Check if a document with this hash already exists (including soft-deleted)
            existing_query = select(DocumentModel).where(DocumentModel.file_hash == file_hash)
            existing_result = await db.execute(existing_query)
            existing_doc = existing_result.scalar_one_or_none()

            if existing_doc:
                # Update existing document (reactivate if soft-deleted)
                existing_doc.filename = path.name
                # Always update original_filename to the new uploaded filename
                # This ensures re-uploads with different names show the correct name
                new_original_filename = metadata.get("original_filename", path.name)
                existing_doc.original_filename = new_original_filename
                existing_doc.file_path = str(file_path)
                existing_doc.file_size = path.stat().st_size if path.exists() else processing_result.file_size
                # Use the actual processing status (could be FAILED if no content extracted)
                existing_doc.processing_status = DBProcessingStatus(processing_result.status.value)
                existing_doc.processing_mode = self.config.processing_mode
                existing_doc.processing_error = processing_result.error_message  # Preserve error message if failed
                existing_doc.page_count = processing_result.page_count
                existing_doc.word_count = processing_result.word_count
                # Merge collection tag with existing tags instead of replacing
                if collection:
                    existing_tags = existing_doc.tags or []
                    if collection not in existing_tags:
                        existing_doc.tags = [collection] + existing_tags
                # else: keep existing tags unchanged
                existing_doc.access_tier_id = access_tier_obj.id
                existing_doc.processed_at = datetime.now()
                # Set folder_id if provided (allows moving document to folder on re-upload)
                if folder_id:
                    existing_doc.folder_id = uuid.UUID(folder_id)

                # Phase 68: Only commit if we own the session (no external session provided)
                if session is None:
                    await db.commit()
                else:
                    await db.flush()

                logger.info(
                    "Document record updated (reactivated)",
                    document_id=str(existing_doc.id),
                    filename=path.name,
                    access_tier=access_tier,
                )
            else:
                # Create new Document record
                document = DocumentModel(
                    id=uuid.UUID(document_id),
                    file_hash=file_hash,
                    filename=path.name,
                    original_filename=metadata.get("original_filename", path.name),
                    file_path=str(file_path),
                    file_type=path.suffix.lower().lstrip("."),
                    file_size=path.stat().st_size if path.exists() else processing_result.file_size,
                    # Use the actual processing status (could be FAILED if no content extracted)
                    processing_status=DBProcessingStatus(processing_result.status.value),
                    processing_mode=self.config.processing_mode,
                    processing_error=processing_result.error_message,  # Preserve error message if failed
                    storage_mode=StorageMode.RAG,
                    page_count=processing_result.page_count,
                    word_count=processing_result.word_count,
                    tags=[collection] if collection else [],  # Initialize with collection, auto-tag will merge more
                    access_tier_id=access_tier_obj.id,
                    processed_at=datetime.now(),
                    # Multi-tenant fields - safely parse UUIDs (skip invalid values like "anonymous")
                    organization_id=self._safe_uuid(organization_id),
                    uploaded_by_id=self._safe_uuid(uploaded_by_id),
                    is_private=is_private,
                    # Folder assignment
                    folder_id=self._safe_uuid(folder_id),
                )

                db.add(document)
                # Phase 68: Only commit if we own the session (no external session provided)
                if session is None:
                    await db.commit()
                else:
                    await db.flush()

                logger.info(
                    "Document record created in database",
                    document_id=document_id,
                    filename=path.name,
                    access_tier=access_tier,
                )

        # Phase 68: Use provided session or create new context
        try:
            if session:
                await _create_record(session)
            else:
                async with async_session_context() as db:
                    await _create_record(db)
        except Exception as e:
            logger.error(
                "Failed to create/update document record",
                document_id=document_id,
                error=str(e),
            )
            # Re-raise so the caller knows the record wasn't created
            raise

    async def process_batch(
        self,
        files: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> BatchProcessingResult:
        """
        Process multiple documents.

        Args:
            files: List of dicts with 'file_path' and optional 'metadata', 'access_tier', 'collection'
            parallel: Whether to process in parallel (uses Ray if enabled)

        Returns:
            BatchProcessingResult with all results
        """
        start_time = datetime.now()

        logger.info(
            "Starting batch processing",
            num_files=len(files),
            parallel=parallel,
        )

        results = []

        if parallel and self.config.use_ray and len(files) > 1:
            # Use Ray for parallel processing
            results = await self._process_batch_ray(files)
        else:
            # Sequential processing
            for i, file_info in enumerate(files):
                result = await self.process_document(
                    file_path=file_info["file_path"],
                    document_id=file_info.get("document_id"),
                    metadata=file_info.get("metadata"),
                    access_tier=file_info.get("access_tier", 10),
                    collection=file_info.get("collection"),
                )
                results.append(result)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)

        logger.info(
            "Batch processing complete",
            total=len(files),
            successful=successful,
            failed=failed,
            processing_time_ms=processing_time,
        )

        return BatchProcessingResult(
            total_documents=len(files),
            successful=successful,
            failed=failed,
            results=results,
            processing_time_ms=processing_time,
        )

    async def _process_batch_ray(
        self,
        files: List[Dict[str, Any]],
    ) -> List[ProcessingResult]:
        """Process batch using Ray for parallelism."""

        @ray.remote
        def process_single(file_info: Dict[str, Any], config_dict: Dict) -> Dict:
            """Ray remote function for processing a single document."""
            import asyncio

            # Reconstruct config
            config = PipelineConfig(
                processing_mode=ProcessingMode(config_dict["processing_mode"]),
                use_ray=False,  # Don't nest Ray calls
                chunk_size=config_dict["chunk_size"],
                chunk_overlap=config_dict["chunk_overlap"],
                embedding_provider=config_dict["embedding_provider"],
                embedding_model=config_dict.get("embedding_model"),
            )

            pipeline = DocumentPipeline(config=config)

            # Run async function
            result = asyncio.run(
                pipeline.process_document(
                    file_path=file_info["file_path"],
                    document_id=file_info.get("document_id"),
                    metadata=file_info.get("metadata"),
                    access_tier=file_info.get("access_tier", 10),
                    collection=file_info.get("collection"),
                )
            )

            # Return serializable dict
            return {
                "document_id": result.document_id,
                "file_path": result.file_path,
                "file_hash": result.file_hash,
                "status": result.status.value,
                "page_count": result.page_count,
                "word_count": result.word_count,
                "chunk_count": result.chunk_count,
                "file_type": result.file_type,
                "file_size": result.file_size,
                "processing_time_ms": result.processing_time_ms,
                "error_message": result.error_message,
                "metadata": result.metadata,
            }

        # Prepare config dict for serialization
        config_dict = {
            "processing_mode": self.config.processing_mode.value,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
        }

        # Submit all tasks
        futures = [
            process_single.remote(file_info, config_dict)
            for file_info in files
        ]

        # Collect results with timeout
        ray_timeout = float(os.getenv("RAY_TASK_TIMEOUT", "300"))
        try:
            raw_results = ray.get(futures, timeout=ray_timeout)
        except ray.exceptions.GetTimeoutError:
            logger.warning(
                "Ray processing timed out, cancelling tasks and falling back",
                timeout=ray_timeout,
                task_count=len(futures),
            )
            for ref in futures:
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    pass
            # Return empty results on timeout - caller should handle fallback
            return []

        # Convert back to ProcessingResult objects
        results = []
        for raw in raw_results:
            result = ProcessingResult(
                document_id=raw["document_id"],
                file_path=raw["file_path"],
                file_hash=raw["file_hash"],
                status=ProcessingStatus(raw["status"]),
                page_count=raw["page_count"],
                word_count=raw["word_count"],
                chunk_count=raw["chunk_count"],
                file_type=raw["file_type"],
                file_size=raw["file_size"],
                processing_time_ms=raw["processing_time_ms"],
                error_message=raw["error_message"],
                metadata=raw["metadata"],
            )
            results.append(result)

        return results

    def set_vector_store(self, vector_store: Any):
        """Set vector store for indexing."""
        self.vector_store = vector_store


# =============================================================================
# Convenience Functions
# =============================================================================

_default_pipeline: Optional[DocumentPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> DocumentPipeline:
    """Get or create default pipeline instance."""
    global _default_pipeline

    if _default_pipeline is None or config is not None:
        _default_pipeline = DocumentPipeline(config=config)

    return _default_pipeline


async def process_file(
    file_path: str,
    access_tier: int = 10,
    collection: Optional[str] = None,
) -> ProcessingResult:
    """
    Quick function to process a single file.

    Args:
        file_path: Path to file
        access_tier: Document access tier
        collection: Collection name

    Returns:
        ProcessingResult
    """
    pipeline = get_pipeline()
    return await pipeline.process_document(
        file_path=file_path,
        access_tier=access_tier,
        collection=collection,
    )
