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

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum
import hashlib
import asyncio
import uuid
import os
import structlog

import ray
from sqlalchemy import select

from backend.processors.universal import UniversalProcessor, ExtractedContent, ExtractedImage
from backend.db.models import ProcessingMode, Document as DocumentModel, AccessTier, ProcessingStatus as DBProcessingStatus, StorageMode
from backend.services.multimodal_rag import MultimodalRAGService, get_multimodal_rag_service
from backend.db.database import async_session_context
from backend.processors.chunker import DocumentChunker, ChunkingConfig, ChunkingStrategy, Chunk
from backend.services.embeddings import EmbeddingService, RayEmbeddingService, EmbeddingResult
from backend.services.vectorstore import VectorStore, get_vector_store
from backend.services.text_preprocessor import TextPreprocessor, PreprocessingConfig, get_text_preprocessor
from backend.services.summarizer import DocumentSummarizer, SummarizationConfig, get_document_summarizer

logger = structlog.get_logger(__name__)


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
        processing_mode: ProcessingMode = ProcessingMode.SMART,
        use_ray: bool = True,

        # Chunking settings
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,

        # Embedding settings (uses env var fallback, can be overridden via Admin UI)
        embedding_provider: str = None,  # Will use DEFAULT_LLM_PROVIDER env var
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

        # Progress callbacks
        on_status_change: Optional[Callable[[str, ProcessingStatus], None]] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.processing_mode = processing_mode
        self.use_ray = use_ray
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        # Use provided value, or fall back to env var, or default to "openai"
        self.embedding_provider = embedding_provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
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
        self._processed_hashes: Dict[str, str] = {}

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

    def _update_status(self, doc_id: str, status: ProcessingStatus):
        """Update processing status via callback."""
        if self.config.on_status_change:
            self.config.on_status_change(doc_id, status)

    def _update_progress(self, doc_id: str, current: int, total: int):
        """Update progress via callback."""
        if self.config.on_progress:
            self.config.on_progress(doc_id, current, total)

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

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
    ) -> ProcessingResult:
        """
        Process a single document through the full pipeline.

        Args:
            file_path: Path to the document file
            document_id: Optional document ID (generated if not provided)
            metadata: Additional metadata to attach
            access_tier: Access tier for the document
            collection: Collection name for organization

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

            result.file_hash = self.compute_file_hash(file_path)

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

            result.page_count = extracted.page_count
            result.word_count = extracted.word_count
            result.metadata = {
                **(metadata or {}),
                **extracted.metadata,
                "access_tier": access_tier,
                "collection": collection,
                "language": extracted.language,
            }

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

            # Step 2.8: Process images for multimodal understanding
            image_chunks = []
            if self._multimodal and extracted.extracted_images:
                logger.info(
                    "Processing images for multimodal understanding",
                    document_id=document_id,
                    image_count=len(extracted.extracted_images),
                )

                image_captions = await self._process_document_images(
                    document_id=document_id,
                    images=extracted.extracted_images,
                )

                if image_captions:
                    # Create chunks for image descriptions
                    for i, (img, caption) in enumerate(zip(extracted.extracted_images, image_captions)):
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
                        "images_processed": len(extracted.extracted_images),
                        "captions_generated": len(image_captions),
                    }

                    logger.info(
                        "Image processing complete",
                        document_id=document_id,
                        captions_generated=len(image_captions),
                    )

            # Step 3: Chunk content
            self._update_status(document_id, ProcessingStatus.CHUNKING)
            self._update_progress(document_id, 2, 5)

            if extracted.pages:
                # Chunk with page preservation
                chunks = self._chunker.chunk_with_pages(
                    pages=[{"content": p.get("text", ""), "page_number": p.get("page", i+1)}
                           for i, p in enumerate(extracted.pages)],
                    metadata=result.metadata,
                    document_id=document_id,
                )
            else:
                # Chunk full text
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

            result.chunks = chunks
            result.chunk_count = len(chunks)

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
                )
                return result

            # Step 4: Generate embeddings
            self._update_status(document_id, ProcessingStatus.EMBEDDING)
            self._update_progress(document_id, 3, 5)

            if chunks:
                if isinstance(self._embeddings, RayEmbeddingService):
                    embeddings = self._embeddings.embed_chunks_parallel(chunks)
                else:
                    embeddings = self._embeddings.embed_chunks(
                        chunks,
                        batch_size=self.config.embedding_batch_size,
                    )
                result.embeddings = embeddings

            # Step 5: Index in vector store (if available)
            self._update_status(document_id, ProcessingStatus.INDEXING)
            self._update_progress(document_id, 4, 5)

            # Index using custom vector store
            if self._custom_vectorstore and chunks and result.embeddings:
                # Look up actual access tier UUID by level
                access_tier_id = await self._get_access_tier_id(access_tier)
                if access_tier_id:
                    # Use original_filename from metadata if available, fallback to path.name
                    original_filename = (metadata or {}).get("original_filename") or path.name
                    await self._index_with_custom_store(
                        document_id=document_id,
                        chunks=chunks,
                        embeddings=result.embeddings,
                        access_tier_id=access_tier_id,
                        filename=original_filename,
                        collection=collection,
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
            )

            # Track hash for deduplication
            self._processed_hashes[result.file_hash] = document_id

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
    ) -> List[str]:
        """
        Process extracted images to generate captions/descriptions.

        Args:
            document_id: Document ID for logging
            images: List of extracted images

        Returns:
            List of caption strings (same order as input images)
        """
        if not self._multimodal or not images:
            return []

        captions = []

        for img in images:
            try:
                # Use the multimodal service to caption the image
                caption = await self._multimodal.caption_image(
                    image_data=img.image_bytes,
                    image_format=img.extension,
                )
                captions.append(caption)

            except Exception as e:
                logger.warning(
                    "Failed to caption image",
                    document_id=document_id,
                    image_index=img.image_index,
                    error=str(e),
                )
                captions.append("")  # Empty caption for failed images

        return captions

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
        Look up access tier UUID by level.

        Args:
            access_tier_level: Integer level of the access tier

        Returns:
            UUID string of the access tier, or None if not found
        """
        try:
            async with async_session_context() as db:
                tier_query = select(AccessTier).where(AccessTier.level == access_tier_level)
                result = await db.execute(tier_query)
                access_tier_obj = result.scalar_one_or_none()

                if access_tier_obj:
                    return str(access_tier_obj.id)

                # Try to find a suitable tier by closest level
                all_tiers_query = select(AccessTier).order_by(AccessTier.level)
                all_result = await db.execute(all_tiers_query)
                all_tiers = all_result.scalars().all()

                if all_tiers:
                    # Find closest tier at or below requested level
                    for tier in reversed(all_tiers):
                        if tier.level <= access_tier_level:
                            logger.info(
                                "Using closest access tier",
                                requested_level=access_tier_level,
                                using_tier=tier.name,
                                using_level=tier.level,
                            )
                            return str(tier.id)

                    # If all tiers are above requested level, use the lowest one
                    logger.info(
                        "Using lowest available access tier",
                        requested_level=access_tier_level,
                        using_tier=all_tiers[0].name,
                    )
                    return str(all_tiers[0].id)

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

            # Add to vector store
            chunk_ids = await self._custom_vectorstore.add_chunks(
                chunks=chunk_data,
                document_id=document_id,
                access_tier_id=access_tier_id,
                document_filename=filename,
                collection=collection,
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
    ) -> None:
        """
        Create or update Document record in database after successful processing.

        This handles both new documents and re-uploads of previously deleted files.
        If a document with the same file_hash exists (even if soft-deleted), we
        update it instead of creating a new one to avoid UNIQUE constraint violations.
        """
        try:
            async with async_session_context() as db:
                # Find or create access tier by level
                tier_query = select(AccessTier).where(AccessTier.level == access_tier)
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

                    await db.commit()

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
                    )

                    db.add(document)
                    await db.commit()

                    logger.info(
                        "Document record created in database",
                        document_id=document_id,
                        filename=path.name,
                        access_tier=access_tier,
                    )

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
            result = asyncio.get_event_loop().run_until_complete(
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

        # Collect results
        raw_results = ray.get(futures)

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
