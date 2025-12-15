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
import structlog

import ray

from backend.processors.universal import UniversalProcessor, ExtractedContent
from backend.db.models import ProcessingMode
from backend.processors.chunker import DocumentChunker, ChunkingConfig, ChunkingStrategy, Chunk
from backend.services.embeddings import EmbeddingService, RayEmbeddingService, EmbeddingResult
from backend.services.vectorstore import VectorStore, get_vector_store

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

        # Embedding settings
        embedding_provider: str = "openai",
        embedding_model: Optional[str] = None,
        embedding_batch_size: int = 100,

        # Deduplication
        check_duplicates: bool = True,

        # Progress callbacks
        on_status_change: Optional[Callable[[str, ProcessingStatus], None]] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.processing_mode = processing_mode
        self.use_ray = use_ray
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.check_duplicates = check_duplicates
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

        # Track processed file hashes for deduplication
        self._processed_hashes: Dict[str, str] = {}

        logger.info(
            "Initialized document pipeline",
            use_ray=self.config.use_ray,
            chunking_strategy=self.config.chunking_strategy.value,
            embedding_provider=self.config.embedding_provider,
            use_custom_vectorstore=use_custom_vectorstore,
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

            # Check for duplicates
            if self.config.check_duplicates:
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

            result.chunks = chunks
            result.chunk_count = len(chunks)

            logger.debug(
                "Document chunked",
                document_id=document_id,
                num_chunks=len(chunks),
            )

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
                await self._index_with_custom_store(
                    document_id=document_id,
                    chunks=chunks,
                    embeddings=result.embeddings,
                    access_tier_id=str(access_tier),  # Will need actual UUID in production
                )
            # Fallback to LangChain vector store
            elif self.vector_store and chunks:
                await self._index_document(document_id, chunks, result.embeddings)

            # Mark as completed
            self._update_status(document_id, ProcessingStatus.COMPLETED)
            self._update_progress(document_id, 5, 5)
            result.status = ProcessingStatus.COMPLETED

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

    async def _index_with_custom_store(
        self,
        document_id: str,
        chunks: List[Chunk],
        embeddings: List[EmbeddingResult],
        access_tier_id: str,
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
                    "token_count": embedding.token_count,
                    "char_count": len(chunk.content),
                })

            # Add to vector store
            chunk_ids = await self._custom_vectorstore.add_chunks(
                chunks=chunk_data,
                document_id=document_id,
                access_tier_id=access_tier_id,
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
