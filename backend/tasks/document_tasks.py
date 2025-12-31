"""
AIDocumentIndexer - Document Processing Tasks
==============================================

Celery tasks for asynchronous document processing.
These tasks handle document ingestion, OCR, chunking, and embedding
in the background, allowing the API to return immediately.
"""

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional
from uuid import UUID

from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


def run_async(coro):
    """Helper to run async code in Celery tasks."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@shared_task(bind=True, name="backend.tasks.document_tasks.process_document_task")
def process_document_task(
    self,
    file_path: str,
    original_filename: str,
    user_id: str,
    collection: Optional[str] = None,
    access_tier: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single document asynchronously.

    Args:
        file_path: Path to the uploaded file
        original_filename: Original name of the file
        user_id: ID of the user who uploaded
        collection: Optional collection/tag
        access_tier: Access tier level
        metadata: Additional metadata

    Returns:
        Dict with document_id and processing result
    """
    logger.info(f"Starting document processing: {original_filename}")

    # Update task state to show progress
    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 10,
            "message": "Initializing document processing...",
            "filename": original_filename,
        }
    )

    try:
        # Import here to avoid circular imports
        from backend.db.database import async_session_context
        from backend.services.pipeline import DocumentPipeline

        async def _process():
            pipeline = DocumentPipeline()

            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": 20,
                    "message": "Reading file...",
                    "filename": original_filename,
                }
            )

            # Read file content
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": 30,
                    "message": "Processing document...",
                    "filename": original_filename,
                }
            )

            # Process the document
            async with async_session_context() as session:
                result = await pipeline.process_document(
                    file_content=file_content,
                    filename=original_filename,
                    user_id=user_id,
                    collection=collection,
                    access_tier=access_tier,
                    metadata=metadata,
                    session=session,
                )

            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": 90,
                    "message": "Finalizing...",
                    "filename": original_filename,
                }
            )

            return result

        result = run_async(_process())

        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

        logger.info(f"Document processed successfully: {original_filename}")
        return {
            "status": "success",
            "document_id": str(result.document_id) if hasattr(result, "document_id") else None,
            "filename": original_filename,
            "chunks": result.chunk_count if hasattr(result, "chunk_count") else 0,
        }

    except Exception as e:
        logger.error(f"Document processing failed: {original_filename} - {str(e)}")
        # Clean up temp file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


@shared_task(bind=True, name="backend.tasks.document_tasks.process_batch_task")
def process_batch_task(
    self,
    file_infos: List[Dict[str, Any]],
    user_id: str,
    collection: Optional[str] = None,
    access_tier: int = 1,
) -> Dict[str, Any]:
    """
    Process multiple documents as a batch.

    Args:
        file_infos: List of dicts with file_path, original_filename
        user_id: ID of the user who uploaded
        collection: Optional collection/tag for all files
        access_tier: Access tier level for all files

    Returns:
        Dict with batch results
    """
    total = len(file_infos)
    logger.info(f"Starting batch processing: {total} documents")

    results = {
        "total": total,
        "successful": 0,
        "failed": 0,
        "documents": [],
        "errors": [],
    }

    for idx, file_info in enumerate(file_infos):
        progress = int((idx / total) * 100)

        self.update_state(
            state="PROGRESS",
            meta={
                "progress": progress,
                "current": idx + 1,
                "total": total,
                "message": f"Processing {file_info['original_filename']}...",
            }
        )

        try:
            # Process each document
            doc_result = process_document_task.apply(
                args=[
                    file_info["file_path"],
                    file_info["original_filename"],
                    user_id,
                    collection,
                    access_tier,
                    file_info.get("metadata"),
                ]
            ).get(timeout=600)  # 10 minute timeout per document

            results["successful"] += 1
            results["documents"].append(doc_result)

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "filename": file_info["original_filename"],
                "error": str(e),
            })
            logger.error(f"Batch item failed: {file_info['original_filename']} - {str(e)}")

    logger.info(
        f"Batch processing complete: {results['successful']}/{total} successful"
    )
    return results


@shared_task(bind=True, name="backend.tasks.document_tasks.reprocess_document_task")
def reprocess_document_task(
    self,
    document_id: str,
    user_id: str,
    force_ocr: bool = False,
) -> Dict[str, Any]:
    """
    Reprocess an existing document.

    Args:
        document_id: UUID of the document to reprocess
        user_id: ID of the user requesting reprocessing
        force_ocr: Force OCR even if text was extracted

    Returns:
        Dict with reprocessing result
    """
    logger.info(f"Starting document reprocessing: {document_id}")

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 10,
            "message": "Loading document...",
            "document_id": document_id,
        }
    )

    try:
        from backend.db.database import async_session_context
        from backend.db.models import Document
        from backend.services.pipeline import DocumentPipeline
        from sqlalchemy import select

        async def _reprocess():
            async with async_session_context() as session:
                # Get document
                result = await session.execute(
                    select(Document).where(Document.id == document_id)
                )
                document = result.scalar_one_or_none()

                if not document:
                    raise ValueError(f"Document not found: {document_id}")

                self.update_state(
                    state="PROGRESS",
                    meta={
                        "progress": 30,
                        "message": "Reprocessing document...",
                        "document_id": document_id,
                    }
                )

                pipeline = DocumentPipeline()
                result = await pipeline.reprocess_document(
                    document_id=document_id,
                    session=session,
                    force_ocr=force_ocr,
                )

                return result

        result = run_async(_reprocess())

        logger.info(f"Document reprocessed successfully: {document_id}")
        return {
            "status": "success",
            "document_id": document_id,
            "chunks": result.chunk_count if hasattr(result, "chunk_count") else 0,
        }

    except Exception as e:
        logger.error(f"Document reprocessing failed: {document_id} - {str(e)}")
        raise


@shared_task(bind=True, name="backend.tasks.document_tasks.ocr_task")
def ocr_task(
    self,
    file_path: str,
    page_numbers: Optional[List[int]] = None,
    language: str = "eng",
) -> Dict[str, Any]:
    """
    Run OCR on a document or specific pages.

    Args:
        file_path: Path to the file
        page_numbers: Optional list of page numbers to OCR
        language: OCR language

    Returns:
        Dict with OCR results
    """
    logger.info(f"Starting OCR task: {file_path}")

    try:
        from backend.processors.universal import UniversalProcessor

        async def _ocr():
            processor = UniversalProcessor()

            # Read file
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Extract text with OCR
            result = await processor.extract_text(
                file_content=file_content,
                filename=os.path.basename(file_path),
                force_ocr=True,
                ocr_language=language,
            )

            return result

        result = run_async(_ocr())

        return {
            "status": "success",
            "text_length": len(result) if result else 0,
            "pages_processed": len(page_numbers) if page_numbers else "all",
        }

    except Exception as e:
        logger.error(f"OCR task failed: {file_path} - {str(e)}")
        raise


@shared_task(bind=True, name="backend.tasks.document_tasks.embedding_task")
def embedding_task(
    self,
    texts: List[str],
    chunk_ids: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate embeddings for a batch of text chunks.

    Args:
        texts: List of text chunks to embed
        chunk_ids: Corresponding chunk IDs
        model: Optional embedding model to use

    Returns:
        Dict with embedding results
    """
    logger.info(f"Starting embedding task: {len(texts)} chunks")

    try:
        from backend.services.embeddings import get_embedding_service

        async def _embed():
            service = get_embedding_service()
            embeddings = await service.embed_texts(texts, model=model)
            return embeddings

        embeddings = run_async(_embed())

        return {
            "status": "success",
            "count": len(embeddings),
            "chunk_ids": chunk_ids,
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    except Exception as e:
        logger.error(f"Embedding task failed: {str(e)}")
        raise
