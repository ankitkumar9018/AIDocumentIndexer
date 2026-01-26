"""
AIDocumentIndexer - Temporary Document Service
===============================================

In-memory storage for temporary documents that allows users to
chat with documents before deciding to save them permanently.

Features:
- Extract text without full processing
- In-memory storage with automatic cleanup
- Optional temporary RAG for large documents
- Session-based document management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid
import asyncio
import structlog
import tiktoken

from backend.processors.universal import UniversalProcessor, ExtractedContent
from backend.processors.chunker import DocumentChunker, ChunkingConfig, ChunkingStrategy
from backend.services.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)

# Token limit for direct context injection (roughly 100K tokens)
MAX_CONTEXT_TOKENS = 100000
# Default session expiry (24 hours)
SESSION_EXPIRY_HOURS = 24


@dataclass
class TempDocument:
    """A temporarily uploaded document."""
    id: str
    filename: str
    content: str  # Extracted text
    token_count: int
    metadata: Dict[str, Any]
    file_path: Optional[str] = None  # Temp file path if stored
    file_size: int = 0
    file_type: str = ""
    # For large docs - temporary chunks and embeddings
    chunks: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TempSession:
    """A temporary document session for a user."""
    id: str
    user_id: str
    documents: Dict[str, TempDocument] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=SESSION_EXPIRY_HOURS))

    @property
    def total_tokens(self) -> int:
        """Get total token count across all documents."""
        return sum(doc.token_count for doc in self.documents.values())

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at


class TempDocumentService:
    """Service for managing temporary document uploads."""

    def __init__(self):
        self._sessions: Dict[str, TempSession] = {}
        self._lock = asyncio.Lock()
        self._processor = UniversalProcessor()
        self._chunker = DocumentChunker()
        self._embedding_service: Optional[EmbeddingService] = None

        # Token counter
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4

    async def create_session(self, user_id: str) -> str:
        """Create a new temporary document session."""
        async with self._lock:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = TempSession(
                id=session_id,
                user_id=user_id,
            )
            logger.info("Created temp session", session_id=session_id, user_id=user_id)
            return session_id

    async def get_session(self, session_id: str) -> Optional[TempSession]:
        """Get a session by ID."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired:
                # Clean up expired session
                del self._sessions[session_id]
                return None
            return session

    async def add_document(
        self,
        session_id: str,
        file_path: str,
        filename: str,
        create_embeddings: bool = False,
    ) -> Optional[TempDocument]:
        """
        Add a document to a temporary session.

        Args:
            session_id: The session ID
            file_path: Path to the uploaded file
            filename: Original filename
            create_embeddings: Whether to create embeddings for large docs

        Returns:
            TempDocument if successful, None otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            logger.warning("Session not found", session_id=session_id)
            return None

        try:
            # Extract content
            path = Path(file_path)
            extracted = await self._processor.process(path)

            if not extracted or not extracted.text:
                logger.warning("No content extracted", filename=filename)
                return None

            # Count tokens
            token_count = self._count_tokens(extracted.text)

            # Create document
            doc_id = str(uuid.uuid4())
            doc = TempDocument(
                id=doc_id,
                filename=filename,
                content=extracted.text,
                token_count=token_count,
                metadata={
                    "page_count": extracted.page_count,
                    "word_count": extracted.word_count,
                    "language": extracted.language,
                    "title": extracted.title,
                    "author": extracted.author,
                },
                file_path=file_path,
                file_size=path.stat().st_size if path.exists() else 0,
                file_type=path.suffix.lower().lstrip('.'),
            )

            # For large documents, create chunks and optionally embeddings
            if token_count > MAX_CONTEXT_TOKENS:
                logger.info(
                    "Document exceeds context limit, creating chunks",
                    filename=filename,
                    token_count=token_count,
                    max_tokens=MAX_CONTEXT_TOKENS,
                )

                # Create chunks
                chunk_config = ChunkingConfig(
                    strategy=ChunkingStrategy.SEMANTIC,
                    chunk_size=512,
                    chunk_overlap=50,
                )
                chunks = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._chunker.chunk_text(extracted.text, chunk_config),
                )
                doc.chunks = [chunk.content for chunk in chunks]

                # Optionally create embeddings
                if create_embeddings and self._embedding_service:
                    try:
                        embedding_results = await self._embedding_service.embed_texts(
                            [chunk.content for chunk in chunks]
                        )
                        doc.embeddings = [r.embedding for r in embedding_results]
                    except Exception as e:
                        logger.warning("Failed to create embeddings", error=str(e))

            # Add to session
            async with self._lock:
                session.documents[doc_id] = doc

            logger.info(
                "Added temp document",
                session_id=session_id,
                doc_id=doc_id,
                filename=filename,
                token_count=token_count,
            )

            return doc

        except Exception as e:
            logger.error("Failed to add temp document", error=str(e), filename=filename)
            return None

    async def remove_document(self, session_id: str, doc_id: str) -> bool:
        """Remove a document from a session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        async with self._lock:
            if doc_id in session.documents:
                del session.documents[doc_id]
                return True
            return False

    async def get_context(
        self,
        session_id: str,
        query: Optional[str] = None,
        max_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> Optional[str]:
        """
        Get the context from temporary documents for a chat.

        Args:
            session_id: The session ID
            query: Optional query for semantic search in large docs
            max_tokens: Maximum tokens to return

        Returns:
            Combined context string or None
        """
        session = await self.get_session(session_id)
        if not session or not session.documents:
            return None

        contexts = []
        remaining_tokens = max_tokens

        for doc in session.documents.values():
            if remaining_tokens <= 0:
                break

            # For small documents, use full content
            if doc.token_count <= MAX_CONTEXT_TOKENS and doc.token_count <= remaining_tokens:
                doc_context = f"\n--- Document: {doc.filename} ---\n{doc.content}\n"
                contexts.append(doc_context)
                remaining_tokens -= doc.token_count

            # For large documents with chunks
            elif doc.chunks:
                if query and doc.embeddings and self._embedding_service:
                    # Semantic search in chunks
                    try:
                        query_embedding = await self._embedding_service.embed_text(query)
                        # Simple cosine similarity search
                        scores = []
                        for i, emb in enumerate(doc.embeddings):
                            score = self._cosine_similarity(query_embedding.embedding, emb)
                            scores.append((i, score))
                        scores.sort(key=lambda x: x[1], reverse=True)

                        # Get top relevant chunks
                        selected_chunks = []
                        chunk_tokens = 0
                        for idx, _ in scores[:10]:  # Top 10 chunks
                            chunk = doc.chunks[idx]
                            chunk_token_count = self._count_tokens(chunk)
                            if chunk_tokens + chunk_token_count <= remaining_tokens:
                                selected_chunks.append(chunk)
                                chunk_tokens += chunk_token_count

                        if selected_chunks:
                            doc_context = f"\n--- Document: {doc.filename} (excerpts) ---\n" + "\n...\n".join(selected_chunks) + "\n"
                            contexts.append(doc_context)
                            remaining_tokens -= chunk_tokens
                    except Exception as e:
                        logger.warning("Semantic search failed", error=str(e))
                else:
                    # Without embeddings, use first chunks that fit
                    selected_chunks = []
                    chunk_tokens = 0
                    for chunk in doc.chunks:
                        chunk_token_count = self._count_tokens(chunk)
                        if chunk_tokens + chunk_token_count <= remaining_tokens:
                            selected_chunks.append(chunk)
                            chunk_tokens += chunk_token_count
                        else:
                            break

                    if selected_chunks:
                        doc_context = f"\n--- Document: {doc.filename} (excerpts) ---\n" + "\n...\n".join(selected_chunks) + "\n"
                        contexts.append(doc_context)
                        remaining_tokens -= chunk_tokens

        if contexts:
            return "\n".join(contexts)
        return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        session = await self.get_session(session_id)
        if not session:
            return None

        return {
            "id": session.id,
            "user_id": session.user_id,
            "document_count": len(session.documents),
            "total_tokens": session.total_tokens,
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "token_count": doc.token_count,
                    "file_size": doc.file_size,
                    "file_type": doc.file_type,
                    "has_chunks": doc.chunks is not None,
                    "has_embeddings": doc.embeddings is not None,
                }
                for doc in session.documents.values()
            ],
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
        }

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its documents."""
        async with self._lock:
            if session_id in self._sessions:
                # Clean up any temp files
                session = self._sessions[session_id]
                for doc in session.documents.values():
                    if doc.file_path:
                        try:
                            Path(doc.file_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                del self._sessions[session_id]
                logger.info("Deleted temp session", session_id=session_id)
                return True
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Clean up all expired sessions. Returns count of cleaned sessions."""
        async with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired
            ]
            for sid in expired:
                session = self._sessions[sid]
                # Clean up temp files
                for doc in session.documents.values():
                    if doc.file_path:
                        try:
                            Path(doc.file_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                del self._sessions[sid]

            if expired:
                logger.info("Cleaned up expired sessions", count=len(expired))
            return len(expired)

    async def promote_to_permanent(
        self,
        session_id: str,
        doc_id: str,
        user_id: str,
        access_tier_id: str,
        collection: Optional[str] = None,
    ) -> Optional[str]:
        """
        Promote a temporary document to permanent storage.

        This triggers the full processing pipeline.

        Returns:
            The permanent document ID if successful, None otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        doc = session.documents.get(doc_id)
        if not doc or not doc.file_path:
            return None

        # Import here to avoid circular dependency
        from backend.services.pipeline import DocumentPipeline, PipelineConfig

        try:
            # Create pipeline and process
            pipeline = DocumentPipeline()
            config = PipelineConfig(
                enable_preprocessing=True,
                enable_embeddings=True,
            )

            result = await pipeline.process_single(
                file_path=doc.file_path,
                user_id=user_id,
                access_tier_id=access_tier_id,
                collection=collection,
                config=config,
            )

            if result and result.status.value == "completed":
                # Remove from temp session
                await self.remove_document(session_id, doc_id)
                return result.document_id

            return None

        except Exception as e:
            logger.error("Failed to promote document", error=str(e), doc_id=doc_id)
            return None


# Global service instance
_temp_document_service: Optional[TempDocumentService] = None


def get_temp_document_service() -> TempDocumentService:
    """Get the global temp document service instance."""
    global _temp_document_service
    if _temp_document_service is None:
        _temp_document_service = TempDocumentService()
    return _temp_document_service
