"""
AIDocumentIndexer - Local Vector Store Service (ChromaDB)
=========================================================

Provides vector storage and similarity search using ChromaDB.
Alternative to pgvector for local development and testing.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import structlog
import chromadb
from chromadb.config import Settings

from backend.services.vectorstore import SearchResult, VectorStoreConfig, SearchType
from backend.db.models import Chunk as ChunkModel, Document
from backend.db.database import async_session_context
from sqlalchemy import select

logger = structlog.get_logger(__name__)


# =============================================================================
# ChromaDB Configuration
# =============================================================================

@dataclass
class ChromaConfig:
    """Configuration for ChromaDB local vector store."""
    persist_directory: str = "./data/chroma"
    collection_name: str = "documents"
    distance_function: str = "cosine"  # cosine, l2, ip


# =============================================================================
# ChromaDB Vector Store
# =============================================================================

class ChromaVectorStore:
    """
    Local vector storage using ChromaDB.

    Features:
    - Persistent local storage (no server required)
    - Vector similarity search
    - Metadata filtering
    - Compatible with VectorStore interface
    """

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        chroma_config: Optional[ChromaConfig] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            config: General vector store configuration
            chroma_config: ChromaDB-specific configuration
        """
        self.config = config or VectorStoreConfig()
        self.chroma_config = chroma_config or ChromaConfig()

        # Get config from environment
        persist_dir = os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            self.chroma_config.persist_directory
        )
        collection_name = os.getenv(
            "CHROMA_COLLECTION_NAME",
            self.chroma_config.collection_name
        )

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.chroma_config.distance_function},
        )

        logger.info(
            "ChromaDB vector store initialized",
            persist_directory=persist_dir,
            collection=collection_name,
        )

    # =========================================================================
    # Storage Operations
    # =========================================================================

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        access_tier_id: str,
        document_filename: Optional[str] = None,
        collection: Optional[str] = None,
        session: Optional[Any] = None,  # Ignored for ChromaDB
    ) -> List[str]:
        """
        Add chunks with embeddings to ChromaDB and SQLite database.

        Args:
            chunks: List of chunk dictionaries with 'content', 'embedding', and metadata
            document_id: ID of the parent document
            access_tier_id: Access tier for permission filtering
            document_filename: Filename of the source document (for display in search results)
            collection: Collection/tag name for document grouping
            session: Ignored (kept for interface compatibility)

        Returns:
            List of created chunk IDs
        """
        chunk_ids = []
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        db_chunks = []

        for i, chunk_data in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            ids.append(chunk_id)
            embeddings.append(chunk_data.get("embedding", []))
            documents.append(chunk_data["content"])

            # Store metadata for filtering
            metadatas.append({
                "document_id": document_id,
                "access_tier_id": access_tier_id,
                "document_filename": document_filename or "",
                "collection": collection or "",  # Collection/tag for document grouping
                "chunk_index": chunk_data.get("chunk_index", i),
                "page_number": chunk_data.get("page_number") or 0,
                "section_title": chunk_data.get("section_title") or "",
                "token_count": chunk_data.get("token_count") or 0,
                "char_count": chunk_data.get("char_count", len(chunk_data["content"])),
                "content_hash": chunk_data.get("content_hash", ""),
            })

            # Prepare SQLite chunk record (without embedding - stored in ChromaDB)
            db_chunks.append(ChunkModel(
                id=uuid.UUID(chunk_id),
                document_id=uuid.UUID(document_id),
                access_tier_id=uuid.UUID(access_tier_id),
                content=chunk_data["content"],
                content_hash=chunk_data.get("content_hash", ""),
                chunk_index=chunk_data.get("chunk_index", i),
                page_number=chunk_data.get("page_number"),
                section_title=chunk_data.get("section_title"),
                token_count=chunk_data.get("token_count") or len(chunk_data["content"]) // 4,
                char_count=chunk_data.get("char_count", len(chunk_data["content"])),
            ))

        # Add to ChromaDB
        if ids:
            self._collection.add(
                ids=ids,
                embeddings=embeddings if any(embeddings) else None,
                documents=documents,
                metadatas=metadatas,
            )

        # Also store in SQLite database for chunk counting and metadata queries
        if db_chunks:
            try:
                async with async_session_context() as db:
                    db.add_all(db_chunks)
                    await db.commit()
                logger.info(
                    "Added chunks to SQLite database",
                    document_id=document_id,
                    chunk_count=len(db_chunks),
                )
            except Exception as e:
                logger.warning(
                    "Failed to store chunks in SQLite (ChromaDB storage succeeded)",
                    document_id=document_id,
                    error=str(e),
                )

        logger.info(
            "Added chunks to ChromaDB",
            document_id=document_id,
            chunk_count=len(chunk_ids),
        )

        return chunk_ids

    async def delete_document_chunks(
        self,
        document_id: str,
        session: Optional[Any] = None,
    ) -> int:
        """
        Delete all chunks for a document from both ChromaDB and SQLite.

        Args:
            document_id: Document ID
            session: Ignored (kept for interface compatibility)

        Returns:
            Number of chunks deleted
        """
        # Query to find all chunks for this document in ChromaDB
        results = self._collection.get(
            where={"document_id": document_id},
        )

        count = len(results["ids"])

        if count > 0:
            self._collection.delete(
                where={"document_id": document_id},
            )

        # Also delete from SQLite database
        try:
            from sqlalchemy import delete
            async with async_session_context() as db:
                stmt = delete(ChunkModel).where(ChunkModel.document_id == uuid.UUID(document_id))
                result = await db.execute(stmt)
                await db.commit()
                sqlite_count = result.rowcount
                logger.info("Deleted chunks from SQLite", document_id=document_id, count=sqlite_count)
        except Exception as e:
            logger.warning(
                "Failed to delete chunks from SQLite",
                document_id=document_id,
                error=str(e),
            )

        logger.info("Deleted document chunks from ChromaDB", document_id=document_id, count=count)
        return count

    async def update_chunk_embedding(
        self,
        chunk_id: str,
        embedding: List[float],
        session: Optional[Any] = None,
    ) -> bool:
        """
        Update embedding for a specific chunk.

        Args:
            chunk_id: Chunk ID
            embedding: New embedding vector
            session: Ignored (kept for interface compatibility)

        Returns:
            True if updated successfully
        """
        try:
            self._collection.update(
                ids=[chunk_id],
                embeddings=[embedding],
            )
            return True
        except Exception as e:
            logger.error("Failed to update chunk embedding", chunk_id=chunk_id, error=str(e))
            return False

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        session: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            similarity_threshold: Minimum similarity score
            session: Ignored (kept for interface compatibility)

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k

        # Build where clause for filtering
        where_clause = None
        if document_ids:
            where_clause = {"document_id": {"$in": document_ids}}

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Convert distance to similarity (for cosine: similarity = 1 - distance)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                # Apply threshold
                threshold = similarity_threshold or self.config.similarity_threshold
                if similarity < threshold:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""

                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    document_filename=metadata.get("document_filename") or None,
                    collection=metadata.get("collection") or None,
                    content=content,
                    score=similarity,
                    metadata={
                        "chunk_index": metadata.get("chunk_index", 0),
                        "token_count": metadata.get("token_count", 0),
                    },
                    page_number=metadata.get("page_number"),
                    section_title=metadata.get("section_title"),
                ))

        return search_results

    async def keyword_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Perform simple keyword search (case-insensitive substring match).

        Note: ChromaDB doesn't have built-in full-text search, so this is a
        basic implementation. For production use, consider using a dedicated
        search engine like Elasticsearch.

        Args:
            query: Search query string
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            session: Ignored (kept for interface compatibility)

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k

        # Build where clause
        where_clause = None
        if document_ids:
            where_clause = {"document_id": {"$in": document_ids}}

        # ChromaDB supports $contains operator for text search
        # Get all documents and filter manually (not efficient for large datasets)
        results = self._collection.get(
            where=where_clause,
            include=["documents", "metadatas"],
        )

        search_results = []
        query_lower = query.lower()

        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}

                # Simple substring match
                if query_lower in content.lower():
                    # Calculate a simple relevance score based on term frequency
                    tf = content.lower().count(query_lower)
                    score = tf / max(len(content.split()), 1)

                    search_results.append(SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata.get("document_id", ""),
                        document_filename=metadata.get("document_filename") or None,
                        collection=metadata.get("collection") or None,
                        content=content,
                        score=score,
                        metadata={
                            "chunk_index": metadata.get("chunk_index", 0),
                            "search_type": "keyword",
                        },
                        page_number=metadata.get("page_number"),
                        section_title=metadata.get("section_title"),
                    ))

        # Sort by score and limit
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:top_k]

    async def enhanced_metadata_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Search documents using enhanced metadata (summaries, keywords, questions).

        This searches the LLM-extracted metadata stored in Document.enhanced_metadata
        including summaries, keywords, topics, and hypothetical questions.

        Args:
            query: Search query string
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            session: Ignored (kept for interface compatibility)

        Returns:
            List of SearchResult objects (one per matching document)
        """
        import uuid as uuid_module
        top_k = top_k or self.config.default_top_k

        async with async_session_context() as db:
            # Get documents with enhanced metadata
            base_query = (
                select(Document)
                .where(Document.enhanced_metadata.isnot(None))
                .where(Document.processing_status == "completed")
            )

            # Filter by document IDs if provided
            if document_ids:
                doc_uuids = [uuid_module.UUID(d) for d in document_ids]
                base_query = base_query.where(Document.id.in_(doc_uuids))

            result = await db.execute(base_query)
            documents = result.scalars().all()

            if not documents:
                return []

            # Score each document based on query match to enhanced metadata
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            scored_docs = []

            for doc in documents:
                metadata = doc.enhanced_metadata or {}
                score = 0.0

                # Search in summary
                summary_short = (metadata.get("summary_short") or "").lower()
                summary_detailed = (metadata.get("summary_detailed") or "").lower()
                if query_lower in summary_short:
                    score += 0.5
                if query_lower in summary_detailed:
                    score += 0.3

                # Search in keywords
                keywords = [k.lower() for k in metadata.get("keywords", [])]
                keyword_matches = sum(1 for term in query_terms if any(term in k for k in keywords))
                score += keyword_matches * 0.2

                # Search in topics
                topics = [t.lower() for t in metadata.get("topics", [])]
                topic_matches = sum(1 for term in query_terms if any(term in t for t in topics))
                score += topic_matches * 0.15

                # Search in hypothetical questions
                questions = [q.lower() for q in metadata.get("hypothetical_questions", [])]
                for question in questions:
                    question_terms = set(question.split())
                    overlap = len(query_terms & question_terms)
                    if overlap > 0:
                        score += overlap * 0.1

                # Search in entities
                entities = metadata.get("entities", {})
                for entity_type, entity_list in entities.items():
                    entity_values = [e.lower() for e in entity_list]
                    entity_matches = sum(1 for term in query_terms if any(term in e for e in entity_values))
                    score += entity_matches * 0.1

                if score > 0:
                    scored_docs.append((doc, score, metadata))

            # Sort by score and take top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = scored_docs[:top_k]

            # Convert to SearchResult (using first chunk from ChromaDB as representative)
            results = []
            for doc, score, metadata in top_docs:
                doc_id_str = str(doc.id)

                # Get first chunk from ChromaDB
                chroma_results = self._collection.get(
                    where={"document_id": doc_id_str},
                    include=["documents", "metadatas"],
                    limit=1,
                )

                first_chunk_id = ""
                first_chunk_content = ""
                if chroma_results["ids"]:
                    first_chunk_id = chroma_results["ids"][0]
                    first_chunk_content = chroma_results["documents"][0] if chroma_results["documents"] else ""

                summary = metadata.get("summary_detailed") or metadata.get("summary_short") or ""
                content = summary if summary else first_chunk_content

                results.append(SearchResult(
                    chunk_id=first_chunk_id,
                    document_id=doc_id_str,
                    content=content,
                    score=score,
                    metadata={
                        "search_type": "enhanced",
                        "topics": metadata.get("topics", []),
                        "document_type": metadata.get("document_type"),
                    },
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    enhanced_summary=metadata.get("summary_short"),
                    enhanced_keywords=metadata.get("keywords", []),
                ))

            return results

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        use_enhanced: Optional[bool] = None,
        session: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity, keyword matching,
        and optionally enhanced metadata search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from all sources.

        Args:
            query: Search query string
            query_embedding: Query embedding vector
            top_k: Number of final results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            vector_weight: Weight for vector results (0-1)
            keyword_weight: Weight for keyword results (0-1)
            use_enhanced: Whether to include enhanced metadata search
            session: Ignored (kept for interface compatibility)

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k
        vec_weight = vector_weight or self.config.vector_weight
        kw_weight = keyword_weight or self.config.keyword_weight
        enhanced_weight = self.config.enhanced_weight
        use_enhanced = use_enhanced if use_enhanced is not None else self.config.use_enhanced_search

        # Get more results from each method for better fusion
        fetch_k = self.config.rerank_top_k if self.config.enable_reranking else top_k * 2

        # Run searches
        vector_results = await self.similarity_search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
        )

        keyword_results = await self.keyword_search(
            query=query,
            top_k=fetch_k,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
        )

        # Optionally search enhanced metadata
        enhanced_results = []
        if use_enhanced:
            enhanced_results = await self.enhanced_metadata_search(
                query=query,
                top_k=fetch_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
            )

        # Reciprocal Rank Fusion
        k = 60
        scores: Dict[str, Tuple[float, SearchResult]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            rrf_score = vec_weight * (1.0 / (k + rank + 1))
            if result.chunk_id in scores:
                scores[result.chunk_id] = (
                    scores[result.chunk_id][0] + rrf_score,
                    result,
                )
            else:
                scores[result.chunk_id] = (rrf_score, result)

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            rrf_score = kw_weight * (1.0 / (k + rank + 1))
            if result.chunk_id in scores:
                scores[result.chunk_id] = (
                    scores[result.chunk_id][0] + rrf_score,
                    result,
                )
            else:
                scores[result.chunk_id] = (rrf_score, result)

        # Process enhanced metadata results
        enhanced_doc_scores: Dict[str, float] = {}
        for rank, result in enumerate(enhanced_results):
            rrf_score = enhanced_weight * (1.0 / (k + rank + 1))
            enhanced_doc_scores[result.document_id] = rrf_score
            if result.chunk_id:
                if result.chunk_id in scores:
                    existing_score, existing_result = scores[result.chunk_id]
                    existing_result.enhanced_summary = result.enhanced_summary
                    existing_result.enhanced_keywords = result.enhanced_keywords
                    scores[result.chunk_id] = (existing_score + rrf_score, existing_result)
                else:
                    scores[result.chunk_id] = (rrf_score, result)

        # Boost existing chunk scores for documents with enhanced metadata matches
        for chunk_id, (current_score, result) in list(scores.items()):
            if result.document_id in enhanced_doc_scores:
                boost = enhanced_doc_scores[result.document_id] * 0.5
                scores[chunk_id] = (current_score + boost, result)

        # Sort by combined score and return top_k
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        # Update scores in results
        final_results = []
        for score, result in sorted_results:
            result.score = score
            result.metadata["search_type"] = "hybrid"
            if use_enhanced and result.document_id in enhanced_doc_scores:
                result.metadata["enhanced_boost"] = True
            final_results.append(result)

        logger.debug(
            "Hybrid search completed (ChromaDB)",
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            enhanced_count=len(enhanced_results),
            final_count=len(final_results),
        )

        return final_results

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        search_type: Optional[SearchType] = None,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Unified search interface.

        Args:
            query: Search query string
            query_embedding: Optional query embedding (required for vector/hybrid)
            search_type: Type of search to perform
            top_k: Number of results
            access_tier_level: Maximum access tier level
            document_ids: Optional document filter
            session: Ignored (kept for interface compatibility)

        Returns:
            List of SearchResult objects
        """
        search_type = search_type or self.config.search_type

        if search_type == SearchType.VECTOR:
            if not query_embedding:
                raise ValueError("query_embedding required for vector search")
            return await self.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
            )

        elif search_type == SearchType.KEYWORD:
            return await self.keyword_search(
                query=query,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
            )

        else:  # HYBRID
            if not query_embedding:
                logger.warning("No embedding provided for hybrid search, using keyword only")
                return await self.keyword_search(
                    query=query,
                    top_k=top_k,
                    access_tier_level=access_tier_level,
                    document_ids=document_ids,
                )

            return await self.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
            )

    # =========================================================================
    # Utility Operations
    # =========================================================================

    async def get_chunk_by_id(
        self,
        chunk_id: str,
        session: Optional[Any] = None,
    ) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        results = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

        if results["ids"]:
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            content = results["documents"][0] if results["documents"] else ""

            return SearchResult(
                chunk_id=chunk_id,
                document_id=metadata.get("document_id", ""),
                document_filename=metadata.get("document_filename") or None,
                content=content,
                score=1.0,
                metadata={"chunk_index": metadata.get("chunk_index", 0)},
                page_number=metadata.get("page_number"),
                section_title=metadata.get("section_title"),
            )

        return None

    async def get_document_chunks(
        self,
        document_id: str,
        session: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Get all chunks for a document."""
        results = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )

        search_results = []
        if results["ids"]:
            # Pair up results and sort by chunk_index
            items = []
            for i, chunk_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                content = results["documents"][i] if results["documents"] else ""
                items.append((metadata.get("chunk_index", 0), chunk_id, content, metadata))

            items.sort(key=lambda x: x[0])

            for chunk_index, chunk_id, content, metadata in items:
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_filename=metadata.get("document_filename") or None,
                    content=content,
                    score=1.0,
                    metadata={"chunk_index": chunk_index},
                    page_number=metadata.get("page_number"),
                    section_title=metadata.get("section_title"),
                ))

        return search_results

    async def get_stats(
        self,
        session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Get vector store statistics."""
        count = self._collection.count()

        # Get unique document count
        results = self._collection.get(include=["metadatas"])
        doc_ids = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if metadata and "document_id" in metadata:
                    doc_ids.add(metadata["document_id"])

        return {
            "total_chunks": count,
            "embedded_chunks": count,  # ChromaDB requires embeddings
            "total_documents": len(doc_ids),
            "embedding_coverage": 100.0 if count > 0 else 0,
            "backend": "chromadb",
            "persist_directory": os.getenv(
                "CHROMA_PERSIST_DIRECTORY",
                self.chroma_config.persist_directory
            ),
        }

    def reset(self) -> None:
        """Reset the collection (delete all data)."""
        collection_name = self._collection.name
        self._client.delete_collection(collection_name)
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.chroma_config.distance_function},
        )
        logger.info("ChromaDB collection reset", collection=collection_name)


# =============================================================================
# Factory Function
# =============================================================================

_chroma_store: Optional[ChromaVectorStore] = None


def get_chroma_vector_store(
    config: Optional[VectorStoreConfig] = None,
    chroma_config: Optional[ChromaConfig] = None,
) -> ChromaVectorStore:
    """
    Get or create ChromaDB vector store instance.

    Args:
        config: Optional general configuration
        chroma_config: Optional ChromaDB-specific configuration

    Returns:
        ChromaVectorStore instance
    """
    global _chroma_store

    if _chroma_store is None or config is not None or chroma_config is not None:
        _chroma_store = ChromaVectorStore(config=config, chroma_config=chroma_config)

    return _chroma_store
