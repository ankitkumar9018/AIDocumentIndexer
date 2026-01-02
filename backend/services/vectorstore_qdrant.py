"""
AIDocumentIndexer - Qdrant Vector Store
========================================

Qdrant vector database integration for high-performance vector search.

Qdrant (https://qdrant.tech) is recommended for 1-50M scale:
- Written in Rust for maximum performance
- Native hybrid search support (dense + sparse vectors)
- Efficient filtering with payload indexes
- Easy migration from pgvector
- Open-source: Apache 2.0 license

Usage:
    store = QdrantVectorStore(url="localhost:6333")
    await store.initialize()
    await store.add_chunks(chunks, document_id, access_tier_id)
    results = await store.search(query, query_embedding)
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import structlog

logger = structlog.get_logger(__name__)

# Check for Qdrant client
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        Range,
        MatchValue,
        SearchRequest,
        SparseVector,
        SparseVectorParams,
        SparseIndexParams,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantClient = None


@dataclass
class QdrantSearchResult:
    """Search result from Qdrant."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_title: Optional[str] = None
    document_filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    collection: Optional[str] = None


class QdrantVectorStore:
    """
    Qdrant vector store for high-performance similarity search.

    Features:
    - HNSW indexing for fast ANN search
    - Hybrid search with sparse vectors
    - Efficient payload filtering
    - Automatic collection management
    """

    def __init__(
        self,
        url: str = "localhost:6333",
        collection_name: str = "documents",
        vector_size: int = 1536,
        enable_hybrid: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            url: Qdrant server URL (host:port)
            collection_name: Name of the collection
            vector_size: Embedding dimension (default: 1536 for OpenAI)
            enable_hybrid: Enable sparse vectors for hybrid search
            api_key: Optional API key for Qdrant Cloud
        """
        if not HAS_QDRANT:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")

        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.enable_hybrid = enable_hybrid
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")

        self._client: Optional[AsyncQdrantClient] = None
        self._initialized = False

        logger.info(
            "Qdrant vector store configured",
            url=url,
            collection=collection_name,
            vector_size=vector_size,
        )

    async def initialize(self) -> bool:
        """
        Initialize Qdrant client and create collection if needed.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Parse URL
            if ":" in self.url:
                host, port = self.url.rsplit(":", 1)
                port = int(port)
            else:
                host = self.url
                port = 6333

            # Create async client
            self._client = AsyncQdrantClient(
                host=host,
                port=port,
                api_key=self.api_key,
            )

            # Check if collection exists
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection
                vectors_config = {
                    "dense": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    )
                }

                sparse_vectors_config = None
                if self.enable_hybrid:
                    sparse_vectors_config = {
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(),
                        )
                    }

                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )

                logger.info("Created Qdrant collection", collection=self.collection_name)

            self._initialized = True
            logger.info("Qdrant initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize Qdrant", error=str(e))
            return False

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        access_tier_id: str,
        **kwargs,
    ) -> List[str]:
        """
        Add chunks with embeddings to Qdrant.

        Args:
            chunks: List of chunk dicts with 'content', 'embedding', etc.
            document_id: Parent document ID
            access_tier_id: Access tier for filtering

        Returns:
            List of chunk IDs
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return []

        points = []
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            embedding = chunk.get("embedding", [])
            if not embedding:
                logger.warning(f"Chunk {i} has no embedding, skipping")
                continue

            # Build payload
            payload = {
                "document_id": document_id,
                "access_tier_id": access_tier_id,
                "content": chunk.get("content", ""),
                "chunk_index": chunk.get("chunk_index", i),
                "page_number": chunk.get("page_number"),
                "section_title": chunk.get("section_title"),
                "document_title": chunk.get("document_title"),
                "document_filename": chunk.get("document_filename"),
                "collection": chunk.get("collection"),
                "metadata": chunk.get("metadata", {}),
            }

            point = PointStruct(
                id=chunk_id,
                vector={"dense": embedding},
                payload=payload,
            )
            points.append(point)

        # Upsert points
        if points:
            await self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info("Added chunks to Qdrant", count=len(points), document_id=document_id)

        return chunk_ids

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[QdrantSearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Search query (for logging)
            query_embedding: Query embedding vector
            top_k: Number of results
            access_tier_level: Access tier filter
            document_ids: Optional document filter

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if not query_embedding:
            logger.warning("No query embedding provided for Qdrant search")
            return []

        # Build filter
        must_conditions = []

        # Access tier filter (simplified - assumes tier IDs map to levels)
        # In production, you'd query tier levels from payload
        if access_tier_level < 100:
            must_conditions.append(
                FieldCondition(
                    key="access_tier_level",
                    range=Range(lte=access_tier_level),
                )
            )

        # Document filter
        if document_ids:
            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(any=document_ids),
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Search
        results = await self._client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_embedding),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        # Convert to SearchResult
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(QdrantSearchResult(
                chunk_id=str(hit.id),
                document_id=payload.get("document_id", ""),
                content=payload.get("content", ""),
                score=float(hit.score),
                similarity_score=float(hit.score),
                metadata=payload.get("metadata", {}),
                document_title=payload.get("document_title"),
                document_filename=payload.get("document_filename"),
                page_number=payload.get("page_number"),
                section_title=payload.get("section_title"),
                collection=payload.get("collection"),
            ))

        logger.debug("Qdrant search complete", results=len(search_results), top_k=top_k)
        return search_results

    async def delete_document_chunks(
        self,
        document_id: str,
        **kwargs,
    ) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted (estimated)
        """
        if not self._initialized:
            await self.initialize()

        # Delete by filter
        result = await self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

        logger.info("Deleted document chunks from Qdrant", document_id=document_id)
        return 1  # Qdrant doesn't return count directly

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._initialized:
            await self.initialize()

        info = await self._client.get_collection(self.collection_name)

        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }

    async def close(self):
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False
            logger.info("Qdrant client closed")
