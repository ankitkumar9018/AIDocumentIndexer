"""
AIDocumentIndexer - Milvus Vector Store
========================================

Milvus distributed vector database integration for large-scale vector search.

Milvus (https://milvus.io) is recommended for 50M+ scale:
- Distributed architecture for horizontal scaling
- Multiple index types (HNSW, IVF, ScaNN)
- GPU acceleration support
- Kubernetes-native deployment
- Open-source: Apache 2.0 license

Usage:
    store = MilvusVectorStore(host="localhost", port=19530)
    await store.initialize()
    await store.add_chunks(chunks, document_id, access_tier_id)
    results = await store.search(query, query_embedding)
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import asyncio

import structlog

logger = structlog.get_logger(__name__)

# Check for Milvus client
try:
    from pymilvus import (
        connections,
        Collection,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
    )
    from pymilvus.orm.types import CONSISTENCY_STRONG
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False


@dataclass
class MilvusSearchResult:
    """Search result from Milvus."""
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


class MilvusVectorStore:
    """
    Milvus vector store for large-scale similarity search.

    Features:
    - Multiple index types (HNSW, IVF_FLAT, IVF_SQ8, etc.)
    - Distributed search across shards
    - Dynamic schema with scalar filtering
    - GPU-accelerated indexing and search
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "documents",
        vector_size: int = 1536,
        index_type: str = "HNSW",
        metric_type: str = "COSINE",
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize Milvus vector store.

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection
            vector_size: Embedding dimension
            index_type: Index type (HNSW, IVF_FLAT, IVF_SQ8)
            metric_type: Distance metric (COSINE, L2, IP)
            user: Optional username
            password: Optional password
        """
        if not HAS_MILVUS:
            raise ImportError("pymilvus not installed. Install with: pip install pymilvus")

        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.index_type = index_type
        self.metric_type = metric_type
        self.user = user or os.getenv("MILVUS_USER")
        self.password = password or os.getenv("MILVUS_PASSWORD")

        self._collection: Optional[Collection] = None
        self._initialized = False
        self._connection_alias = f"milvus_{collection_name}"

        logger.info(
            "Milvus vector store configured",
            host=host,
            port=port,
            collection=collection_name,
            index_type=index_type,
        )

    async def initialize(self) -> bool:
        """
        Initialize Milvus connection and create collection if needed.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Run sync operations in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_initialize)
            self._initialized = True
            logger.info("Milvus initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize Milvus", error=str(e))
            return False

    def _sync_initialize(self):
        """Synchronous initialization (Milvus SDK is sync)."""
        # Connect to Milvus
        connect_params = {
            "alias": self._connection_alias,
            "host": self.host,
            "port": self.port,
        }

        if self.user and self.password:
            connect_params["user"] = self.user
            connect_params["password"] = self.password

        connections.connect(**connect_params)

        # Check if collection exists
        if utility.has_collection(self.collection_name, using=self._connection_alias):
            self._collection = Collection(
                name=self.collection_name,
                using=self._connection_alias,
            )
            logger.info("Connected to existing Milvus collection", collection=self.collection_name)
        else:
            # Create collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="access_tier_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="document_filename", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_size),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Document chunks for RAG",
            )

            self._collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self._connection_alias,
            )

            # Create index
            index_params = self._get_index_params()
            self._collection.create_index(
                field_name="embedding",
                index_params=index_params,
            )

            # Create scalar indexes for filtering
            self._collection.create_index(
                field_name="document_id",
                index_name="idx_document_id",
            )

            logger.info("Created Milvus collection with index", collection=self.collection_name)

        # Load collection into memory
        self._collection.load()

    def _get_index_params(self) -> Dict[str, Any]:
        """Get index parameters based on index type."""
        if self.index_type == "HNSW":
            return {
                "index_type": "HNSW",
                "metric_type": self.metric_type,
                "params": {
                    "M": 16,
                    "efConstruction": 200,
                },
            }
        elif self.index_type == "IVF_FLAT":
            return {
                "index_type": "IVF_FLAT",
                "metric_type": self.metric_type,
                "params": {
                    "nlist": 1024,
                },
            }
        elif self.index_type == "IVF_SQ8":
            return {
                "index_type": "IVF_SQ8",
                "metric_type": self.metric_type,
                "params": {
                    "nlist": 1024,
                },
            }
        else:
            # Default to HNSW
            return {
                "index_type": "HNSW",
                "metric_type": self.metric_type,
                "params": {
                    "M": 16,
                    "efConstruction": 200,
                },
            }

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        access_tier_id: str,
        **kwargs,
    ) -> List[str]:
        """
        Add chunks with embeddings to Milvus.

        Args:
            chunks: List of chunk dicts
            document_id: Parent document ID
            access_tier_id: Access tier for filtering

        Returns:
            List of chunk IDs
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return []

        # Prepare data for insertion
        ids = []
        document_ids = []
        access_tier_ids = []
        contents = []
        chunk_indices = []
        page_numbers = []
        document_titles = []
        document_filenames = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            embedding = chunk.get("embedding", [])
            if not embedding or len(embedding) != self.vector_size:
                logger.warning(f"Chunk {i} has invalid embedding, skipping")
                continue

            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            document_ids.append(document_id)
            access_tier_ids.append(access_tier_id)
            contents.append(chunk.get("content", "")[:65535])  # Truncate to max length
            chunk_indices.append(chunk.get("chunk_index", i))
            page_numbers.append(chunk.get("page_number") or 0)
            document_titles.append((chunk.get("document_title") or "")[:1024])
            document_filenames.append((chunk.get("document_filename") or "")[:1024])
            embeddings.append(embedding)

        if not ids:
            return []

        # Insert data
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.insert([
                ids,
                document_ids,
                access_tier_ids,
                contents,
                chunk_indices,
                page_numbers,
                document_titles,
                document_filenames,
                embeddings,
            ])
        )

        logger.info("Added chunks to Milvus", count=len(ids), document_id=document_id)
        return ids

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[MilvusSearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Search query (for logging)
            query_embedding: Query embedding vector
            top_k: Number of results
            access_tier_level: Access tier filter (currently not used directly)
            document_ids: Optional document filter

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if not query_embedding:
            logger.warning("No query embedding provided for Milvus search")
            return []

        # Build expression for filtering
        expr = None
        if document_ids:
            doc_ids_str = ", ".join([f'"{d}"' for d in document_ids])
            expr = f"document_id in [{doc_ids_str}]"

        # Search parameters
        search_params = {
            "metric_type": self.metric_type,
            "params": {"ef": 100} if self.index_type == "HNSW" else {"nprobe": 16},
        }

        # Run search
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["document_id", "content", "chunk_index", "page_number", "document_title", "document_filename"],
            )
        )

        # Convert results
        search_results = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                search_results.append(MilvusSearchResult(
                    chunk_id=str(hit.id),
                    document_id=entity.get("document_id", ""),
                    content=entity.get("content", ""),
                    score=float(hit.score),
                    similarity_score=float(hit.score),
                    metadata={"chunk_index": entity.get("chunk_index")},
                    document_title=entity.get("document_title"),
                    document_filename=entity.get("document_filename"),
                    page_number=entity.get("page_number"),
                ))

        logger.debug("Milvus search complete", results=len(search_results), top_k=top_k)
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
            Number of chunks deleted
        """
        if not self._initialized:
            await self.initialize()

        expr = f'document_id == "{document_id}"'

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._collection.delete(expr)
        )

        logger.info("Deleted document chunks from Milvus", document_id=document_id)
        return 1  # Milvus doesn't return count directly

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            lambda: self._collection.get_stats()
        )

        return {
            "name": self.collection_name,
            "row_count": stats.get("row_count", 0),
            "index_type": self.index_type,
            "metric_type": self.metric_type,
        }

    async def close(self):
        """Close the connection."""
        if self._collection:
            self._collection.release()

        connections.disconnect(self._connection_alias)
        self._collection = None
        self._initialized = False
        logger.info("Milvus connection closed")
