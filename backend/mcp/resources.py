"""
AIDocumentIndexer - MCP Resources
=================================

Resource providers for Model Context Protocol.
Resources allow LLM agents to access document content directly.

Resource URIs:
- aidoc://documents - List all documents
- aidoc://documents/{id} - Get document content
- aidoc://collections - List all collections
- aidoc://collections/{id} - Get collection with documents
- aidoc://search?q={query} - Search results as resource
"""

import json
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Resource Templates
# =============================================================================

RESOURCE_TEMPLATES = [
    {
        "uri_template": "aidoc://documents",
        "name": "All Documents",
        "description": "List of all indexed documents",
        "mime_type": "application/json",
        "handler": "list_documents",
    },
    {
        "uri_template": "aidoc://documents/{document_id}",
        "name": "Document Content",
        "description": "Full content and metadata of a specific document",
        "mime_type": "text/plain",
        "handler": "get_document",
    },
    {
        "uri_template": "aidoc://collections",
        "name": "All Collections",
        "description": "List of all document collections",
        "mime_type": "application/json",
        "handler": "list_collections",
    },
    {
        "uri_template": "aidoc://collections/{collection_id}",
        "name": "Collection Details",
        "description": "Collection metadata and document list",
        "mime_type": "application/json",
        "handler": "get_collection",
    },
    {
        "uri_template": "aidoc://search",
        "name": "Search Results",
        "description": "Search documents (use ?q=query parameter)",
        "mime_type": "application/json",
        "handler": "search",
    },
    {
        "uri_template": "aidoc://knowledge-graph",
        "name": "Knowledge Graph",
        "description": "Entities and relationships from knowledge graph",
        "mime_type": "application/json",
        "handler": "get_knowledge_graph",
    },
]


# =============================================================================
# Resource Provider
# =============================================================================

class MCPResourceProvider:
    """
    Provider for MCP resources.

    Exposes AIDocumentIndexer data as addressable resources
    that can be accessed by LLM agents.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._templates: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize resource provider."""
        if self._initialized:
            return

        # Register templates
        self._templates = RESOURCE_TEMPLATES

        # Register handlers
        self._handlers["list_documents"] = self._handle_list_documents
        self._handlers["get_document"] = self._handle_get_document
        self._handlers["list_collections"] = self._handle_list_collections
        self._handlers["get_collection"] = self._handle_get_collection
        self._handlers["search"] = self._handle_search
        self._handlers["get_knowledge_graph"] = self._handle_get_knowledge_graph

        self._initialized = True
        logger.info(
            "MCP resources initialized",
            templates=len(self._templates),
        )

    def get_handlers(self) -> Dict[str, Callable]:
        """Get all resource handlers."""
        return self._handlers

    def get_templates(self) -> List[Dict[str, Any]]:
        """Get all resource templates."""
        return self._templates

    # =========================================================================
    # Resource Handlers
    # =========================================================================

    async def _handle_list_documents(
        self,
        uri: Optional[str] = None,
        list_only: bool = False,
    ) -> Any:
        """Handle documents list resource."""
        try:
            from backend.db.database import get_async_session
            from backend.db.models import Document
            from sqlalchemy import select

            async with get_async_session() as session:
                query = select(Document).limit(100)
                result = await session.execute(query)
                documents = result.scalars().all()

                if list_only:
                    # Return list of resource descriptors
                    return [
                        {
                            "uri": f"aidoc://documents/{doc.id}",
                            "name": doc.name,
                            "description": f"{doc.file_type} - {doc.file_size} bytes",
                            "mimeType": self._get_mime_type(doc.file_type),
                        }
                        for doc in documents
                    ]

                # Return full content
                return {
                    "uri": "aidoc://documents",
                    "mimeType": "application/json",
                    "text": json.dumps({
                        "documents": [
                            {
                                "id": str(doc.id),
                                "name": doc.name,
                                "file_type": doc.file_type,
                                "file_size": doc.file_size,
                                "chunk_count": doc.chunk_count,
                            }
                            for doc in documents
                        ],
                        "total": len(documents),
                    }, indent=2),
                }

        except Exception as e:
            logger.error("List documents resource error", error=str(e))
            return {"uri": uri, "mimeType": "text/plain", "text": f"Error: {str(e)}"}

    async def _handle_get_document(
        self,
        uri: Optional[str] = None,
        list_only: bool = False,
    ) -> Any:
        """Handle single document resource."""
        if list_only:
            return []  # Individual documents listed via list_documents

        if not uri:
            return {"uri": uri, "mimeType": "text/plain", "text": "Error: No URI provided"}

        # Extract document ID from URI
        # Format: aidoc://documents/{document_id}
        parts = uri.replace("aidoc://documents/", "").split("/")
        document_id_str = parts[0] if parts else None

        if not document_id_str:
            return {"uri": uri, "mimeType": "text/plain", "text": "Error: Invalid document URI"}

        try:
            from uuid import UUID as PyUUID
            from backend.db.database import get_async_session
            from backend.db.models import Document, Chunk
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            try:
                document_id = PyUUID(document_id_str)
            except ValueError:
                return {"uri": uri, "mimeType": "text/plain", "text": "Error: Invalid document ID format"}

            async with get_async_session() as session:
                query = (
                    select(Document)
                    .where(Document.id == document_id)
                    .options(selectinload(Document.chunks))
                )
                result = await session.execute(query)
                doc = result.scalar_one_or_none()

                if not doc:
                    return {
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": f"Document not found: {document_id}",
                    }

                # Combine all chunks into full content
                content_parts = []
                content_parts.append(f"# {doc.name}")
                content_parts.append(f"\nFile Type: {doc.file_type}")
                content_parts.append(f"Size: {doc.file_size} bytes")
                if doc.summary:
                    content_parts.append(f"\n## Summary\n{doc.summary}")
                if doc.tags:
                    content_parts.append(f"\nTags: {', '.join(doc.tags)}")

                content_parts.append("\n\n## Content\n")

                if doc.chunks:
                    for chunk in sorted(doc.chunks, key=lambda c: c.chunk_index):
                        content_parts.append(chunk.content)
                        content_parts.append("\n---\n")

                return {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": "\n".join(content_parts),
                }

        except Exception as e:
            logger.error("Get document resource error", error=str(e))
            return {"uri": uri, "mimeType": "text/plain", "text": f"Error: {str(e)}"}

    async def _handle_list_collections(
        self,
        uri: Optional[str] = None,
        list_only: bool = False,
    ) -> Any:
        """Handle collections list resource."""
        try:
            from backend.db.database import get_async_session
            from backend.db.models import Collection
            from sqlalchemy import select

            async with get_async_session() as session:
                query = select(Collection).limit(50)
                result = await session.execute(query)
                collections = result.scalars().all()

                if list_only:
                    return [
                        {
                            "uri": f"aidoc://collections/{col.id}",
                            "name": col.name,
                            "description": col.description or "No description",
                            "mimeType": "application/json",
                        }
                        for col in collections
                    ]

                return {
                    "uri": "aidoc://collections",
                    "mimeType": "application/json",
                    "text": json.dumps({
                        "collections": [
                            {
                                "id": str(col.id),
                                "name": col.name,
                                "description": col.description,
                            }
                            for col in collections
                        ],
                        "total": len(collections),
                    }, indent=2),
                }

        except Exception as e:
            logger.error("List collections resource error", error=str(e))
            return {"uri": uri, "mimeType": "text/plain", "text": f"Error: {str(e)}"}

    async def _handle_get_collection(
        self,
        uri: Optional[str] = None,
        list_only: bool = False,
    ) -> Any:
        """Handle single collection resource."""
        if list_only:
            return []

        if not uri:
            return {"uri": uri, "mimeType": "text/plain", "text": "Error: No URI provided"}

        # Extract collection ID
        parts = uri.replace("aidoc://collections/", "").split("/")
        collection_id = parts[0] if parts else None

        if not collection_id:
            return {"uri": uri, "mimeType": "text/plain", "text": "Error: Invalid collection URI"}

        try:
            from backend.db.database import get_async_session
            from backend.db.models import Collection, Document
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with get_async_session() as session:
                query = (
                    select(Collection)
                    .where(Collection.id == collection_id)
                    .options(selectinload(Collection.documents))
                )
                result = await session.execute(query)
                col = result.scalar_one_or_none()

                if not col:
                    return {
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": f"Collection not found: {collection_id}",
                    }

                return {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps({
                        "id": str(col.id),
                        "name": col.name,
                        "description": col.description,
                        "documents": [
                            {
                                "id": str(doc.id),
                                "name": doc.name,
                                "file_type": doc.file_type,
                            }
                            for doc in (col.documents or [])[:50]
                        ],
                        "document_count": len(col.documents) if col.documents else 0,
                    }, indent=2),
                }

        except Exception as e:
            logger.error("Get collection resource error", error=str(e))
            return {"uri": uri, "mimeType": "text/plain", "text": f"Error: {str(e)}"}

    async def _handle_search(
        self,
        uri: Optional[str] = None,
        list_only: bool = False,
    ) -> Any:
        """Handle search resource."""
        if list_only:
            return []  # Search is query-based, not listable

        if not uri:
            return {"uri": uri, "mimeType": "text/plain", "text": "Error: No URI provided"}

        # Parse query parameter
        parsed = urlparse(uri)
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]

        if not query:
            return {
                "uri": uri,
                "mimeType": "text/plain",
                "text": "Error: Missing 'q' query parameter. Use aidoc://search?q=your+query",
            }

        try:
            from backend.services.rag import get_rag_service

            rag = get_rag_service()
            results = await rag.search(query=query, top_k=20)

            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps({
                    "query": query,
                    "results": [
                        {
                            "content": r.get("content", ""),
                            "score": r.get("score", 0),
                            "document_name": r.get("document_name"),
                            "document_id": r.get("document_id"),
                        }
                        for r in results
                    ],
                    "total": len(results),
                }, indent=2),
            }

        except Exception as e:
            logger.error("Search resource error", error=str(e))
            return {"uri": uri, "mimeType": "text/plain", "text": f"Error: {str(e)}"}

    async def _handle_get_knowledge_graph(
        self,
        uri: Optional[str] = None,
        list_only: bool = False,
    ) -> Any:
        """Handle knowledge graph resource."""
        if list_only:
            return [{
                "uri": "aidoc://knowledge-graph",
                "name": "Knowledge Graph",
                "description": "Entities and relationships extracted from documents",
                "mimeType": "application/json",
            }]

        try:
            from backend.services.knowledge_graph import get_knowledge_graph_service

            kg_service = get_knowledge_graph_service()
            entities = await kg_service.search_entities(limit=100)

            return {
                "uri": "aidoc://knowledge-graph",
                "mimeType": "application/json",
                "text": json.dumps({
                    "entities": [
                        {
                            "id": str(e.get("id")),
                            "name": e.get("name"),
                            "type": e.get("type"),
                            "description": e.get("description"),
                        }
                        for e in entities
                    ],
                    "total": len(entities),
                }, indent=2),
            }

        except Exception as e:
            logger.error("Knowledge graph resource error", error=str(e))
            return {"uri": uri, "mimeType": "text/plain", "text": f"Error: {str(e)}"}

    def _get_mime_type(self, file_type: str) -> str:
        """Get MIME type for file type."""
        mime_map = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc": "application/msword",
            "txt": "text/plain",
            "md": "text/markdown",
            "html": "text/html",
            "csv": "text/csv",
            "json": "application/json",
            "xml": "application/xml",
        }
        return mime_map.get(file_type.lower(), "application/octet-stream")


# =============================================================================
# Singleton
# =============================================================================

_resource_provider: Optional[MCPResourceProvider] = None


def get_resource_provider() -> MCPResourceProvider:
    """Get or create the resource provider singleton."""
    global _resource_provider
    if _resource_provider is None:
        _resource_provider = MCPResourceProvider()
    return _resource_provider
