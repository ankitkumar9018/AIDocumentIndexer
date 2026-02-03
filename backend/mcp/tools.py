"""
AIDocumentIndexer - MCP Tools
=============================

Tool definitions for Model Context Protocol.
These tools enable LLM agents to interact with AIDocumentIndexer.

Available Tools:
- search_documents: Semantic search across documents
- chat: RAG-powered chat with documents
- list_documents: List available documents
- get_document: Get document content
- upload_document: Upload a new document
- list_collections: List document collections
- get_collection: Get collection details
"""

import os
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# =============================================================================
# Tool Schema Definitions
# =============================================================================

TOOL_SCHEMAS = {
    "search_documents": {
        "name": "search_documents",
        "description": "Search through indexed documents using semantic search. Returns relevant passages with sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10,
                },
                "collection_id": {
                    "type": "string",
                    "description": "Optional: Filter by collection ID",
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: Filter by file types (e.g., ['pdf', 'docx'])",
                },
            },
            "required": ["query"],
        },
    },

    "chat": {
        "name": "chat",
        "description": "Have a RAG-powered conversation about your documents. Uses semantic search to find relevant context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Your message or question",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional: Session ID for conversation continuity",
                },
                "collection_id": {
                    "type": "string",
                    "description": "Optional: Limit search to specific collection",
                },
            },
            "required": ["message"],
        },
    },

    "list_documents": {
        "name": "list_documents",
        "description": "List all indexed documents with metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents (default: 50)",
                    "default": 50,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0)",
                    "default": 0,
                },
                "collection_id": {
                    "type": "string",
                    "description": "Optional: Filter by collection",
                },
                "file_type": {
                    "type": "string",
                    "description": "Optional: Filter by file type",
                },
            },
            "required": [],
        },
    },

    "get_document": {
        "name": "get_document",
        "description": "Get full content and metadata of a specific document.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The document ID",
                },
                "include_chunks": {
                    "type": "boolean",
                    "description": "Include document chunks (default: false)",
                    "default": False,
                },
            },
            "required": ["document_id"],
        },
    },

    "list_collections": {
        "name": "list_collections",
        "description": "List all document collections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of collections (default: 50)",
                    "default": 50,
                },
            },
            "required": [],
        },
    },

    "get_knowledge_graph": {
        "name": "get_knowledge_graph",
        "description": "Get entities and relationships from the knowledge graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional: Filter entities by query",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Optional: Filter by entity type (person, organization, concept, etc.)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum entities to return (default: 50)",
                    "default": 50,
                },
            },
            "required": [],
        },
    },
}


# =============================================================================
# Tool Registry
# =============================================================================

class MCPToolRegistry:
    """
    Registry for MCP tools.

    Manages tool schemas and handler functions.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize tool registry with handlers."""
        if self._initialized:
            return

        # Register all tool schemas
        for name, schema in TOOL_SCHEMAS.items():
            self._schemas[name] = schema

        # Register handlers
        self._handlers["search_documents"] = self._handle_search_documents
        self._handlers["chat"] = self._handle_chat
        self._handlers["list_documents"] = self._handle_list_documents
        self._handlers["get_document"] = self._handle_get_document
        self._handlers["list_collections"] = self._handle_list_collections
        self._handlers["get_knowledge_graph"] = self._handle_get_knowledge_graph

        self._initialized = True
        logger.info("MCP tools initialized", tools=list(self._handlers.keys()))

    def get_handlers(self) -> Dict[str, Callable]:
        """Get all tool handlers."""
        return self._handlers

    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all tool schemas as MCPTool-compatible dicts."""
        from backend.mcp.server import MCPTool
        return {
            name: MCPTool(
                name=schema["name"],
                description=schema["description"],
                input_schema=schema["input_schema"],
            )
            for name, schema in self._schemas.items()
        }

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    async def _handle_search_documents(
        self,
        query: str,
        limit: int = 10,
        collection_id: Optional[str] = None,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Handle search_documents tool."""
        try:
            from backend.services.rag import get_rag_service

            rag = get_rag_service()

            # Build filter
            filters = {}
            if collection_id:
                filters["collection_id"] = collection_id
            if file_types:
                filters["file_type"] = {"$in": file_types}

            # Perform search
            results = await rag.search(
                query=query,
                top_k=limit,
                filters=filters if filters else None,
            )

            # Format results
            formatted = []
            for r in results:
                formatted.append({
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                    "document_id": r.get("document_id"),
                    "document_name": r.get("document_name"),
                    "chunk_index": r.get("chunk_index"),
                })

            return {
                "results": formatted,
                "total": len(formatted),
                "query": query,
            }

        except Exception as e:
            logger.error("Search error", error=str(e))
            return {"error": str(e), "results": []}

    async def _handle_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle chat tool."""
        try:
            from backend.services.rag import get_rag_service
            from uuid import uuid4

            rag = get_rag_service()

            # Generate session if not provided
            if not session_id:
                session_id = str(uuid4())

            # Build filter
            filters = None
            if collection_id:
                filters = {"collection_id": collection_id}

            # Get RAG response
            response = await rag.query(
                query=message,
                session_id=session_id,
                filters=filters,
            )

            # Format response
            sources = []
            for source in response.get("sources", []):
                sources.append({
                    "document_name": source.get("document_name"),
                    "content": source.get("content", "")[:200] + "...",
                    "score": source.get("score"),
                })

            return {
                "response": response.get("answer", ""),
                "sources": sources,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error("Chat error", error=str(e))
            return {"error": str(e), "response": ""}

    async def _handle_list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        collection_id: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle list_documents tool."""
        try:
            from backend.db.database import get_async_session
            from backend.db.models import Document
            from sqlalchemy import select, func

            async with get_async_session() as session:
                query = select(Document)

                if collection_id:
                    query = query.where(Document.collection_id == collection_id)
                if file_type:
                    query = query.where(Document.file_type == file_type)

                query = query.offset(offset).limit(limit)
                result = await session.execute(query)
                documents = result.scalars().all()

                # Get total count
                count_query = select(func.count(Document.id))
                if collection_id:
                    count_query = count_query.where(Document.collection_id == collection_id)
                if file_type:
                    count_query = count_query.where(Document.file_type == file_type)

                total_result = await session.execute(count_query)
                total = total_result.scalar() or 0

                return {
                    "documents": [
                        {
                            "id": str(doc.id),
                            "name": doc.name,
                            "file_type": doc.file_type,
                            "file_size": doc.file_size,
                            "created_at": doc.created_at.isoformat() if doc.created_at else None,
                            "chunk_count": doc.chunk_count,
                        }
                        for doc in documents
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }

        except Exception as e:
            logger.error("List documents error", error=str(e))
            return {"error": str(e), "documents": []}

    async def _handle_get_document(
        self,
        document_id: str,
        include_chunks: bool = False,
    ) -> Dict[str, Any]:
        """Handle get_document tool."""
        try:
            from backend.db.database import get_async_session
            from backend.db.models import Document, Chunk
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with get_async_session() as session:
                query = select(Document).where(Document.id == document_id)

                if include_chunks:
                    query = query.options(selectinload(Document.chunks))

                result = await session.execute(query)
                doc = result.scalar_one_or_none()

                if not doc:
                    return {"error": f"Document not found: {document_id}"}

                response = {
                    "id": str(doc.id),
                    "name": doc.name,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "chunk_count": doc.chunk_count,
                    "tags": doc.tags or [],
                    "summary": doc.summary,
                }

                if include_chunks and doc.chunks:
                    response["chunks"] = [
                        {
                            "index": chunk.chunk_index,
                            "content": chunk.content,
                        }
                        for chunk in sorted(doc.chunks, key=lambda c: c.chunk_index)
                    ]

                return response

        except Exception as e:
            logger.error("Get document error", error=str(e))
            return {"error": str(e)}

    async def _handle_list_collections(
        self,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Handle list_collections tool."""
        try:
            from backend.db.database import get_async_session
            from backend.db.models import Collection
            from sqlalchemy import select

            async with get_async_session() as session:
                query = select(Collection).limit(limit)
                result = await session.execute(query)
                collections = result.scalars().all()

                return {
                    "collections": [
                        {
                            "id": str(col.id),
                            "name": col.name,
                            "description": col.description,
                            "document_count": col.document_count if hasattr(col, "document_count") else 0,
                            "created_at": col.created_at.isoformat() if col.created_at else None,
                        }
                        for col in collections
                    ],
                    "total": len(collections),
                }

        except Exception as e:
            logger.error("List collections error", error=str(e))
            return {"error": str(e), "collections": []}

    async def _handle_get_knowledge_graph(
        self,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Handle get_knowledge_graph tool."""
        try:
            from backend.services.knowledge_graph import get_knowledge_graph_service

            kg_service = get_knowledge_graph_service()

            # Get entities
            entities = await kg_service.search_entities(
                query=query,
                entity_type=entity_type,
                limit=limit,
            )

            return {
                "entities": [
                    {
                        "id": str(e.get("id")),
                        "name": e.get("name"),
                        "type": e.get("type"),
                        "description": e.get("description"),
                        "relationships": e.get("relationships", []),
                    }
                    for e in entities
                ],
                "total": len(entities),
            }

        except Exception as e:
            logger.error("Knowledge graph error", error=str(e))
            return {"error": str(e), "entities": []}


# =============================================================================
# Singleton
# =============================================================================

_tool_registry: Optional[MCPToolRegistry] = None


def get_tool_registry() -> MCPToolRegistry:
    """Get or create the tool registry singleton."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = MCPToolRegistry()
    return _tool_registry
