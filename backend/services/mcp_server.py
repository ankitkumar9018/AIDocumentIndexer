"""
AIDocumentIndexer - MCP (Model Context Protocol) Server
========================================================

Exposes AIDocumentIndexer capabilities as MCP tools that can be called
by Claude Desktop, external AI assistants, or any MCP-compatible client.

Tools exposed:
1. rag_query - Query documents with RAG
2. search_documents - Semantic document search
3. knowledge_graph_query - Query the knowledge graph
4. scrape_url - Scrape and optionally index a URL
5. text_to_sql - Natural language to SQL
6. execute_skill - Run a skill (summarize, translate, etc.)
7. calculate - Mathematical calculations
8. fact_check - Verify claims against documents
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable
from enum import Enum
import asyncio
import json
import time

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


class MCPToolType(str, Enum):
    """Types of MCP tools available."""
    RAG_QUERY = "rag_query"
    SEARCH_DOCUMENTS = "search_documents"
    KNOWLEDGE_GRAPH = "knowledge_graph_query"
    SCRAPE_URL = "scrape_url"
    TEXT_TO_SQL = "text_to_sql"
    EXECUTE_SKILL = "execute_skill"
    CALCULATE = "calculate"
    FACT_CHECK = "fact_check"


@dataclass
class MCPToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable[..., Awaitable[Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class MCPToolResult:
    """Result from an MCP tool call."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


class MCPServer:
    """
    MCP Server that exposes AIDocumentIndexer capabilities as tools.

    This server can be:
    1. Used via HTTP endpoints (/api/v1/mcp/*)
    2. Used via stdio for Claude Desktop integration
    3. Called programmatically by agents
    """

    def __init__(self):
        """Initialize the MCP server with tool definitions."""
        self._tools: Dict[str, MCPToolDefinition] = {}
        self._register_tools()

    def _register_tools(self):
        """Register all available MCP tools."""

        # RAG Query Tool
        self._tools["rag_query"] = MCPToolDefinition(
            name="rag_query",
            description="Query documents using RAG (Retrieval-Augmented Generation). Returns an answer based on indexed documents with source citations.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to answer based on documents",
                    },
                    "collection": {
                        "type": "string",
                        "description": "Optional collection/folder to search within",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to retrieve (default: 5)",
                        "default": 5,
                    },
                    "include_sources": {
                        "type": "boolean",
                        "description": "Whether to include source documents in response",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
            handler=self._handle_rag_query,
        )

        # Search Documents Tool
        self._tools["search_documents"] = MCPToolDefinition(
            name="search_documents",
            description="Search for documents using semantic similarity. Returns matching documents with relevance scores.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10,
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by file types (e.g., ['pdf', 'docx'])",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Filter documents from this date (YYYY-MM-DD)",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Filter documents until this date (YYYY-MM-DD)",
                    },
                },
                "required": ["query"],
            },
            handler=self._handle_search_documents,
        )

        # Knowledge Graph Query Tool
        self._tools["knowledge_graph_query"] = MCPToolDefinition(
            name="knowledge_graph_query",
            description="Query the knowledge graph for entities, relationships, and connections between concepts.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about entities or relationships",
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by entity types (e.g., ['person', 'organization'])",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum relationship hops to traverse (default: 2)",
                        "default": 2,
                    },
                },
                "required": ["query"],
            },
            handler=self._handle_knowledge_graph_query,
        )

        # Scrape URL Tool
        self._tools["scrape_url"] = MCPToolDefinition(
            name="scrape_url",
            description="Scrape content from a URL. Can optionally index the content for future RAG queries.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape",
                    },
                    "index_content": {
                        "type": "boolean",
                        "description": "Whether to index the scraped content (default: False)",
                        "default": False,
                    },
                    "extract_links": {
                        "type": "boolean",
                        "description": "Whether to extract and return links from the page",
                        "default": False,
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum crawl depth if following links (default: 1)",
                        "default": 1,
                    },
                },
                "required": ["url"],
            },
            handler=self._handle_scrape_url,
        )

        # Text-to-SQL Tool
        self._tools["text_to_sql"] = MCPToolDefinition(
            name="text_to_sql",
            description="Convert natural language questions to SQL queries and execute them against connected databases.",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the data",
                    },
                    "database_id": {
                        "type": "string",
                        "description": "ID of the database to query (optional if only one connected)",
                    },
                    "execute": {
                        "type": "boolean",
                        "description": "Whether to execute the SQL or just return it (default: True)",
                        "default": True,
                    },
                },
                "required": ["question"],
            },
            handler=self._handle_text_to_sql,
        )

        # Execute Skill Tool
        self._tools["execute_skill"] = MCPToolDefinition(
            name="execute_skill",
            description="Execute an AI skill like summarization, translation, sentiment analysis, or entity extraction.",
            input_schema={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "enum": ["summarize", "translate", "sentiment", "entities", "compare", "fact_check"],
                        "description": "The skill to execute",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to process",
                    },
                    "options": {
                        "type": "object",
                        "description": "Skill-specific options (e.g., target_language for translate)",
                    },
                },
                "required": ["skill", "text"],
            },
            handler=self._handle_execute_skill,
        )

        # Calculate Tool
        self._tools["calculate"] = MCPToolDefinition(
            name="calculate",
            description="Perform mathematical calculations safely. Supports basic arithmetic, functions (sqrt, sin, cos, log), and constants (pi, e).",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., 'sqrt(16) + 2**3')",
                    },
                },
                "required": ["expression"],
            },
            handler=self._handle_calculate,
        )

        # Fact Check Tool
        self._tools["fact_check"] = MCPToolDefinition(
            name="fact_check",
            description="Verify a claim against indexed documents. Returns verdict (supported/contradicted/unverifiable) with evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The claim to verify",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for verification",
                    },
                },
                "required": ["claim"],
            },
            handler=self._handle_fact_check,
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available MCP tools."""
        return [tool.to_dict() for tool in self._tools.values()]

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> MCPToolResult:
        """
        Call an MCP tool by name.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            user_id: Optional user ID for access control

        Returns:
            MCPToolResult with the tool output
        """
        start_time = time.time()

        if tool_name not in self._tools:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}",
                execution_time_ms=0,
            )

        tool = self._tools[tool_name]

        try:
            logger.info(
                "MCP tool call",
                tool=tool_name,
                arguments=arguments,
                user_id=user_id,
            )

            result = await tool.handler(arguments, user_id)

            execution_time = int((time.time() - start_time) * 1000)

            logger.info(
                "MCP tool call complete",
                tool=tool_name,
                execution_time_ms=execution_time,
            )

            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(
                "MCP tool call failed",
                tool=tool_name,
                error=str(e),
            )
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    # ==========================================================================
    # Tool Handlers
    # ==========================================================================

    async def _handle_rag_query(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle RAG query tool."""
        from backend.services.rag import get_rag_service

        rag_service = get_rag_service()

        response = await rag_service.query(
            question=arguments["query"],
            collection_filter=arguments.get("collection"),
            top_k=arguments.get("top_k", 5),
            user_id=user_id,
        )

        result = {
            "answer": response.content,
            "confidence": response.confidence_score,
            "model": response.model,
        }

        if arguments.get("include_sources", True):
            result["sources"] = [
                {
                    "document": s.get("document_name", "Unknown"),
                    "content": s.get("content", "")[:500],
                    "score": s.get("score", 0),
                }
                for s in response.sources[:5]
            ]

        return result

    async def _handle_search_documents(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle document search tool."""
        from backend.services.vectorstore import get_vectorstore

        vectorstore = get_vectorstore()

        results = await vectorstore.similarity_search(
            query=arguments["query"],
            k=arguments.get("limit", 10),
        )

        return {
            "documents": [
                {
                    "id": doc.metadata.get("document_id", ""),
                    "name": doc.metadata.get("document_name", "Unknown"),
                    "content": doc.page_content[:500],
                    "score": doc.metadata.get("score", 0),
                    "file_type": doc.metadata.get("file_type", ""),
                }
                for doc in results
            ],
            "total": len(results),
        }

    async def _handle_knowledge_graph_query(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle knowledge graph query tool."""
        from backend.services.knowledge_graph import get_knowledge_graph_service

        kg_service = get_knowledge_graph_service()

        # Search for entities matching the query
        entities = await kg_service.search_entities(
            query=arguments["query"],
            entity_types=arguments.get("entity_types"),
            limit=10,
        )

        # Get relationships for found entities
        relationships = []
        for entity in entities[:5]:
            rels = await kg_service.get_entity_relationships(
                entity_id=entity.get("id"),
                max_hops=arguments.get("max_hops", 2),
            )
            relationships.extend(rels)

        return {
            "entities": entities,
            "relationships": relationships[:20],
            "query": arguments["query"],
        }

    async def _handle_scrape_url(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle URL scraping tool."""
        from backend.services.web_crawler import WebCrawler

        crawler = WebCrawler()

        result = await crawler.crawl_url(
            url=arguments["url"],
            max_depth=arguments.get("max_depth", 1),
            extract_links=arguments.get("extract_links", False),
        )

        response = {
            "url": arguments["url"],
            "title": result.get("title", ""),
            "content": result.get("content", "")[:5000],
            "word_count": len(result.get("content", "").split()),
        }

        if arguments.get("extract_links", False):
            response["links"] = result.get("links", [])[:50]

        if arguments.get("index_content", False):
            # Trigger document indexing via the pipeline
            try:
                from backend.services.pipeline import get_pipeline_service

                pipeline = get_pipeline_service()
                content = result.get("content", "")
                title = result.get("title", arguments["url"])

                if content:
                    # Create a document from the scraped content
                    index_result = await pipeline.process_text(
                        text=content,
                        title=title,
                        metadata={
                            "source_url": arguments["url"],
                            "source_type": "web_scrape",
                        },
                    )
                    response["indexed"] = True
                    response["document_id"] = index_result.get("document_id")
                else:
                    response["indexed"] = False
                    response["index_error"] = "No content to index"
            except Exception as e:
                logger.error(f"Failed to index scraped content: {e}")
                response["indexed"] = False
                response["index_error"] = str(e)

        return response

    async def _handle_text_to_sql(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle text-to-SQL tool."""
        from backend.services.text_to_sql.service import TextToSQLService

        service = TextToSQLService()

        result = await service.query(
            question=arguments["question"],
            database_id=arguments.get("database_id"),
            execute=arguments.get("execute", True),
        )

        return {
            "question": arguments["question"],
            "sql": result.get("sql", ""),
            "result": result.get("result") if arguments.get("execute", True) else None,
            "explanation": result.get("explanation", ""),
        }

    async def _handle_execute_skill(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle skill execution tool."""
        from backend.services.llm import LLMFactory

        skill = arguments["skill"]
        text = arguments["text"]
        options = arguments.get("options", {})

        # Skill prompts
        skill_prompts = {
            "summarize": f"Summarize the following text concisely:\n\n{text}",
            "translate": f"Translate the following text to {options.get('target_language', 'English')}:\n\n{text}",
            "sentiment": f"Analyze the sentiment of the following text. Return: positive, negative, or neutral with confidence:\n\n{text}",
            "entities": f"Extract all named entities (people, organizations, locations, dates) from:\n\n{text}",
            "compare": f"Compare and contrast the following texts:\n\n{text}",
            "fact_check": f"Identify any factual claims in this text and assess their verifiability:\n\n{text}",
        }

        if skill not in skill_prompts:
            raise ValueError(f"Unknown skill: {skill}")

        llm = LLMFactory.get_chat_model(
            provider=settings.DEFAULT_LLM_PROVIDER,
            model=settings.DEFAULT_CHAT_MODEL,
            temperature=0.3,
        )

        response = await llm.ainvoke(skill_prompts[skill])

        return {
            "skill": skill,
            "input_length": len(text),
            "result": response.content,
        }

    async def _handle_calculate(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle calculation tool."""
        from backend.services.tool_augmentation import get_tool_augmentation_service

        service = get_tool_augmentation_service()
        result = service.calculate(arguments["expression"])

        return {
            "expression": arguments["expression"],
            "result": result.result if result.success else None,
            "error": result.error,
        }

    async def _handle_fact_check(
        self,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle fact checking tool."""
        from backend.services.tool_augmentation import get_tool_augmentation_service

        service = get_tool_augmentation_service()
        result = await service.check_fact(
            claim=arguments["claim"],
            context=arguments.get("context", ""),
        )

        if result.success:
            return result.result
        else:
            return {
                "claim": arguments["claim"],
                "verdict": "error",
                "error": result.error,
            }


# =============================================================================
# MCP Message Protocol Handler (for stdio/HTTP)
# =============================================================================

class MCPProtocolHandler:
    """
    Handles MCP protocol messages (JSON-RPC style).

    Supports:
    - tools/list: List available tools
    - tools/call: Call a tool with arguments
    """

    def __init__(self, server: MCPServer):
        self.server = server

    async def handle_message(
        self,
        message: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle an MCP protocol message.

        Args:
            message: JSON-RPC style message with method and params
            user_id: Optional user ID for access control

        Returns:
            JSON-RPC style response
        """
        method = message.get("method", "")
        params = message.get("params", {})
        msg_id = message.get("id")

        try:
            if method == "tools/list":
                result = self.server.list_tools()
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                tool_result = await self.server.call_tool(tool_name, arguments, user_id)
                result = tool_result.to_dict()
            elif method == "ping":
                result = {"status": "ok", "server": "AIDocumentIndexer MCP"}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}",
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            }

        except Exception as e:
            logger.error("MCP message handling failed", error=str(e))
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": str(e),
                },
            }


# =============================================================================
# Singleton
# =============================================================================

_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create the MCP server singleton."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server


def get_mcp_protocol_handler() -> MCPProtocolHandler:
    """Get the MCP protocol handler."""
    return MCPProtocolHandler(get_mcp_server())
