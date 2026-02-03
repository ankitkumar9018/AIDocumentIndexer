"""
AIDocumentIndexer - MCP Server
==============================

Model Context Protocol server implementation.
Exposes AIDocumentIndexer functionality as MCP tools and resources.

MCP Specification: https://modelcontextprotocol.io/
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# =============================================================================
# MCP Protocol Models
# =============================================================================

class MCPCapabilities(BaseModel):
    """Server capabilities advertised to clients."""
    tools: bool = True
    resources: bool = True
    prompts: bool = False
    logging: bool = True


class MCPServerInfo(BaseModel):
    """Server information."""
    name: str = "aidocindexer"
    version: str = "1.0.0"
    protocol_version: str = "2024-11-05"
    capabilities: MCPCapabilities = Field(default_factory=MCPCapabilities)


class MCPTool(BaseModel):
    """Tool definition for MCP."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPResource(BaseModel):
    """Resource definition for MCP."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


class MCPToolCall(BaseModel):
    """Tool invocation request."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    """Tool execution result."""
    content: List[Dict[str, Any]]
    is_error: bool = False


class MCPResourceContent(BaseModel):
    """Resource content response."""
    uri: str
    mime_type: str
    text: Optional[str] = None
    blob: Optional[str] = None  # Base64 encoded


class MCPMessage(BaseModel):
    """MCP protocol message."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


# =============================================================================
# MCP Server
# =============================================================================

class MCPServer:
    """
    Model Context Protocol server.

    Exposes AIDocumentIndexer as an MCP server that can be connected
    to by Claude Desktop, Claude Code, and other MCP clients.

    Features:
    - Tool execution (search, chat, upload)
    - Resource access (documents, collections)
    - Streaming support
    - Session management
    """

    def __init__(self):
        self.server_info = MCPServerInfo()
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, MCPTool] = {}
        self._resources: Dict[str, Callable] = {}
        self._resource_templates: List[Dict[str, Any]] = []
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MCP server with tools and resources."""
        if self._initialized:
            return

        from backend.mcp.tools import get_tool_registry
        from backend.mcp.resources import get_resource_provider

        # Register tools
        tool_registry = get_tool_registry()
        await tool_registry.initialize()
        self._tools = tool_registry.get_handlers()
        self._tool_schemas = tool_registry.get_schemas()

        # Register resources
        resource_provider = get_resource_provider()
        await resource_provider.initialize()
        self._resources = resource_provider.get_handlers()
        self._resource_templates = resource_provider.get_templates()

        self._initialized = True
        logger.info(
            "MCP server initialized",
            tools=list(self._tools.keys()),
            resource_templates=len(self._resource_templates),
        )

    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming MCP message.

        Args:
            message: JSON-RPC message

        Returns:
            JSON-RPC response
        """
        msg = MCPMessage(**message)

        try:
            if msg.method == "initialize":
                return await self._handle_initialize(msg)
            elif msg.method == "tools/list":
                return await self._handle_list_tools(msg)
            elif msg.method == "tools/call":
                return await self._handle_call_tool(msg)
            elif msg.method == "resources/list":
                return await self._handle_list_resources(msg)
            elif msg.method == "resources/read":
                return await self._handle_read_resource(msg)
            elif msg.method == "resources/templates/list":
                return await self._handle_list_resource_templates(msg)
            elif msg.method == "logging/setLevel":
                return await self._handle_set_log_level(msg)
            elif msg.method == "ping":
                return self._success_response(msg.id, {})
            else:
                return self._error_response(
                    msg.id,
                    -32601,
                    f"Method not found: {msg.method}"
                )
        except Exception as e:
            logger.error("MCP message handling error", error=str(e), method=msg.method)
            return self._error_response(msg.id, -32603, str(e))

    async def _handle_initialize(self, msg: MCPMessage) -> Dict[str, Any]:
        """Handle initialize request."""
        await self.initialize()

        return self._success_response(msg.id, {
            "protocolVersion": self.server_info.protocol_version,
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True, "listChanged": True},
                "logging": {},
            },
            "serverInfo": {
                "name": self.server_info.name,
                "version": self.server_info.version,
            },
        })

    async def _handle_list_tools(self, msg: MCPMessage) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tool_schemas.values()
        ]

        return self._success_response(msg.id, {"tools": tools})

    async def _handle_call_tool(self, msg: MCPMessage) -> Dict[str, Any]:
        """Handle tools/call request."""
        params = msg.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            return self._error_response(
                msg.id,
                -32602,
                f"Unknown tool: {tool_name}"
            )

        try:
            handler = self._tools[tool_name]
            result = await handler(**arguments)

            # Format result as MCP content
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result, indent=2)}]
            elif isinstance(result, list):
                content = [{"type": "text", "text": json.dumps(result, indent=2)}]
            else:
                content = [{"type": "text", "text": str(result)}]

            return self._success_response(msg.id, {
                "content": content,
                "isError": False,
            })

        except Exception as e:
            logger.error("Tool execution error", tool=tool_name, error=str(e))
            return self._success_response(msg.id, {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            })

    async def _handle_list_resources(self, msg: MCPMessage) -> Dict[str, Any]:
        """Handle resources/list request."""
        resources = []

        for template in self._resource_templates:
            # Get dynamic resources from handlers
            handler = self._resources.get(template.get("handler"))
            if handler:
                try:
                    items = await handler(list_only=True)
                    resources.extend(items)
                except Exception as e:
                    logger.warning(
                        "Failed to list resources",
                        template=template.get("uri_template"),
                        error=str(e),
                    )

        return self._success_response(msg.id, {"resources": resources})

    async def _handle_read_resource(self, msg: MCPMessage) -> Dict[str, Any]:
        """Handle resources/read request."""
        params = msg.params or {}
        uri = params.get("uri")

        if not uri:
            return self._error_response(msg.id, -32602, "Missing uri parameter")

        # Find matching handler
        for template in self._resource_templates:
            handler_name = template.get("handler")
            uri_template = template.get("uri_template", "")

            if self._uri_matches_template(uri, uri_template):
                handler = self._resources.get(handler_name)
                if handler:
                    try:
                        content = await handler(uri=uri)
                        return self._success_response(msg.id, {
                            "contents": [content]
                        })
                    except Exception as e:
                        logger.error("Resource read error", uri=uri, error=str(e))
                        return self._error_response(msg.id, -32603, str(e))

        return self._error_response(msg.id, -32602, f"Resource not found: {uri}")

    async def _handle_list_resource_templates(self, msg: MCPMessage) -> Dict[str, Any]:
        """Handle resources/templates/list request."""
        templates = [
            {
                "uriTemplate": t.get("uri_template"),
                "name": t.get("name"),
                "description": t.get("description"),
                "mimeType": t.get("mime_type", "text/plain"),
            }
            for t in self._resource_templates
        ]

        return self._success_response(msg.id, {"resourceTemplates": templates})

    async def _handle_set_log_level(self, msg: MCPMessage) -> Dict[str, Any]:
        """Handle logging/setLevel request."""
        params = msg.params or {}
        level = params.get("level", "info")

        # Map MCP levels to Python logging
        level_map = {
            "debug": "DEBUG",
            "info": "INFO",
            "notice": "INFO",
            "warning": "WARNING",
            "error": "ERROR",
            "critical": "CRITICAL",
            "alert": "CRITICAL",
            "emergency": "CRITICAL",
        }

        python_level = level_map.get(level.lower(), "INFO")
        # Note: Would need to update structlog config here

        return self._success_response(msg.id, {})

    def _uri_matches_template(self, uri: str, template: str) -> bool:
        """Check if a URI matches a template pattern."""
        # Simple template matching (supports {param} patterns)
        import re

        # Convert template to regex
        pattern = re.escape(template)
        pattern = pattern.replace(r"\{", "(?P<").replace(r"\}", ">[^/]+)")
        pattern = f"^{pattern}$"

        return bool(re.match(pattern, uri))

    def _success_response(self, msg_id: Optional[str], result: Any) -> Dict[str, Any]:
        """Create a success response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result,
        }

    def _error_response(
        self,
        msg_id: Optional[str],
        code: int,
        message: str,
    ) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    async def create_session(self) -> str:
        """Create a new MCP session."""
        session_id = str(uuid4())
        self._sessions[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
        }
        return session_id

    async def close_session(self, session_id: str) -> None:
        """Close an MCP session."""
        self._sessions.pop(session_id, None)

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information for clients."""
        return {
            "name": self.server_info.name,
            "version": self.server_info.version,
            "protocol_version": self.server_info.protocol_version,
            "capabilities": self.server_info.capabilities.model_dump(),
            "tools_count": len(self._tool_schemas),
            "resource_templates_count": len(self._resource_templates),
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


async def initialize_mcp_server() -> MCPServer:
    """Initialize and return the MCP server."""
    server = get_mcp_server()
    await server.initialize()
    return server
