"""
AIDocumentIndexer - MCP (Model Context Protocol) API Routes
============================================================

HTTP endpoints for the MCP server, allowing external AI assistants
to discover and call AIDocumentIndexer tools.

Endpoints:
- GET /tools - List available MCP tools
- POST /tools/call - Call an MCP tool
- POST /message - Handle raw MCP protocol messages
- GET /health - Check MCP server health
- GET /resources - List available MCP resources
- WS /ws - WebSocket transport for MCP

References:
- MCP Spec: https://modelcontextprotocol.io/
- JSON-RPC 2.0: https://www.jsonrpc.org/specification
"""

import json
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from backend.mcp import get_mcp_server, MCPServer
from backend.api.middleware.auth import get_current_user_optional
from backend.db.models import User

logger = structlog.get_logger(__name__)

# Global MCP server instance
_mcp_server_instance: Optional[MCPServer] = None


async def get_initialized_mcp_server() -> MCPServer:
    """Get or initialize the MCP server singleton."""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        _mcp_server_instance = get_mcp_server()
        await _mcp_server_instance.initialize()
    return _mcp_server_instance

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool."""
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class MCPToolResult(BaseModel):
    """Result from an MCP tool call."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


class MCPToolDefinition(BaseModel):
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPMessageRequest(BaseModel):
    """Raw MCP protocol message (JSON-RPC 2.0)."""
    jsonrpc: str = Field(default="2.0")
    id: Optional[str | int] = None
    method: str = Field(..., description="Method to call (tools/list, tools/call, resources/list, etc.)")
    params: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/info")
async def get_mcp_info():
    """Get MCP server information and capabilities."""
    server = await get_initialized_mcp_server()
    return {
        "name": server.name,
        "version": server.version,
        "protocol_version": server.protocol_version,
        "capabilities": server.capabilities,
        "status": "ready" if server._initialized else "initializing",
    }


@router.get("/tools", response_model=List[MCPToolDefinition])
async def list_mcp_tools(
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    List all available MCP tools.

    Returns a list of tool definitions with their names, descriptions,
    and input schemas. This follows the MCP protocol format.
    """
    server = await get_initialized_mcp_server()
    tools = server._tool_registry.get_tools()

    logger.info(
        "MCP tools listed",
        tool_count=len(tools),
        user_id=str(user.id) if user else None,
    )

    return [
        MCPToolDefinition(
            name=tool["name"],
            description=tool["description"],
            input_schema=tool["inputSchema"],
        )
        for tool in tools
    ]


@router.get("/resources")
async def list_mcp_resources(
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    List all available MCP resources.

    Returns resource templates with URI patterns and descriptions.
    """
    server = await get_initialized_mcp_server()
    templates = server._resource_provider.get_templates()

    logger.info(
        "MCP resources listed",
        resource_count=len(templates),
        user_id=str(user.id) if user else None,
    )

    return {
        "resources": [
            {
                "uri_template": t["uri_template"],
                "name": t["name"],
                "description": t["description"],
                "mime_type": t.get("mime_type"),
            }
            for t in templates
        ],
        "total": len(templates),
    }


@router.post("/tools/call", response_model=MCPToolResult)
async def call_mcp_tool(
    request: MCPToolCallRequest,
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Call an MCP tool by name.

    This endpoint allows external AI assistants to invoke AIDocumentIndexer
    capabilities like RAG queries, document search, knowledge graph queries,
    and more.
    """
    server = await get_initialized_mcp_server()

    logger.info(
        "MCP tool call request",
        tool=request.name,
        user_id=str(user.id) if user else None,
    )

    try:
        result = await server._tool_registry.call_tool(request.name, request.arguments)
        is_error = result.get("isError", False)

        return MCPToolResult(
            tool_name=request.name,
            success=not is_error,
            result=result.get("content", []),
            error=result.get("content", [{}])[0].get("text") if is_error else None,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error("MCP tool call error", tool=request.name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/message")
async def handle_mcp_message(
    request: MCPMessageRequest,
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Handle a raw MCP protocol message (JSON-RPC 2.0).

    This endpoint accepts JSON-RPC style messages as per the MCP protocol:
    - initialize: Initialize the MCP session
    - tools/list: List available tools
    - tools/call: Call a tool with arguments
    - resources/list: List available resources
    - resources/read: Read a specific resource
    - resources/templates/list: List resource templates

    This is useful for direct MCP protocol integration.
    """
    server = await get_initialized_mcp_server()

    logger.info(
        "MCP message received",
        method=request.method,
        request_id=request.id,
        user_id=str(user.id) if user else None,
    )

    try:
        response = await server.handle_request({
            "jsonrpc": request.jsonrpc,
            "id": request.id,
            "method": request.method,
            "params": request.params,
        })
        return JSONResponse(content=response)
    except Exception as e:
        logger.error("MCP message error", method=request.method, error=str(e))
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}",
                },
                "id": request.id,
            },
            status_code=500,
        )


@router.get("/health")
async def mcp_health_check():
    """
    Check MCP server health.

    Returns server status, available tool count, and resource count.
    """
    try:
        server = await get_initialized_mcp_server()
        tools = server._tool_registry.get_tools()
        resources = server._resource_provider.get_templates()

        return {
            "status": "healthy",
            "server": server.name,
            "version": server.version,
            "protocol_version": server.protocol_version,
            "tools_available": len(tools),
            "resources_available": len(resources),
            "initialized": server._initialized,
        }
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503,
        )


@router.get("/tools/{tool_name}")
async def get_tool_details(
    tool_name: str,
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Get detailed information about a specific MCP tool.
    """
    server = await get_initialized_mcp_server()
    tools = server._tool_registry.get_tools()

    for tool in tools:
        if tool["name"] == tool_name:
            return {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tool not found: {tool_name}",
    )


@router.get("/schema")
async def get_mcp_schema():
    """
    Get the full MCP schema for AIDocumentIndexer.

    This returns a schema document that can be used by MCP clients
    to understand the available capabilities.
    """
    server = await get_initialized_mcp_server()
    tools = server._tool_registry.get_tools()
    resources = server._resource_provider.get_templates()

    return {
        "name": server.name,
        "description": "Intelligent Document Archive with RAG capabilities",
        "version": server.version,
        "protocol_version": server.protocol_version,
        "capabilities": server.capabilities,
        "tools": [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in tools
        ],
        "resources": [
            {
                "uri_template": r["uri_template"],
                "name": r["name"],
                "description": r["description"],
            }
            for r in resources
        ],
        "endpoints": {
            "info": "/api/v1/mcp/info",
            "list_tools": "/api/v1/mcp/tools",
            "call_tool": "/api/v1/mcp/tools/call",
            "list_resources": "/api/v1/mcp/resources",
            "message": "/api/v1/mcp/message",
            "health": "/api/v1/mcp/health",
            "websocket": "/api/v1/mcp/ws",
        },
    }


# =============================================================================
# WebSocket Transport
# =============================================================================

@router.websocket("/ws")
async def mcp_websocket(websocket: WebSocket):
    """
    WebSocket transport for MCP protocol.

    Allows bidirectional JSON-RPC communication for:
    - Real-time tool responses
    - Streaming resource updates
    - Server-initiated notifications
    """
    await websocket.accept()
    server = await get_initialized_mcp_server()

    logger.info("MCP WebSocket connection established")

    try:
        while True:
            # Receive JSON-RPC request
            data = await websocket.receive_text()

            try:
                request = json.loads(data)

                # Handle the request
                response = await server.handle_request(request)

                # Send response
                await websocket.send_json(response)

            except json.JSONDecodeError:
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error: Invalid JSON",
                    },
                    "id": None,
                })
            except Exception as e:
                logger.error("MCP WebSocket request error", error=str(e))
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}",
                    },
                    "id": None,
                })

    except WebSocketDisconnect:
        logger.info("MCP WebSocket connection closed")
    except Exception as e:
        logger.error("MCP WebSocket fatal error", error=str(e))
