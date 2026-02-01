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
"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.mcp_server import (
    get_mcp_server,
    get_mcp_protocol_handler,
    MCPServer,
    MCPProtocolHandler,
)
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.api.middleware.auth import AuthenticatedUser, get_current_user_optional

logger = structlog.get_logger(__name__)

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
    execution_time_ms: int


class MCPToolDefinition(BaseModel):
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPMessageRequest(BaseModel):
    """Raw MCP protocol message."""
    jsonrpc: str = Field(default="2.0")
    id: Optional[str] = None
    method: str = Field(..., description="Method to call (tools/list, tools/call)")
    params: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/tools", response_model=List[MCPToolDefinition])
async def list_mcp_tools(
    user: Optional[AuthenticatedUser] = Depends(get_current_user_optional),
):
    """
    List all available MCP tools.

    Returns a list of tool definitions with their names, descriptions,
    and input schemas. This follows the MCP protocol format.
    """
    server = get_mcp_server()
    tools = server.list_tools()

    logger.info(
        "MCP tools listed",
        tool_count=len(tools),
        user_id=user.user_id if user else None,
    )

    return tools


@router.post("/tools/call", response_model=MCPToolResult)
async def call_mcp_tool(
    request: MCPToolCallRequest,
    user: Optional[AuthenticatedUser] = Depends(get_current_user_optional),
):
    """
    Call an MCP tool by name.

    This endpoint allows external AI assistants to invoke AIDocumentIndexer
    capabilities like RAG queries, document search, knowledge graph queries,
    and more.
    """
    server = get_mcp_server()

    logger.info(
        "MCP tool call request",
        tool=request.name,
        user_id=user.user_id if user else None,
    )

    result = await server.call_tool(
        tool_name=request.name,
        arguments=request.arguments,
        user_id=user.user_id if user else None,
    )

    if not result.success:
        # Return result even on failure (with error details)
        pass

    return MCPToolResult(
        tool_name=result.tool_name,
        success=result.success,
        result=result.result,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
    )


@router.post("/message")
async def handle_mcp_message(
    request: MCPMessageRequest,
    user: Optional[AuthenticatedUser] = Depends(get_current_user_optional),
):
    """
    Handle a raw MCP protocol message.

    This endpoint accepts JSON-RPC style messages as per the MCP protocol:
    - tools/list: List available tools
    - tools/call: Call a tool with arguments
    - ping: Check server health

    This is useful for direct MCP protocol integration.
    """
    handler = get_mcp_protocol_handler()

    response = await handler.handle_message(
        message={
            "jsonrpc": request.jsonrpc,
            "id": request.id,
            "method": request.method,
            "params": request.params,
        },
        user_id=user.user_id if user else None,
    )

    return response


@router.get("/health")
async def mcp_health_check():
    """
    Check MCP server health.

    Returns server status and available tool count.
    """
    server = get_mcp_server()
    tools = server.list_tools()

    return {
        "status": "healthy",
        "server": "AIDocumentIndexer MCP",
        "version": "1.0.0",
        "tools_available": len(tools),
        "protocol": "MCP/1.0",
    }


@router.get("/tools/{tool_name}")
async def get_tool_details(
    tool_name: str,
    user: Optional[AuthenticatedUser] = Depends(get_current_user_optional),
):
    """
    Get detailed information about a specific MCP tool.
    """
    server = get_mcp_server()
    tools = server.list_tools()

    for tool in tools:
        if tool["name"] == tool_name:
            return tool

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
    server = get_mcp_server()
    tools = server.list_tools()

    return {
        "name": "AIDocumentIndexer",
        "description": "Intelligent Document Archive with RAG capabilities",
        "version": "1.0.0",
        "protocol_version": "MCP/1.0",
        "capabilities": {
            "tools": True,
            "resources": False,  # Not implemented yet
            "prompts": False,    # Not implemented yet
        },
        "tools": tools,
        "endpoints": {
            "list_tools": "/api/v1/mcp/tools",
            "call_tool": "/api/v1/mcp/tools/call",
            "message": "/api/v1/mcp/message",
            "health": "/api/v1/mcp/health",
        },
    }
