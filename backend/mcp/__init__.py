"""
AIDocumentIndexer - Model Context Protocol (MCP) Support
=========================================================

Implements Anthropic's Model Context Protocol for tool and resource access.
Enables integration with Claude Desktop, Claude Code, and other MCP clients.

Components:
- server.py: MCP server implementation
- tools.py: Tool definitions (search, chat, upload)
- resources.py: Resource providers (documents, collections)
"""

from backend.mcp.server import MCPServer, get_mcp_server
from backend.mcp.tools import MCPToolRegistry, get_tool_registry
from backend.mcp.resources import MCPResourceProvider, get_resource_provider

__all__ = [
    "MCPServer",
    "get_mcp_server",
    "MCPToolRegistry",
    "get_tool_registry",
    "MCPResourceProvider",
    "get_resource_provider",
]
