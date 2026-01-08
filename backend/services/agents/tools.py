"""
AIDocumentIndexer - Agent Tool Framework
==========================================

Provides a pluggable tool system for agents to extend their capabilities:
- Tool registration and discovery
- Type-safe parameter handling
- Async execution support
- Usage tracking and approval workflows

Tools enable agents to:
- Search documents via RAG
- Perform web searches
- Execute calculations
- Query external APIs
- Generate files
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Tool Types and Configuration
# =============================================================================

class ToolCategory(str, Enum):
    """Categories of tools for organization and filtering."""
    RETRIEVAL = "retrieval"       # Document/web search
    COMPUTATION = "computation"   # Calculations, data processing
    GENERATION = "generation"     # Content creation
    COMMUNICATION = "communication"  # Email, notifications
    FILE_OPERATION = "file_operation"  # File read/write/export
    EXTERNAL_API = "external_api"     # Third-party integrations


class ApprovalLevel(str, Enum):
    """Approval requirements for tool execution."""
    NONE = "none"              # Auto-execute
    LOG_ONLY = "log_only"      # Execute but log for audit
    USER_APPROVAL = "user"     # Require user confirmation
    ADMIN_APPROVAL = "admin"   # Require admin confirmation


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    description: str
    type: str  # "string", "number", "boolean", "array", "object"
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None  # Allowed values
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
        }


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""
    id: str
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[ToolResult] = None
    invoked_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    approved: bool = False
    approved_by: Optional[str] = None


# =============================================================================
# Base Tool Class
# =============================================================================

class BaseTool(ABC):
    """
    Abstract base class for agent tools.

    Subclasses must implement:
    - execute(): Core tool logic
    """

    name: str = "base_tool"
    description: str = "Base tool"
    category: ToolCategory = ToolCategory.RETRIEVAL
    approval_level: ApprovalLevel = ApprovalLevel.NONE
    parameters: List[ToolParameter] = []

    def __init__(self):
        """Initialize tool."""
        self._invocation_count = 0
        self._total_execution_time_ms = 0

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameters against schema.

        Returns list of error messages (empty if valid).
        """
        errors = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                if param.default is None:
                    errors.append(f"Missing required parameter: {param.name}")

            if param.name in params:
                value = params[param.name]

                # Type validation
                if param.type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter {param.name} must be a string")
                elif param.type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter {param.name} must be a number")
                elif param.type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Parameter {param.name} must be a boolean")
                elif param.type == "array" and not isinstance(value, list):
                    errors.append(f"Parameter {param.name} must be an array")

                # Enum validation
                if param.enum and value not in param.enum:
                    errors.append(f"Parameter {param.name} must be one of: {param.enum}")

                # Range validation
                if param.min_value is not None and isinstance(value, (int, float)):
                    if value < param.min_value:
                        errors.append(f"Parameter {param.name} must be >= {param.min_value}")
                if param.max_value is not None and isinstance(value, (int, float)):
                    if value > param.max_value:
                        errors.append(f"Parameter {param.name} must be <= {param.max_value}")

        return errors

    async def invoke(
        self,
        params: Dict[str, Any],
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Invoke tool with parameters.

        Handles validation, timing, and error handling.
        """
        import time

        # Validate parameters
        errors = self.validate_parameters(params)
        if errors:
            return ToolResult(
                success=False,
                output=None,
                error=f"Parameter validation failed: {'; '.join(errors)}",
            )

        # Apply defaults
        full_params = {}
        for param in self.parameters:
            if param.name in params:
                full_params[param.name] = params[param.name]
            elif param.default is not None:
                full_params[param.name] = param.default

        # Execute with timing
        start_time = time.time()
        try:
            result = await self.execute(**full_params)
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Update stats
            self._invocation_count += 1
            self._total_execution_time_ms += execution_time_ms

            if isinstance(result, ToolResult):
                result.execution_time_ms = execution_time_ms
                return result

            return ToolResult(
                success=True,
                output=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Tool execution failed",
                tool=self.name,
                error=str(e),
                params=list(full_params.keys()),
            )
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool logic.

        Must be implemented by subclasses.
        """
        pass


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Registry for managing available tools.

    Provides:
    - Tool registration and discovery
    - Category-based filtering
    - Schema export for LLM
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {}
        return cls._instance

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """Register a tool instance."""
        cls._tools[tool.name] = tool
        logger.info(
            "Tool registered",
            name=tool.name,
            category=tool.category.value,
        )

    @classmethod
    def register_class(cls, tool_class: Type[BaseTool]) -> None:
        """Register a tool class (instantiates automatically)."""
        tool = tool_class()
        cls.register(tool)

    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return cls._tools.get(name)

    @classmethod
    def has_tool(cls, name: str) -> bool:
        """Check if a tool is registered by name."""
        return name in cls._tools

    @classmethod
    def get_function_definitions(cls) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function definitions for all registered tools."""
        return [tool.get_schema() for tool in cls._tools.values()]

    @classmethod
    def list_tools(
        cls,
        category: Optional[ToolCategory] = None,
        approval_level: Optional[ApprovalLevel] = None,
    ) -> List[BaseTool]:
        """List tools with optional filtering."""
        tools = list(cls._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if approval_level:
            tools = [t for t in tools if t.approval_level == approval_level]

        return tools

    @classmethod
    def get_schemas(
        cls,
        category: Optional[ToolCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Get tool schemas for LLM function calling."""
        tools = cls.list_tools(category=category)
        return [t.get_schema() for t in tools]

    @classmethod
    def get_descriptions(cls) -> str:
        """Get formatted tool descriptions for prompts."""
        lines = []
        for tool in cls._tools.values():
            params = ", ".join([
                f"{p.name}: {p.type}" + ("?" if not p.required else "")
                for p in tool.parameters
            ])
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)


# =============================================================================
# Built-in Tools
# =============================================================================

class DocumentSearchTool(BaseTool):
    """Search user's uploaded documents."""

    name = "search_documents"
    description = "Search the user's uploaded documents for relevant information"
    category = ToolCategory.RETRIEVAL
    approval_level = ApprovalLevel.NONE
    parameters = [
        ToolParameter(
            name="query",
            description="Search query to find relevant documents",
            type="string",
            required=True,
        ),
        ToolParameter(
            name="collection",
            description="Optional collection name to search within",
            type="string",
            required=False,
        ),
        ToolParameter(
            name="limit",
            description="Maximum number of results to return",
            type="number",
            required=False,
            default=5,
            min_value=1,
            max_value=20,
        ),
    ]

    def __init__(self, rag_service=None):
        super().__init__()
        self.rag_service = rag_service

    def set_rag_service(self, rag_service):
        """Set RAG service for document search."""
        self.rag_service = rag_service

    async def execute(
        self,
        query: str,
        collection: Optional[str] = None,
        limit: int = 5,
    ) -> Any:
        """Execute document search."""
        if not self.rag_service:
            return {"error": "RAG service not available", "results": []}

        try:
            results = await self.rag_service.search(
                query=query,
                collection_filter=collection,
                limit=limit,
            )

            return {
                "query": query,
                "result_count": len(results),
                "results": [
                    {
                        "document_name": r.get("document_name", "Unknown"),
                        "content": r.get("content", "")[:500],
                        "score": r.get("score", 0),
                        "collection": r.get("metadata", {}).get("collection") if r.get("metadata") else None,
                    }
                    for r in results
                ],
            }

        except Exception as e:
            logger.error("Document search failed", error=str(e))
            return {"error": str(e), "results": []}


class CalculatorTool(BaseTool):
    """Perform mathematical calculations."""

    name = "calculator"
    description = "Perform mathematical calculations and evaluations"
    category = ToolCategory.COMPUTATION
    approval_level = ApprovalLevel.NONE
    parameters = [
        ToolParameter(
            name="expression",
            description="Mathematical expression to evaluate (e.g., '2 + 2 * 3')",
            type="string",
            required=True,
        ),
    ]

    async def execute(self, expression: str) -> Any:
        """Evaluate mathematical expression safely."""
        import ast
        import operator

        # Safe operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_expr(node.left)
                right = eval_expr(node.right)
                return operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_expr(node.operand)
                return operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")

        try:
            tree = ast.parse(expression, mode='eval')
            result = eval_expr(tree.body)
            return {
                "expression": expression,
                "result": result,
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": f"Failed to evaluate: {str(e)}",
            }


class WebSearchTool(BaseTool):
    """Search the web for information."""

    name = "web_search"
    description = "Search the web for current information and news"
    category = ToolCategory.RETRIEVAL
    approval_level = ApprovalLevel.LOG_ONLY  # Log for audit
    parameters = [
        ToolParameter(
            name="query",
            description="Search query",
            type="string",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            description="Maximum number of results",
            type="number",
            required=False,
            default=5,
            min_value=1,
            max_value=10,
        ),
    ]

    def __init__(self, scraper_service=None):
        super().__init__()
        self.scraper_service = scraper_service

    def set_scraper_service(self, scraper_service):
        """Set scraper service for web search."""
        self.scraper_service = scraper_service

    async def execute(
        self,
        query: str,
        max_results: int = 5,
    ) -> Any:
        """Execute web search."""
        if not self.scraper_service:
            return {"error": "Web search service not available", "results": []}

        try:
            results = await self.scraper_service.scrape_and_query(
                urls=[],
                query=query,
                max_results=max_results,
            )

            return {
                "query": query,
                "result_count": len(results),
                "results": [
                    {
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "content": r.get("content", "")[:300],
                    }
                    for r in results
                ],
            }

        except Exception as e:
            logger.error("Web search failed", error=str(e))
            return {"error": str(e), "results": []}


class CurrentDateTimeTool(BaseTool):
    """Get current date and time."""

    name = "get_datetime"
    description = "Get the current date and time in various formats"
    category = ToolCategory.COMPUTATION
    approval_level = ApprovalLevel.NONE
    parameters = [
        ToolParameter(
            name="timezone",
            description="Timezone (e.g., 'UTC', 'US/Pacific', 'Europe/London')",
            type="string",
            required=False,
            default="UTC",
        ),
        ToolParameter(
            name="format",
            description="Output format",
            type="string",
            required=False,
            default="iso",
            enum=["iso", "human", "date_only", "time_only"],
        ),
    ]

    async def execute(
        self,
        timezone: str = "UTC",
        format: str = "iso",
    ) -> Any:
        """Get current datetime."""
        from datetime import datetime
        try:
            import pytz
            tz = pytz.timezone(timezone)
        except Exception:
            from datetime import timezone as tz_module
            tz = tz_module.utc

        now = datetime.now(tz)

        if format == "iso":
            formatted = now.isoformat()
        elif format == "human":
            formatted = now.strftime("%A, %B %d, %Y at %I:%M %p")
        elif format == "date_only":
            formatted = now.strftime("%Y-%m-%d")
        elif format == "time_only":
            formatted = now.strftime("%H:%M:%S")
        else:
            formatted = now.isoformat()

        return {
            "datetime": formatted,
            "timezone": timezone,
            "format": format,
            "unix_timestamp": int(now.timestamp()),
        }


# =============================================================================
# Tool Executor
# =============================================================================

class ToolExecutor:
    """
    Executes tools with proper handling of approvals and logging.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        approval_callback: Optional[Callable] = None,
    ):
        """
        Initialize executor.

        Args:
            registry: Tool registry (uses singleton if not provided)
            approval_callback: Async callback for approval requests
        """
        self.registry = registry or ToolRegistry()
        self.approval_callback = approval_callback
        self._invocation_history: List[ToolInvocation] = []

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        skip_approval: bool = False,
    ) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            agent_id: ID of the calling agent
            session_id: Current session ID
            skip_approval: Skip approval check (for internal use)

        Returns:
            ToolResult with execution output
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}",
            )

        # Create invocation record
        invocation = ToolInvocation(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            parameters=parameters,
            agent_id=agent_id,
            session_id=session_id,
        )

        # Check approval requirements
        if not skip_approval and tool.approval_level == ApprovalLevel.USER_APPROVAL:
            if self.approval_callback:
                approved = await self.approval_callback(tool_name, parameters)
                if not approved:
                    invocation.result = ToolResult(
                        success=False,
                        output=None,
                        error="Tool execution not approved by user",
                    )
                    self._invocation_history.append(invocation)
                    return invocation.result
                invocation.approved = True

        # Execute tool
        result = await tool.invoke(
            params=parameters,
            agent_id=agent_id,
            session_id=session_id,
        )

        # Record completion
        invocation.result = result
        invocation.completed_at = datetime.utcnow()
        self._invocation_history.append(invocation)

        # Log for audit
        if tool.approval_level in (ApprovalLevel.LOG_ONLY, ApprovalLevel.USER_APPROVAL):
            logger.info(
                "Tool executed",
                tool=tool_name,
                agent=agent_id,
                session=session_id,
                success=result.success,
                execution_time_ms=result.execution_time_ms,
            )

        return result

    def get_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ToolInvocation]:
        """Get invocation history with optional filtering."""
        history = self._invocation_history

        if session_id:
            history = [h for h in history if h.session_id == session_id]

        return history[-limit:]


# =============================================================================
# Factory Functions
# =============================================================================

def register_default_tools(
    rag_service=None,
    scraper_service=None,
) -> None:
    """Register all default tools with optional service injection."""
    registry = ToolRegistry()

    # Document search
    doc_search = DocumentSearchTool(rag_service=rag_service)
    registry.register(doc_search)

    # Calculator
    registry.register(CalculatorTool())

    # Web search
    web_search = WebSearchTool(scraper_service=scraper_service)
    registry.register(web_search)

    # Date/time
    registry.register(CurrentDateTimeTool())

    logger.info(
        "Default tools registered",
        tool_count=len(registry.list_tools()),
    )


def get_tool_registry() -> ToolRegistry:
    """Get the tool registry singleton."""
    return ToolRegistry()


def get_tool_executor(
    approval_callback: Optional[Callable] = None,
) -> ToolExecutor:
    """Create a tool executor instance."""
    return ToolExecutor(approval_callback=approval_callback)
