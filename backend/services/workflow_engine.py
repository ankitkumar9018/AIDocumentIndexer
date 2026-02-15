"""
AIDocumentIndexer - Workflow Engine Service
===========================================

Provides workflow management and execution:
- CRUD operations for workflows, nodes, edges
- Workflow execution with node traversal
- Trigger management (manual, scheduled, webhook)
- Execution tracking and logging

Architecture:
- WorkflowService: CRUD for workflows
- WorkflowExecutionEngine: Executes workflow graphs
- NodeExecutor: Executes individual node types
- TriggerManager: Handles workflow triggers
"""

import uuid
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, AsyncGenerator, Callable
from enum import Enum

import structlog
from sqlalchemy import select, func, and_, or_, desc, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.services.base import (
    BaseService,
    CRUDService,
    ServiceException,
    ValidationException,
    NotFoundException,
    PermissionException,
)
from backend.db.models import (
    Workflow,
    WorkflowNode,
    WorkflowEdge,
    WorkflowExecution,
    WorkflowNodeExecution,
    WorkflowStatus,
    WorkflowNodeType,
    WorkflowTriggerType,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Workflow Service (CRUD)
# =============================================================================


class WorkflowService(CRUDService[Workflow]):
    """
    Service for workflow CRUD operations.

    Handles creating, reading, updating, and deleting workflows
    along with their nodes and edges.
    """

    model_class = Workflow
    model_name = "Workflow"

    async def get_by_id(self, id: uuid.UUID) -> Optional[Workflow]:
        """Get workflow with nodes and edges loaded."""
        session = await self.get_session()

        query = (
            select(Workflow)
            .where(Workflow.id == id)
            .options(
                selectinload(Workflow.nodes),
                selectinload(Workflow.edges),
            )
        )

        # Add organization filter - check both matching org_id and workflows belonging to user
        # This handles single-tenant mode where organization_id == user_id
        if self._organization_id:
            from sqlalchemy import or_
            query = query.where(
                or_(
                    Workflow.organization_id == self._organization_id,
                    Workflow.created_by_id == self._organization_id,  # Single-tenant fallback
                )
            )

        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def list_with_stats(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        trigger_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> tuple[List[Dict], int]:
        """
        List workflows with execution statistics.

        Returns workflows with:
        - Total executions count
        - Last execution time
        - Success/failure counts
        """
        session = await self.get_session()

        # Base query
        query = select(Workflow)

        # Organization filter - check both org_id and created_by_id
        if self._organization_id:
            query = query.where(
                or_(
                    Workflow.organization_id == self._organization_id,
                    Workflow.created_by_id == self._organization_id,  # Single-tenant fallback
                )
            )

        # Status filter
        if status == "active":
            query = query.where(Workflow.is_active == True)
        elif status == "inactive":
            query = query.where(Workflow.is_active == False)
        elif status == "draft":
            query = query.where(Workflow.is_draft == True)

        # Trigger type filter
        if trigger_type:
            query = query.where(Workflow.trigger_type == trigger_type)

        # Search filter
        if search:
            safe = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            search_pattern = f"%{safe}%"
            query = query.where(
                or_(
                    Workflow.name.ilike(search_pattern),
                    Workflow.description.ilike(search_pattern),
                )
            )

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination and ordering
        query = query.order_by(desc(Workflow.updated_at))
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        result = await session.execute(query)
        workflows = list(result.scalars().all())

        # Get execution stats for each workflow
        workflow_ids = [w.id for w in workflows]
        stats = await self._get_execution_stats(session, workflow_ids)

        # Build response with stats
        workflow_list = []
        for workflow in workflows:
            wf_stats = stats.get(workflow.id, {})
            workflow_list.append(
                {
                    "id": str(workflow.id),
                    "name": workflow.name,
                    "description": workflow.description,
                    "is_active": workflow.is_active,
                    "is_draft": workflow.is_draft,
                    "trigger_type": workflow.trigger_type,
                    "version": workflow.version,
                    "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
                    "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None,
                    "total_executions": wf_stats.get("total", 0),
                    "successful_executions": wf_stats.get("successful", 0),
                    "failed_executions": wf_stats.get("failed", 0),
                    "last_execution_at": wf_stats.get("last_execution_at"),
                }
            )

        return workflow_list, total

    async def _get_execution_stats(
        self, session: AsyncSession, workflow_ids: List[uuid.UUID]
    ) -> Dict[uuid.UUID, Dict]:
        """Get execution statistics for workflows."""
        if not workflow_ids:
            return {}

        # Query execution counts
        query = (
            select(
                WorkflowExecution.workflow_id,
                func.count(WorkflowExecution.id).label("total"),
                func.sum(
                    func.cast(WorkflowExecution.status == WorkflowStatus.COMPLETED.value, Integer)
                ).label("successful"),
                func.sum(
                    func.cast(WorkflowExecution.status == WorkflowStatus.FAILED.value, Integer)
                ).label("failed"),
                func.max(WorkflowExecution.started_at).label("last_execution_at"),
            )
            .where(WorkflowExecution.workflow_id.in_(workflow_ids))
            .group_by(WorkflowExecution.workflow_id)
        )

        try:
            result = await session.execute(query)
            rows = result.all()

            stats = {}
            for row in rows:
                stats[row.workflow_id] = {
                    "total": row.total or 0,
                    "successful": row.successful or 0,
                    "failed": row.failed or 0,
                    "last_execution_at": row.last_execution_at.isoformat() if row.last_execution_at else None,
                }

            return stats
        except Exception as e:
            logger.warning("Failed to get execution stats", error=str(e))
            return {}

    async def create_workflow(
        self,
        name: str,
        description: Optional[str] = None,
        trigger_type: str = WorkflowTriggerType.MANUAL.value,
        trigger_config: Optional[Dict] = None,
        nodes: Optional[List[Dict]] = None,
        edges: Optional[List[Dict]] = None,
        created_by_id: Optional[uuid.UUID] = None,
    ) -> Workflow:
        """
        Create a new workflow with nodes and edges.

        Args:
            name: Workflow name
            description: Optional description
            trigger_type: Type of trigger (manual, scheduled, webhook)
            trigger_config: Trigger-specific configuration
            nodes: List of node definitions
            edges: List of edge definitions
            created_by_id: User ID who created the workflow

        Returns:
            Created workflow with nodes and edges
        """
        session = await self.get_session()

        # Validate trigger type
        if trigger_type not in [t.value for t in WorkflowTriggerType]:
            raise ValidationException(f"Invalid trigger type: {trigger_type}", field="trigger_type")

        # Ensure organization_id is set - fall back to created_by_id if not provided
        # This ensures workflow is always associated with an organization
        org_id = self._organization_id
        if not org_id and created_by_id:
            org_id = created_by_id  # Use user ID as organization ID in single-tenant mode
            self.log_info("Using created_by_id as organization_id", created_by_id=str(created_by_id))

        # Create workflow
        workflow = Workflow(
            id=uuid.uuid4(),
            organization_id=org_id,
            name=name,
            description=description,
            trigger_type=trigger_type,
            trigger_config=trigger_config or {},
            is_active=False,
            is_draft=True,
            version=1,
            created_by_id=created_by_id or self._user_id,
        )
        session.add(workflow)

        # Create nodes
        node_id_map = {}  # Map temp IDs to actual IDs
        if nodes:
            for node_data in nodes:
                temp_id = node_data.get("id") or node_data.get("temp_id")
                node = WorkflowNode(
                    id=uuid.uuid4(),
                    workflow_id=workflow.id,
                    node_type=node_data.get("node_type", WorkflowNodeType.ACTION.value),
                    name=node_data.get("name", "Untitled Node"),
                    description=node_data.get("description"),
                    position_x=node_data.get("position_x", 0),
                    position_y=node_data.get("position_y", 0),
                    config=node_data.get("config", {}),
                )
                session.add(node)
                if temp_id:
                    node_id_map[temp_id] = node.id

        await session.flush()  # Ensure nodes have IDs

        # Create edges
        if edges:
            for edge_data in edges:
                source_id = edge_data.get("source_node_id")
                target_id = edge_data.get("target_node_id")

                # Resolve temp IDs if used
                if source_id in node_id_map:
                    source_id = node_id_map[source_id]
                if target_id in node_id_map:
                    target_id = node_id_map[target_id]

                edge = WorkflowEdge(
                    id=uuid.uuid4(),
                    workflow_id=workflow.id,
                    source_node_id=source_id,
                    target_node_id=target_id,
                    label=edge_data.get("label"),
                    condition=edge_data.get("condition"),
                    edge_type=edge_data.get("edge_type", "default"),
                )
                session.add(edge)

        await session.commit()
        await session.refresh(workflow)

        # Reload with relationships
        return await self.get_by_id(workflow.id)

    async def update_workflow(
        self,
        workflow_id: uuid.UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        trigger_type: Optional[str] = None,
        trigger_config: Optional[Dict] = None,
        is_active: Optional[bool] = None,
        is_draft: Optional[bool] = None,
        config: Optional[Dict] = None,
    ) -> Workflow:
        """Update workflow metadata."""
        session = await self.get_session()

        workflow = await self.get_by_id_or_raise(workflow_id)

        if name is not None:
            workflow.name = name
        if description is not None:
            workflow.description = description
        if trigger_type is not None:
            if trigger_type not in [t.value for t in WorkflowTriggerType]:
                raise ValidationException(f"Invalid trigger type: {trigger_type}", field="trigger_type")
            workflow.trigger_type = trigger_type
        if trigger_config is not None:
            workflow.trigger_config = trigger_config
        if is_active is not None:
            workflow.is_active = is_active
        if is_draft is not None:
            workflow.is_draft = is_draft
        if config is not None:
            workflow.config = config

        await session.commit()
        await session.refresh(workflow)

        self.log_info("Workflow updated", workflow_id=str(workflow_id))

        return await self.get_by_id(workflow_id)

    async def update_nodes_and_edges(
        self,
        workflow_id: uuid.UUID,
        nodes: List[Dict],
        edges: List[Dict],
    ) -> Workflow:
        """
        Update workflow nodes and edges (replace all).

        This is an atomic operation - all existing nodes and edges
        are replaced with the new ones.
        """
        session = await self.get_session()

        workflow = await self.get_by_id_or_raise(workflow_id)

        # Delete existing edges first (due to foreign key)
        from sqlalchemy import delete

        await session.execute(delete(WorkflowEdge).where(WorkflowEdge.workflow_id == workflow_id))
        await session.execute(delete(WorkflowNode).where(WorkflowNode.workflow_id == workflow_id))

        # Create new nodes
        node_id_map = {}
        for node_data in nodes:
            temp_id = node_data.get("id") or node_data.get("temp_id")
            node = WorkflowNode(
                id=uuid.uuid4(),
                workflow_id=workflow_id,
                node_type=node_data.get("node_type", WorkflowNodeType.ACTION.value),
                name=node_data.get("name", "Untitled Node"),
                description=node_data.get("description"),
                position_x=node_data.get("position_x", 0),
                position_y=node_data.get("position_y", 0),
                config=node_data.get("config", {}),
            )
            session.add(node)
            if temp_id:
                node_id_map[str(temp_id)] = node.id

        await session.flush()

        # Create new edges
        for edge_data in edges:
            source_id = str(edge_data.get("source_node_id", ""))
            target_id = str(edge_data.get("target_node_id", ""))

            # Resolve temp IDs
            resolved_source = node_id_map.get(source_id, source_id)
            resolved_target = node_id_map.get(target_id, target_id)

            edge = WorkflowEdge(
                id=uuid.uuid4(),
                workflow_id=workflow_id,
                source_node_id=resolved_source if isinstance(resolved_source, uuid.UUID) else uuid.UUID(resolved_source),
                target_node_id=resolved_target if isinstance(resolved_target, uuid.UUID) else uuid.UUID(resolved_target),
                label=edge_data.get("label"),
                condition=edge_data.get("condition"),
                edge_type=edge_data.get("edge_type", "default"),
            )
            session.add(edge)

        # Increment version
        workflow.version += 1
        workflow.is_draft = True  # Mark as draft after changes

        await session.commit()

        self.log_info(
            "Workflow nodes and edges updated",
            workflow_id=str(workflow_id),
            node_count=len(nodes),
            edge_count=len(edges),
        )

        return await self.get_by_id(workflow_id)

    async def duplicate_workflow(
        self,
        workflow_id: uuid.UUID,
        new_name: Optional[str] = None,
    ) -> Workflow:
        """Create a copy of a workflow."""
        session = await self.get_session()

        original = await self.get_by_id_or_raise(workflow_id)

        # Prepare nodes and edges for duplication
        nodes = [
            {
                "temp_id": str(node.id),
                "node_type": node.node_type,
                "name": node.name,
                "description": node.description,
                "position_x": node.position_x,
                "position_y": node.position_y,
                "config": node.config,
            }
            for node in original.nodes
        ]

        edges = [
            {
                "source_node_id": str(edge.source_node_id),
                "target_node_id": str(edge.target_node_id),
                "label": edge.label,
                "condition": edge.condition,
                "edge_type": edge.edge_type,
            }
            for edge in original.edges
        ]

        # Create duplicate
        return await self.create_workflow(
            name=new_name or f"{original.name} (Copy)",
            description=original.description,
            trigger_type=original.trigger_type,
            trigger_config=original.trigger_config,
            nodes=nodes,
            edges=edges,
            created_by_id=self._user_id,
        )

    async def publish_workflow(self, workflow_id: uuid.UUID) -> Workflow:
        """Publish a draft workflow (make it active and non-draft)."""
        workflow = await self.get_by_id_or_raise(workflow_id)

        # Validate workflow has start and end nodes (case-insensitive)
        node_types = [n.node_type.lower() if n.node_type else "" for n in workflow.nodes]
        if WorkflowNodeType.START.value not in node_types:
            raise ValidationException("Workflow must have a START node")
        if WorkflowNodeType.END.value not in node_types:
            raise ValidationException("Workflow must have an END node")

        return await self.update_workflow(
            workflow_id,
            is_active=True,
            is_draft=False,
        )


# =============================================================================
# Workflow Execution Engine
# =============================================================================


class ExecutionContext:
    """Context for workflow execution, holds variables and state."""

    def __init__(
        self,
        execution_id: uuid.UUID,
        workflow_id: uuid.UUID,
        trigger_type: str,
        trigger_data: Optional[Dict] = None,
        input_data: Optional[Dict] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session: Optional[Any] = None,
    ):
        self.execution_id = execution_id
        self.workflow_id = workflow_id
        self.trigger_type = trigger_type
        self.trigger_data = trigger_data or {}
        self.input_data = input_data or {}

        # Caller-provided context (used by _action_create_document, etc.)
        self.organization_id = organization_id
        self.user_id = user_id
        self.session = session

        # Runtime state
        self.variables: Dict[str, Any] = {}
        self.node_outputs: Dict[uuid.UUID, Any] = {}
        self.current_node_id: Optional[uuid.UUID] = None
        self.error: Optional[str] = None
        self.cancelled: bool = False

    def set_variable(self, name: str, value: Any):
        """Set a context variable."""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(name, default)

    def set_node_output(self, node_id: uuid.UUID, output: Any):
        """Store output from a node."""
        self.node_outputs[node_id] = output

    def get_node_output(self, node_id: uuid.UUID) -> Any:
        """Get output from a previous node."""
        return self.node_outputs.get(node_id)

    def resolve_template(self, template: str) -> str:
        """
        Resolve template variables in a string.

        Supports:
        - {{input.field}} - Access input data
        - {{trigger.field}} - Access trigger data
        - {{vars.name}} - Access runtime variables
        - {{nodes.node_id.field}} - Access node outputs
        """
        import re

        def replacer(match):
            path = match.group(1)
            parts = path.split(".")

            if parts[0] == "input":
                return str(self._get_nested(self.input_data, parts[1:]))
            elif parts[0] == "trigger":
                return str(self._get_nested(self.trigger_data, parts[1:]))
            elif parts[0] == "vars":
                return str(self.variables.get(parts[1], ""))
            elif parts[0] == "nodes" and len(parts) >= 2:
                node_id = uuid.UUID(parts[1])
                output = self.node_outputs.get(node_id, {})
                return str(self._get_nested(output, parts[2:]))

            return match.group(0)  # Return unchanged if not matched

        pattern = r"\{\{([^}]+)\}\}"
        return re.sub(pattern, replacer, template)

    def _get_nested(self, data: Any, path: List[str]) -> Any:
        """Get nested value from dict/list."""
        for key in path:
            if isinstance(data, dict):
                data = data.get(key)
            elif isinstance(data, list) and key.isdigit():
                data = data[int(key)] if int(key) < len(data) else None
            else:
                return None
        return data


class NodeExecutor:
    """
    Executes individual workflow nodes.

    Each node type has a specific executor method.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = structlog.get_logger(__name__)

    @staticmethod
    def _validate_outbound_url(url: str) -> None:
        """Validate outbound URL to prevent SSRF attacks."""
        import ipaddress
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP/HTTPS protocols are allowed")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Invalid URL: no hostname")

        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            raise ValueError("Requests to localhost are not allowed")

        try:
            resolved_ip = ipaddress.ip_address(socket.gethostbyname(hostname))
            if resolved_ip.is_private or resolved_ip.is_loopback or resolved_ip.is_link_local or resolved_ip.is_reserved:
                raise ValueError("Requests to private/internal IPs are not allowed")
        except socket.gaierror:
            pass  # Let the HTTP client handle DNS resolution failures

    async def execute(
        self,
        node: WorkflowNode,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Execute a node and return its output.

        Args:
            node: The node to execute
            context: Execution context with variables

        Returns:
            Dict with node output
        """
        node_type = node.node_type

        # Route to specific executor
        executor_map = {
            WorkflowNodeType.START.value: self._execute_start,
            WorkflowNodeType.END.value: self._execute_end,
            WorkflowNodeType.ACTION.value: self._execute_action,
            WorkflowNodeType.CONDITION.value: self._execute_condition,
            WorkflowNodeType.LOOP.value: self._execute_loop,
            WorkflowNodeType.CODE.value: self._execute_code,
            WorkflowNodeType.DELAY.value: self._execute_delay,
            WorkflowNodeType.HTTP.value: self._execute_http,
            WorkflowNodeType.NOTIFICATION.value: self._execute_notification,
            WorkflowNodeType.AGENT.value: self._execute_agent,
            WorkflowNodeType.VOICE_AGENT.value: self._execute_voice_agent,
            WorkflowNodeType.CHAT_AGENT.value: self._execute_chat_agent,
            WorkflowNodeType.HUMAN_APPROVAL.value: self._execute_human_approval,
            # Lobster-style advanced nodes
            WorkflowNodeType.PARALLEL.value: self._execute_parallel,
            WorkflowNodeType.JOIN.value: self._execute_join,
            WorkflowNodeType.SUBWORKFLOW.value: self._execute_subworkflow,
            WorkflowNodeType.TRANSFORM.value: self._execute_transform,
            WorkflowNodeType.SWITCH.value: self._execute_switch,
            WorkflowNodeType.RETRY.value: self._execute_retry,
            WorkflowNodeType.AGGREGATE.value: self._execute_aggregate,
        }

        executor = executor_map.get(node_type)
        if not executor:
            raise ServiceException(f"Unknown node type: {node_type}")

        self.logger.info(
            "Executing node",
            node_id=str(node.id),
            node_type=node_type,
            node_name=node.name,
        )

        return await executor(node, context)

    async def _execute_start(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """Start node - passes through input data."""
        return {"status": "started", "input": context.input_data}

    async def _execute_end(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """End node - collects final output."""
        return {
            "status": "completed",
            "output": context.variables,
            "node_outputs": {str(k): v for k, v in context.node_outputs.items()},
        }

    async def _execute_action(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Action node - performs a predefined action.

        Enhanced Config (from NodeConfigPanel):
            action_type: create_document, update_document, delete_document,
                        generate_pdf, generate_docx, generate_pptx,
                        send_email, run_query, embed_text, transform_data, set_variable
            template_id: Template ID for document generation
            data_mapping: JSON mapping of input data to template variables
            email_to, email_subject, email_body, email_html: Email configuration
            query, top_k, folder_filter: RAG query configuration
            transform_type, transform_expression: Data transformation
            on_error: fail, continue, or retry
            max_retries, retry_delay: Retry configuration
        """
        config = node.config or {}
        action_type = config.get("action_type", "noop")

        # Build params from the enhanced config structure
        params = {}

        # Document generation actions
        if action_type in ["create_document", "update_document", "generate_pdf", "generate_docx", "generate_pptx"]:
            params["template_id"] = config.get("template_id", "")
            params["format"] = action_type.replace("generate_", "") if action_type.startswith("generate_") else "docx"
            # Parse data mapping from JSON string
            data_mapping_str = config.get("data_mapping", "{}")
            try:
                params["data_mapping"] = json.loads(data_mapping_str) if isinstance(data_mapping_str, str) else data_mapping_str
            except json.JSONDecodeError:
                params["data_mapping"] = {}

        # Email action
        elif action_type == "send_email":
            params["to"] = config.get("email_to", "").split(",")
            params["subject"] = config.get("email_subject", "")
            params["body"] = config.get("email_body", "")
            params["html"] = config.get("email_html", False)

        # RAG query action
        elif action_type == "run_query":
            params["query"] = config.get("query", "")
            params["top_k"] = config.get("top_k", 5)
            params["folder_filter"] = config.get("folder_filter", "")

        # Transform data action
        elif action_type == "transform_data":
            params["transform_type"] = config.get("transform_type", "jq")
            params["expression"] = config.get("transform_expression", "")

        # Set variable action
        elif action_type == "set_variable":
            params["name"] = config.get("variable_name", "")
            params["value"] = config.get("variable_value", "")

        # Resolve template variables in params
        resolved_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved_params[key] = context.resolve_template(value)
            elif isinstance(value, dict):
                resolved_params[key] = {
                    k: context.resolve_template(v) if isinstance(v, str) else v
                    for k, v in value.items()
                }
            else:
                resolved_params[key] = value

        # Error handling configuration
        on_error = config.get("on_error", "fail")
        max_retries = config.get("max_retries", 3)
        retry_delay = config.get("retry_delay", 5)

        # Execute with retry logic if configured
        result = None
        last_error = None
        attempts = 1 if on_error != "retry" else max_retries

        for attempt in range(attempts):
            try:
                result = await self._dispatch_action(action_type, resolved_params, context)
                if result.get("status") == "success":
                    break
                last_error = result.get("error")
            except Exception as e:
                last_error = str(e)
                if attempt < attempts - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                if on_error == "fail":
                    raise
                result = {"status": "error", "error": last_error}

        self.logger.info(
            "Executed action",
            action_type=action_type,
            result_status=result.get("status") if result else "unknown",
            attempts=attempt + 1
        )

        return {
            "action_type": action_type,
            "params": resolved_params,
            "result": result,
            "attempts": attempt + 1 if on_error == "retry" else 1,
        }

    async def _dispatch_action(
        self,
        action_type: str,
        params: Dict[str, Any],
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Dispatch action to appropriate handler."""
        action_handlers = {
            "noop": self._action_noop,
            "log": self._action_log,
            "set_variable": self._action_set_variable,
            "create_document": self._action_create_document,
            "update_document": self._action_update_document,
            "delete_document": self._action_delete_document,
            "send_email": self._action_send_email,
            "call_api": self._action_call_api,
            "run_query": self._action_run_query,
            "generate_document": self._action_generate_document,
            "trigger_workflow": self._action_trigger_workflow,
        }

        handler = action_handlers.get(action_type, self._action_unknown)
        return await handler(params, context)

    async def _action_noop(self, params: Dict, context: "ExecutionContext") -> Dict:
        """No operation - does nothing."""
        return {"status": "success", "message": "No operation performed"}

    async def _action_log(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Log a message."""
        level = params.get("level", "info")
        message = params.get("message", "")
        self.logger.log(level.upper(), message, workflow_id=str(context.workflow_id))
        return {"status": "success", "logged": message}

    async def _action_set_variable(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Set a workflow variable."""
        name = params.get("name")
        value = params.get("value")
        if name:
            context.variables[name] = value
        return {"status": "success", "variable": name, "value": value}

    async def _action_create_document(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Create a new document."""
        try:
            from backend.db.models import Document
            doc = Document(
                name=params.get("name", "Workflow Generated Document"),
                content=params.get("content", ""),
                organization_id=context.organization_id,
                created_by_id=context.user_id,
            )
            context.session.add(doc)
            await context.session.flush()
            return {"status": "success", "document_id": str(doc.id)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _action_update_document(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Update an existing document."""
        try:
            from backend.db.models import Document
            doc_id = params.get("document_id")
            if not doc_id:
                return {"status": "error", "error": "document_id required"}

            result = await context.session.execute(
                select(Document).where(Document.id == uuid.UUID(doc_id))
            )
            doc = result.scalar_one_or_none()
            if not doc:
                return {"status": "error", "error": "Document not found"}

            if "name" in params:
                doc.name = params["name"]
            if "content" in params:
                doc.content = params["content"]

            return {"status": "success", "document_id": doc_id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _action_delete_document(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Delete a document."""
        try:
            from backend.db.models import Document
            doc_id = params.get("document_id")
            if not doc_id:
                return {"status": "error", "error": "document_id required"}

            result = await context.session.execute(
                select(Document).where(Document.id == uuid.UUID(doc_id))
            )
            doc = result.scalar_one_or_none()
            if doc:
                await context.session.delete(doc)
            return {"status": "success", "deleted": doc_id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _action_send_email(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Send an email (placeholder - integrate with email service)."""
        to = params.get("to", [])
        subject = params.get("subject", "")
        body = params.get("body", "")

        # Log for now - integrate with actual email service
        self.logger.info("Email action", to=to, subject=subject, body_length=len(body))
        return {"status": "success", "to": to, "subject": subject, "sent": True}

    async def _action_call_api(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Make an API call."""
        import aiohttp
        url = params.get("url", "")
        method = params.get("method", "GET")
        headers = params.get("headers", {})
        body = params.get("body")

        try:
            self._validate_outbound_url(url)
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=headers, json=body,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return {
                        "status": "success",
                        "status_code": response.status,
                        "body": await response.text()
                    }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _action_run_query(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Run a RAG query with full parameter support."""
        try:
            from backend.services.rag import RAGService
            rag = RAGService()
            query = params.get("query", "")

            # Build RAG query kwargs with all supported parameters
            rag_kwargs = {
                "query": query,
                "organization_id": str(context.organization_id) if context.organization_id else None,
            }

            # Pass through top_k if specified
            if params.get("top_k"):
                rag_kwargs["top_k"] = params.get("top_k")

            # Pass through folder_filter if specified
            if params.get("folder_filter"):
                rag_kwargs["folder_filter"] = params.get("folder_filter")

            # Pass through enable_knowledge_graph if specified
            if "enable_kg" in params:
                rag_kwargs["enable_knowledge_graph"] = params.get("enable_kg", True)

            result = await rag.query(**rag_kwargs)
            return {
                "status": "success",
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "kg_entities": result.get("kg_entities", []),  # Include KG entities if available
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _action_generate_document(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Trigger document generation."""
        # Placeholder - integrate with generation service
        format_type = params.get("format", "docx")
        topic = params.get("topic", "")
        self.logger.info("Document generation action", format=format_type, topic=topic)
        return {"status": "success", "format": format_type, "topic": topic, "job_id": "pending"}

    async def _action_trigger_workflow(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Trigger another workflow."""
        workflow_id = params.get("workflow_id")
        input_data = params.get("input", {})
        if not workflow_id:
            return {"status": "error", "error": "workflow_id required"}

        self.logger.info("Triggering sub-workflow", workflow_id=workflow_id)
        return {"status": "success", "triggered_workflow": workflow_id, "input": input_data}

    async def _action_unknown(self, params: Dict, context: "ExecutionContext") -> Dict:
        """Handle unknown action type."""
        return {"status": "error", "error": "Unknown action type"}

    async def _execute_condition(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Condition node - evaluates an expression.

        Enhanced Config (from NodeConfigPanel):
            condition_type: expression, compare, exists, type_check, regex, all, any
            expression: JavaScript expression for 'expression' type
            left_value, operator, right_value: For 'compare' type
            regex_input, regex_pattern, regex_case_insensitive: For 'regex' type
            true_label, false_label: Branch labels

        Returns:
            result: True/False based on expression
        """
        config = node.config or {}
        condition_type = config.get("condition_type", "expression")
        result = False

        try:
            if condition_type == "expression":
                # JavaScript-like expression evaluation
                expression = config.get("expression", "true")
                resolved = context.resolve_template(expression)
                result = self._evaluate_expression(resolved)

            elif condition_type == "compare":
                # Compare two values with operator
                left = context.resolve_template(config.get("left_value", ""))
                right = context.resolve_template(config.get("right_value", ""))
                operator = config.get("operator", "equals")

                result = self._compare_values(left, right, operator)

            elif condition_type == "exists":
                # Check if value exists (not None, not empty)
                check_value = context.resolve_template(config.get("check_value", ""))
                result = bool(check_value and check_value.strip())

            elif condition_type == "type_check":
                # Check the type of a value
                check_value = context.resolve_template(config.get("check_value", ""))
                expected_type = config.get("expected_type", "string")
                result = self._check_type(check_value, expected_type)

            elif condition_type == "regex":
                # Regex pattern matching
                input_value = context.resolve_template(config.get("regex_input", ""))
                pattern = config.get("regex_pattern", "")
                case_insensitive = config.get("regex_case_insensitive", False)

                import re
                flags = re.IGNORECASE if case_insensitive else 0
                try:
                    result = bool(re.search(pattern, input_value, flags))
                except re.error:
                    result = False

            elif condition_type in ("all", "any"):
                # Multiple conditions (AND/OR)
                conditions = config.get("conditions", [])
                results = []
                for cond in conditions:
                    sub_result = self._evaluate_sub_condition(cond, context)
                    results.append(sub_result)

                if condition_type == "all":
                    result = all(results) if results else False
                else:
                    result = any(results) if results else False

            else:
                # Fallback to expression
                expression = config.get("expression", "true")
                resolved = context.resolve_template(expression)
                result = self._evaluate_expression(resolved)

        except Exception as e:
            self.logger.warning(
                "Condition evaluation failed",
                condition_type=condition_type,
                error=str(e)
            )
            result = False

        return {
            "result": result,
            "condition_type": condition_type,
            "true_label": config.get("true_label", "Yes"),
            "false_label": config.get("false_label", "No"),
        }

    def _compare_values(self, left: str, right: str, operator: str) -> bool:
        """Compare two values with the given operator."""
        # Try numeric comparison first
        try:
            left_num = float(left)
            right_num = float(right)

            if operator == "equals":
                return left_num == right_num
            elif operator == "not_equals":
                return left_num != right_num
            elif operator == "greater":
                return left_num > right_num
            elif operator == "greater_eq":
                return left_num >= right_num
            elif operator == "less":
                return left_num < right_num
            elif operator == "less_eq":
                return left_num <= right_num
        except (ValueError, TypeError):
            pass

        # Fall back to string comparison
        if operator == "equals":
            return left == right
        elif operator == "not_equals":
            return left != right
        elif operator == "contains":
            return right in left
        elif operator == "starts_with":
            return left.startswith(right)
        elif operator == "ends_with":
            return left.endswith(right)
        elif operator == "in":
            # Check if left is in right (comma-separated list)
            items = [i.strip() for i in right.split(",")]
            return left in items

        return False

    def _check_type(self, value: str, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if expected_type == "string":
            return True  # Everything resolved is a string
        elif expected_type == "number":
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == "integer":
            try:
                int(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == "boolean":
            return value.lower() in ("true", "false", "1", "0", "yes", "no")
        elif expected_type == "array":
            try:
                parsed = json.loads(value)
                return isinstance(parsed, list)
            except (json.JSONDecodeError, TypeError):
                return False
        elif expected_type == "object":
            try:
                parsed = json.loads(value)
                return isinstance(parsed, dict)
            except (json.JSONDecodeError, TypeError):
                return False
        return False

    def _evaluate_sub_condition(self, cond: Dict, context: "ExecutionContext") -> bool:
        """Evaluate a sub-condition for all/any conditions."""
        left = context.resolve_template(cond.get("left", ""))
        right = context.resolve_template(cond.get("right", ""))
        operator = cond.get("operator", "equals")
        return self._compare_values(left, right, operator)

    def _evaluate_expression(self, expr: str) -> bool:
        """Safely evaluate a simple boolean expression."""
        expr = expr.strip().lower()

        # Boolean literals
        if expr in ("true", "1", "yes"):
            return True
        if expr in ("false", "0", "no", ""):
            return False

        # Simple comparisons
        import re

        # Numeric comparisons
        match = re.match(r"(\d+(?:\.\d+)?)\s*([<>=!]+)\s*(\d+(?:\.\d+)?)", expr)
        if match:
            left = float(match.group(1))
            op = match.group(2)
            right = float(match.group(3))

            if op == "==":
                return left == right
            elif op == "!=":
                return left != right
            elif op == ">":
                return left > right
            elif op == ">=":
                return left >= right
            elif op == "<":
                return left < right
            elif op == "<=":
                return left <= right

        # String comparisons
        match = re.match(r"'([^']*)'\s*([<>=!]+)\s*'([^']*)'", expr)
        if match:
            left = match.group(1)
            op = match.group(2)
            right = match.group(3)

            if op == "==":
                return left == right
            elif op == "!=":
                return left != right

        return False

    async def _execute_loop(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Loop node - iterates over items.

        Enhanced Config (from NodeConfigPanel):
            loop_type: for_each, while, count, batch
            items_source: Path to array (for for_each, batch)
            item_var: Variable name for current item
            while_condition: Condition for while loop
            count: Number of iterations for count loop
            batch_size: Size of each batch (for batch)
            max_iterations: Maximum iterations (safety limit)
            delay_ms: Delay between iterations
            parallel: Run iterations in parallel
            concurrency: Max parallel executions

        This node sets up loop context; actual iteration is handled by engine.
        """
        config = node.config or {}
        loop_type = config.get("loop_type", "for_each")
        item_var = config.get("item_var", "item")
        max_iterations = config.get("max_iterations", 100)
        delay_ms = config.get("delay_ms", 0)
        parallel = config.get("parallel", False)
        concurrency = config.get("concurrency", 5)

        items = []
        loop_info = {}

        if loop_type == "for_each":
            # Iterate over array items
            items_source = config.get("items_source", "[]")
            resolved = context.resolve_template(items_source)
            try:
                items = json.loads(resolved) if isinstance(resolved, str) else resolved
            except json.JSONDecodeError:
                items = []

            if not isinstance(items, list):
                items = [items] if items else []

        elif loop_type == "while":
            # Generate items based on while condition
            while_condition = config.get("while_condition", "false")
            iteration = 0

            while iteration < max_iterations:
                # Set loop context for condition evaluation
                context.set_variable("loop", {"index": iteration, "iteration": iteration + 1})

                # Evaluate while condition
                resolved_condition = context.resolve_template(while_condition)
                if not self._evaluate_expression(resolved_condition):
                    break

                items.append({"index": iteration, "iteration": iteration + 1})
                iteration += 1

            loop_info["iterations_run"] = iteration

        elif loop_type == "count":
            # Fixed number of iterations
            count = min(config.get("count", 5), max_iterations)
            items = [{"index": i, "iteration": i + 1} for i in range(count)]

        elif loop_type == "batch":
            # Batch processing
            items_source = config.get("items_source", "[]")
            batch_size = config.get("batch_size", 10)

            resolved = context.resolve_template(items_source)
            try:
                all_items = json.loads(resolved) if isinstance(resolved, str) else resolved
            except json.JSONDecodeError:
                all_items = []

            if not isinstance(all_items, list):
                all_items = [all_items] if all_items else []

            # Split into batches
            items = [
                all_items[i:i + batch_size]
                for i in range(0, len(all_items), batch_size)
            ]
            loop_info["total_items"] = len(all_items)
            loop_info["batch_count"] = len(items)

        # Apply max_iterations limit
        items = items[:max_iterations]

        return {
            "items": items,
            "item_var": item_var,
            "total": len(items),
            "loop_type": loop_type,
            "parallel": parallel,
            "concurrency": concurrency,
            "delay_ms": delay_ms,
            **loop_info,
        }

    async def _execute_code(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Code node - executes custom code.

        Enhanced Config (from NodeConfigPanel):
            language: python, javascript
            mode: all_items, each_item
            code: Code to execute
            timeout: Execution timeout in seconds
            memory_limit: Memory limit in MB
            sandbox: Whether to run in sandbox

        WARNING: Code execution should be sandboxed in production!
        """
        config = node.config or {}
        language = config.get("language", "python")
        code = config.get("code", "")
        mode = config.get("mode", "all_items")
        timeout_seconds = config.get("timeout", 30)
        memory_limit_mb = config.get("memory_limit", 256)
        use_sandbox = config.get("sandbox", True)

        # Sandboxed code execution
        if not code.strip():
            return {"status": "skipped", "reason": "No code provided", "language": language}

        # Prepare execution context with workflow variables
        exec_context = {
            "variables": dict(context.variables),
            "input": context.input_data or {},
            "output": {},
            "nodes": {str(k): v for k, v in context.node_outputs.items()},
        }

        # Add context object for accessing workflow info
        exec_context["context"] = {
            "execution_id": str(context.execution_id),
            "workflow_id": str(context.workflow_id),
            "trigger_type": context.trigger_type,
        }

        try:
            if language == "python":
                if use_sandbox:
                    result = await asyncio.wait_for(
                        self._execute_python_sandboxed(code, exec_context),
                        timeout=timeout_seconds
                    )
                else:
                    result = await asyncio.wait_for(
                        self._execute_python_basic(code, exec_context),
                        timeout=timeout_seconds
                    )
            elif language == "javascript":
                result = await asyncio.wait_for(
                    self._execute_javascript_sandboxed(code, exec_context),
                    timeout=timeout_seconds
                )
            else:
                return {"status": "error", "error": f"Unsupported language: {language}"}

            # Update context variables from output
            if "output" in result and isinstance(result["output"], dict):
                context.variables.update(result["output"])

            return {
                "status": "success",
                "language": language,
                "mode": mode,
                "result": result.get("result"),
                "output": result.get("output", {}),
            }
        except asyncio.TimeoutError:
            self.logger.error("Code execution timeout", language=language, timeout=timeout_seconds)
            return {"status": "error", "language": language, "error": f"Execution timed out after {timeout_seconds}s"}
        except Exception as e:
            self.logger.error("Code execution failed", language=language, error=str(e))
            return {"status": "error", "language": language, "error": str(e)}

    async def _execute_python_sandboxed(self, code: str, context: Dict) -> Dict:
        """
        Execute Python code in a restricted sandbox.

        Uses RestrictedPython for safe execution with limited builtins.
        """
        try:
            from RestrictedPython import compile_restricted, safe_globals
            from RestrictedPython.Eval import default_guarded_getitem
            from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence
        except ImportError:
            # Fallback to basic exec with timeout if RestrictedPython not available
            return await self._execute_python_basic(code, context)

        # Prepare restricted globals with safe module wrappers
        # NOTE: Never pass raw module objects — they expose __builtins__/__import__
        import math as _math

        class _SafeMath:
            """Restricted math interface."""
            ceil = staticmethod(_math.ceil)
            floor = staticmethod(_math.floor)
            sqrt = staticmethod(_math.sqrt)
            pow = staticmethod(_math.pow)
            log = staticmethod(_math.log)
            log10 = staticmethod(_math.log10)
            abs = staticmethod(_math.fabs)
            pi = _math.pi
            e = _math.e
            inf = _math.inf
            def __getattr__(self, name):
                raise AttributeError(f"Access to 'math.{name}' is not allowed in sandbox")

        class _SafeJson:
            """Restricted JSON interface."""
            @staticmethod
            def loads(s, **kwargs): return json.loads(s, **kwargs)
            @staticmethod
            def dumps(obj, indent=None, ensure_ascii=True, default=None, sort_keys=False):
                return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=default, sort_keys=sort_keys)
            def __getattr__(self, name):
                raise AttributeError(f"Access to 'json.{name}' is not allowed in sandbox")

        import re as _re

        class _SafeRegex:
            """Restricted regex interface."""
            @staticmethod
            def search(pattern, string, flags=0): return _re.search(pattern, string, flags)
            @staticmethod
            def match(pattern, string, flags=0): return _re.match(pattern, string, flags)
            @staticmethod
            def findall(pattern, string, flags=0): return _re.findall(pattern, string, flags)
            @staticmethod
            def sub(pattern, repl, string, count=0, flags=0): return _re.sub(pattern, repl, string, count, flags)
            @staticmethod
            def split(pattern, string, maxsplit=0, flags=0): return _re.split(pattern, string, maxsplit, flags)
            IGNORECASE = _re.IGNORECASE
            MULTILINE = _re.MULTILINE
            DOTALL = _re.DOTALL
            def __getattr__(self, name):
                raise AttributeError(f"Access to 're.{name}' is not allowed in sandbox")

        class _SafeDatetime:
            """Restricted datetime interface — no access to module internals."""
            @staticmethod
            def now(): return datetime.now()
            @staticmethod
            def utcnow(): return datetime.utcnow()
            @staticmethod
            def fromisoformat(s): return datetime.fromisoformat(s)
            @staticmethod
            def strptime(date_string, fmt): return datetime.strptime(date_string, fmt)
            @staticmethod
            def today(): return datetime.today()
            def __getattr__(self, name):
                raise AttributeError(f"Access to 'datetime.{name}' is not allowed in sandbox")

        restricted_globals = {
            "__builtins__": safe_builtins,
            "_getitem_": default_guarded_getitem,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            # Safe module wrappers (no __builtins__/__import__ access)
            "json": _SafeJson(),
            "math": _SafeMath(),
            "datetime": _SafeDatetime(),
            "re": _SafeRegex(),
            # Add context
            "variables": context.get("variables", {}),
            "input": context.get("input", {}),
            "output": {},
        }

        # Compile with restrictions
        try:
            byte_code = compile_restricted(code, "<workflow_code>", "exec")
        except SyntaxError as e:
            return {"status": "error", "error": f"Syntax error: {e}"}

        # Execute with timeout
        local_vars = {}
        try:
            exec(byte_code, restricted_globals, local_vars)
        except Exception as e:
            return {"status": "error", "error": f"Execution error: {e}"}

        return {
            "status": "success",
            "result": local_vars.get("result"),
            "output": restricted_globals.get("output", {}),
        }

    async def _execute_python_basic(self, code: str, context: Dict) -> Dict:
        """Basic Python execution with safety restrictions (no RestrictedPython)."""
        import ast

        # Parse and validate - only allow simple expressions and assignments
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"status": "error", "error": f"Syntax error: {e}"}

        # Check for dangerous constructs
        dangerous_nodes = (ast.Import, ast.ImportFrom, ast.Global)
        # Also block ast.Exec on Python < 3.x (doesn't exist in 3.x but safe to check)
        if hasattr(ast, 'Exec'):
            dangerous_nodes = dangerous_nodes + (ast.Exec,)

        # Blocked dunder attributes that enable sandbox escape
        blocked_attrs = {
            "__builtins__", "__import__", "__class__", "__bases__",
            "__subclasses__", "__mro__", "__globals__", "__code__",
            "__getattribute__", "__dict__", "__module__", "__loader__",
            "__spec__", "__init_subclass__",
        }

        for node in ast.walk(tree):
            if isinstance(node, dangerous_nodes):
                return {"status": "error", "error": "Imports and global statements not allowed"}
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "open", "__import__",
                                         "getattr", "setattr", "delattr", "globals", "locals",
                                         "vars", "dir", "breakpoint"):
                        return {"status": "error", "error": f"Function '{node.func.id}' not allowed"}
            # Block access to dunder attributes (prevents __class__.__bases__ escape)
            if isinstance(node, ast.Attribute):
                if node.attr in blocked_attrs:
                    return {"status": "error", "error": f"Access to '{node.attr}' is not allowed"}

        # Execute with limited builtins
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "float": float, "int": int, "len": len, "list": list, "max": max,
            "min": min, "range": range, "round": round, "set": set, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "zip": zip, "True": True,
            "False": False, "None": None,
        }

        local_vars = {
            "variables": context.get("variables", {}),
            "input": context.get("input", {}),
            "output": {},
        }

        try:
            exec(code, {"__builtins__": safe_builtins}, local_vars)
        except Exception as e:
            return {"status": "error", "error": f"Execution error: {e}"}

        return {
            "status": "success",
            "result": local_vars.get("result"),
            "output": local_vars.get("output", {}),
        }

    async def _execute_javascript_sandboxed(self, code: str, context: Dict) -> Dict:
        """
        Execute JavaScript code in a sandbox.

        Uses subprocess with a Node.js runner or a Python JS engine.
        """
        try:
            # Try using py_mini_racer (V8 engine for Python)
            from py_mini_racer import MiniRacer

            ctx = MiniRacer()

            # Prepare context
            setup_code = f"""
            var variables = {json.dumps(context.get('variables', {}))};
            var input = {json.dumps(context.get('input', {}))};
            var output = {{}};
            """

            ctx.eval(setup_code)
            result = ctx.eval(code)

            return {
                "status": "success",
                "result": result,
                "output": ctx.eval("output"),
            }
        except ImportError:
            # Fallback: Use subprocess with Node.js if available
            return await self._execute_javascript_subprocess(code, context)
        except Exception as e:
            return {"status": "error", "error": f"JavaScript execution error: {e}"}

    async def _execute_javascript_subprocess(self, code: str, context: Dict) -> Dict:
        """Execute JavaScript using Node.js subprocess."""
        import subprocess
        import tempfile
        import os

        # Wrap code with context setup
        wrapper_code = f"""
        const variables = {json.dumps(context.get('variables', {}))};
        const input = {json.dumps(context.get('input', {}))};
        let output = {{}};

        {code}

        console.log(JSON.stringify({{ result: typeof result !== 'undefined' ? result : null, output }}));
        """

        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(wrapper_code)
                temp_path = f.name

            # Execute with timeout
            proc = await asyncio.create_subprocess_exec(
                'node', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                return {"status": "error", "error": "Execution timeout (5s)"}
            finally:
                os.unlink(temp_path)

            if proc.returncode != 0:
                return {"status": "error", "error": stderr.decode().strip()}

            output = json.loads(stdout.decode().strip())
            return {"status": "success", **output}

        except FileNotFoundError:
            return {"status": "error", "error": "Node.js not available for JavaScript execution"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _execute_delay(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Delay node - waits for specified time.

        Enhanced Config (from NodeConfigPanel):
            delay_type: fixed, until, cron
            delay_value: Duration value (for fixed)
            delay_unit: seconds, minutes, hours, days (for fixed)
            until_time: ISO datetime or expression (for until)
            cron: Cron expression (for cron)
        """
        config = node.config or {}
        delay_type = config.get("delay_type", "fixed")
        seconds = 0

        if delay_type == "fixed":
            # Fixed duration delay
            delay_value = config.get("delay_value", 0)
            delay_unit = config.get("delay_unit", "seconds")

            # Convert to seconds
            multipliers = {
                "seconds": 1,
                "minutes": 60,
                "hours": 3600,
                "days": 86400,
            }
            seconds = delay_value * multipliers.get(delay_unit, 1)

        elif delay_type == "until":
            # Wait until specific time
            until_time = config.get("until_time", "")
            resolved = context.resolve_template(until_time)
            try:
                target_time = datetime.fromisoformat(resolved.replace("Z", "+00:00"))
                wait_seconds = (target_time - datetime.utcnow()).total_seconds()
                seconds = max(0, wait_seconds)
            except (ValueError, TypeError) as e:
                self.logger.warning("Invalid until_time format", until_time=until_time, error=str(e))
                seconds = 0

        elif delay_type == "cron":
            # Wait until next cron occurrence
            cron_expression = config.get("cron", "")
            if cron_expression:
                try:
                    from croniter import croniter
                    cron = croniter(cron_expression, datetime.utcnow())
                    next_run = cron.get_next(datetime)
                    seconds = (next_run - datetime.utcnow()).total_seconds()
                except ImportError:
                    self.logger.warning("croniter not installed, using fixed delay")
                    seconds = 60  # Default to 1 minute if croniter not available
                except Exception as e:
                    self.logger.warning("Invalid cron expression", cron=cron_expression, error=str(e))
                    seconds = 0

        if seconds > 0:
            # Cap at 1 hour for safety (can be configured)
            max_wait = config.get("max_wait_seconds", 3600)
            seconds = min(seconds, max_wait)
            await asyncio.sleep(seconds)

        return {
            "waited_seconds": seconds,
            "delay_type": delay_type,
        }

    async def _execute_http(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        HTTP node - makes an HTTP request.

        Enhanced Config (from NodeConfigPanel):
            method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS
            url: URL to request
            auth_type: none, basic, bearer, api_key, oauth2
            auth_username, auth_password: For basic auth
            auth_token: For bearer auth
            api_key_name, api_key_value, api_key_location: For API key auth
            headers: Dict of headers
            query_params: Dict of query parameters
            content_type: json, form, multipart, raw, xml
            body: Request body
            response_type: json, text, binary, auto
            extract_path: JSONPath to extract from response
            timeout: Request timeout in seconds
            retries: Number of retries
            follow_redirects: Whether to follow redirects
            ignore_ssl: Whether to ignore SSL errors
        """
        import aiohttp
        import base64
        import ipaddress as _ipaddress
        import socket as _socket

        config = node.config or {}
        url = context.resolve_template(config.get("url", ""))
        method = config.get("method", "GET").upper()
        timeout_seconds = min(config.get("timeout", 30), 120)
        retries = min(config.get("retries", 0), 5)
        follow_redirects = config.get("follow_redirects", True)
        ignore_ssl = config.get("ignore_ssl", False)

        # SSRF protection: block requests to private/internal IPs
        self._validate_outbound_url(url)

        # Build headers
        headers = {}
        for key, value in config.get("headers", {}).items():
            headers[key] = context.resolve_template(value) if isinstance(value, str) else value

        # Add authentication headers
        auth_type = config.get("auth_type", "none")
        if auth_type == "basic":
            username = context.resolve_template(config.get("auth_username", ""))
            password = context.resolve_template(config.get("auth_password", ""))
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        elif auth_type == "bearer":
            token = context.resolve_template(config.get("auth_token", ""))
            headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "api_key":
            key_name = config.get("api_key_name", "X-API-Key")
            key_value = context.resolve_template(config.get("api_key_value", ""))
            key_location = config.get("api_key_location", "header")
            if key_location == "header":
                headers[key_name] = key_value
            # Query param handled below

        # Add query parameters to URL
        query_params = config.get("query_params", {})
        if auth_type == "api_key" and config.get("api_key_location") == "query":
            query_params[config.get("api_key_name", "api_key")] = context.resolve_template(config.get("api_key_value", ""))

        if query_params:
            resolved_params = {k: context.resolve_template(v) if isinstance(v, str) else v for k, v in query_params.items()}
            from urllib.parse import urlencode, urlparse, urlunparse, parse_qs
            parsed = urlparse(url)
            existing_params = parse_qs(parsed.query)
            existing_params.update(resolved_params)
            new_query = urlencode(existing_params, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))

        # Prepare request body
        body = None
        body_data = config.get("body", "")
        content_type = config.get("content_type", "json")

        if body_data and method in ("POST", "PUT", "PATCH"):
            resolved_body = context.resolve_template(body_data) if isinstance(body_data, str) else body_data

            if content_type == "json":
                try:
                    body = json.loads(resolved_body) if isinstance(resolved_body, str) else resolved_body
                except json.JSONDecodeError:
                    body = resolved_body
                headers.setdefault("Content-Type", "application/json")
            elif content_type == "form":
                body = resolved_body
                headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
            elif content_type == "xml":
                body = resolved_body
                headers.setdefault("Content-Type", "application/xml")
            else:
                body = resolved_body

        # SSL context for ignore_ssl
        ssl_context = False if ignore_ssl else None

        # Execute request with retries
        last_error = None
        for attempt in range(retries + 1):
            try:
                connector = aiohttp.TCPConnector(ssl=ssl_context) if ignore_ssl else None
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        json=body if content_type == "json" and isinstance(body, (dict, list)) else None,
                        data=body if content_type != "json" or not isinstance(body, (dict, list)) else None,
                        timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                        allow_redirects=follow_redirects,
                    ) as response:
                        # Limit response body size to prevent OOM
                        MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB
                        response_bytes = await response.content.read(MAX_RESPONSE_SIZE + 1)
                        if len(response_bytes) > MAX_RESPONSE_SIZE:
                            raise ValueError(f"Response body exceeds {MAX_RESPONSE_SIZE // (1024*1024)}MB limit")
                        response_text = response_bytes.decode('utf-8', errors='replace')

                        # Parse response based on type
                        response_type = config.get("response_type", "auto")
                        parsed_body = response_text

                        if response_type == "json" or (response_type == "auto" and "application/json" in response.headers.get("Content-Type", "")):
                            try:
                                parsed_body = json.loads(response_text)
                            except json.JSONDecodeError:
                                parsed_body = response_text

                        # Extract path if specified
                        extract_path = config.get("extract_path", "")
                        extracted = None
                        if extract_path and isinstance(parsed_body, (dict, list)):
                            extracted = self._extract_jsonpath(parsed_body, extract_path)

                        return {
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "body": parsed_body,
                            "extracted": extracted,
                            "ok": response.ok,
                            "attempts": attempt + 1,
                        }

            except Exception as e:
                last_error = str(e)
                if attempt < retries:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue

        self.logger.error("HTTP request failed", url=url, error=last_error, attempts=retries + 1)
        return {
            "status_code": 0,
            "error": last_error,
            "ok": False,
            "attempts": retries + 1,
        }

    def _extract_jsonpath(self, data: Any, path: str) -> Any:
        """Extract value from data using a simple JSONPath-like syntax."""
        # Simple implementation for $.key.subkey syntax
        if not path.startswith("$."):
            return None

        parts = path[2:].split(".")
        result = data
        for part in parts:
            if isinstance(result, dict):
                result = result.get(part)
            elif isinstance(result, list) and part.isdigit():
                idx = int(part)
                result = result[idx] if idx < len(result) else None
            else:
                return None
        return result

    async def _execute_notification(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Notification node - sends a notification.

        Enhanced Config (from NodeConfigPanel):
            channel: email, slack, teams, discord, webhook, sms
            recipients: Comma-separated recipients
            subject: Subject/title
            message: Message body
            html_email: Whether email is HTML formatted
            webhook_url: URL for Slack/Teams/Discord/webhook
            include_output: Include workflow output as attachment
            attachment_path: Path to file to attach
        """
        config = node.config or {}
        channel = config.get("channel", "email")

        # Parse recipients
        recipients_str = config.get("recipients", "")
        recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

        # Build message
        subject = context.resolve_template(config.get("subject", "Workflow Notification"))
        message = context.resolve_template(config.get("message", ""))

        # Handle attachments
        include_output = config.get("include_output", False)
        attachment_path = config.get("attachment_path", "")

        attachment = None
        if include_output:
            attachment = json.dumps(context.variables, indent=2)
        elif attachment_path:
            resolved_path = context.resolve_template(attachment_path)
            attachment = resolved_path  # Path to file

        # Build notification config
        notification_config = {
            **config,
            "subject": subject,
            "html_email": config.get("html_email", False),
            "webhook_url": config.get("webhook_url", ""),
            "attachment": attachment,
        }

        # Dispatch to appropriate notification channel
        try:
            result = await self._send_notification(channel, message, recipients, notification_config, context)
            self.logger.info(
                "Notification sent",
                channel=channel,
                recipients_count=len(recipients),
                success=result.get("sent", False),
            )
            return result
        except Exception as e:
            self.logger.error("Notification failed", channel=channel, error=str(e))
            return {
                "channel": channel,
                "message": message,
                "recipients": recipients,
                "sent": False,
                "error": str(e),
            }

    async def _send_notification(
        self,
        channel: str,
        message: str,
        recipients: List[str],
        config: Dict,
        context: "ExecutionContext",
    ) -> Dict:
        """Send notification via the specified channel."""
        handlers = {
            "log": self._notify_log,
            "email": self._notify_email,
            "slack": self._notify_slack,
            "webhook": self._notify_webhook,
            "teams": self._notify_teams,
            "discord": self._notify_discord,
        }

        handler = handlers.get(channel, self._notify_log)
        return await handler(message, recipients, config, context)

    async def _notify_log(self, message: str, recipients: List, config: Dict, context: "ExecutionContext") -> Dict:
        """Log notification (default fallback)."""
        self.logger.info("Workflow notification", message=message, recipients=recipients)
        return {"channel": "log", "message": message, "recipients": recipients, "sent": True}

    async def _notify_email(self, message: str, recipients: List, config: Dict, context: "ExecutionContext") -> Dict:
        """
        Send email notification via SMTP.

        Config options:
            - subject: Email subject line
            - html: If True, send as HTML email
            - from_name: Sender display name
            - reply_to: Reply-to address

        SMTP settings from environment or database settings:
            - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
        """
        import os
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        subject = config.get("subject", "Workflow Notification")
        is_html = config.get("html", False)

        # Get SMTP settings
        smtp_settings = await self._get_smtp_settings()
        smtp_host = smtp_settings.get("host")
        smtp_port = smtp_settings.get("port", 587)
        smtp_user = smtp_settings.get("user")
        smtp_pass = smtp_settings.get("password")
        smtp_from = smtp_settings.get("from_email")
        from_name = config.get("from_name", smtp_settings.get("from_name", "AIDocumentIndexer"))

        if not smtp_host:
            self.logger.warning("SMTP not configured, email notification skipped")
            return {
                "channel": "email",
                "sent": False,
                "error": "SMTP not configured. Set SMTP_HOST in environment or settings.",
                "recipients": recipients,
            }

        try:
            import aiosmtplib

            # Build email message
            msg = MIMEMultipart("alternative") if is_html else MIMEText(message, "plain")
            msg["Subject"] = subject
            msg["To"] = ", ".join(recipients)
            msg["From"] = f"{from_name} <{smtp_from}>" if from_name else smtp_from

            if config.get("reply_to"):
                msg["Reply-To"] = config["reply_to"]

            if is_html:
                # Add both plain text and HTML versions
                plain_part = MIMEText(message, "plain")
                html_part = MIMEText(message, "html")
                msg.attach(plain_part)
                msg.attach(html_part)

            # Send email
            await aiosmtplib.send(
                msg,
                hostname=smtp_host,
                port=smtp_port,
                username=smtp_user,
                password=smtp_pass,
                use_tls=smtp_port == 465,
                start_tls=smtp_port == 587,
                timeout=30,
            )

            self.logger.info(
                "Email notification sent",
                to=recipients,
                subject=subject,
                smtp_host=smtp_host,
            )
            return {
                "channel": "email",
                "sent": True,
                "message": message,
                "recipients": recipients,
                "subject": subject,
            }

        except ImportError:
            self.logger.warning("aiosmtplib not installed, email skipped")
            return {"channel": "email", "sent": False, "error": "aiosmtplib not installed"}
        except Exception as e:
            self.logger.error("Email send failed", error=str(e))
            return {"channel": "email", "sent": False, "error": str(e), "recipients": recipients}

    async def _get_smtp_settings(self) -> Dict:
        """Get SMTP settings from environment or database."""
        import os

        # Try environment variables first
        settings = {
            "host": os.getenv("SMTP_HOST"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "user": os.getenv("SMTP_USER"),
            "password": os.getenv("SMTP_PASS") or os.getenv("SMTP_PASSWORD"),
            "from_email": os.getenv("SMTP_FROM") or os.getenv("SMTP_FROM_EMAIL"),
            "from_name": os.getenv("SMTP_FROM_NAME", "AIDocumentIndexer"),
        }

        # If not in environment, try database settings
        if not settings["host"]:
            try:
                from backend.services.settings import get_settings_service
                svc = get_settings_service()
                settings["host"] = await svc.get_setting("notifications.smtp_host")
                settings["port"] = await svc.get_setting("notifications.smtp_port") or 587
                settings["user"] = await svc.get_setting("notifications.smtp_user")
                settings["password"] = await svc.get_setting("notifications.smtp_password")
                settings["from_email"] = await svc.get_setting("notifications.smtp_from_email")
                settings["from_name"] = await svc.get_setting("notifications.smtp_from_name")
            except Exception:
                pass

        return settings

    async def _notify_slack(self, message: str, recipients: List, config: Dict, context: "ExecutionContext") -> Dict:
        """Send Slack notification."""
        import aiohttp

        webhook_url = config.get("webhook_url") or config.get("slack_webhook")
        if not webhook_url:
            return {"channel": "slack", "sent": False, "error": "No Slack webhook URL configured"}

        try:
            self._validate_outbound_url(webhook_url)
        except ValueError as e:
            return {"channel": "slack", "sent": False, "error": str(e)}

        payload = {
            "text": message,
            "username": config.get("username", "Workflow Bot"),
        }

        if recipients:
            mentions = " ".join([f"<@{r}>" for r in recipients])
            payload["text"] = f"{mentions}\n{message}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return {"channel": "slack", "message": message, "sent": response.status == 200}
        except Exception as e:
            return {"channel": "slack", "sent": False, "error": str(e)}

    async def _notify_webhook(self, message: str, recipients: List, config: Dict, context: "ExecutionContext") -> Dict:
        """Send to a custom webhook."""
        import aiohttp

        webhook_url = config.get("webhook_url") or config.get("url")
        if not webhook_url:
            return {"channel": "webhook", "sent": False, "error": "No webhook URL configured"}

        try:
            self._validate_outbound_url(webhook_url)
        except ValueError as e:
            return {"channel": "webhook", "sent": False, "error": str(e)}

        payload = {
            "message": message,
            "recipients": recipients,
            "workflow_id": str(context.workflow_id),
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return {"channel": "webhook", "message": message, "sent": response.status < 400}
        except Exception as e:
            return {"channel": "webhook", "sent": False, "error": str(e)}

    async def _notify_teams(self, message: str, recipients: List, config: Dict, context: "ExecutionContext") -> Dict:
        """Send Microsoft Teams notification."""
        import aiohttp

        webhook_url = config.get("webhook_url") or config.get("teams_webhook")
        if not webhook_url:
            return {"channel": "teams", "sent": False, "error": "No Teams webhook URL configured"}

        try:
            self._validate_outbound_url(webhook_url)
        except ValueError as e:
            return {"channel": "teams", "sent": False, "error": str(e)}

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": "Workflow Notification",
            "sections": [{"text": message}],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return {"channel": "teams", "message": message, "sent": response.status < 400}
        except Exception as e:
            return {"channel": "teams", "sent": False, "error": str(e)}

    async def _notify_discord(self, message: str, recipients: List, config: Dict, context: "ExecutionContext") -> Dict:
        """Send Discord notification."""
        import aiohttp

        webhook_url = config.get("webhook_url") or config.get("discord_webhook")
        if not webhook_url:
            return {"channel": "discord", "sent": False, "error": "No Discord webhook URL configured"}

        try:
            self._validate_outbound_url(webhook_url)
        except ValueError as e:
            return {"channel": "discord", "sent": False, "error": str(e)}

        payload = {"content": message, "username": config.get("username", "Workflow Bot")}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return {"channel": "discord", "message": message, "sent": response.status < 400}
        except Exception as e:
            return {"channel": "discord", "sent": False, "error": str(e)}

    async def _execute_agent(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Agent node - executes an AI agent.

        Enhanced Config (from NodeConfigPanel):
            agent_type: default, researcher, writer, coder, analyst, custom
            agent_id: ID for custom agent
            prompt: User prompt/task
            system_prompt: System instructions
            model: LLM model to use (default, gpt-4o, gpt-4o-mini, claude-3-5-sonnet, etc.)
            temperature: Model temperature (0-2)
            max_tokens: Maximum response tokens
            use_rag: Enable document search
            doc_filter: Filter for RAG documents
            use_web: Enable web search
            use_code: Enable code execution
            wait_for_result: Block until completion
            timeout: Execution timeout
            output_format: text, json, markdown
        """
        config = node.config or {}
        agent_type = config.get("agent_type", "default")
        agent_id = config.get("agent_id") if agent_type == "custom" else None

        # Build agent input from config
        prompt = context.resolve_template(config.get("prompt", ""))
        system_prompt = context.resolve_template(config.get("system_prompt", ""))

        agent_input = {
            "prompt": prompt,
            "system_prompt": system_prompt,
        }

        # Model settings
        model = config.get("model", "default")
        if model == "default":
            from backend.services.llm import llm_config
            model = llm_config.default_chat_model

        model_settings = {
            "model": model,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 4096),
        }

        # Tool settings
        tools_config = {
            "use_rag": config.get("use_rag", True),
            "doc_filter": config.get("doc_filter", ""),
            "use_web": config.get("use_web", False),
            "use_code": config.get("use_code", False),
        }

        # Execution settings
        wait_for_result = config.get("wait_for_result", True)
        timeout_seconds = config.get("timeout", 120)
        output_format = config.get("output_format", "text")

        # Integrate with agent service
        try:
            if wait_for_result:
                result = await asyncio.wait_for(
                    self._run_agent(agent_id, agent_input, {
                        **config,
                        **model_settings,
                        **tools_config,
                        "agent_type": agent_type,
                        "output_format": output_format,
                    }, context),
                    timeout=timeout_seconds
                )
            else:
                # Fire and forget (return immediately)
                asyncio.create_task(
                    self._run_agent(agent_id, agent_input, {
                        **config,
                        **model_settings,
                        **tools_config,
                        "agent_type": agent_type,
                    }, context)
                )
                result = {"status": "started", "async": True}

            self.logger.info(
                "Agent execution completed",
                agent_type=agent_type,
                agent_id=agent_id,
                success=result.get("status") == "success",
            )

            # Format output if needed
            output = result
            if output_format == "json" and isinstance(result.get("response"), str):
                try:
                    output["response"] = json.loads(result["response"])
                except json.JSONDecodeError:
                    pass  # Keep as string

            return {
                "agent_type": agent_type,
                "agent_id": agent_id,
                "input": agent_input,
                "model": model,
                "output": output,
            }
        except asyncio.TimeoutError:
            self.logger.error("Agent execution timeout", agent_type=agent_type, timeout=timeout_seconds)
            return {
                "agent_type": agent_type,
                "agent_id": agent_id,
                "input": agent_input,
                "output": {"status": "error", "error": f"Agent timed out after {timeout_seconds}s"},
            }
        except Exception as e:
            self.logger.error("Agent execution failed", agent_type=agent_type, error=str(e))
            return {
                "agent_type": agent_type,
                "agent_id": agent_id,
                "input": agent_input,
                "output": {"status": "error", "error": str(e)},
            }

    async def _execute_voice_agent(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Voice Agent node - executes an AI agent with text-to-speech output.

        Config:
            agent_id: ID of the agent to use (optional)
            prompt: User prompt/task
            system_prompt: System instructions
            model: LLM model to use
            temperature: Model temperature (0-2)
            max_tokens: Maximum response tokens
            tts_provider: TTS provider (openai, elevenlabs, cartesia, edge)
            voice_id: Voice ID for TTS
            speed: Speech speed (0.5-2.0)
            output_format: audio, text_and_audio
            use_rag: Enable document search
            use_streaming: Enable streaming TTS (for real-time)

            # Knowledge Sources (comprehensive knowledge integration)
            knowledge_bases: Array of knowledge base/collection IDs to use
            folder_sources: Array of folder IDs to search
            url_sources: Array of URLs to scrape and use as context
            file_sources: Array of file paths/configs to process
            text_sources: Array of raw text blocks as knowledge
            database_sources: Array of database query configs
            api_sources: Array of API endpoint configs
            use_knowledge_graph: Enable KG-enhanced retrieval
        """
        config = node.config or {}

        # Check if we have knowledge sources - if so, use chat agent logic first
        has_knowledge_sources = (
            config.get("knowledge_bases") or
            config.get("folder_sources") or
            config.get("url_sources") or
            config.get("file_sources") or
            config.get("text_sources") or
            config.get("database_sources") or
            config.get("api_sources") or
            config.get("use_rag")
        )

        agent_result = None
        text_response = ""
        knowledge_metadata = {}

        if has_knowledge_sources:
            # Create a temporary chat agent node config for knowledge processing
            chat_config = {
                **config,
                "use_memory": False,  # Voice agent doesn't need memory
                "enable_citations": False,  # No citations in voice
            }
            temp_node = WorkflowNode(
                id=node.id,
                config=chat_config,
                node_type="chat_agent",
            )

            # Execute as chat agent to get knowledge-grounded response
            chat_result = await self._execute_chat_agent(temp_node, context)

            if chat_result.get("status") == "success":
                text_response = chat_result.get("response", "")
                knowledge_metadata = chat_result.get("knowledge_metadata", {})
            else:
                # Fall back to regular agent
                agent_result = await self._execute_agent(node, context)
                if agent_result.get("output", {}).get("status") == "error":
                    return agent_result
                text_response = agent_result.get("output", {}).get("response", "")
                if not text_response:
                    text_response = agent_result.get("output", {}).get("answer", "")
        else:
            # No knowledge sources - use regular agent
            agent_result = await self._execute_agent(node, context)

            if agent_result.get("output", {}).get("status") == "error":
                return agent_result

            # Extract text response
            text_response = agent_result.get("output", {}).get("response", "")
            if not text_response:
                text_response = agent_result.get("output", {}).get("answer", "")

        # TTS configuration
        tts_provider = config.get("tts_provider", "openai")
        voice_id = config.get("voice_id", "alloy")
        speed = config.get("speed", 1.0)
        output_format = config.get("output_format", "text_and_audio")

        # Generate audio
        audio_data = None
        audio_url = None
        try:
            from backend.services.audio.tts_service import TTSService, TTSProvider

            tts = TTSService()
            provider_enum = TTSProvider(tts_provider) if tts_provider else TTSProvider.OPENAI

            audio_bytes = await tts.synthesize_text(
                text=text_response,
                voice_id=voice_id,
                provider=provider_enum,
                speed=speed,
            )

            if audio_bytes:
                # Save audio to temp file and return URL
                import tempfile
                import os
                from pathlib import Path

                audio_dir = Path(settings.UPLOAD_DIR) / "audio" / "workflow"
                audio_dir.mkdir(parents=True, exist_ok=True)

                audio_filename = f"voice_agent_{node.id}_{context.execution_id}.mp3"
                audio_path = audio_dir / audio_filename

                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)

                audio_url = f"/api/v1/audio/workflow/{audio_filename}"
                audio_data = {
                    "path": str(audio_path),
                    "url": audio_url,
                    "size_bytes": len(audio_bytes),
                    "provider": tts_provider,
                    "voice_id": voice_id,
                }

                self.logger.info(
                    "Voice agent audio generated",
                    node_id=str(node.id),
                    audio_size=len(audio_bytes),
                    provider=tts_provider,
                )

        except Exception as e:
            self.logger.error("Voice agent TTS failed", error=str(e))
            # Continue without audio - return text at minimum

        result = {
            "status": "success",
            "node_type": "voice_agent",
            "audio": audio_data,
            "audio_url": audio_url,
            "text_response": text_response,
            "tts_provider": tts_provider,
            "voice_id": voice_id,
        }

        # Include agent result if available
        if agent_result:
            result["agent_output"] = agent_result.get("output", {})

        # Include knowledge metadata if available
        if knowledge_metadata:
            result["knowledge_metadata"] = knowledge_metadata

        return result

    async def _execute_chat_agent(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Chat Agent node - executes a conversational AI agent with comprehensive knowledge access.

        Config:
            agent_id: ID of the agent to use (optional)
            prompt: User prompt/task
            system_prompt: System instructions for chat personality
            model: LLM model to use
            temperature: Model temperature (0-2)
            max_tokens: Maximum response tokens

            # Knowledge Sources (NEW - comprehensive knowledge integration)
            knowledge_bases: Array of knowledge base/collection IDs to use
            folder_sources: Array of folder IDs to search
            url_sources: Array of URLs to scrape and use as context
            file_sources: Array of file paths/configs to process
            text_sources: Array of raw text blocks as knowledge
            database_sources: Array of database query configs
            api_sources: Array of API endpoint configs

            # Chat Settings
            conversation_id: ID to maintain conversation history
            use_memory: Enable conversation memory
            memory_window: Number of turns to remember
            enable_citations: Include source citations in response
            response_style: formal, casual, technical, friendly
            use_knowledge_graph: Enable KG-enhanced retrieval
            max_context_length: Maximum context size (words)
        """
        config = node.config or {}

        # Extract chat-specific config
        knowledge_bases = config.get("knowledge_bases", [])
        folder_sources = config.get("folder_sources", [])
        url_sources = config.get("url_sources", [])
        file_sources = config.get("file_sources", [])
        text_sources = config.get("text_sources", [])
        database_sources = config.get("database_sources", [])
        api_sources = config.get("api_sources", [])

        conversation_id = config.get("conversation_id") or str(context.execution_id)
        use_memory = config.get("use_memory", True)
        memory_window = config.get("memory_window", 10)
        enable_citations = config.get("enable_citations", True)
        response_style = config.get("response_style", "friendly")
        max_context_length = config.get("max_context_length", 8000)

        # Build prompt with style instruction
        style_instructions = {
            "formal": "Respond in a formal, professional tone.",
            "casual": "Respond in a casual, conversational tone.",
            "technical": "Respond with technical precision and detail.",
            "friendly": "Respond in a warm, friendly, and helpful manner.",
        }

        prompt = context.resolve_template(config.get("prompt", ""))
        system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")

        # Add style instruction to system prompt
        if response_style in style_instructions:
            system_prompt = f"{system_prompt}\n\n{style_instructions[response_style]}"

        # Get conversation history if memory is enabled
        conversation_history = []
        if use_memory:
            try:
                from backend.services.conversation_memory import ConversationMemoryService
                memory_service = ConversationMemoryService()
                history = await memory_service.get_history(
                    conversation_id=conversation_id,
                    limit=memory_window,
                )
                conversation_history = history or []
            except Exception as e:
                self.logger.warning("Failed to load conversation history", error=str(e))

        # Process additional knowledge sources using WorkflowKnowledgeService
        additional_context = ""
        additional_sources = []
        knowledge_metadata = {}

        has_additional_sources = (
            url_sources or file_sources or text_sources or
            database_sources or api_sources
        )

        if has_additional_sources:
            try:
                from backend.services.workflow_knowledge_service import (
                    WorkflowKnowledgeService,
                    KnowledgeSourceConfig,
                    KnowledgeSourceType,
                )

                knowledge_service = WorkflowKnowledgeService(session=context.session)

                # Build list of knowledge source configs
                source_configs = []

                # URL sources
                for url in url_sources:
                    if isinstance(url, str):
                        source_configs.append(KnowledgeSourceConfig(
                            source_type=KnowledgeSourceType.URL,
                            value=url,
                            options={"max_length": 5000},
                        ))
                    elif isinstance(url, dict):
                        source_configs.append(KnowledgeSourceConfig(
                            source_type=KnowledgeSourceType.URL,
                            value=url.get("url"),
                            options=url.get("options", {}),
                        ))

                # File sources
                for file_config in file_sources:
                    if isinstance(file_config, str):
                        source_configs.append(KnowledgeSourceConfig(
                            source_type=KnowledgeSourceType.FILE,
                            value=file_config,
                        ))
                    elif isinstance(file_config, dict):
                        source_configs.append(KnowledgeSourceConfig(
                            source_type=KnowledgeSourceType.FILE,
                            value=file_config.get("path") or file_config.get("file_path"),
                            options=file_config.get("options", {}),
                        ))

                # Text sources
                for text_config in text_sources:
                    if isinstance(text_config, str):
                        source_configs.append(KnowledgeSourceConfig(
                            source_type=KnowledgeSourceType.TEXT,
                            value=text_config,
                        ))
                    elif isinstance(text_config, dict):
                        source_configs.append(KnowledgeSourceConfig(
                            source_type=KnowledgeSourceType.TEXT,
                            value=text_config.get("content") or text_config.get("text"),
                            options={"title": text_config.get("title", "Inline Text")},
                        ))

                # Database sources
                for db_config in database_sources:
                    source_configs.append(KnowledgeSourceConfig(
                        source_type=KnowledgeSourceType.DATABASE,
                        value=db_config,
                    ))

                # API sources
                for api_config in api_sources:
                    source_configs.append(KnowledgeSourceConfig(
                        source_type=KnowledgeSourceType.API,
                        value=api_config,
                    ))

                # Process all sources
                if source_configs:
                    org_id = str(context.organization_id) if context.organization_id else None
                    additional_context, additional_sources, knowledge_metadata = await knowledge_service.process_knowledge_sources(
                        sources=source_configs,
                        organization_id=org_id,
                        max_context_length=max_context_length // 2,  # Reserve half for RAG
                    )

                    self.logger.info(
                        "Processed additional knowledge sources",
                        source_count=len(source_configs),
                        context_words=knowledge_metadata.get("total_words", 0),
                    )

                await knowledge_service.close()

            except Exception as e:
                self.logger.warning(
                    "Failed to process additional knowledge sources",
                    error=str(e),
                )

        # Determine if we should use RAG
        use_rag = bool(knowledge_bases) or bool(folder_sources) or config.get("use_rag", False)

        if use_rag or additional_context:
            # Use RAG service for knowledge-grounded response
            try:
                from backend.services.rag import RAGService
                rag = RAGService()

                # Build query with additional context if available
                enhanced_query = prompt
                if additional_context:
                    enhanced_query = f"""Additional Context (from URLs, files, text):
{additional_context}

---

User Question: {prompt}"""

                # Combine collection and folder filters
                collection_filter = knowledge_bases if knowledge_bases else None
                folder_filter = folder_sources[0] if folder_sources else None  # RAG supports single folder

                rag_result = await rag.query(
                    query=enhanced_query,
                    organization_id=str(context.organization_id) if context.organization_id else None,
                    collection_filter=collection_filter,
                    folder_filter=folder_filter,
                    enable_knowledge_graph=config.get("use_knowledge_graph", True),
                    conversation_history=conversation_history,
                    system_prompt=system_prompt,
                )

                response_text = rag_result.get("answer", "")
                sources = rag_result.get("sources", []) if enable_citations else []

                # Add additional sources to citations
                if enable_citations and additional_sources:
                    sources.extend(additional_sources)

                # Save to conversation memory
                if use_memory:
                    try:
                        from backend.services.conversation_memory import ConversationMemoryService
                        memory_service = ConversationMemoryService()
                        await memory_service.add_turn(
                            conversation_id=conversation_id,
                            role="user",
                            content=prompt,
                        )
                        await memory_service.add_turn(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=response_text,
                        )
                    except Exception as e:
                        self.logger.warning("Failed to save conversation turn", error=str(e))

                return {
                    "status": "success",
                    "node_type": "chat_agent",
                    "response": response_text,
                    "sources": sources,
                    "conversation_id": conversation_id,
                    "knowledge_bases_used": knowledge_bases,
                    "folder_sources_used": folder_sources,
                    "additional_sources_used": len(additional_sources),
                    "knowledge_metadata": knowledge_metadata,
                    "model": config.get("model", "default"),
                    "response_style": response_style,
                }

            except Exception as e:
                self.logger.error("Chat agent RAG failed", error=str(e))
                # Fall through to regular agent execution

        # Fall back to regular agent execution without RAG
        agent_result = await self._execute_agent(node, context)

        # Extract response
        output = agent_result.get("output", {})
        response_text = output.get("response", "") or output.get("answer", "")

        # Save to conversation memory
        if use_memory and response_text:
            try:
                from backend.services.conversation_memory import ConversationMemoryService
                memory_service = ConversationMemoryService()
                await memory_service.add_turn(
                    conversation_id=conversation_id,
                    role="user",
                    content=prompt,
                )
                await memory_service.add_turn(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response_text,
                )
            except Exception as e:
                self.logger.warning("Failed to save conversation turn", error=str(e))

        return {
            **agent_result,
            "node_type": "chat_agent",
            "conversation_id": conversation_id,
            "response_style": response_style,
        }

    async def _run_agent(
        self,
        agent_id: Optional[str],
        agent_input: Dict,
        config: Dict,
        context: "ExecutionContext",
    ) -> Dict:
        """Execute an agent and return the result."""
        # If no agent_id, use inline LLM call
        if not agent_id:
            return await self._run_inline_llm(agent_input, config, context)

        # Try to load agent from database
        try:
            from backend.db.models import AgentDefinition
            result = await context.session.execute(
                select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
            )
            agent_def = result.scalar_one_or_none()

            if not agent_def:
                return {"status": "error", "error": f"Agent not found: {agent_id}"}

            # Execute based on agent type
            agent_type = agent_def.agent_type or "general"

            if agent_type == "rag":
                return await self._run_rag_agent(agent_def, agent_input, context)
            elif agent_type == "chat":
                return await self._run_chat_agent(agent_def, agent_input, context)
            elif agent_type == "tool":
                return await self._run_tool_agent(agent_def, agent_input, context)
            else:
                return await self._run_general_agent(agent_def, agent_input, context)

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_inline_llm(self, agent_input: Dict, config: Dict, context: "ExecutionContext") -> Dict:
        """Run an inline LLM call without a predefined agent.

        Phase 59: Enhanced to support advanced RAG features when use_rag=True.
        """
        try:
            from backend.services.llm import EnhancedLLMFactory
            from langchain_core.messages import HumanMessage, SystemMessage
            from backend.services.rag_module.prompts import enhance_agent_system_prompt

            prompt = agent_input.get("prompt", "")
            system_prompt = agent_input.get("system_prompt", "")
            model = config.get("model", "gpt-4")
            temperature = config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens", 2000)

            # Phase 59: Use RAG service when use_rag is enabled
            use_rag = config.get("use_rag", False)
            rag_context = ""
            sources = []

            if use_rag:
                try:
                    from backend.services.rag import RAGService
                    from backend.core.config import settings as core_settings

                    rag = RAGService()

                    # Build RAG query with advanced features
                    rag_kwargs = {
                        "query": prompt,
                        "organization_id": str(context.organization_id) if context.organization_id else None,
                    }

                    # Apply document filter if specified
                    doc_filter = config.get("doc_filter", "")
                    if doc_filter:
                        rag_kwargs["folder_filter"] = doc_filter

                    # Phase 59: Enable advanced RAG features from config
                    # Self-RAG for hallucination detection
                    if config.get("use_self_rag", core_settings.ENABLE_SELF_RAG):
                        rag_kwargs["use_self_rag"] = True

                    # Hybrid retrieval (LightRAG + RAPTOR)
                    if config.get("use_hybrid", True):
                        rag_kwargs["use_hybrid_retrieval"] = True

                    # Knowledge graph
                    if config.get("use_knowledge_graph", True):
                        rag_kwargs["enable_knowledge_graph"] = True

                    # Query expansion
                    if config.get("use_query_expansion", core_settings.ENABLE_QUERY_EXPANSION):
                        rag_kwargs["enable_query_expansion"] = True

                    # Execute RAG query with full pipeline
                    rag_result = await rag.query(**rag_kwargs)

                    # Extract context and sources
                    rag_context = rag_result.get("context", "") or rag_result.get("answer", "")
                    sources = rag_result.get("sources", [])

                    # If RAG returned a direct answer, we can use it
                    if rag_result.get("answer"):
                        return {
                            "status": "success",
                            "response": rag_result.get("answer"),
                            "sources": sources,
                            "kg_entities": rag_result.get("kg_entities", []),
                            "model": model,
                            "rag_used": True,
                        }

                except Exception as rag_error:
                    self.logger.warning(f"RAG retrieval failed, falling back to LLM only: {rag_error}")
                    # Continue with LLM-only response

            # Get LLM using EnhancedLLMFactory
            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="workflow",
                user_id=None,
                track_usage=True,
            )

            # PHASE 15: Apply model-specific enhancements for small models
            # Enhance the prompt with model-specific base instructions
            enhanced_prompt = enhance_agent_system_prompt(prompt, model)

            # Build messages with RAG context if available
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            if rag_context:
                # Include RAG context in the prompt
                context_prompt = f"""Based on the following context from relevant documents:

{rag_context}

User question: {enhanced_prompt}

Please provide a comprehensive answer based on the context above."""
                messages.append(HumanMessage(content=context_prompt))
            else:
                messages.append(HumanMessage(content=enhanced_prompt))

            # Invoke LLM with configured parameters
            response = await llm.ainvoke(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result = {
                "status": "success",
                "response": response.content if hasattr(response, 'content') else str(response),
                "model": model,
            }

            # Include sources if RAG was used
            if sources:
                result["sources"] = sources
                result["rag_used"] = True

            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_rag_agent(self, agent_def, agent_input: Dict, context: "ExecutionContext") -> Dict:
        """Run a RAG-based agent with full agent definition support.

        Phase 59: Enhanced to support advanced RAG features (Self-RAG, LightRAG, RAPTOR).
        """
        try:
            from backend.services.rag import RAGService
            from backend.core.config import settings as core_settings

            rag = RAGService()

            query = agent_input.get("prompt") or agent_input.get("query", "")

            # Extract RAG configuration from agent definition
            agent_config = agent_def.config if hasattr(agent_def, 'config') and agent_def.config else {}

            # Build RAG query kwargs from agent definition
            rag_kwargs = {
                "query": query,
                "organization_id": str(context.organization_id) if context.organization_id else None,
            }

            # Apply agent-specific RAG settings
            if agent_config.get("top_k"):
                rag_kwargs["top_k"] = agent_config.get("top_k")

            if agent_config.get("folder_filter"):
                rag_kwargs["folder_filter"] = agent_config.get("folder_filter")

            # KG settings from agent config
            if "enable_knowledge_graph" in agent_config:
                rag_kwargs["enable_knowledge_graph"] = agent_config.get("enable_knowledge_graph", True)

            if agent_config.get("kg_max_hops"):
                rag_kwargs["knowledge_graph_max_hops"] = agent_config.get("kg_max_hops")

            # Query enhancement settings
            if "enable_query_expansion" in agent_config:
                rag_kwargs["enable_query_expansion"] = agent_config.get("enable_query_expansion", False)

            # Phase 59: Add advanced RAG features
            # Self-RAG for hallucination detection (default from global settings)
            use_self_rag = agent_config.get(
                "use_self_rag",
                core_settings.ENABLE_SELF_RAG
            )
            if use_self_rag:
                rag_kwargs["use_self_rag"] = True

            # Hybrid retrieval (LightRAG + RAPTOR fusion)
            use_hybrid = agent_config.get(
                "use_hybrid_retrieval",
                core_settings.ENABLE_LIGHTRAG or
                core_settings.ENABLE_RAPTOR
            )
            if use_hybrid:
                rag_kwargs["use_hybrid_retrieval"] = True

            # Tiered reranking
            use_reranking = agent_config.get(
                "use_tiered_reranking",
                core_settings.ENABLE_TIERED_RERANKING
            )
            if use_reranking:
                rag_kwargs["use_tiered_reranking"] = True

            # Context compression for long responses
            use_compression = agent_config.get(
                "use_context_compression",
                core_settings.ENABLE_CONTEXT_COMPRESSION
            )
            if use_compression:
                rag_kwargs["use_context_compression"] = True

            result = await rag.query(**rag_kwargs)

            return {
                "status": "success",
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "kg_entities": result.get("kg_entities", []),
                "agent_type": "rag",
                "agent_name": agent_def.name if hasattr(agent_def, 'name') else None,
                # Phase 59: Include advanced RAG metadata
                "self_rag_verified": result.get("self_rag_verified", False),
                "retrieval_method": result.get("retrieval_method", "standard"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_chat_agent(self, agent_def, agent_input: Dict, context: "ExecutionContext") -> Dict:
        """Run a chat-based agent."""
        try:
            from backend.services.llm import EnhancedLLMFactory
            from langchain_core.messages import HumanMessage, SystemMessage
            from backend.services.rag_module.prompts import enhance_agent_system_prompt

            system_prompt = agent_def.system_prompt or "You are a helpful assistant."
            user_prompt = agent_input.get("prompt", "")
            model = agent_def.model or "gpt-4"

            # Get LLM using EnhancedLLMFactory
            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="workflow",
                user_id=None,
                track_usage=True,
            )

            # PHASE 15: Apply model-specific enhancements for small models
            # Enhance the system prompt with model-specific base instructions
            enhanced_system_prompt = enhance_agent_system_prompt(system_prompt, model)

            # Build messages
            messages = [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=user_prompt),
            ]

            # Invoke LLM
            response = await llm.ainvoke(messages)

            return {
                "status": "success",
                "response": response.content if hasattr(response, 'content') else str(response),
                "agent_type": "chat",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_tool_agent(self, agent_def, agent_input: Dict, context: "ExecutionContext") -> Dict:
        """
        Run a tool-using agent that can execute external tools.

        Supported tool types:
        - http: Make HTTP API calls (GET, POST, PUT, DELETE)
        - command: Execute whitelisted shell commands
        - python: Execute Python functions from allowed modules

        Agent config (from agent_def.config):
            - tools: List of tool definitions
            - tool_timeout: Max execution time per tool (default 30s)
            - allow_parallel: Execute independent tools in parallel

        Tool definition format:
            {
                "name": "tool_name",
                "type": "http|command|python",
                "config": { ... tool-specific config ... }
            }
        """
        import aiohttp
        import asyncio
        import shlex

        tools = agent_def.config.get("tools", []) if agent_def.config else []
        tool_timeout = agent_def.config.get("tool_timeout", 30) if agent_def.config else 30
        results = []

        # Allowed CLI commands (security whitelist)
        ALLOWED_COMMANDS = {
            "curl", "wget", "python", "python3", "node", "npx",
            "jq", "grep", "awk", "sed", "cat", "head", "tail",
            "ls", "find", "wc", "sort", "uniq", "date", "echo",
        }

        for tool in tools:
            tool_name = tool.get("name", "unnamed")
            tool_type = tool.get("type", "http")
            tool_config = tool.get("config", {})

            try:
                if tool_type == "http":
                    result = await self._execute_http_tool(tool_config, tool_timeout, context)
                elif tool_type == "command":
                    result = await self._execute_command_tool(
                        tool_config, tool_timeout, ALLOWED_COMMANDS
                    )
                elif tool_type == "python":
                    result = await self._execute_python_tool(tool_config, tool_timeout)
                else:
                    result = {"status": "error", "error": f"Unknown tool type: {tool_type}"}

                result["tool_name"] = tool_name
                results.append(result)

            except asyncio.TimeoutError:
                results.append({
                    "tool_name": tool_name,
                    "status": "error",
                    "error": f"Tool execution timed out after {tool_timeout}s",
                })
            except Exception as e:
                results.append({
                    "tool_name": tool_name,
                    "status": "error",
                    "error": str(e),
                })

        # Aggregate results
        success_count = sum(1 for r in results if r.get("status") == "success")
        return {
            "status": "success" if success_count == len(results) else "partial",
            "response": f"Executed {success_count}/{len(results)} tools successfully",
            "agent_type": "tool",
            "tool_results": results,
        }

    async def _execute_http_tool(
        self, config: Dict, timeout: int, context: "ExecutionContext"
    ) -> Dict:
        """
        Execute HTTP API tool.

        Config:
            - url: Target URL (required)
            - method: HTTP method (default: GET)
            - headers: Request headers
            - body: Request body (for POST/PUT)
            - auth_type: "bearer", "basic", or None
            - auth_token: Token or "user:pass" for basic
        """
        import aiohttp

        url = config.get("url")
        if not url:
            return {"status": "error", "error": "No URL specified"}

        # SSRF prevention: validate URL before making request
        try:
            from backend.services.web_crawler import _validate_crawl_url
            _validate_crawl_url(url)
        except ValueError as ssrf_err:
            return {"status": "error", "error": f"URL blocked: {ssrf_err}"}

        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        body = config.get("body")

        # Handle authentication
        auth_type = config.get("auth_type")
        auth_token = config.get("auth_token")
        if auth_type == "bearer" and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        elif auth_type == "basic" and auth_token:
            import base64
            encoded = base64.b64encode(auth_token.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        try:
            async with aiohttp.ClientSession() as session:
                kwargs = {
                    "headers": headers,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                }
                if body and method in ("POST", "PUT", "PATCH"):
                    if isinstance(body, dict):
                        kwargs["json"] = body
                    else:
                        kwargs["data"] = body

                async with session.request(method, url, **kwargs) as response:
                    try:
                        response_data = await response.json()
                    except Exception:
                        response_data = await response.text()

                    return {
                        "status": "success" if response.status < 400 else "error",
                        "http_status": response.status,
                        "output": response_data,
                    }

        except aiohttp.ClientError as e:
            return {"status": "error", "error": f"HTTP error: {str(e)}"}

    async def _execute_command_tool(
        self, config: Dict, timeout: int, allowed_commands: set
    ) -> Dict:
        """
        Execute whitelisted shell command.

        Config:
            - command: Shell command to execute (required)
            - working_dir: Working directory (optional)
        """
        import asyncio
        import shlex

        command = config.get("command")
        if not command:
            return {"status": "error", "error": "No command specified"}

        # Security: Extract the base command and check whitelist
        try:
            parts = shlex.split(command)
            base_cmd = parts[0].split("/")[-1]  # Handle full paths
        except ValueError:
            return {"status": "error", "error": "Invalid command syntax"}

        if base_cmd not in allowed_commands:
            return {
                "status": "error",
                "error": f"Command not allowed: {base_cmd}. Allowed: {', '.join(sorted(allowed_commands))}",
            }

        working_dir = config.get("working_dir")

        try:
            # Use create_subprocess_exec with parsed args to prevent shell injection.
            # create_subprocess_shell would allow "ls; rm -rf /" to bypass whitelist.
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            return {
                "status": "success" if proc.returncode == 0 else "error",
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }

        except asyncio.TimeoutError:
            proc.kill()
            return {"status": "error", "error": f"Command timed out after {timeout}s"}

    async def _execute_python_tool(self, config: Dict, timeout: int) -> Dict:
        """
        Execute Python function from allowed modules.

        Config:
            - module: Module path (must be in allowed list)
            - function: Function name to call
            - args: Positional arguments (list)
            - kwargs: Keyword arguments (dict)
        """
        import asyncio
        import importlib

        # Allowed modules (security whitelist)
        ALLOWED_MODULES = {
            "json", "datetime", "re", "math", "statistics",
            "urllib.parse", "base64", "hashlib", "uuid",
            # Add specific backend modules as needed
            "backend.services.rag",
            "backend.services.vectorstore",
        }

        module_path = config.get("module")
        function_name = config.get("function")

        if not module_path or not function_name:
            return {"status": "error", "error": "Module and function are required"}

        # Check module whitelist
        if module_path not in ALLOWED_MODULES:
            return {
                "status": "error",
                "error": f"Module not allowed: {module_path}",
            }

        try:
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)

            args = config.get("args", [])
            kwargs = config.get("kwargs", {})

            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=timeout,
                )

            return {
                "status": "success",
                "output": result if isinstance(result, (dict, list, str, int, float, bool)) else str(result),
            }

        except ImportError as e:
            return {"status": "error", "error": f"Import error: {str(e)}"}
        except AttributeError as e:
            return {"status": "error", "error": f"Function not found: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_general_agent(self, agent_def, agent_input: Dict, context: "ExecutionContext") -> Dict:
        """Run a general-purpose agent."""
        try:
            from backend.services.llm import EnhancedLLMFactory
            from langchain_core.messages import HumanMessage
            from backend.services.rag_module.prompts import enhance_agent_system_prompt

            prompt = agent_input.get("prompt", "")
            model = agent_def.model or "gpt-4"

            # Get LLM using EnhancedLLMFactory
            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="workflow",
                user_id=None,
                track_usage=True,
            )

            # PHASE 15: Apply model-specific enhancements for small models
            # Enhance the prompt with model-specific base instructions
            enhanced_prompt = enhance_agent_system_prompt(prompt, model)

            # Build messages
            messages = [HumanMessage(content=enhanced_prompt)]

            # Invoke LLM
            response = await llm.ainvoke(messages)

            return {
                "status": "success",
                "response": response.content if hasattr(response, 'content') else str(response),
                "agent_type": "general",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _execute_human_approval(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Human approval node - waits for human input.

        Enhanced Config (from NodeConfigPanel):
            approvers: Comma-separated approver emails/IDs
            approval_type: any (one approver), all (all approvers), majority
            title: Approval request title
            message: Message to show approvers
            allow_comments: Allow approvers to add comments
            timeout_value: Timeout value
            timeout_unit: minutes, hours, days
            on_timeout: reject, approve, escalate, remind
            escalation_to: Email for escalation
            notify_channels: Array of channels (email, slack, teams, in_app)
            send_reminder: Enable reminders
            reminder_interval: Hours between reminders

        This node pauses execution until approved/rejected.
        """
        config = node.config or {}

        # Parse approvers
        approvers_str = config.get("approvers", "")
        approvers = [a.strip() for a in approvers_str.split(",") if a.strip()]

        # Build timeout
        timeout_value = config.get("timeout_value", 24)
        timeout_unit = config.get("timeout_unit", "hours")
        timeout_multipliers = {"minutes": 1/60, "hours": 1, "days": 24}
        timeout_hours = timeout_value * timeout_multipliers.get(timeout_unit, 1)

        # Build message
        title = context.resolve_template(config.get("title", "Approval Required"))
        message = context.resolve_template(config.get("message", "Please review and approve this workflow step."))

        # Approval settings
        approval_type = config.get("approval_type", "any")
        allow_comments = config.get("allow_comments", True)
        on_timeout = config.get("on_timeout", "reject")
        escalation_to = config.get("escalation_to", "")
        notify_channels = config.get("notify_channels", ["email"])
        send_reminder = config.get("send_reminder", True)
        reminder_interval = config.get("reminder_interval", 4)

        # Enhanced config for approval
        enhanced_config = {
            **config,
            "approval_type": approval_type,
            "allow_comments": allow_comments,
            "on_timeout": on_timeout,
            "escalation_to": escalation_to,
            "notify_channels": notify_channels,
            "send_reminder": send_reminder,
            "reminder_interval": reminder_interval,
            "title": title,
        }

        # Create approval request in database
        try:
            approval_result = await self._create_approval_request(
                node=node,
                context=context,
                approvers=approvers,
                timeout_hours=timeout_hours,
                message=message,
                config=enhanced_config,
            )

            self.logger.info(
                "Approval request created",
                approval_id=approval_result.get("approval_id"),
                approvers=approvers,
                approval_type=approval_type,
                status=approval_result.get("status"),
            )

            return approval_result

        except Exception as e:
            self.logger.error("Failed to create approval request", error=str(e))
            # On error, auto-approve to not block workflow
            return {
                "status": "approved",
                "auto_approved": True,
                "message": message,
                "approvers": approvers,
                "error": str(e),
            }

    async def _create_approval_request(
        self,
        node: WorkflowNode,
        context: "ExecutionContext",
        approvers: List[str],
        timeout_hours: int,
        message: str,
        config: Dict,
    ) -> Dict:
        """Create an approval request and optionally wait for it."""
        from backend.db.models import WorkflowNodeExecution

        # Check if there's already an approval for this execution
        if context.execution_id:
            result = await context.session.execute(
                select(WorkflowNodeExecution).where(
                    WorkflowNodeExecution.execution_id == context.execution_id,
                    WorkflowNodeExecution.node_id == node.id,
                )
            )
            existing = result.scalar_one_or_none()

            if existing and existing.output_data:
                approval_data = existing.output_data.get("approval", {})
                if approval_data.get("decision"):
                    # Already has a decision
                    return {
                        "status": approval_data.get("decision"),
                        "auto_approved": False,
                        "decided_by": approval_data.get("decided_by"),
                        "decided_at": approval_data.get("decided_at"),
                        "message": message,
                        "approvers": approvers,
                    }

        # Calculate timeout
        timeout_at = datetime.utcnow() + timedelta(hours=timeout_hours)

        # Store approval request in node execution
        approval_id = str(uuid.uuid4())
        approval_data = {
            "approval_id": approval_id,
            "approvers": approvers,
            "message": message,
            "timeout_at": timeout_at.isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "decision": None,
            "decided_by": None,
            "decided_at": None,
        }

        # Check for auto-approve conditions
        auto_approve = config.get("auto_approve_if_empty", False) and not approvers

        if auto_approve:
            approval_data["decision"] = "approved"
            approval_data["auto_approved"] = True
            approval_data["decided_at"] = datetime.utcnow().isoformat()

        # Send notifications to approvers
        if approvers and not auto_approve:
            await self._notify_approvers(
                approvers=approvers,
                message=message,
                approval_id=approval_id,
                workflow_id=context.workflow_id,
                execution_id=context.execution_id,
                config=config,
            )

        # If wait_for_approval is True and not auto-approved, pause execution
        wait_for_approval = config.get("wait_for_approval", True)

        if wait_for_approval and not auto_approve:
            # Check for timeout
            poll_interval = config.get("poll_interval_seconds", 30)
            max_polls = int((timeout_hours * 3600) / poll_interval)

            for _ in range(min(max_polls, 10)):  # Cap at 10 polls (~5 mins max wait)
                await asyncio.sleep(poll_interval)

                # Check if approval decision made
                if context.execution_id:
                    result = await context.session.execute(
                        select(WorkflowNodeExecution).where(
                            WorkflowNodeExecution.execution_id == context.execution_id,
                            WorkflowNodeExecution.node_id == node.id,
                        )
                    )
                    node_exec = result.scalar_one_or_none()

                    if node_exec and node_exec.output_data:
                        saved_approval = node_exec.output_data.get("approval", {})
                        if saved_approval.get("decision"):
                            return {
                                "status": saved_approval["decision"],
                                "auto_approved": False,
                                "decided_by": saved_approval.get("decided_by"),
                                "decided_at": saved_approval.get("decided_at"),
                                "message": message,
                                "approvers": approvers,
                                "approval_id": approval_id,
                            }

            # Timeout reached - apply timeout action
            timeout_action = config.get("timeout_action", "reject")
            approval_data["decision"] = timeout_action == "approve" and "approved" or "rejected"
            approval_data["timeout_reached"] = True
            approval_data["decided_at"] = datetime.utcnow().isoformat()

        return {
            "status": approval_data.get("decision", "pending"),
            "auto_approved": approval_data.get("auto_approved", False),
            "message": message,
            "approvers": approvers,
            "approval_id": approval_id,
            "timeout_at": timeout_at.isoformat(),
            "approval": approval_data,
        }

    async def _notify_approvers(
        self,
        approvers: List[str],
        message: str,
        approval_id: str,
        workflow_id: uuid.UUID,
        execution_id: Optional[uuid.UUID],
        config: Dict,
    ):
        """Send approval notifications to designated approvers."""
        notification_channel = config.get("notification_channel", "email")

        # Build approval URLs
        base_url = config.get("app_base_url", "")
        approve_url = f"{base_url}/api/v1/workflows/approve/{approval_id}?action=approve"
        reject_url = f"{base_url}/api/v1/workflows/approve/{approval_id}?action=reject"

        notification_message = f"""
{message}

Workflow Execution Approval Required

[Approve]({approve_url})
[Reject]({reject_url})

Approval ID: {approval_id}
"""

        # Send via notification channel
        if notification_channel == "slack":
            webhook_url = config.get("slack_webhook")
            if webhook_url:
                import aiohttp
                payload = {
                    "text": notification_message,
                    "attachments": [{
                        "fallback": "Approval Required",
                        "color": "#ff9900",
                        "actions": [
                            {"type": "button", "text": "Approve", "url": approve_url, "style": "primary"},
                            {"type": "button", "text": "Reject", "url": reject_url, "style": "danger"},
                        ],
                    }],
                }
                async with aiohttp.ClientSession() as session:
                    await session.post(webhook_url, json=payload)

        # Log notification
        self.logger.info(
            "Approval notification sent",
            approvers=approvers,
            approval_id=approval_id,
            channel=notification_channel,
        )

    # =========================================================================
    # Lobster-Style Advanced Node Executors
    # =========================================================================

    async def _execute_parallel(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Parallel node - fork execution into multiple parallel branches.

        Config:
            branches: List of branch configurations
            wait_for_all: Whether to wait for all branches (default: True)
            timeout_seconds: Timeout for parallel execution
        """
        config = node.config or {}
        branches = config.get("branches", [])
        wait_for_all = config.get("wait_for_all", True)
        timeout = config.get("timeout_seconds", 300)

        if not branches:
            return {"status": "completed", "branches": [], "message": "No branches configured"}

        # Create tasks for each branch
        branch_results = {}
        branch_tasks = []

        for i, branch in enumerate(branches):
            branch_id = branch.get("id", f"branch_{i}")
            branch_tasks.append(
                self._execute_branch(branch_id, branch, context.copy())
            )

        # Execute branches in parallel
        try:
            if wait_for_all:
                results = await asyncio.wait_for(
                    asyncio.gather(*branch_tasks, return_exceptions=True),
                    timeout=timeout
                )
                for i, result in enumerate(results):
                    branch_id = branches[i].get("id", f"branch_{i}")
                    if isinstance(result, Exception):
                        branch_results[branch_id] = {"status": "error", "error": str(result)}
                    else:
                        branch_results[branch_id] = result
            else:
                # Return as soon as first branch completes
                done, pending = await asyncio.wait(
                    [asyncio.create_task(t) for t in branch_tasks],
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    result = task.result()
                    branch_results["first_completed"] = result
                    break
                # Cancel pending
                for task in pending:
                    task.cancel()

        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "branches": branch_results,
                "message": f"Parallel execution timed out after {timeout}s",
            }

        return {
            "status": "completed",
            "branches": branch_results,
            "branch_count": len(branches),
        }

    async def _execute_branch(self, branch_id: str, branch_config: Dict, context: ExecutionContext) -> Dict:
        """Execute a single branch in parallel execution."""
        # Branch can have inline actions or reference nodes
        action_type = branch_config.get("action_type", "noop")
        params = branch_config.get("params", {})

        # Simple inline action execution
        result = await self._execute_inline_action(action_type, params, context)

        return {
            "branch_id": branch_id,
            "status": "completed",
            "result": result,
        }

    async def _execute_inline_action(self, action_type: str, params: Dict, context: ExecutionContext) -> Any:
        """Execute an inline action within a branch."""
        if action_type == "http":
            import aiohttp
            async with aiohttp.ClientSession() as session:
                method = params.get("method", "GET").upper()
                url = context.resolve_template(params.get("url", ""))
                headers = params.get("headers", {})
                body = params.get("body")

                async with session.request(method, url, headers=headers, json=body) as resp:
                    return {"status_code": resp.status, "body": await resp.text()}

        elif action_type == "delay":
            delay = params.get("seconds", 1)
            await asyncio.sleep(delay)
            return {"delayed": delay}

        return {"action": action_type, "params": params}

    async def _execute_join(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Join node - wait for all parallel branches to complete.

        Config:
            required_branches: List of branch IDs that must complete
            merge_strategy: how to merge results (all, first, latest)
        """
        config = node.config or {}
        required_branches = config.get("required_branches", [])
        merge_strategy = config.get("merge_strategy", "all")

        # Get results from parallel branches stored in context
        parallel_results = context.variables.get("_parallel_results", {})

        if required_branches:
            # Check if all required branches completed
            completed = all(b in parallel_results for b in required_branches)
            if not completed:
                return {
                    "status": "waiting",
                    "message": "Waiting for required branches",
                    "required": required_branches,
                    "completed": list(parallel_results.keys()),
                }

        # Merge results based on strategy
        if merge_strategy == "all":
            merged = parallel_results
        elif merge_strategy == "first":
            merged = next(iter(parallel_results.values()), {})
        else:
            merged = parallel_results

        return {
            "status": "completed",
            "merged_results": merged,
            "branch_count": len(parallel_results),
        }

    async def _execute_subworkflow(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Subworkflow node - execute another workflow as a step.

        Config:
            workflow_id: ID of workflow to execute
            input_mapping: Map current context to subworkflow input
            output_mapping: Map subworkflow output to current context
            wait_for_completion: Whether to wait for subworkflow
        """
        config = node.config or {}
        subworkflow_id = config.get("workflow_id")
        input_mapping = config.get("input_mapping", {})
        output_mapping = config.get("output_mapping", {})
        wait = config.get("wait_for_completion", True)

        if not subworkflow_id:
            return {"status": "error", "message": "No workflow_id configured"}

        # Map inputs
        subworkflow_input = {}
        for target_key, source_expr in input_mapping.items():
            subworkflow_input[target_key] = context.resolve_template(source_expr)

        # Execute subworkflow
        try:
            # Import engine locally to avoid circular import
            from backend.services.workflow_engine import WorkflowExecutionEngine

            engine = WorkflowExecutionEngine(session=self.session)
            execution = await engine.execute(
                workflow_id=uuid.UUID(subworkflow_id),
                trigger_type="subworkflow",
                input_data=subworkflow_input,
            )

            if wait:
                # Wait for completion and get results
                final_status = execution.status
                final_output = execution.output_data or {}

                # Map outputs back
                mapped_output = {}
                for target_key, source_key in output_mapping.items():
                    mapped_output[target_key] = final_output.get(source_key)

                return {
                    "status": "completed",
                    "subworkflow_id": subworkflow_id,
                    "execution_id": str(execution.id),
                    "subworkflow_status": final_status,
                    "output": mapped_output,
                }
            else:
                return {
                    "status": "started",
                    "subworkflow_id": subworkflow_id,
                    "execution_id": str(execution.id),
                    "message": "Subworkflow started in background",
                }

        except Exception as e:
            self.logger.error("Subworkflow execution failed", error=str(e))
            return {"status": "error", "message": str(e)}

    async def _execute_transform(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Transform node - data transformation/mapping.

        Config:
            transform_type: jmespath, jsonata, python, template
            expression: Transformation expression
            input_path: Path to input data in context
            output_variable: Variable name for output
        """
        config = node.config or {}
        transform_type = config.get("transform_type", "template")
        expression = config.get("expression", "")
        input_path = config.get("input_path", "")
        output_var = config.get("output_variable", "transformed")

        # Get input data
        if input_path:
            input_data = context.get_variable(input_path)
        else:
            input_data = context.variables

        result = None

        try:
            if transform_type == "template":
                result = context.resolve_template(expression)

            elif transform_type == "jmespath":
                import jmespath
                result = jmespath.search(expression, input_data)

            elif transform_type == "python":
                import ast

                # Validate expression AST before eval
                try:
                    tree = ast.parse(expression, mode='eval')
                except SyntaxError:
                    raise ValueError("Invalid expression syntax")

                blocked_attrs = {
                    "__builtins__", "__import__", "__class__", "__bases__",
                    "__subclasses__", "__mro__", "__globals__", "__code__",
                    "__getattribute__", "__dict__", "__module__",
                }
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and node.attr in blocked_attrs:
                        raise ValueError(f"Access to '{node.attr}' is not allowed")
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id in ("eval", "exec", "compile", "open", "__import__",
                                             "getattr", "setattr", "delattr", "globals", "locals"):
                            raise ValueError(f"Function '{node.func.id}' not allowed")

                # Safe eval with limited scope
                safe_globals = {
                    "__builtins__": {},
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "sorted": sorted,
                    "json": json,
                }
                result = eval(expression, safe_globals, {"data": input_data, "ctx": context.variables})

            elif transform_type == "map":
                # Simple field mapping
                mappings = config.get("mappings", {})
                result = {}
                for target, source in mappings.items():
                    result[target] = context.resolve_template(source)

            else:
                result = input_data

            # Store result
            context.variables[output_var] = result

            return {
                "status": "completed",
                "transform_type": transform_type,
                "output_variable": output_var,
                "result": result,
            }

        except Exception as e:
            self.logger.error("Transform failed", error=str(e))
            return {"status": "error", "message": str(e)}

    async def _execute_switch(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Switch node - multi-way conditional branching (like switch/case).

        Config:
            expression: Expression to evaluate
            cases: List of {value: X, target_node_id: Y}
            default_target: Default target if no case matches
        """
        config = node.config or {}
        expression = config.get("expression", "")
        cases = config.get("cases", [])
        default_target = config.get("default_target")

        # Evaluate expression
        value = context.resolve_template(expression)

        # Find matching case
        matched_target = None
        for case in cases:
            case_value = case.get("value")
            if str(value) == str(case_value):
                matched_target = case.get("target_node_id")
                break

        if not matched_target:
            matched_target = default_target

        return {
            "status": "completed",
            "evaluated_value": value,
            "matched_case": matched_target,
            "next_node": matched_target,
        }

    async def _execute_retry(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Retry node - retry a failed action with exponential backoff.

        Config:
            action_config: Configuration for action to retry
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            backoff_multiplier: Multiplier for each retry
            max_delay: Maximum delay between retries
        """
        config = node.config or {}
        action_config = config.get("action_config", {})
        max_retries = config.get("max_retries", 3)
        initial_delay = config.get("initial_delay", 1)
        backoff_multiplier = config.get("backoff_multiplier", 2)
        max_delay = config.get("max_delay", 60)

        delay = initial_delay
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Execute the action
                action_type = action_config.get("action_type", "noop")
                result = await self._execute_inline_action(action_type, action_config, context)

                return {
                    "status": "completed",
                    "attempts": attempt + 1,
                    "result": result,
                }

            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"Retry attempt {attempt + 1} failed",
                    error=last_error,
                    next_delay=delay,
                )

                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_multiplier, max_delay)

        return {
            "status": "failed",
            "attempts": max_retries + 1,
            "last_error": last_error,
        }

    async def _execute_aggregate(self, node: WorkflowNode, context: ExecutionContext) -> Dict:
        """
        Aggregate node - combine results from parallel branches.

        Config:
            source_variable: Variable containing array of results
            aggregation_type: sum, count, avg, min, max, concat, merge
            field_path: Path to field to aggregate (for objects)
            output_variable: Variable to store result
        """
        config = node.config or {}
        source_var = config.get("source_variable", "_parallel_results")
        agg_type = config.get("aggregation_type", "merge")
        field_path = config.get("field_path")
        output_var = config.get("output_variable", "aggregated")

        source_data = context.get_variable(source_var)
        if not source_data:
            return {"status": "error", "message": f"Source variable '{source_var}' not found"}

        # Extract field if specified
        if field_path and isinstance(source_data, dict):
            values = [v.get(field_path) for v in source_data.values() if isinstance(v, dict)]
        elif isinstance(source_data, dict):
            values = list(source_data.values())
        else:
            values = list(source_data) if source_data else []

        # Aggregate
        result = None
        try:
            if agg_type == "sum":
                result = sum(v for v in values if isinstance(v, (int, float)))
            elif agg_type == "count":
                result = len(values)
            elif agg_type == "avg":
                nums = [v for v in values if isinstance(v, (int, float))]
                result = sum(nums) / len(nums) if nums else 0
            elif agg_type == "min":
                nums = [v for v in values if isinstance(v, (int, float))]
                result = min(nums) if nums else None
            elif agg_type == "max":
                nums = [v for v in values if isinstance(v, (int, float))]
                result = max(nums) if nums else None
            elif agg_type == "concat":
                result = [item for v in values for item in (v if isinstance(v, list) else [v])]
            elif agg_type == "merge":
                result = {}
                for v in values:
                    if isinstance(v, dict):
                        result.update(v)
            else:
                result = values

            context.variables[output_var] = result

            return {
                "status": "completed",
                "aggregation_type": agg_type,
                "output_variable": output_var,
                "result": result,
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}


class WorkflowExecutionEngine(BaseService):
    """
    Engine for executing workflows.

    Handles:
    - Graph traversal
    - Node execution
    - State management
    - Error handling
    - Progress tracking
    """

    async def execute(
        self,
        workflow_id: uuid.UUID,
        trigger_type: str = WorkflowTriggerType.MANUAL.value,
        trigger_data: Optional[Dict] = None,
        input_data: Optional[Dict] = None,
        triggered_by_id: Optional[uuid.UUID] = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow_id: ID of workflow to execute
            trigger_type: Type of trigger that started execution
            trigger_data: Trigger-specific data
            input_data: Input data for the workflow
            triggered_by_id: User ID who triggered execution

        Returns:
            WorkflowExecution record with results
        """
        session = await self.get_session()

        # Load workflow with nodes and edges
        workflow_service = WorkflowService(session=session, organization_id=self._organization_id)
        workflow = await workflow_service.get_by_id(workflow_id)

        if not workflow:
            raise NotFoundException("Workflow", str(workflow_id))

        if not workflow.is_active and not workflow.is_draft:
            raise ValidationException("Workflow is not active")

        # Create execution record
        execution = WorkflowExecution(
            id=uuid.uuid4(),
            organization_id=self._organization_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING.value,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            input_data=input_data,
            triggered_by_id=triggered_by_id or self._user_id,
        )
        session.add(execution)
        await session.commit()

        # Create execution context
        context = ExecutionContext(
            execution_id=execution.id,
            workflow_id=workflow_id,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            input_data=input_data,
        )

        try:
            # Update status to running
            execution.status = WorkflowStatus.RUNNING.value
            execution.started_at = datetime.utcnow()
            await session.commit()

            # Execute the workflow graph
            await self._execute_graph(workflow, context, session)

            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED.value
            execution.completed_at = datetime.utcnow()
            execution.duration_ms = int(
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )
            execution.output_data = {
                "variables": context.variables,
                "node_outputs": {str(k): v for k, v in context.node_outputs.items()},
            }

        except Exception as e:
            self.log_error("Workflow execution failed", error=e, workflow_id=str(workflow_id))

            execution.status = WorkflowStatus.FAILED.value
            execution.completed_at = datetime.utcnow()
            execution.error_message = str(e)
            execution.error_node_id = context.current_node_id

        finally:
            await session.commit()
            await session.refresh(execution)

        return execution

    async def execute_stream(
        self,
        workflow_id: uuid.UUID,
        trigger_type: str = WorkflowTriggerType.MANUAL.value,
        trigger_data: Optional[Dict] = None,
        input_data: Optional[Dict] = None,
        triggered_by_id: Optional[uuid.UUID] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Execute a workflow with streaming progress updates.

        Yields:
            Progress updates for each node execution
        """
        session = await self.get_session()

        # Load workflow
        workflow_service = WorkflowService(session=session, organization_id=self._organization_id)
        workflow = await workflow_service.get_by_id(workflow_id)

        if not workflow:
            yield {"type": "error", "error": f"Workflow not found: {workflow_id}"}
            return

        # Create execution record
        execution = WorkflowExecution(
            id=uuid.uuid4(),
            organization_id=self._organization_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING.value,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            input_data=input_data,
            triggered_by_id=triggered_by_id or self._user_id,
            started_at=datetime.utcnow(),
        )
        session.add(execution)
        await session.commit()

        yield {
            "type": "started",
            "execution_id": str(execution.id),
            "workflow_id": str(workflow_id),
        }

        # Create context
        context = ExecutionContext(
            execution_id=execution.id,
            workflow_id=workflow_id,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            input_data=input_data,
        )

        try:
            # Execute with progress
            async for progress in self._execute_graph_stream(workflow, context, session):
                yield progress

            # Mark completed
            execution.status = WorkflowStatus.COMPLETED.value
            execution.completed_at = datetime.utcnow()
            execution.output_data = {"variables": context.variables}

            yield {
                "type": "completed",
                "execution_id": str(execution.id),
                "output": execution.output_data,
            }

        except Exception as e:
            execution.status = WorkflowStatus.FAILED.value
            execution.completed_at = datetime.utcnow()
            execution.error_message = str(e)

            yield {
                "type": "error",
                "execution_id": str(execution.id),
                "error": str(e),
            }

        finally:
            await session.commit()

    async def _execute_graph(
        self,
        workflow: Workflow,
        context: ExecutionContext,
        session: AsyncSession,
    ):
        """Execute workflow graph by traversing nodes."""
        node_executor = NodeExecutor(session)

        # Build node map and adjacency list
        nodes = {node.id: node for node in workflow.nodes}
        edges_by_source: Dict[uuid.UUID, List[WorkflowEdge]] = {}
        for edge in workflow.edges:
            if edge.source_node_id not in edges_by_source:
                edges_by_source[edge.source_node_id] = []
            edges_by_source[edge.source_node_id].append(edge)

        # Find start node
        start_node = next(
            (n for n in workflow.nodes if n.node_type == WorkflowNodeType.START.value),
            None,
        )

        if not start_node:
            raise ValidationException("Workflow has no START node")

        # Execute nodes in order
        current_nodes = [start_node]
        visited = set()

        while current_nodes:
            node = current_nodes.pop(0)

            if node.id in visited:
                continue
            visited.add(node.id)

            context.current_node_id = node.id

            # Record node execution start
            node_execution = WorkflowNodeExecution(
                id=uuid.uuid4(),
                execution_id=context.execution_id,
                node_id=node.id,
                status="running",
                input_data={"context_vars": context.variables},
                started_at=datetime.utcnow(),
            )
            session.add(node_execution)
            await session.flush()

            try:
                # Execute node
                output = await node_executor.execute(node, context)
                context.set_node_output(node.id, output)

                # Update node execution
                node_execution.status = "completed"
                node_execution.output_data = output
                node_execution.completed_at = datetime.utcnow()
                node_execution.duration_ms = int(
                    (node_execution.completed_at - node_execution.started_at).total_seconds() * 1000
                )

            except Exception as e:
                node_execution.status = "failed"
                node_execution.error_message = str(e)
                node_execution.completed_at = datetime.utcnow()
                raise

            # Find next nodes
            outgoing_edges = edges_by_source.get(node.id, [])
            for edge in outgoing_edges:
                # Check condition if present
                if edge.condition:
                    condition_result = context.resolve_template(edge.condition)
                    # For condition nodes, check the result
                    if node.node_type == WorkflowNodeType.CONDITION.value:
                        node_output = context.get_node_output(node.id)
                        if node_output and "result" in node_output:
                            expected = str(node_output["result"]).lower()
                            if condition_result.lower() != expected:
                                continue

                target_node = nodes.get(edge.target_node_id)
                if target_node and target_node.id not in visited:
                    current_nodes.append(target_node)

        await session.commit()

    async def _execute_graph_stream(
        self,
        workflow: Workflow,
        context: ExecutionContext,
        session: AsyncSession,
    ) -> AsyncGenerator[Dict, None]:
        """Execute workflow graph with streaming progress."""
        node_executor = NodeExecutor(session)

        nodes = {node.id: node for node in workflow.nodes}
        edges_by_source: Dict[uuid.UUID, List[WorkflowEdge]] = {}
        for edge in workflow.edges:
            if edge.source_node_id not in edges_by_source:
                edges_by_source[edge.source_node_id] = []
            edges_by_source[edge.source_node_id].append(edge)

        start_node = next(
            (n for n in workflow.nodes if n.node_type == WorkflowNodeType.START.value),
            None,
        )

        if not start_node:
            raise ValidationException("Workflow has no START node")

        current_nodes = [start_node]
        visited = set()
        total_nodes = len(workflow.nodes)
        executed_count = 0

        while current_nodes:
            node = current_nodes.pop(0)

            if node.id in visited:
                continue
            visited.add(node.id)

            context.current_node_id = node.id
            executed_count += 1

            yield {
                "type": "node_started",
                "node_id": str(node.id),
                "node_name": node.name,
                "node_type": node.node_type,
                "progress": executed_count / total_nodes,
            }

            try:
                output = await node_executor.execute(node, context)
                context.set_node_output(node.id, output)

                yield {
                    "type": "node_completed",
                    "node_id": str(node.id),
                    "node_name": node.name,
                    "output": output,
                    "progress": executed_count / total_nodes,
                }

            except Exception as e:
                yield {
                    "type": "node_failed",
                    "node_id": str(node.id),
                    "node_name": node.name,
                    "error": str(e),
                }
                raise

            # Find next nodes
            outgoing_edges = edges_by_source.get(node.id, [])
            for edge in outgoing_edges:
                if edge.condition:
                    condition_result = context.resolve_template(edge.condition)
                    if node.node_type == WorkflowNodeType.CONDITION.value:
                        node_output = context.get_node_output(node.id)
                        if node_output and "result" in node_output:
                            expected = str(node_output["result"]).lower()
                            if condition_result.lower() != expected:
                                continue

                target_node = nodes.get(edge.target_node_id)
                if target_node and target_node.id not in visited:
                    current_nodes.append(target_node)


# =============================================================================
# Service Singletons
# =============================================================================

_workflow_service: Optional[WorkflowService] = None
_execution_engine: Optional[WorkflowExecutionEngine] = None


def get_workflow_service(
    session: Optional[AsyncSession] = None,
    organization_id: Optional[uuid.UUID] = None,
) -> WorkflowService:
    """Get workflow service instance."""
    global _workflow_service
    if _workflow_service is None or session is not None:
        _workflow_service = WorkflowService(session=session, organization_id=organization_id)
    return _workflow_service


def get_execution_engine(
    session: Optional[AsyncSession] = None,
    organization_id: Optional[uuid.UUID] = None,
) -> WorkflowExecutionEngine:
    """Get workflow execution engine instance."""
    global _execution_engine
    if _execution_engine is None or session is not None:
        _execution_engine = WorkflowExecutionEngine(session=session, organization_id=organization_id)
    return _execution_engine
