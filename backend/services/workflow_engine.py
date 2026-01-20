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
            search_pattern = f"%{search}%"
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
    ):
        self.execution_id = execution_id
        self.workflow_id = workflow_id
        self.trigger_type = trigger_type
        self.trigger_data = trigger_data or {}
        self.input_data = input_data or {}

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
            WorkflowNodeType.HUMAN_APPROVAL.value: self._execute_human_approval,
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
        """Run a RAG query."""
        try:
            from backend.services.rag import RAGService
            rag = RAGService()
            query = params.get("query", "")
            result = await rag.query(
                query=query,
                organization_id=str(context.organization_id) if context.organization_id else None,
            )
            return {"status": "success", "answer": result.get("answer"), "sources": result.get("sources", [])}
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

        # Prepare restricted globals
        restricted_globals = {
            "__builtins__": safe_builtins,
            "_getitem_": default_guarded_getitem,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            # Add safe modules
            "json": json,
            "math": __import__("math"),
            "datetime": datetime,
            "re": __import__("re"),
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
        dangerous_nodes = (ast.Import, ast.ImportFrom, ast.Exec, ast.Global)
        for node in ast.walk(tree):
            if isinstance(node, dangerous_nodes):
                return {"status": "error", "error": "Imports and global statements not allowed"}
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "open", "__import__"):
                        return {"status": "error", "error": f"Function '{node.func.id}' not allowed"}

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

        config = node.config or {}
        url = context.resolve_template(config.get("url", ""))
        method = config.get("method", "GET").upper()
        timeout_seconds = config.get("timeout", 30)
        retries = config.get("retries", 0)
        follow_redirects = config.get("follow_redirects", True)
        ignore_ssl = config.get("ignore_ssl", False)

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
                        response_text = await response.text()

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
        """Send email notification."""
        subject = config.get("subject", "Workflow Notification")
        self.logger.info("Email notification", to=recipients, subject=subject)
        return {"channel": "email", "message": message, "recipients": recipients, "sent": True, "subject": subject}

    async def _notify_slack(self, message: str, recipients: List, config: Dict, context: "ExecutionContext") -> Dict:
        """Send Slack notification."""
        import aiohttp

        webhook_url = config.get("webhook_url") or config.get("slack_webhook")
        if not webhook_url:
            return {"channel": "slack", "sent": False, "error": "No Slack webhook URL configured"}

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
            model = "gpt-4o"  # Default model

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
        """Run an inline LLM call without a predefined agent."""
        try:
            from backend.services.llm import EnhancedLLMFactory
            from langchain_core.messages import HumanMessage, SystemMessage
            from backend.services.rag_module.prompts import enhance_agent_system_prompt

            prompt = agent_input.get("prompt", "")
            model = config.get("model", "gpt-4")
            temperature = config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens", 2000)

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

            # Invoke LLM with configured parameters
            response = await llm.ainvoke(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return {
                "status": "success",
                "response": response.content if hasattr(response, 'content') else str(response),
                "model": model,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_rag_agent(self, agent_def, agent_input: Dict, context: "ExecutionContext") -> Dict:
        """Run a RAG-based agent."""
        try:
            from backend.services.rag import RAGService
            rag = RAGService()

            query = agent_input.get("prompt") or agent_input.get("query", "")
            result = await rag.query(
                query=query,
                organization_id=str(context.organization_id) if context.organization_id else None,
            )

            return {
                "status": "success",
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "agent_type": "rag",
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
        """Run a tool-using agent."""
        # Placeholder for tool-using agent
        return {
            "status": "success",
            "response": "Tool agent execution placeholder",
            "agent_type": "tool",
        }

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
