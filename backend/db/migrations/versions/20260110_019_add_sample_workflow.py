"""
Add sample workflow

Revision ID: 019_add_sample_workflow
Revises: 20260110_018_add_model_registry
Create Date: 2026-01-10

This migration creates a sample complex workflow that demonstrates
all node types and configurations available in the workflow builder.
"""

import uuid
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


# Sample workflow configuration
SAMPLE_WORKFLOW_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
SAMPLE_ORG_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")  # Default org


def create_sample_workflow_data():
    """
    Create a complex document processing workflow that demonstrates:
    - START/END nodes
    - HTTP node for external API calls
    - CONDITION nodes for branching logic
    - LOOP nodes for iteration
    - AGENT nodes for AI processing
    - CODE nodes for custom transformations
    - NOTIFICATION nodes for alerts
    - DELAY nodes for timing
    - HUMAN_APPROVAL for manual gates
    - ACTION nodes for document operations
    """

    workflow_id = SAMPLE_WORKFLOW_ID
    now = datetime.utcnow()

    # Define nodes with positions and configurations
    nodes = [
        # START - Workflow entry point
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000001"),
            "workflow_id": workflow_id,
            "node_type": "start",
            "name": "Start",
            "description": "Workflow begins when documents are uploaded",
            "position_x": 400,
            "position_y": 50,
            "config": {
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_ids": {"type": "array", "items": {"type": "string"}},
                        "processing_mode": {"type": "string", "enum": ["fast", "thorough"]},
                        "notify_on_completion": {"type": "boolean", "default": True}
                    },
                    "required": ["document_ids"]
                }
            },
        },

        # HTTP - Validate documents via external API
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000002"),
            "workflow_id": workflow_id,
            "node_type": "http",
            "name": "Validate Documents",
            "description": "Call document validation service",
            "position_x": 400,
            "position_y": 150,
            "config": {
                "method": "POST",
                "url": "{{env.VALIDATION_API_URL}}/validate",
                "auth_type": "bearer",
                "auth_token": "{{env.VALIDATION_API_TOKEN}}",
                "content_type": "json",
                "body": '{"document_ids": {{input.document_ids}}}',
                "timeout": 30,
                "retries": 2,
                "response_type": "json",
                "extract_path": "$.validation_results"
            },
        },

        # CONDITION - Check if all documents are valid
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000003"),
            "workflow_id": workflow_id,
            "node_type": "condition",
            "name": "All Valid?",
            "description": "Check if all documents passed validation",
            "position_x": 400,
            "position_y": 250,
            "config": {
                "condition_type": "expression",
                "expression": "{{nodes.validate_documents.extracted.every(r => r.valid)}}",
                "true_label": "Yes",
                "false_label": "No"
            },
        },

        # NOTIFICATION - Alert on validation failure (false branch)
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000004"),
            "workflow_id": workflow_id,
            "node_type": "notification",
            "name": "Alert: Validation Failed",
            "description": "Send notification about validation failure",
            "position_x": 200,
            "position_y": 350,
            "config": {
                "channel": "slack",
                "recipients": "{{env.SLACK_CHANNEL_ALERTS}}",
                "subject": "Document Validation Failed",
                "message": "Documents failed validation:\n{{nodes.validate_documents.extracted.filter(r => !r.valid).map(r => r.document_id).join('\\n')}}",
                "webhook_url": "{{env.SLACK_WEBHOOK_URL}}"
            },
        },

        # LOOP - Process each valid document (true branch)
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000005"),
            "workflow_id": workflow_id,
            "node_type": "loop",
            "name": "Process Each Document",
            "description": "Iterate over valid documents for processing",
            "position_x": 600,
            "position_y": 350,
            "config": {
                "loop_type": "for_each",
                "items_source": "{{input.document_ids}}",
                "item_var": "current_doc",
                "max_iterations": 100,
                "parallel": True,
                "concurrency": 5
            },
        },

        # AGENT - Extract key information from document
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000006"),
            "workflow_id": workflow_id,
            "node_type": "agent",
            "name": "Extract Information",
            "description": "Use AI to extract key information from document",
            "position_x": 600,
            "position_y": 450,
            "config": {
                "agent_type": "analyst",
                "prompt": "Extract the following information from document {{loop.current_doc}}:\n- Title\n- Author\n- Key Topics\n- Summary (max 200 words)\n- Named Entities (people, organizations, locations)\n\nReturn as JSON.",
                "system_prompt": "You are a document analyst. Extract structured information from documents accurately and concisely.",
                "model": "gpt-4o",
                "temperature": 0.3,
                "max_tokens": 2000,
                "use_rag": True,
                "doc_filter": "{{loop.current_doc}}",
                "output_format": "json",
                "wait_for_result": True,
                "timeout": 120
            },
        },

        # CODE - Transform and validate extracted data
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000007"),
            "workflow_id": workflow_id,
            "node_type": "code",
            "name": "Validate & Transform",
            "description": "Validate extracted data and add metadata",
            "position_x": 600,
            "position_y": 550,
            "config": {
                "language": "python",
                "mode": "each_item",
                "timeout": 30,
                "sandbox": True,
                "code": """
import json
from datetime import datetime

# Get input from previous node
extracted = context.nodes.get('extract_information', {}).get('output', {})

# Validate required fields
required_fields = ['title', 'author', 'key_topics', 'summary']
missing = [f for f in required_fields if not extracted.get(f)]

if missing:
    output['status'] = 'error'
    output['error'] = f'Missing required fields: {missing}'
else:
    # Add metadata
    output['status'] = 'success'
    output['document_id'] = context.variables.get('current_doc')
    output['extracted'] = extracted
    output['processed_at'] = datetime.utcnow().isoformat()
    output['word_count'] = len(extracted.get('summary', '').split())
    output['entity_count'] = len(extracted.get('named_entities', []))
"""
            },
        },

        # CONDITION - Check processing mode for quality check
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000008"),
            "workflow_id": workflow_id,
            "node_type": "condition",
            "name": "Thorough Mode?",
            "description": "Check if thorough processing is requested",
            "position_x": 600,
            "position_y": 650,
            "config": {
                "condition_type": "compare",
                "left_value": "{{input.processing_mode}}",
                "operator": "equals",
                "right_value": "thorough",
                "true_label": "Thorough",
                "false_label": "Fast"
            },
        },

        # HUMAN_APPROVAL - Quality review for thorough mode
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000009"),
            "workflow_id": workflow_id,
            "node_type": "human_approval",
            "name": "Quality Review",
            "description": "Manual review of extracted information",
            "position_x": 800,
            "position_y": 750,
            "config": {
                "approvers": "{{env.QUALITY_REVIEWERS}}",
                "approval_type": "any",
                "title": "Review Document Extraction",
                "message": "Please review the extracted information for document {{loop.current_doc}}:\n\nTitle: {{nodes.validate_transform.output.extracted.title}}\nSummary: {{nodes.validate_transform.output.extracted.summary}}\n\nApprove if accurate, reject with corrections if needed.",
                "allow_comments": True,
                "timeout_value": 4,
                "timeout_unit": "hours",
                "on_timeout": "escalate",
                "escalation_to": "{{env.ESCALATION_EMAIL}}",
                "notify_channels": ["email", "slack"],
                "send_reminder": True,
                "reminder_interval": 2
            },
        },

        # ACTION - Store processed results
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000010"),
            "workflow_id": workflow_id,
            "node_type": "action",
            "name": "Store Results",
            "description": "Save extracted information to document metadata",
            "position_x": 600,
            "position_y": 850,
            "config": {
                "action_type": "update_document",
                "template_id": "",
                "data_mapping": '{"metadata": {"extraction": "{{nodes.validate_transform.output}}", "approved": "{{nodes.quality_review.approved or true}}"}}',
                "on_error": "retry",
                "max_retries": 3,
                "retry_delay": 5
            },
        },

        # Loop exit point (implicit - connects to merge)

        # DELAY - Wait for all parallel processing to complete
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000011"),
            "workflow_id": workflow_id,
            "node_type": "delay",
            "name": "Wait for Completion",
            "description": "Brief wait to ensure all updates are committed",
            "position_x": 400,
            "position_y": 950,
            "config": {
                "delay_type": "fixed",
                "delay_value": 5,
                "delay_unit": "seconds"
            },
        },

        # AGENT - Generate summary report
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000012"),
            "workflow_id": workflow_id,
            "node_type": "agent",
            "name": "Generate Report",
            "description": "Create a summary report of all processed documents",
            "position_x": 400,
            "position_y": 1050,
            "config": {
                "agent_type": "writer",
                "prompt": "Generate a professional summary report of the document processing batch.\n\nDocuments processed: {{input.document_ids.length}}\nProcessing mode: {{input.processing_mode}}\n\nInclude:\n1. Executive summary\n2. Key findings across all documents\n3. Common themes identified\n4. Recommendations for follow-up\n\nFormat as a well-structured report.",
                "system_prompt": "You are a professional report writer. Create clear, concise, and actionable reports.",
                "model": "gpt-4o",
                "temperature": 0.5,
                "max_tokens": 4000,
                "use_rag": True,
                "output_format": "markdown",
                "wait_for_result": True,
                "timeout": 180
            },
        },

        # ACTION - Create report document
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000013"),
            "workflow_id": workflow_id,
            "node_type": "action",
            "name": "Create Report Document",
            "description": "Generate PDF report from markdown",
            "position_x": 400,
            "position_y": 1150,
            "config": {
                "action_type": "generate_pdf",
                "template_id": "report_template",
                "data_mapping": '{"content": "{{nodes.generate_report.output.response}}", "title": "Document Processing Report", "date": "{{now}}"}',
                "on_error": "continue"
            },
        },

        # CONDITION - Check if notification requested
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000014"),
            "workflow_id": workflow_id,
            "node_type": "condition",
            "name": "Notify?",
            "description": "Check if completion notification is requested",
            "position_x": 400,
            "position_y": 1250,
            "config": {
                "condition_type": "compare",
                "left_value": "{{input.notify_on_completion}}",
                "operator": "equals",
                "right_value": "true",
                "true_label": "Yes",
                "false_label": "No"
            },
        },

        # NOTIFICATION - Send completion notification
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000015"),
            "workflow_id": workflow_id,
            "node_type": "notification",
            "name": "Completion Notification",
            "description": "Send email with report attached",
            "position_x": 200,
            "position_y": 1350,
            "config": {
                "channel": "email",
                "recipients": "{{env.NOTIFICATION_EMAIL}}",
                "subject": "Document Processing Complete - {{input.document_ids.length}} Documents",
                "message": "The document processing workflow has completed.\n\nDocuments processed: {{input.document_ids.length}}\nProcessing mode: {{input.processing_mode}}\n\nPlease find the summary report attached.",
                "html_email": True,
                "include_output": True
            },
        },

        # END - Workflow completion
        {
            "id": uuid.UUID("10000000-0000-0000-0000-000000000016"),
            "workflow_id": workflow_id,
            "node_type": "end",
            "name": "End",
            "description": "Workflow complete",
            "position_x": 400,
            "position_y": 1450,
            "config": {
                "output_path": "$.result",
                "include_metadata": True
            },
        },
    ]

    # Define edges (connections between nodes)
    edges = [
        # Start -> Validate
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000001"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000001"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000002"),
         "edge_type": "default"},

        # Validate -> All Valid?
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000002"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000002"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000003"),
         "edge_type": "default"},

        # All Valid? -> Alert (false)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000003"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000003"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000004"),
         "edge_type": "false", "condition": "false"},

        # Alert -> End (early termination)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000004"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000004"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000016"),
         "edge_type": "default"},

        # All Valid? -> Loop (true)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000005"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000003"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000005"),
         "edge_type": "true", "condition": "true"},

        # Loop -> Extract Info (loop body)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000006"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000005"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000006"),
         "edge_type": "true", "condition": "loop_body"},

        # Extract -> Validate & Transform
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000007"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000006"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000007"),
         "edge_type": "default"},

        # Validate & Transform -> Thorough Mode?
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000008"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000007"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000008"),
         "edge_type": "default"},

        # Thorough Mode? -> Quality Review (true/thorough)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000009"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000008"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000009"),
         "edge_type": "true", "condition": "true"},

        # Thorough Mode? -> Store Results (false/fast)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000010"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000008"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000010"),
         "edge_type": "false", "condition": "false"},

        # Quality Review -> Store Results
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000011"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000009"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000010"),
         "edge_type": "default"},

        # Store Results -> Loop (continue iteration)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000012"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000010"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000005"),
         "edge_type": "loop_continue"},

        # Loop -> Wait (loop exit)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000013"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000005"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000011"),
         "edge_type": "false", "condition": "loop_exit"},

        # Wait -> Generate Report
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000014"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000011"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000012"),
         "edge_type": "default"},

        # Generate Report -> Create Report Document
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000015"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000012"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000013"),
         "edge_type": "default"},

        # Create Report Document -> Notify?
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000016"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000013"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000014"),
         "edge_type": "default"},

        # Notify? -> Completion Notification (true)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000017"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000014"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000015"),
         "edge_type": "true", "condition": "true"},

        # Notify? -> End (false)
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000018"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000014"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000016"),
         "edge_type": "false", "condition": "false"},

        # Completion Notification -> End
        {"id": uuid.UUID("20000000-0000-0000-0000-000000000019"), "workflow_id": workflow_id,
         "source_node_id": uuid.UUID("10000000-0000-0000-0000-000000000015"),
         "target_node_id": uuid.UUID("10000000-0000-0000-0000-000000000016"),
         "edge_type": "default"},
    ]

    return {
        "workflow": {
            "id": workflow_id,
            "organization_id": SAMPLE_ORG_ID,
            "name": "Document Processing Pipeline",
            "description": "A comprehensive workflow demonstrating all node types: validates documents via external API, processes each document with AI extraction, includes optional human review, generates summary reports, and sends notifications.",
            "trigger_type": "manual",
            "trigger_config": {},
            "is_active": False,
            "is_draft": True,
            "version": 1,
            "config": {
                "sample": True,
                "description": "This is a sample workflow to demonstrate the workflow builder capabilities."
            },
            "created_at": now,
            "updated_at": now,
        },
        "nodes": nodes,
        "edges": edges,
    }


def upgrade() -> None:
    """Create sample workflow with all node types."""
    data = create_sample_workflow_data()

    # Insert workflow
    workflow = data["workflow"]
    op.execute(
        sa.text("""
            INSERT INTO workflows (id, organization_id, name, description, trigger_type, trigger_config, is_active, is_draft, version, config, created_at, updated_at)
            VALUES (:id, :organization_id, :name, :description, :trigger_type, :trigger_config, :is_active, :is_draft, :version, :config, :created_at, :updated_at)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                config = EXCLUDED.config,
                updated_at = EXCLUDED.updated_at
        """).bindparams(
            id=str(workflow["id"]),
            organization_id=str(workflow["organization_id"]),
            name=workflow["name"],
            description=workflow["description"],
            trigger_type=workflow["trigger_type"],
            trigger_config=sa.type_coerce(workflow["trigger_config"], JSONB),
            is_active=workflow["is_active"],
            is_draft=workflow["is_draft"],
            version=workflow["version"],
            config=sa.type_coerce(workflow["config"], JSONB),
            created_at=workflow["created_at"],
            updated_at=workflow["updated_at"],
        )
    )

    # Insert nodes
    for node in data["nodes"]:
        op.execute(
            sa.text("""
                INSERT INTO workflow_nodes (id, workflow_id, node_type, name, description, position_x, position_y, config)
                VALUES (:id, :workflow_id, :node_type, :name, :description, :position_x, :position_y, :config)
                ON CONFLICT (id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    position_x = EXCLUDED.position_x,
                    position_y = EXCLUDED.position_y,
                    config = EXCLUDED.config
            """).bindparams(
                id=str(node["id"]),
                workflow_id=str(node["workflow_id"]),
                node_type=node["node_type"],
                name=node["name"],
                description=node.get("description"),
                position_x=node["position_x"],
                position_y=node["position_y"],
                config=sa.type_coerce(node["config"], JSONB),
            )
        )

    # Insert edges
    for edge in data["edges"]:
        op.execute(
            sa.text("""
                INSERT INTO workflow_edges (id, workflow_id, source_node_id, target_node_id, label, condition, edge_type)
                VALUES (:id, :workflow_id, :source_node_id, :target_node_id, :label, :condition, :edge_type)
                ON CONFLICT (id) DO UPDATE SET
                    source_node_id = EXCLUDED.source_node_id,
                    target_node_id = EXCLUDED.target_node_id,
                    label = EXCLUDED.label,
                    condition = EXCLUDED.condition,
                    edge_type = EXCLUDED.edge_type
            """).bindparams(
                id=str(edge["id"]),
                workflow_id=str(edge["workflow_id"]),
                source_node_id=str(edge["source_node_id"]),
                target_node_id=str(edge["target_node_id"]),
                label=edge.get("label"),
                condition=edge.get("condition"),
                edge_type=edge.get("edge_type", "default"),
            )
        )


def downgrade() -> None:
    """Remove sample workflow."""
    workflow_id = str(SAMPLE_WORKFLOW_ID)

    # Delete edges first (foreign key constraint)
    op.execute(
        sa.text("DELETE FROM workflow_edges WHERE workflow_id = :workflow_id").bindparams(
            workflow_id=workflow_id
        )
    )

    # Delete nodes
    op.execute(
        sa.text("DELETE FROM workflow_nodes WHERE workflow_id = :workflow_id").bindparams(
            workflow_id=workflow_id
        )
    )

    # Delete workflow
    op.execute(
        sa.text("DELETE FROM workflows WHERE id = :id").bindparams(
            id=workflow_id
        )
    )
