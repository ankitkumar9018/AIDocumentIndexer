#!/usr/bin/env python3
"""
Seed Sample Workflow

Creates a sample complex workflow that demonstrates all node types
in the workflow builder. Run this script directly to seed the database.

Usage:
    python -m backend.scripts.seed_sample_workflow
    or
    python backend/scripts/seed_sample_workflow.py
"""

import asyncio
import uuid
import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///backend/data/aidocindexer.db")

# Convert to async URL if needed
if DATABASE_URL.startswith("sqlite:///"):
    DATABASE_URL = DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


def create_sample_workflow_data(organization_id: uuid.UUID):
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

    workflow_id = uuid.uuid4()
    now = datetime.utcnow()

    # Generate node IDs
    node_ids = {f"node_{i}": uuid.uuid4() for i in range(1, 17)}

    # Define nodes with positions and configurations
    nodes = [
        # START - Workflow entry point
        {
            "id": node_ids["node_1"],
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
            "id": node_ids["node_2"],
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
            "id": node_ids["node_3"],
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
            "id": node_ids["node_4"],
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
                "message": "Documents failed validation. Please check the logs.",
                "webhook_url": "{{env.SLACK_WEBHOOK_URL}}"
            },
        },

        # LOOP - Process each valid document (true branch)
        {
            "id": node_ids["node_5"],
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
            "id": node_ids["node_6"],
            "workflow_id": workflow_id,
            "node_type": "agent",
            "name": "Extract Information",
            "description": "Use AI to extract key information from document",
            "position_x": 600,
            "position_y": 450,
            "config": {
                "agent_type": "analyst",
                "prompt": "Extract the following information from document {{loop.current_doc}}:\n- Title\n- Author\n- Key Topics\n- Summary (max 200 words)\n- Named Entities\n\nReturn as JSON.",
                "system_prompt": "You are a document analyst. Extract structured information accurately.",
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
            "id": node_ids["node_7"],
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
# Get input from previous node
extracted = context['nodes'].get('extract_information', {}).get('output', {})

# Validate required fields
required_fields = ['title', 'author', 'key_topics', 'summary']
missing = [f for f in required_fields if not extracted.get(f)]

if missing:
    output['status'] = 'error'
    output['error'] = f'Missing required fields: {missing}'
else:
    output['status'] = 'success'
    output['extracted'] = extracted
    output['word_count'] = len(extracted.get('summary', '').split())
"""
            },
        },

        # CONDITION - Check processing mode for quality check
        {
            "id": node_ids["node_8"],
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
            "id": node_ids["node_9"],
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
                "message": "Please review the extracted information for accuracy.",
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
            "id": node_ids["node_10"],
            "workflow_id": workflow_id,
            "node_type": "action",
            "name": "Store Results",
            "description": "Save extracted information to document metadata",
            "position_x": 600,
            "position_y": 850,
            "config": {
                "action_type": "update_document",
                "template_id": "",
                "data_mapping": '{"metadata": {"extraction": "{{nodes.validate_transform.output}}"}}',
                "on_error": "retry",
                "max_retries": 3,
                "retry_delay": 5
            },
        },

        # DELAY - Wait for all parallel processing to complete
        {
            "id": node_ids["node_11"],
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
            "id": node_ids["node_12"],
            "workflow_id": workflow_id,
            "node_type": "agent",
            "name": "Generate Report",
            "description": "Create a summary report of all processed documents",
            "position_x": 400,
            "position_y": 1050,
            "config": {
                "agent_type": "writer",
                "prompt": "Generate a professional summary report of the document processing batch.\n\nDocuments processed: {{input.document_ids.length}}\nProcessing mode: {{input.processing_mode}}\n\nInclude:\n1. Executive summary\n2. Key findings\n3. Common themes\n4. Recommendations",
                "system_prompt": "You are a professional report writer.",
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
            "id": node_ids["node_13"],
            "workflow_id": workflow_id,
            "node_type": "action",
            "name": "Create Report Document",
            "description": "Generate PDF report from markdown",
            "position_x": 400,
            "position_y": 1150,
            "config": {
                "action_type": "generate_pdf",
                "template_id": "report_template",
                "data_mapping": '{"content": "{{nodes.generate_report.output.response}}", "title": "Document Processing Report"}',
                "on_error": "continue"
            },
        },

        # CONDITION - Check if notification requested
        {
            "id": node_ids["node_14"],
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
            "id": node_ids["node_15"],
            "workflow_id": workflow_id,
            "node_type": "notification",
            "name": "Completion Notification",
            "description": "Send email with report attached",
            "position_x": 200,
            "position_y": 1350,
            "config": {
                "channel": "email",
                "recipients": "{{env.NOTIFICATION_EMAIL}}",
                "subject": "Document Processing Complete",
                "message": "The document processing workflow has completed successfully.",
                "html_email": True,
                "include_output": True
            },
        },

        # END - Workflow completion
        {
            "id": node_ids["node_16"],
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
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_1"], "target_node_id": node_ids["node_2"],
         "edge_type": "default"},

        # Validate -> All Valid?
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_2"], "target_node_id": node_ids["node_3"],
         "edge_type": "default"},

        # All Valid? -> Alert (false)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_3"], "target_node_id": node_ids["node_4"],
         "edge_type": "false", "condition": "false"},

        # Alert -> End (early termination)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_4"], "target_node_id": node_ids["node_16"],
         "edge_type": "default"},

        # All Valid? -> Loop (true)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_3"], "target_node_id": node_ids["node_5"],
         "edge_type": "true", "condition": "true"},

        # Loop -> Extract Info (loop body)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_5"], "target_node_id": node_ids["node_6"],
         "edge_type": "true", "condition": "loop_body"},

        # Extract -> Validate & Transform
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_6"], "target_node_id": node_ids["node_7"],
         "edge_type": "default"},

        # Validate & Transform -> Thorough Mode?
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_7"], "target_node_id": node_ids["node_8"],
         "edge_type": "default"},

        # Thorough Mode? -> Quality Review (true)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_8"], "target_node_id": node_ids["node_9"],
         "edge_type": "true", "condition": "true"},

        # Thorough Mode? -> Store Results (false)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_8"], "target_node_id": node_ids["node_10"],
         "edge_type": "false", "condition": "false"},

        # Quality Review -> Store Results
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_9"], "target_node_id": node_ids["node_10"],
         "edge_type": "default"},

        # Store Results -> Loop (continue)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_10"], "target_node_id": node_ids["node_5"],
         "edge_type": "loop_continue"},

        # Loop -> Wait (exit)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_5"], "target_node_id": node_ids["node_11"],
         "edge_type": "false", "condition": "loop_exit"},

        # Wait -> Generate Report
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_11"], "target_node_id": node_ids["node_12"],
         "edge_type": "default"},

        # Generate Report -> Create Report Document
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_12"], "target_node_id": node_ids["node_13"],
         "edge_type": "default"},

        # Create Report Document -> Notify?
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_13"], "target_node_id": node_ids["node_14"],
         "edge_type": "default"},

        # Notify? -> Completion Notification (true)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_14"], "target_node_id": node_ids["node_15"],
         "edge_type": "true", "condition": "true"},

        # Notify? -> End (false)
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_14"], "target_node_id": node_ids["node_16"],
         "edge_type": "false", "condition": "false"},

        # Completion Notification -> End
        {"id": uuid.uuid4(), "workflow_id": workflow_id,
         "source_node_id": node_ids["node_15"], "target_node_id": node_ids["node_16"],
         "edge_type": "default"},
    ]

    return {
        "workflow": {
            "id": workflow_id,
            "organization_id": organization_id,
            "name": "Document Processing Pipeline (Sample)",
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


async def seed_sample_workflow():
    """Seed the sample workflow into the database."""

    # Create engine and session
    engine = create_async_engine(DATABASE_URL, echo=True)
    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as session:
        # Get or create a default organization_id (use first user's ID)
        result = await session.execute(text("SELECT id FROM users LIMIT 1"))
        row = result.fetchone()

        if row:
            organization_id = uuid.UUID(row[0]) if isinstance(row[0], str) else row[0]
        else:
            # Create a default org ID
            organization_id = uuid.uuid4()
            print(f"No users found, using generated organization_id: {organization_id}")

        print(f"Using organization_id: {organization_id}")

        # Check if sample workflow already exists
        check_result = await session.execute(
            text("SELECT id FROM workflows WHERE name LIKE '%Sample%' LIMIT 1")
        )
        if check_result.fetchone():
            print("Sample workflow already exists, skipping...")
            return

        # Create workflow data
        data = create_sample_workflow_data(organization_id)
        workflow = data["workflow"]

        # Insert workflow
        await session.execute(
            text("""
                INSERT INTO workflows (id, organization_id, name, description, trigger_type, trigger_config, is_active, is_draft, version, config, created_at, updated_at)
                VALUES (:id, :organization_id, :name, :description, :trigger_type, :trigger_config, :is_active, :is_draft, :version, :config, :created_at, :updated_at)
            """),
            {
                "id": str(workflow["id"]),
                "organization_id": str(workflow["organization_id"]),
                "name": workflow["name"],
                "description": workflow["description"],
                "trigger_type": workflow["trigger_type"],
                "trigger_config": json.dumps(workflow["trigger_config"]),
                "is_active": workflow["is_active"],
                "is_draft": workflow["is_draft"],
                "version": workflow["version"],
                "config": json.dumps(workflow["config"]),
                "created_at": workflow["created_at"],
                "updated_at": workflow["updated_at"],
            }
        )

        # Insert nodes
        for node in data["nodes"]:
            await session.execute(
                text("""
                    INSERT INTO workflow_nodes (id, workflow_id, node_type, name, description, position_x, position_y, config)
                    VALUES (:id, :workflow_id, :node_type, :name, :description, :position_x, :position_y, :config)
                """),
                {
                    "id": str(node["id"]),
                    "workflow_id": str(node["workflow_id"]),
                    "node_type": node["node_type"],
                    "name": node["name"],
                    "description": node.get("description"),
                    "position_x": node["position_x"],
                    "position_y": node["position_y"],
                    "config": json.dumps(node["config"]),
                }
            )

        # Insert edges
        for edge in data["edges"]:
            await session.execute(
                text("""
                    INSERT INTO workflow_edges (id, workflow_id, source_node_id, target_node_id, label, condition, edge_type)
                    VALUES (:id, :workflow_id, :source_node_id, :target_node_id, :label, :condition, :edge_type)
                """),
                {
                    "id": str(edge["id"]),
                    "workflow_id": str(edge["workflow_id"]),
                    "source_node_id": str(edge["source_node_id"]),
                    "target_node_id": str(edge["target_node_id"]),
                    "label": edge.get("label"),
                    "condition": edge.get("condition"),
                    "edge_type": edge.get("edge_type", "default"),
                }
            )

        await session.commit()
        print(f"Successfully created sample workflow: {workflow['name']} (ID: {workflow['id']})")


if __name__ == "__main__":
    asyncio.run(seed_sample_workflow())
