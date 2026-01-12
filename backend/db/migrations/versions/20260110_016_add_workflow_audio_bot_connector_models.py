"""Add Workflow, Audio, Bot, Connector, Budget, and Image models.

Revision ID: 20260110_016
Revises: 20260109_015
Create Date: 2026-01-10

This migration adds:
1. Workflow tables (workflows, workflow_nodes, workflow_edges, workflow_executions, workflow_node_executions)
2. Audio Overview tables (audio_overviews)
3. Bot Connection tables (bot_connections)
4. Connector tables (connector_instances, synced_resources)
5. LLM Gateway tables (budgets, virtual_api_keys)
6. Image Generation tables (generated_images)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '016b'
down_revision = '016a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Detect database type
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # Helper for UUID type
    def uuid_type():
        if is_postgresql:
            return postgresql.UUID(as_uuid=True)
        return sa.String(36)

    # Helper for JSON type
    def json_type():
        if is_postgresql:
            return postgresql.JSONB()
        return sa.Text()

    # =========================================================================
    # 1. Workflows table
    # =========================================================================
    op.create_table(
        'workflows',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_draft', sa.Boolean(), default=True, nullable=False),
        sa.Column('version', sa.Integer(), default=1, nullable=False),
        sa.Column('trigger_type', sa.String(50), default='manual', nullable=False),
        sa.Column('trigger_config', json_type(), nullable=True),
        sa.Column('config', json_type(), nullable=True),
        sa.Column('created_by_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_workflows_organization', 'workflows', ['organization_id'])
    op.create_index('idx_workflows_active', 'workflows', ['is_active'])
    op.create_index('idx_workflows_trigger', 'workflows', ['trigger_type'])
    op.create_index('idx_workflows_created_by', 'workflows', ['created_by_id'])

    # =========================================================================
    # 2. Workflow Nodes table
    # =========================================================================
    op.create_table(
        'workflow_nodes',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('workflow_id', uuid_type(),
                  sa.ForeignKey('workflows.id', ondelete='CASCADE'), nullable=False),
        sa.Column('node_type', sa.String(50), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('position_x', sa.Float(), default=0.0),
        sa.Column('position_y', sa.Float(), default=0.0),
        sa.Column('config', json_type(), nullable=True),
    )
    op.create_index('idx_workflow_nodes_workflow', 'workflow_nodes', ['workflow_id'])
    op.create_index('idx_workflow_nodes_type', 'workflow_nodes', ['node_type'])

    # =========================================================================
    # 3. Workflow Edges table
    # =========================================================================
    op.create_table(
        'workflow_edges',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('workflow_id', uuid_type(),
                  sa.ForeignKey('workflows.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_node_id', uuid_type(),
                  sa.ForeignKey('workflow_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('target_node_id', uuid_type(),
                  sa.ForeignKey('workflow_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('label', sa.String(100), nullable=True),
        sa.Column('condition', sa.Text(), nullable=True),
        sa.Column('edge_type', sa.String(20), default='default'),
    )
    op.create_index('idx_workflow_edges_workflow', 'workflow_edges', ['workflow_id'])
    op.create_index('idx_workflow_edges_source', 'workflow_edges', ['source_node_id'])
    op.create_index('idx_workflow_edges_target', 'workflow_edges', ['target_node_id'])

    # =========================================================================
    # 4. Workflow Executions table
    # =========================================================================
    op.create_table(
        'workflow_executions',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('workflow_id', uuid_type(),
                  sa.ForeignKey('workflows.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(50), default='pending', nullable=False),
        sa.Column('current_node_id', uuid_type(), nullable=True),
        sa.Column('trigger_type', sa.String(50), nullable=False),
        sa.Column('trigger_data', json_type(), nullable=True),
        sa.Column('input_data', json_type(), nullable=True),
        sa.Column('output_data', json_type(), nullable=True),
        sa.Column('context', json_type(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_node_id', uuid_type(), nullable=True),
        sa.Column('retry_count', sa.Integer(), default=0),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('triggered_by_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_workflow_executions_org', 'workflow_executions', ['organization_id'])
    op.create_index('idx_workflow_executions_workflow', 'workflow_executions', ['workflow_id'])
    op.create_index('idx_workflow_executions_status', 'workflow_executions', ['status'])
    op.create_index('idx_workflow_executions_started', 'workflow_executions', ['started_at'])

    # =========================================================================
    # 5. Workflow Node Executions table
    # =========================================================================
    op.create_table(
        'workflow_node_executions',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('execution_id', uuid_type(),
                  sa.ForeignKey('workflow_executions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('node_id', uuid_type(),
                  sa.ForeignKey('workflow_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('input_data', json_type(), nullable=True),
        sa.Column('output_data', json_type(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
    )
    op.create_index('idx_workflow_node_exec_execution', 'workflow_node_executions', ['execution_id'])
    op.create_index('idx_workflow_node_exec_node', 'workflow_node_executions', ['node_id'])

    # =========================================================================
    # 6. Audio Overviews table
    # =========================================================================
    # Helper for array type
    def array_type(element_type):
        if is_postgresql:
            return postgresql.ARRAY(element_type)
        return sa.Text()  # Store as JSON string for SQLite

    op.create_table(
        'audio_overviews',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('document_ids', sa.Text() if not is_postgresql else postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('folder_id', uuid_type(),
                  sa.ForeignKey('folders.id', ondelete='SET NULL'), nullable=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('format', sa.String(50), default='deep_dive', nullable=False),
        sa.Column('language', sa.String(10), default='en'),
        sa.Column('host_config', json_type(), nullable=True),
        sa.Column('target_duration_minutes', sa.Integer(), nullable=True),
        sa.Column('tone', sa.String(50), nullable=True),
        sa.Column('script', json_type(), nullable=True),
        sa.Column('transcript', sa.Text(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('audio_url', sa.String(1000), nullable=True),
        sa.Column('storage_path', sa.String(1000), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('audio_format', sa.String(10), default='mp3'),
        sa.Column('tts_provider', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), default='pending', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('progress_percent', sa.Integer(), default=0),
        sa.Column('current_step', sa.String(100), nullable=True),
        sa.Column('created_by_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('play_count', sa.Integer(), default=0),
        sa.Column('last_played_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_audio_overviews_org', 'audio_overviews', ['organization_id'])
    op.create_index('idx_audio_overviews_status', 'audio_overviews', ['status'])
    op.create_index('idx_audio_overviews_format', 'audio_overviews', ['format'])
    op.create_index('idx_audio_overviews_created_by', 'audio_overviews', ['created_by_id'])

    # =========================================================================
    # 7. Bot Connections table
    # =========================================================================
    op.create_table(
        'bot_connections',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('platform', sa.String(50), nullable=False),
        sa.Column('workspace_id', sa.String(255), nullable=True),
        sa.Column('bot_user_id', sa.String(255), nullable=True),
        sa.Column('bot_token_encrypted', sa.Text(), nullable=True),
        sa.Column('refresh_token_encrypted', sa.Text(), nullable=True),
        sa.Column('signing_secret_encrypted', sa.Text(), nullable=True),
        sa.Column('webhook_url', sa.String(1000), nullable=True),
        sa.Column('config', json_type(), nullable=True),
        sa.Column('allowed_users', sa.Text() if not is_postgresql else postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('allowed_channels', sa.Text() if not is_postgresql else postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('last_verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_event_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_count', sa.Integer(), default=0),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_by_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_bot_connections_org', 'bot_connections', ['organization_id'])
    op.create_index('idx_bot_connections_platform', 'bot_connections', ['platform'])
    op.create_index('idx_bot_connections_active', 'bot_connections', ['is_active'])
    op.create_index('idx_bot_connections_workspace', 'bot_connections', ['platform', 'workspace_id'])

    # =========================================================================
    # 8. Connector Instances table
    # =========================================================================
    op.create_table(
        'connector_instances',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('connector_type', sa.String(50), nullable=False),
        sa.Column('access_token_encrypted', sa.Text(), nullable=True),
        sa.Column('refresh_token_encrypted', sa.Text(), nullable=True),
        sa.Column('token_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('api_key_encrypted', sa.Text(), nullable=True),
        sa.Column('external_account_id', sa.String(255), nullable=True),
        sa.Column('external_account_email', sa.String(255), nullable=True),
        sa.Column('config', json_type(), nullable=True),
        sa.Column('target_folder_id', uuid_type(),
                  sa.ForeignKey('folders.id', ondelete='SET NULL'), nullable=True),
        sa.Column('auto_sync', sa.Boolean(), default=True),
        sa.Column('sync_interval_minutes', sa.Integer(), default=60),
        sa.Column('sync_mode', sa.String(20), default='incremental'),
        sa.Column('last_sync_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_sync_cursor', sa.Text(), nullable=True),
        sa.Column('next_sync_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_resources_synced', sa.Integer(), default=0),
        sa.Column('total_bytes_synced', sa.BigInteger(), default=0),
        sa.Column('status', sa.String(50), default='connected', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_count', sa.Integer(), default=0),
        sa.Column('rate_limit_reset_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_connector_instances_org', 'connector_instances', ['organization_id'])
    op.create_index('idx_connector_instances_type', 'connector_instances', ['connector_type'])
    op.create_index('idx_connector_instances_status', 'connector_instances', ['status'])
    op.create_index('idx_connector_instances_next_sync', 'connector_instances', ['next_sync_at'])

    # =========================================================================
    # 9. Synced Resources table
    # =========================================================================
    op.create_table(
        'synced_resources',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('connector_id', uuid_type(),
                  sa.ForeignKey('connector_instances.id', ondelete='CASCADE'), nullable=False),
        sa.Column('external_id', sa.String(500), nullable=False),
        sa.Column('external_parent_id', sa.String(500), nullable=True),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('mime_type', sa.String(100), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('external_url', sa.String(2000), nullable=True),
        sa.Column('external_version', sa.String(100), nullable=True),
        sa.Column('external_modified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('external_modified_by', sa.String(255), nullable=True),
        sa.Column('document_id', uuid_type(),
                  sa.ForeignKey('documents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('metadata', json_type(), nullable=True),
        sa.Column('sync_status', sa.String(50), default='synced'),
        sa.Column('last_synced_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('sync_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_synced_resources_org', 'synced_resources', ['organization_id'])
    op.create_index('idx_synced_resources_connector', 'synced_resources', ['connector_id'])
    op.create_index('idx_synced_resources_external', 'synced_resources', ['connector_id', 'external_id'], unique=True)
    op.create_index('idx_synced_resources_document', 'synced_resources', ['document_id'])
    op.create_index('idx_synced_resources_status', 'synced_resources', ['sync_status'])

    # =========================================================================
    # 10. Budgets table
    # =========================================================================
    op.create_table(
        'budgets',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('user_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('period', sa.String(20), default='monthly', nullable=False),
        sa.Column('limit_amount', sa.Float(), nullable=False),
        sa.Column('soft_limit_amount', sa.Float(), nullable=True),
        sa.Column('current_spend', sa.Float(), default=0.0),
        sa.Column('current_tokens', sa.BigInteger(), default=0),
        sa.Column('current_requests', sa.Integer(), default=0),
        sa.Column('period_start', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_hard_limit', sa.Boolean(), default=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('paused_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('alert_thresholds', json_type(), nullable=True),
        sa.Column('last_alert_threshold', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_budgets_org', 'budgets', ['organization_id'])
    op.create_index('idx_budgets_user', 'budgets', ['user_id'])
    op.create_index('idx_budgets_active', 'budgets', ['is_active'])
    op.create_index('idx_budgets_period_end', 'budgets', ['period_end'])

    # =========================================================================
    # 11. Virtual API Keys table
    # =========================================================================
    op.create_table(
        'virtual_api_keys',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('user_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('key_prefix', sa.String(12), nullable=False),
        sa.Column('key_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('scopes', sa.Text() if not is_postgresql else postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('allowed_models', sa.Text() if not is_postgresql else postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('allowed_ips', sa.Text() if not is_postgresql else postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('rate_limit_rpm', sa.Integer(), nullable=True),
        sa.Column('rate_limit_tpd', sa.Integer(), nullable=True),
        sa.Column('monthly_budget', sa.Float(), nullable=True),
        sa.Column('current_month_spend', sa.Float(), default=0.0),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_requests', sa.Integer(), default=0),
        sa.Column('total_tokens', sa.BigInteger(), default=0),
        sa.Column('total_cost', sa.Float(), default=0.0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_virtual_api_keys_org', 'virtual_api_keys', ['organization_id'])
    op.create_index('idx_virtual_api_keys_user', 'virtual_api_keys', ['user_id'])
    op.create_index('idx_virtual_api_keys_hash', 'virtual_api_keys', ['key_hash'])
    op.create_index('idx_virtual_api_keys_active', 'virtual_api_keys', ['is_active'])
    op.create_index('idx_virtual_api_keys_prefix', 'virtual_api_keys', ['key_prefix'])

    # =========================================================================
    # 12. Generated Images table
    # =========================================================================
    op.create_table(
        'generated_images',
        sa.Column('id', uuid_type(), primary_key=True),
        sa.Column('organization_id', uuid_type(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('enhanced_prompt', sa.Text(), nullable=True),
        sa.Column('negative_prompt', sa.Text(), nullable=True),
        sa.Column('provider', sa.String(50), default='openai_dalle', nullable=False),
        sa.Column('model', sa.String(100), nullable=True),
        sa.Column('width', sa.Integer(), default=1024),
        sa.Column('height', sa.Integer(), default=1024),
        sa.Column('style', sa.String(50), nullable=True),
        sa.Column('quality', sa.String(20), nullable=True),
        sa.Column('context_document_ids', sa.Text() if not is_postgresql else postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('context_text', sa.Text(), nullable=True),
        sa.Column('image_url', sa.String(2000), nullable=True),
        sa.Column('storage_path', sa.String(1000), nullable=True),
        sa.Column('thumbnail_path', sa.String(1000), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('status', sa.String(50), default='pending', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('cost_usd', sa.Float(), nullable=True),
        sa.Column('created_by_id', uuid_type(),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('download_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('idx_generated_images_org', 'generated_images', ['organization_id'])
    op.create_index('idx_generated_images_status', 'generated_images', ['status'])
    op.create_index('idx_generated_images_provider', 'generated_images', ['provider'])
    op.create_index('idx_generated_images_created_by', 'generated_images', ['created_by_id'])

    # =========================================================================
    # 13. Set up PostgreSQL Row-Level Security (RLS) for new tables
    # =========================================================================
    if is_postgresql:
        tables_with_rls = [
            'workflows', 'workflow_executions', 'audio_overviews',
            'bot_connections', 'connector_instances', 'synced_resources',
            'budgets', 'virtual_api_keys', 'generated_images'
        ]

        for table in tables_with_rls:
            # Enable RLS
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

            # Create policy for organization isolation
            op.execute(f"""
                CREATE POLICY org_isolation_{table} ON {table}
                FOR ALL
                USING (organization_id IS NULL OR organization_id = current_setting('app.current_org_id', true)::uuid)
                WITH CHECK (organization_id IS NULL OR organization_id = current_setting('app.current_org_id', true)::uuid)
            """)


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # Drop RLS policies if PostgreSQL
    if is_postgresql:
        tables_with_rls = [
            'workflows', 'workflow_executions', 'audio_overviews',
            'bot_connections', 'connector_instances', 'synced_resources',
            'budgets', 'virtual_api_keys', 'generated_images'
        ]
        for table in tables_with_rls:
            op.execute(f"DROP POLICY IF EXISTS org_isolation_{table} ON {table}")
            op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    # Drop tables in reverse order of creation (due to foreign keys)
    op.drop_table('generated_images')
    op.drop_table('virtual_api_keys')
    op.drop_table('budgets')
    op.drop_table('synced_resources')
    op.drop_table('connector_instances')
    op.drop_table('bot_connections')
    op.drop_table('audio_overviews')
    op.drop_table('workflow_node_executions')
    op.drop_table('workflow_executions')
    op.drop_table('workflow_edges')
    op.drop_table('workflow_nodes')
    op.drop_table('workflows')
