"""Add chat privacy controls and user-level RLS.

Revision ID: 016
Revises: 015
Create Date: 2026-01-10

This migration adds:
1. User privacy preferences table for controlling chat history/memory
2. User-level RLS policies for chat_sessions (in addition to org-level)
3. Admin override policies for support access
4. Chat history retention settings
5. Memory isolation controls

Based on best practices from:
- OpenAI ChatGPT memory controls
- PostgreSQL RLS for multi-tenant isolation
- GDPR/privacy compliance requirements
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '016a'
down_revision = '015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # =========================================================================
    # 1. Create User Privacy Preferences Table
    # =========================================================================
    op.create_table(
        'user_privacy_preferences',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),
        sa.Column('user_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'),
                  nullable=False, unique=True),

        # Chat History Controls (like ChatGPT's controls)
        sa.Column('chat_history_enabled', sa.Boolean(), default=True, nullable=False,
                  server_default='true',
                  comment='Whether to save chat history'),
        sa.Column('chat_history_visible_to_admins', sa.Boolean(), default=True, nullable=False,
                  server_default='true',
                  comment='Whether org admins can view chat history (for support)'),

        # Memory Controls (like ChatGPT memory feature)
        sa.Column('memory_enabled', sa.Boolean(), default=True, nullable=False,
                  server_default='true',
                  comment='Whether AI can remember context across sessions'),
        sa.Column('memory_include_chat_history', sa.Boolean(), default=True, nullable=False,
                  server_default='true',
                  comment='Whether memory draws from past conversations'),
        sa.Column('memory_include_saved_facts', sa.Boolean(), default=True, nullable=False,
                  server_default='true',
                  comment='Whether to use explicitly saved memories'),

        # Data Training Controls
        sa.Column('allow_training_data', sa.Boolean(), default=False, nullable=False,
                  server_default='false',
                  comment='Whether conversations can be used for model improvement'),

        # Retention Settings
        sa.Column('auto_delete_history_days', sa.Integer(), nullable=True,
                  comment='Auto-delete chat history after N days (null = never)'),
        sa.Column('auto_delete_memory_days', sa.Integer(), nullable=True,
                  comment='Auto-delete semantic memories after N days (null = never)'),

        # Export/Portability
        sa.Column('last_export_at', sa.DateTime(timezone=True), nullable=True,
                  comment='Last time user exported their data'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    # Indexes
    op.create_index('idx_user_privacy_preferences_user_id', 'user_privacy_preferences', ['user_id'], unique=True)

    # =========================================================================
    # 2. Create Semantic Memory Store Table (for user-isolated long-term memory)
    # =========================================================================
    op.create_table(
        'user_semantic_memories',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),
        sa.Column('user_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        # Memory content
        sa.Column('content', sa.Text(), nullable=False,
                  comment='The memory content (fact, preference, etc.)'),
        sa.Column('memory_type', sa.String(50), nullable=False, default='fact',
                  comment='Type: fact, preference, entity, insight'),
        sa.Column('importance', sa.String(20), nullable=False, default='medium',
                  comment='Importance: low, medium, high, critical'),

        # Embedding for semantic search
        sa.Column('embedding', postgresql.ARRAY(sa.Float()) if is_postgresql else sa.Text(),
                  nullable=True,
                  comment='Vector embedding for semantic retrieval'),

        # Metadata
        sa.Column('metadata', sa.Text() if not is_postgresql else postgresql.JSONB(),
                  nullable=True,
                  comment='Additional metadata (source, context, etc.)'),
        sa.Column('source_session_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  nullable=True,
                  comment='Chat session that generated this memory'),

        # Access tracking
        sa.Column('access_count', sa.Integer(), default=0, nullable=False),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When this memory should be auto-deleted'),
    )

    # Indexes for semantic memories
    op.create_index('idx_user_semantic_memories_user_id', 'user_semantic_memories', ['user_id'])
    op.create_index('idx_user_semantic_memories_org_id', 'user_semantic_memories', ['organization_id'])
    op.create_index('idx_user_semantic_memories_type', 'user_semantic_memories', ['memory_type'])
    op.create_index('idx_user_semantic_memories_importance', 'user_semantic_memories', ['importance'])
    op.create_index('idx_user_semantic_memories_expires', 'user_semantic_memories', ['expires_at'])

    # =========================================================================
    # 3. Create Chat Data Export Requests Table
    # =========================================================================
    op.create_table(
        'chat_data_export_requests',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),
        sa.Column('user_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        # Export details
        sa.Column('export_type', sa.String(50), nullable=False,
                  comment='Type: chat_history, memories, all_data'),
        sa.Column('status', sa.String(50), nullable=False, default='pending',
                  comment='Status: pending, processing, completed, failed, expired'),
        sa.Column('format', sa.String(20), nullable=False, default='json',
                  comment='Export format: json, csv, zip'),

        # File location (when completed)
        sa.Column('file_path', sa.String(512), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('download_count', sa.Integer(), default=0, nullable=False),

        # Timestamps
        sa.Column('requested_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When the export file expires'),
        sa.Column('error_message', sa.Text(), nullable=True),
    )

    op.create_index('idx_chat_data_export_requests_user_id', 'chat_data_export_requests', ['user_id'])
    op.create_index('idx_chat_data_export_requests_status', 'chat_data_export_requests', ['status'])

    # =========================================================================
    # 4. Add metadata column to chat_sessions for privacy tracking
    # =========================================================================
    # Check if column exists first
    try:
        op.add_column('chat_sessions', sa.Column(
            'metadata',
            sa.Text() if not is_postgresql else postgresql.JSONB(),
            nullable=True,
            comment='Session metadata including summaries, preferences, etc.'
        ))
    except Exception:
        pass  # Column might already exist

    # Add is_temporary column for temporary/incognito chats
    try:
        op.add_column('chat_sessions', sa.Column(
            'is_temporary',
            sa.Boolean(),
            default=False,
            nullable=False,
            server_default='false',
            comment='Temporary chats are auto-deleted and not used for memory'
        ))
    except Exception:
        pass

    # Add retention_override column
    try:
        op.add_column('chat_sessions', sa.Column(
            'retention_days',
            sa.Integer(),
            nullable=True,
            comment='Override retention period for this session (null = use user default)'
        ))
    except Exception:
        pass

    # =========================================================================
    # 5. Set up User-Level RLS Policies (PostgreSQL only)
    # =========================================================================
    if is_postgresql:
        # Create user-level RLS for chat_sessions
        # This is IN ADDITION to the org-level policy from migration 015
        # Users can only see their own chats, unless they're an admin

        # First, drop the existing org-only policy if it exists
        op.execute("DROP POLICY IF EXISTS org_isolation_chat_sessions ON chat_sessions")

        # Create combined org + user isolation policy
        # Users can see:
        # 1. Their own sessions (user_id match)
        # 2. If they're an org admin AND the session owner allows admin viewing
        op.execute("""
            CREATE POLICY chat_session_user_isolation ON chat_sessions
            FOR ALL
            USING (
                -- Must be in the same organization
                organization_id = current_setting('app.current_org_id', true)::uuid
                AND (
                    -- User can see their own sessions
                    user_id = current_setting('app.current_user_id', true)::uuid
                    OR
                    -- Or admin can see if user allows it (checked via join in app layer)
                    current_setting('app.is_admin', true)::boolean = true
                )
            )
            WITH CHECK (
                organization_id = current_setting('app.current_org_id', true)::uuid
                AND user_id = current_setting('app.current_user_id', true)::uuid
            )
        """)

        # Enable RLS on new tables
        op.execute("ALTER TABLE user_privacy_preferences ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE user_semantic_memories ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE chat_data_export_requests ENABLE ROW LEVEL SECURITY")

        # Create policies for user_privacy_preferences (user can only see/edit their own)
        op.execute("""
            CREATE POLICY user_privacy_own ON user_privacy_preferences
            FOR ALL
            USING (user_id = current_setting('app.current_user_id', true)::uuid)
            WITH CHECK (user_id = current_setting('app.current_user_id', true)::uuid)
        """)

        # Create policies for user_semantic_memories (strict user isolation)
        op.execute("""
            CREATE POLICY user_memories_isolation ON user_semantic_memories
            FOR ALL
            USING (
                organization_id = current_setting('app.current_org_id', true)::uuid
                AND user_id = current_setting('app.current_user_id', true)::uuid
            )
            WITH CHECK (
                organization_id = current_setting('app.current_org_id', true)::uuid
                AND user_id = current_setting('app.current_user_id', true)::uuid
            )
        """)

        # Create policies for chat_data_export_requests
        op.execute("""
            CREATE POLICY user_exports_own ON chat_data_export_requests
            FOR ALL
            USING (user_id = current_setting('app.current_user_id', true)::uuid)
            WITH CHECK (user_id = current_setting('app.current_user_id', true)::uuid)
        """)

    # =========================================================================
    # 6. Create function for auto-cleanup of expired data (PostgreSQL only)
    # =========================================================================
    if is_postgresql:
        # Function to clean up expired temporary chats
        op.execute("""
            CREATE OR REPLACE FUNCTION cleanup_expired_chat_data()
            RETURNS void AS $$
            BEGIN
                -- Delete temporary chats older than 30 days (OpenAI-style)
                DELETE FROM chat_sessions
                WHERE is_temporary = true
                AND created_at < NOW() - INTERVAL '30 days';

                -- Delete sessions based on user retention preferences
                DELETE FROM chat_sessions cs
                USING user_privacy_preferences upp
                WHERE cs.user_id = upp.user_id
                AND upp.auto_delete_history_days IS NOT NULL
                AND cs.created_at < NOW() - (upp.auto_delete_history_days || ' days')::interval;

                -- Delete expired semantic memories
                DELETE FROM user_semantic_memories
                WHERE expires_at IS NOT NULL AND expires_at < NOW();

                -- Delete expired export files
                UPDATE chat_data_export_requests
                SET status = 'expired', file_path = NULL
                WHERE expires_at IS NOT NULL AND expires_at < NOW() AND status = 'completed';
            END;
            $$ LANGUAGE plpgsql;
        """)

        # Note: Schedule this function with pg_cron or external scheduler:
        # SELECT cron.schedule('cleanup_chat_data', '0 3 * * *', 'SELECT cleanup_expired_chat_data()');


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # Drop RLS policies
    if is_postgresql:
        op.execute("DROP POLICY IF EXISTS chat_session_user_isolation ON chat_sessions")
        op.execute("DROP POLICY IF EXISTS user_privacy_own ON user_privacy_preferences")
        op.execute("DROP POLICY IF EXISTS user_memories_isolation ON user_semantic_memories")
        op.execute("DROP POLICY IF EXISTS user_exports_own ON chat_data_export_requests")

        # Re-create the original org-only policy
        op.execute("""
            CREATE POLICY org_isolation_chat_sessions ON chat_sessions
            FOR ALL
            USING (organization_id = current_setting('app.current_org_id', true)::uuid)
            WITH CHECK (organization_id = current_setting('app.current_org_id', true)::uuid)
        """)

        # Disable RLS on new tables
        op.execute("ALTER TABLE user_privacy_preferences DISABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE user_semantic_memories DISABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE chat_data_export_requests DISABLE ROW LEVEL SECURITY")

        # Drop cleanup function
        op.execute("DROP FUNCTION IF EXISTS cleanup_expired_chat_data()")

    # Drop columns from chat_sessions
    try:
        op.drop_column('chat_sessions', 'retention_days')
    except Exception:
        pass
    try:
        op.drop_column('chat_sessions', 'is_temporary')
    except Exception:
        pass
    try:
        op.drop_column('chat_sessions', 'metadata')
    except Exception:
        pass

    # Drop tables
    op.drop_index('idx_chat_data_export_requests_status', 'chat_data_export_requests')
    op.drop_index('idx_chat_data_export_requests_user_id', 'chat_data_export_requests')
    op.drop_table('chat_data_export_requests')

    op.drop_index('idx_user_semantic_memories_expires', 'user_semantic_memories')
    op.drop_index('idx_user_semantic_memories_importance', 'user_semantic_memories')
    op.drop_index('idx_user_semantic_memories_type', 'user_semantic_memories')
    op.drop_index('idx_user_semantic_memories_org_id', 'user_semantic_memories')
    op.drop_index('idx_user_semantic_memories_user_id', 'user_semantic_memories')
    op.drop_table('user_semantic_memories')

    op.drop_index('idx_user_privacy_preferences_user_id', 'user_privacy_preferences')
    op.drop_table('user_privacy_preferences')
