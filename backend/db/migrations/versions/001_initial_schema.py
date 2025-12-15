"""Initial schema with pgvector support

Revision ID: 001_initial
Revises: None
Create Date: 2024-12-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create access_tiers table
    op.create_table(
        'access_tiers',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(50), unique=True, nullable=False),
        sa.Column('level', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('permissions', postgresql.JSONB()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(255)),
        sa.Column('full_name', sa.String(255)),
        sa.Column('role', sa.String(50), server_default='user'),
        sa.Column('access_tier_id', sa.Integer(), sa.ForeignKey('access_tiers.id')),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('is_verified', sa.Boolean(), server_default='false'),
        sa.Column('avatar_url', sa.String(512)),
        sa.Column('preferences', postgresql.JSONB(), server_default='{}'),
        sa.Column('last_login', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('original_filename', sa.String(255)),
        sa.Column('file_path', sa.String(1024), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=False, index=True),
        sa.Column('mime_type', sa.String(128)),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('file_hash', sa.String(64), unique=True, nullable=False, index=True),
        sa.Column('collection', sa.String(128), index=True),
        sa.Column('access_tier', sa.Integer(), server_default='1', index=True),
        sa.Column('status', sa.String(50), server_default='pending', index=True),
        sa.Column('page_count', sa.Integer()),
        sa.Column('word_count', sa.Integer()),
        sa.Column('chunk_count', sa.Integer(), server_default='0'),
        sa.Column('language', sa.String(10)),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('processing_error', sa.Text()),
        sa.Column('uploaded_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), index=True),
        sa.Column('processed_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create chunks table with vector embedding
    op.create_table(
        'chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('chunk_hash', sa.String(32)),
        sa.Column('page_number', sa.Integer()),
        sa.Column('slide_number', sa.Integer()),
        sa.Column('section', sa.String(255)),
        sa.Column('char_count', sa.Integer()),
        sa.Column('word_count', sa.Integer()),
        sa.Column('access_tier', sa.Integer(), server_default='1', index=True),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Add vector column (1536 dimensions for OpenAI embeddings)
    op.execute('ALTER TABLE chunks ADD COLUMN embedding vector(1536)')

    # Create vector index for similarity search
    op.execute('''
        CREATE INDEX chunks_embedding_idx ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    ''')

    # Create chat_sessions table
    op.create_table(
        'chat_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('title', sa.String(255)),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('sources', postgresql.JSONB(), server_default='[]'),
        sa.Column('tokens_used', sa.Integer()),
        sa.Column('model', sa.String(64)),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Create scraped_content table
    op.create_table(
        'scraped_content',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('url', sa.String(2048), nullable=False, index=True),
        sa.Column('title', sa.String(512)),
        sa.Column('content', sa.Text()),
        sa.Column('content_hash', sa.String(64)),
        sa.Column('stored_permanently', sa.Boolean(), server_default='false'),
        sa.Column('access_tier', sa.Integer(), server_default='1'),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('scraped_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Add vector column for scraped content
    op.execute('ALTER TABLE scraped_content ADD COLUMN embedding vector(1536)')

    # Create audit_log table
    op.create_table(
        'audit_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), index=True),
        sa.Column('action', sa.String(50), nullable=False, index=True),
        sa.Column('resource_type', sa.String(50), index=True),
        sa.Column('resource_id', sa.String(64)),
        sa.Column('details', postgresql.JSONB()),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.String(512)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
    )

    # Create processing_queue table
    op.create_table(
        'processing_queue',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), index=True),
        sa.Column('status', sa.String(50), server_default='pending', index=True),
        sa.Column('priority', sa.Integer(), server_default='0'),
        sa.Column('progress', sa.Integer(), server_default='0'),
        sa.Column('current_step', sa.String(100)),
        sa.Column('error_message', sa.Text()),
        sa.Column('retry_count', sa.Integer(), server_default='0'),
        sa.Column('worker_id', sa.String(64)),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Insert default access tiers
    op.execute("""
        INSERT INTO access_tiers (name, level, description, permissions) VALUES
        ('intern', 10, 'Basic access for interns', '{"can_upload": false, "can_delete": false}'::jsonb),
        ('staff', 30, 'Standard staff access', '{"can_upload": true, "can_delete": false}'::jsonb),
        ('manager', 50, 'Manager-level access', '{"can_upload": true, "can_delete": true}'::jsonb),
        ('executive', 80, 'Executive-level access', '{"can_upload": true, "can_delete": true, "can_manage_users": false}'::jsonb),
        ('admin', 100, 'Full administrative access', '{"can_upload": true, "can_delete": true, "can_manage_users": true}'::jsonb)
    """)

    # Create Row Level Security policies
    op.execute("""
        -- Enable RLS on documents
        ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

        -- Enable RLS on chunks
        ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

        -- Policy: Users can only see documents at or below their access tier
        CREATE POLICY documents_access_policy ON documents
            FOR SELECT
            USING (
                access_tier <= (
                    SELECT COALESCE(at.level, 100)
                    FROM users u
                    LEFT JOIN access_tiers at ON u.access_tier_id = at.id
                    WHERE u.id = current_setting('app.current_user_id', true)::uuid
                )
            );

        -- Policy: Users can only see chunks at or below their access tier
        CREATE POLICY chunks_access_policy ON chunks
            FOR SELECT
            USING (
                access_tier <= (
                    SELECT COALESCE(at.level, 100)
                    FROM users u
                    LEFT JOIN access_tiers at ON u.access_tier_id = at.id
                    WHERE u.id = current_setting('app.current_user_id', true)::uuid
                )
            );
    """)


def downgrade() -> None:
    # Drop RLS policies
    op.execute('DROP POLICY IF EXISTS documents_access_policy ON documents')
    op.execute('DROP POLICY IF EXISTS chunks_access_policy ON chunks')
    op.execute('ALTER TABLE documents DISABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE chunks DISABLE ROW LEVEL SECURITY')

    # Drop tables in reverse order
    op.drop_table('processing_queue')
    op.drop_table('audit_log')
    op.drop_table('scraped_content')
    op.drop_table('chat_messages')
    op.drop_table('chat_sessions')
    op.drop_table('chunks')
    op.drop_table('documents')
    op.drop_table('users')
    op.drop_table('access_tiers')

    # Drop pgvector extension (optional - might be used by other DBs)
    # op.execute('DROP EXTENSION IF EXISTS vector')
