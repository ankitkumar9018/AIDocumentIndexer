"""Add multi-organization architecture with RLS support.

Revision ID: 015
Revises: 014
Create Date: 2026-01-09

This migration adds:
1. Organizations table for multi-tenant support
2. Organization settings for feature toggles and provider configuration
3. Adds organization_id to all tenant tables (users, documents, folders, etc.)
4. Sets up PostgreSQL Row-Level Security (RLS) policies for data isolation
5. Adds share links and enhanced permission models
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '015'
down_revision = '20260109_014'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Detect database type
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # =========================================================================
    # 1. Create Organizations table
    # =========================================================================
    op.create_table(
        'organizations',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), unique=True, nullable=False),  # For subdomain
        sa.Column('plan', sa.String(50), default='free', nullable=False),
        sa.Column('settings', sa.Text() if not is_postgresql else postgresql.JSONB(), default='{}'),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('owner_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  nullable=True),  # Will be set after users table updated
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    # Create indexes for organizations
    op.create_index('idx_organizations_slug', 'organizations', ['slug'], unique=True)
    op.create_index('idx_organizations_plan', 'organizations', ['plan'])
    op.create_index('idx_organizations_is_active', 'organizations', ['is_active'])

    # =========================================================================
    # 2. Create a default organization for existing data
    # =========================================================================
    if is_postgresql:
        # Create default organization with a fixed UUID for existing data
        op.execute("""
            INSERT INTO organizations (id, name, slug, plan, settings, is_active, created_at, updated_at)
            VALUES (
                '00000000-0000-0000-0000-000000000001'::uuid,
                'Default Organization',
                'default',
                'business',
                '{"features": {"audioOverviews": {"enabled": true}, "imageGeneration": {"enabled": true}, "workflowBuilder": {"enabled": true}}}',
                true,
                NOW(),
                NOW()
            )
            ON CONFLICT (slug) DO NOTHING
        """)
    else:
        # SQLite version
        op.execute("""
            INSERT OR IGNORE INTO organizations (id, name, slug, plan, settings, is_active, created_at, updated_at)
            VALUES (
                '00000000-0000-0000-0000-000000000001',
                'Default Organization',
                'default',
                'business',
                '{"features": {"audioOverviews": {"enabled": true}, "imageGeneration": {"enabled": true}, "workflowBuilder": {"enabled": true}}}',
                1,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            )
        """)

    # =========================================================================
    # 3. Add organization_id to Users table
    # =========================================================================
    op.add_column('users', sa.Column(
        'organization_id',
        sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
        nullable=True,  # Initially nullable for migration
    ))

    # Update existing users to belong to default organization
    if is_postgresql:
        op.execute("""
            UPDATE users
            SET organization_id = '00000000-0000-0000-0000-000000000001'::uuid
            WHERE organization_id IS NULL
        """)
    else:
        op.execute("""
            UPDATE users
            SET organization_id = '00000000-0000-0000-0000-000000000001'
            WHERE organization_id IS NULL
        """)

    # Add role_in_org column to users
    op.add_column('users', sa.Column(
        'role_in_org',
        sa.String(50),
        default='member',
        nullable=False,
        server_default='member'
    ))

    # Create foreign key and index for organization_id
    op.create_foreign_key(
        'fk_users_organization_id',
        'users', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('idx_users_organization_id', 'users', ['organization_id'])
    op.create_index('idx_users_org_email', 'users', ['organization_id', 'email'], unique=True)

    # =========================================================================
    # 4. Add organization_id to Documents table
    # =========================================================================
    op.add_column('documents', sa.Column(
        'organization_id',
        sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
        nullable=True,
    ))

    # Update existing documents
    if is_postgresql:
        op.execute("""
            UPDATE documents
            SET organization_id = '00000000-0000-0000-0000-000000000001'::uuid
            WHERE organization_id IS NULL
        """)
    else:
        op.execute("""
            UPDATE documents
            SET organization_id = '00000000-0000-0000-0000-000000000001'
            WHERE organization_id IS NULL
        """)

    op.create_foreign_key(
        'fk_documents_organization_id',
        'documents', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('idx_documents_organization_id', 'documents', ['organization_id'])

    # =========================================================================
    # 5. Add organization_id to Folders table
    # =========================================================================
    op.add_column('folders', sa.Column(
        'organization_id',
        sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
        nullable=True,
    ))

    if is_postgresql:
        op.execute("""
            UPDATE folders
            SET organization_id = '00000000-0000-0000-0000-000000000001'::uuid
            WHERE organization_id IS NULL
        """)
    else:
        op.execute("""
            UPDATE folders
            SET organization_id = '00000000-0000-0000-0000-000000000001'
            WHERE organization_id IS NULL
        """)

    op.create_foreign_key(
        'fk_folders_organization_id',
        'folders', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('idx_folders_organization_id', 'folders', ['organization_id'])

    # Add limited_access column for Google Drive-like folder isolation
    op.add_column('folders', sa.Column(
        'limited_access',
        sa.Boolean(),
        default=False,
        nullable=False,
        server_default='false'
    ))

    # =========================================================================
    # 6. Add organization_id to Chunks table
    # =========================================================================
    op.add_column('chunks', sa.Column(
        'organization_id',
        sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
        nullable=True,
    ))

    if is_postgresql:
        op.execute("""
            UPDATE chunks
            SET organization_id = '00000000-0000-0000-0000-000000000001'::uuid
            WHERE organization_id IS NULL
        """)
    else:
        op.execute("""
            UPDATE chunks
            SET organization_id = '00000000-0000-0000-0000-000000000001'
            WHERE organization_id IS NULL
        """)

    op.create_foreign_key(
        'fk_chunks_organization_id',
        'chunks', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('idx_chunks_organization_id', 'chunks', ['organization_id'])

    # =========================================================================
    # 7. Add organization_id to ChatSessions table
    # =========================================================================
    op.add_column('chat_sessions', sa.Column(
        'organization_id',
        sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
        nullable=True,
    ))

    if is_postgresql:
        op.execute("""
            UPDATE chat_sessions
            SET organization_id = '00000000-0000-0000-0000-000000000001'::uuid
            WHERE organization_id IS NULL
        """)
    else:
        op.execute("""
            UPDATE chat_sessions
            SET organization_id = '00000000-0000-0000-0000-000000000001'
            WHERE organization_id IS NULL
        """)

    op.create_foreign_key(
        'fk_chat_sessions_organization_id',
        'chat_sessions', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('idx_chat_sessions_organization_id', 'chat_sessions', ['organization_id'])

    # =========================================================================
    # 8. Add organization_id to FolderPermissions table
    # =========================================================================
    op.add_column('folder_permissions', sa.Column(
        'organization_id',
        sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
        nullable=True,
    ))

    if is_postgresql:
        op.execute("""
            UPDATE folder_permissions
            SET organization_id = '00000000-0000-0000-0000-000000000001'::uuid
            WHERE organization_id IS NULL
        """)
    else:
        op.execute("""
            UPDATE folder_permissions
            SET organization_id = '00000000-0000-0000-0000-000000000001'
            WHERE organization_id IS NULL
        """)

    op.create_foreign_key(
        'fk_folder_permissions_organization_id',
        'folder_permissions', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('idx_folder_permissions_organization_id', 'folder_permissions', ['organization_id'])

    # =========================================================================
    # 9. Create Share Links table for Google Drive-like sharing
    # =========================================================================
    op.create_table(
        'share_links',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True),
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('resource_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  nullable=False),  # Document or Folder ID
        sa.Column('resource_type', sa.String(20), nullable=False),  # 'document' or 'folder'
        sa.Column('token', sa.String(64), unique=True, nullable=False),  # Secure token
        sa.Column('permission_level', sa.String(20), default='viewer', nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('max_uses', sa.Integer(), nullable=True),
        sa.Column('use_count', sa.Integer(), default=0, nullable=False),
        sa.Column('allow_download', sa.Boolean(), default=True, nullable=False),
        sa.Column('require_login', sa.Boolean(), default=False, nullable=False),
        sa.Column('allowed_domains', sa.Text() if not is_postgresql else postgresql.JSONB(), nullable=True),
        sa.Column('created_by_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    op.create_index('idx_share_links_token', 'share_links', ['token'], unique=True)
    op.create_index('idx_share_links_organization_id', 'share_links', ['organization_id'])
    op.create_index('idx_share_links_resource', 'share_links', ['resource_id', 'resource_type'])
    op.create_index('idx_share_links_created_by', 'share_links', ['created_by_id'])

    # =========================================================================
    # 10. Create File Activity table for tracking
    # =========================================================================
    op.create_table(
        'file_activities',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True),
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('resource_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  nullable=False),
        sa.Column('resource_type', sa.String(20), nullable=False),
        sa.Column('action', sa.String(50), nullable=False),  # view, edit, share, download, move, delete
        sa.Column('actor_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('actor_name', sa.String(255), nullable=True),  # Denormalized
        sa.Column('details', sa.Text() if not is_postgresql else postgresql.JSONB(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_index('idx_file_activities_organization_id', 'file_activities', ['organization_id'])
    op.create_index('idx_file_activities_resource', 'file_activities', ['resource_id', 'resource_type'])
    op.create_index('idx_file_activities_actor', 'file_activities', ['actor_id'])
    op.create_index('idx_file_activities_action', 'file_activities', ['action'])
    op.create_index('idx_file_activities_created_at', 'file_activities', ['created_at'])

    # =========================================================================
    # 11. Create File Versions table for version history
    # =========================================================================
    op.create_table(
        'file_versions',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True),
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('version_number', sa.Integer(), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('file_hash', sa.String(64), nullable=False),
        sa.Column('storage_path', sa.String(1000), nullable=False),
        sa.Column('created_by_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_by_name', sa.String(255), nullable=True),  # Denormalized
        sa.Column('change_summary', sa.Text(), nullable=True),
        sa.Column('is_current', sa.Boolean(), default=False, nullable=False),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_index('idx_file_versions_organization_id', 'file_versions', ['organization_id'])
    op.create_index('idx_file_versions_document_id', 'file_versions', ['document_id'])
    op.create_index('idx_file_versions_is_current', 'file_versions', ['document_id', 'is_current'])

    # =========================================================================
    # 12. Create Feature Flags table for admin-controlled features
    # =========================================================================
    op.create_table(
        'feature_flags',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True),
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('feature_name', sa.String(100), nullable=False),
        sa.Column('enabled', sa.Boolean(), default=True, nullable=False),
        sa.Column('config', sa.Text() if not is_postgresql else postgresql.JSONB(), nullable=True),
        sa.Column('enabled_for_tiers', sa.Text() if not is_postgresql else postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    op.create_index('idx_feature_flags_organization_id', 'feature_flags', ['organization_id'])
    op.create_index('idx_feature_flags_org_feature', 'feature_flags',
                    ['organization_id', 'feature_name'], unique=True)

    # =========================================================================
    # 13. Set up PostgreSQL Row-Level Security (RLS) - PostgreSQL only
    # =========================================================================
    if is_postgresql:
        # Enable RLS on tenant tables
        tables_with_rls = [
            'users', 'documents', 'folders', 'chunks', 'chat_sessions',
            'folder_permissions', 'share_links', 'file_activities', 'file_versions',
            'feature_flags'
        ]

        for table in tables_with_rls:
            # Enable RLS
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

            # Create policy for organization isolation
            # Note: The actual policy uses current_setting('app.current_org_id')
            # which must be set by the application before each request
            op.execute(f"""
                CREATE POLICY org_isolation_{table} ON {table}
                FOR ALL
                USING (organization_id = current_setting('app.current_org_id', true)::uuid)
                WITH CHECK (organization_id = current_setting('app.current_org_id', true)::uuid)
            """)

        # Create a bypass policy for the app user to work without RLS during migrations
        # In production, use a non-superuser role with NOBYPASSRLS
        op.execute("""
            -- Note: In production, create a dedicated app_user role:
            -- CREATE ROLE app_user WITH LOGIN PASSWORD 'xxx' NOBYPASSRLS;
            -- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
        """)

    # =========================================================================
    # 14. Update owner_id foreign key on organizations table
    # =========================================================================
    op.create_foreign_key(
        'fk_organizations_owner_id',
        'organizations', 'users',
        ['owner_id'], ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # Drop RLS policies if PostgreSQL
    if is_postgresql:
        tables_with_rls = [
            'users', 'documents', 'folders', 'chunks', 'chat_sessions',
            'folder_permissions', 'share_links', 'file_activities', 'file_versions',
            'feature_flags'
        ]
        for table in tables_with_rls:
            op.execute(f"DROP POLICY IF EXISTS org_isolation_{table} ON {table}")
            op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    # Drop new tables
    op.drop_table('feature_flags')
    op.drop_table('file_versions')
    op.drop_table('file_activities')
    op.drop_table('share_links')

    # Remove organization_id from existing tables
    op.drop_constraint('fk_folder_permissions_organization_id', 'folder_permissions', type_='foreignkey')
    op.drop_index('idx_folder_permissions_organization_id', 'folder_permissions')
    op.drop_column('folder_permissions', 'organization_id')

    op.drop_constraint('fk_chat_sessions_organization_id', 'chat_sessions', type_='foreignkey')
    op.drop_index('idx_chat_sessions_organization_id', 'chat_sessions')
    op.drop_column('chat_sessions', 'organization_id')

    op.drop_constraint('fk_chunks_organization_id', 'chunks', type_='foreignkey')
    op.drop_index('idx_chunks_organization_id', 'chunks')
    op.drop_column('chunks', 'organization_id')

    op.drop_column('folders', 'limited_access')
    op.drop_constraint('fk_folders_organization_id', 'folders', type_='foreignkey')
    op.drop_index('idx_folders_organization_id', 'folders')
    op.drop_column('folders', 'organization_id')

    op.drop_constraint('fk_documents_organization_id', 'documents', type_='foreignkey')
    op.drop_index('idx_documents_organization_id', 'documents')
    op.drop_column('documents', 'organization_id')

    op.drop_index('idx_users_org_email', 'users')
    op.drop_index('idx_users_organization_id', 'users')
    op.drop_constraint('fk_users_organization_id', 'users', type_='foreignkey')
    op.drop_column('users', 'role_in_org')
    op.drop_column('users', 'organization_id')

    # Drop organizations table
    op.drop_constraint('fk_organizations_owner_id', 'organizations', type_='foreignkey')
    op.drop_index('idx_organizations_is_active', 'organizations')
    op.drop_index('idx_organizations_plan', 'organizations')
    op.drop_index('idx_organizations_slug', 'organizations')
    op.drop_table('organizations')
