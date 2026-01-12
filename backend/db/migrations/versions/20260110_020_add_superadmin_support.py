"""Add superadmin support.

Revision ID: 020
Revises: 019
Create Date: 2026-01-10

This migration adds:
1. is_superadmin flag to users table
2. current_organization_id to users table for org context switching
3. Sets admin@example.com as superadmin
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '020'
down_revision = '019'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Detect database type
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # =========================================================================
    # 1. Add is_superadmin column to users
    # =========================================================================
    try:
        op.add_column('users', sa.Column(
            'is_superadmin',
            sa.Boolean(),
            default=False,
            nullable=False,
            server_default='false'
        ))
    except Exception:
        pass  # Column may already exist

    try:
        op.create_index('idx_users_is_superadmin', 'users', ['is_superadmin'])
    except Exception:
        pass  # Index may already exist

    # =========================================================================
    # 2. Add current_organization_id to users for context switching
    # =========================================================================
    try:
        op.add_column('users', sa.Column(
            'current_organization_id',
            sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
            nullable=True,
        ))
    except Exception:
        pass  # Column may already exist

    # Only add foreign key on PostgreSQL (SQLite doesn't support ALTER constraints)
    if is_postgresql:
        try:
            op.create_foreign_key(
                'fk_users_current_organization_id',
                'users', 'organizations',
                ['current_organization_id'], ['id'],
                ondelete='SET NULL'
            )
        except Exception:
            pass  # FK may already exist

    # =========================================================================
    # 3. Add max_users and max_storage_gb to organizations if not exists
    # =========================================================================
    try:
        op.add_column('organizations', sa.Column(
            'max_users',
            sa.Integer(),
            default=5,
            nullable=False,
            server_default='5'
        ))
    except Exception:
        pass  # Column might already exist

    try:
        op.add_column('organizations', sa.Column(
            'max_storage_gb',
            sa.Integer(),
            default=10,
            nullable=False,
            server_default='10'
        ))
    except Exception:
        pass  # Column might already exist

    # =========================================================================
    # 4. Create organization_members table if not exists
    # =========================================================================
    try:
        op.create_table(
            'organization_members',
            sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                      primary_key=True),
            sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                      nullable=False),
            sa.Column('user_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                      nullable=False),
            sa.Column('role', sa.String(20), default='member', nullable=False, server_default='member'),
            sa.Column('joined_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index('idx_org_members_org_id', 'organization_members', ['organization_id'])
        op.create_index('idx_org_members_user_id', 'organization_members', ['user_id'])
        op.create_index('ix_org_members_org_user', 'organization_members',
                        ['organization_id', 'user_id'], unique=True)
    except Exception:
        pass  # Table might already exist

    # =========================================================================
    # 5. Set admin@example.com as superadmin
    # =========================================================================
    try:
        if is_postgresql:
            op.execute("""
                UPDATE users
                SET is_superadmin = true
                WHERE email = 'admin@example.com'
            """)
        else:
            op.execute("""
                UPDATE users
                SET is_superadmin = 1
                WHERE email = 'admin@example.com'
            """)
    except Exception:
        pass  # May fail if no matching user


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # Remove superadmin status
    try:
        if is_postgresql:
            op.execute("UPDATE users SET is_superadmin = false")
        else:
            op.execute("UPDATE users SET is_superadmin = 0")
    except Exception:
        pass

    # Drop organization_members table
    try:
        op.drop_index('ix_org_members_org_user', 'organization_members')
        op.drop_index('idx_org_members_user_id', 'organization_members')
        op.drop_index('idx_org_members_org_id', 'organization_members')
        op.drop_table('organization_members')
    except Exception:
        pass

    # Drop columns
    if is_postgresql:
        try:
            op.drop_constraint('fk_users_current_organization_id', 'users', type_='foreignkey')
        except Exception:
            pass

    try:
        op.drop_column('users', 'current_organization_id')
    except Exception:
        pass

    try:
        op.drop_index('idx_users_is_superadmin', 'users')
        op.drop_column('users', 'is_superadmin')
    except Exception:
        pass

    try:
        op.drop_column('organizations', 'max_users')
        op.drop_column('organizations', 'max_storage_gb')
    except Exception:
        pass
