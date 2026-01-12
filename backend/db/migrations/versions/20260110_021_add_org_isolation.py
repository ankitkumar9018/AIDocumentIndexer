"""Add organization isolation to all tables.

Revision ID: 021
Revises: 020
Create Date: 2026-01-10

This migration:
1. Creates the default "Mandala Labs" organization
2. Adds organization_id to all relevant tables (documents, chunks, chat_sessions, etc.)
3. Assigns all existing data to the default organization
4. Assigns all existing users to the default organization as members
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime
import uuid

# revision identifiers
revision = '021'
down_revision = '020'
branch_labels = None
depends_on = None

# Default organization ID (consistent UUID for Mandala Labs)
DEFAULT_ORG_ID = 'a0000000-0000-0000-0000-000000000001'
DEFAULT_ORG_NAME = 'Mandala Labs'
DEFAULT_ORG_SLUG = 'mandala-labs'


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Create the default "Mandala Labs" organization
    print(f"Creating default organization: {DEFAULT_ORG_NAME}")
    try:
        conn.execute(sa.text("""
            INSERT INTO organizations (id, name, slug, plan, is_active, max_users, max_storage_gb, created_at, updated_at)
            VALUES (:id, :name, :slug, 'enterprise', 1, 1000, 1000, :now, :now)
        """), {
            'id': DEFAULT_ORG_ID,
            'name': DEFAULT_ORG_NAME,
            'slug': DEFAULT_ORG_SLUG,
            'now': datetime.utcnow().isoformat()
        })
        print(f"  Created organization: {DEFAULT_ORG_NAME}")
    except Exception as e:
        print(f"  Organization may already exist: {e}")

    # 2. Add organization_id to documents table
    print("Adding organization_id to documents table...")
    try:
        op.add_column('documents', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE documents SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to documents")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 3. Add organization_id to chunks table
    print("Adding organization_id to chunks table...")
    try:
        op.add_column('chunks', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE chunks SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to chunks")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 4. Add organization_id to chat_sessions table
    print("Adding organization_id to chat_sessions table...")
    try:
        op.add_column('chat_sessions', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE chat_sessions SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to chat_sessions")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 5. Add organization_id to chat_messages table
    print("Adding organization_id to chat_messages table...")
    try:
        op.add_column('chat_messages', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE chat_messages SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to chat_messages")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 6. Add organization_id to entities table
    print("Adding organization_id to entities table...")
    try:
        op.add_column('entities', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE entities SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to entities")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 7. Add organization_id to entity_relations table
    print("Adding organization_id to entity_relations table...")
    try:
        op.add_column('entity_relations', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE entity_relations SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to entity_relations")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 8. Add organization_id to folders table
    print("Adding organization_id to folders table...")
    try:
        op.add_column('folders', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE folders SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to folders")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 9. Add organization_id to workflows table
    print("Adding organization_id to workflows table...")
    try:
        op.add_column('workflows', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE workflows SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to workflows")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 10. Add organization_id to audio_overviews table
    print("Adding organization_id to audio_overviews table...")
    try:
        op.add_column('audio_overviews', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE audio_overviews SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to audio_overviews")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 11. Add organization_id to scraped_content table
    print("Adding organization_id to scraped_content table...")
    try:
        op.add_column('scraped_content', sa.Column('organization_id', sa.String(36), nullable=True))
        conn.execute(sa.text(f"UPDATE scraped_content SET organization_id = '{DEFAULT_ORG_ID}' WHERE organization_id IS NULL"))
        print("  Added organization_id to scraped_content")
    except Exception as e:
        print(f"  Column may already exist: {e}")

    # 12. Update all users to have current_organization_id set to default org
    print("Setting current_organization_id for all users...")
    try:
        conn.execute(sa.text(f"""
            UPDATE users SET current_organization_id = '{DEFAULT_ORG_ID}'
            WHERE current_organization_id IS NULL
        """))
        print("  Updated users with default organization")
    except Exception as e:
        print(f"  Error updating users: {e}")

    # 13. Add all users as members of the default organization
    print("Adding all users as members of default organization...")
    try:
        # Get all users
        result = conn.execute(sa.text("SELECT id, email, is_superadmin FROM users"))
        users = result.fetchall()

        for user in users:
            user_id = user[0]
            email = user[1]
            is_superadmin = user[2]

            # Check if membership already exists
            existing = conn.execute(sa.text("""
                SELECT id FROM organization_members
                WHERE user_id = :user_id AND organization_id = :org_id
            """), {'user_id': user_id, 'org_id': DEFAULT_ORG_ID}).fetchone()

            if not existing:
                # Superadmins are owners, others are members
                role = 'owner' if is_superadmin else 'member'
                member_id = str(uuid.uuid4())

                conn.execute(sa.text("""
                    INSERT INTO organization_members (id, organization_id, user_id, role, joined_at)
                    VALUES (:id, :org_id, :user_id, :role, :now)
                """), {
                    'id': member_id,
                    'org_id': DEFAULT_ORG_ID,
                    'user_id': user_id,
                    'role': role,
                    'now': datetime.utcnow().isoformat()
                })
                print(f"  Added {email} as {role} of {DEFAULT_ORG_NAME}")
    except Exception as e:
        print(f"  Error adding members: {e}")

    print("Migration complete!")


def downgrade() -> None:
    # Remove organization_id columns (data loss!)
    op.drop_column('documents', 'organization_id')
    op.drop_column('chunks', 'organization_id')
    op.drop_column('chat_sessions', 'organization_id')
    op.drop_column('chat_messages', 'organization_id')
    op.drop_column('entities', 'organization_id')
    op.drop_column('entity_relations', 'organization_id')
    op.drop_column('folders', 'organization_id')
    op.drop_column('workflows', 'organization_id')
    op.drop_column('audio_overviews', 'organization_id')
    op.drop_column('scraped_content', 'organization_id')
