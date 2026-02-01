"""Add Skills and SkillExecution tables

Revision ID: 029_add_skills_tables
Revises: 028_add_dspy_tables
Create Date: 2026-01-31

Adds database persistence for Skills Marketplace feature.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '029_add_skills_tables'
down_revision: Union[str, None] = '028_add_dspy_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create skills table
    op.create_table(
        'skills',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('skill_key', sa.String(100), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(50), default='custom', index=True),
        sa.Column('icon', sa.String(50), default='zap'),
        sa.Column('tags', sa.Text(), nullable=True),  # JSON array stored as text for SQLite compatibility
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('inputs', sa.Text(), nullable=True),  # JSON array
        sa.Column('outputs', sa.Text(), nullable=True),  # JSON array
        sa.Column('config', sa.Text(), nullable=True),  # JSON object
        sa.Column('is_public', sa.Boolean(), default=False),
        sa.Column('is_builtin', sa.Boolean(), default=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('version', sa.String(20), default='1.0.0'),
        sa.Column('use_count', sa.Integer(), default=0),
        sa.Column('avg_execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create indexes for skills
    op.create_index('idx_skills_user', 'skills', ['user_id'])
    op.create_index('idx_skills_org', 'skills', ['organization_id'])
    op.create_index('idx_skills_key', 'skills', ['skill_key'])
    op.create_index('idx_skills_category', 'skills', ['category'])
    op.create_index('idx_skills_public', 'skills', ['is_public'])
    op.create_index('idx_skills_builtin', 'skills', ['is_builtin'])

    # Create skill_executions table
    op.create_table(
        'skill_executions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('skill_id', sa.String(36), sa.ForeignKey('skills.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('inputs', sa.Text(), nullable=True),  # JSON object
        sa.Column('output', sa.Text(), nullable=True),  # JSON object
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('model_used', sa.String(100), nullable=True),
        sa.Column('provider_used', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create indexes for skill_executions
    op.create_index('idx_skill_executions_skill', 'skill_executions', ['skill_id'])
    op.create_index('idx_skill_executions_user', 'skill_executions', ['user_id'])
    op.create_index('idx_skill_executions_status', 'skill_executions', ['status'])
    op.create_index('idx_skill_executions_created', 'skill_executions', ['created_at'])


def downgrade() -> None:
    # Drop skill_executions indexes
    op.drop_index('idx_skill_executions_created', table_name='skill_executions')
    op.drop_index('idx_skill_executions_status', table_name='skill_executions')
    op.drop_index('idx_skill_executions_user', table_name='skill_executions')
    op.drop_index('idx_skill_executions_skill', table_name='skill_executions')

    # Drop skill_executions table
    op.drop_table('skill_executions')

    # Drop skills indexes
    op.drop_index('idx_skills_builtin', table_name='skills')
    op.drop_index('idx_skills_public', table_name='skills')
    op.drop_index('idx_skills_category', table_name='skills')
    op.drop_index('idx_skills_key', table_name='skills')
    op.drop_index('idx_skills_org', table_name='skills')
    op.drop_index('idx_skills_user', table_name='skills')

    # Drop skills table
    op.drop_table('skills')
