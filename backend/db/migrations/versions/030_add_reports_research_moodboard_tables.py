"""Add Reports, ResearchResults, and MoodBoards tables

Revision ID: 030_add_reports_research_moodboard
Revises: 029_add_skills_tables
Create Date: 2026-01-31

Adds database persistence for Reports (Sparkpages), Deep Research, and MoodBoard features.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '030_add_reports_research_moodboard'
down_revision: Union[str, None] = '029_add_skills_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create reports table
    op.create_table(
        'reports',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), default='draft'),
        sa.Column('sections', sa.Text(), nullable=True),  # JSON array
        sa.Column('citations', sa.Text(), nullable=True),  # JSON array
        sa.Column('extra_data', sa.Text(), nullable=True),  # JSON object
        sa.Column('thumbnail_url', sa.String(500), nullable=True),
        sa.Column('is_starred', sa.Boolean(), default=False),
        sa.Column('is_public', sa.Boolean(), default=False),
        sa.Column('section_count', sa.Integer(), default=0),
        sa.Column('citation_count', sa.Integer(), default=0),
        sa.Column('view_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create indexes for reports
    op.create_index('idx_reports_user', 'reports', ['user_id'])
    op.create_index('idx_reports_org', 'reports', ['organization_id'])
    op.create_index('idx_reports_status', 'reports', ['status'])
    op.create_index('idx_reports_starred', 'reports', ['is_starred'])

    # Create research_results table
    op.create_table(
        'research_results',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('findings', sa.Text(), nullable=True),  # JSON array
        sa.Column('sources', sa.Text(), nullable=True),  # JSON array
        sa.Column('verification', sa.Text(), nullable=True),  # JSON object
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('models_used', sa.Text(), nullable=True),  # JSON array
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('is_starred', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create indexes for research_results
    op.create_index('idx_research_user', 'research_results', ['user_id'])
    op.create_index('idx_research_org', 'research_results', ['organization_id'])
    op.create_index('idx_research_status', 'research_results', ['status'])
    op.create_index('idx_research_starred', 'research_results', ['is_starred'])

    # Create mood_boards table
    op.create_table(
        'mood_boards',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('status', sa.String(20), default='generating'),
        sa.Column('images', sa.Text(), nullable=True),  # JSON array
        sa.Column('themes', sa.Text(), nullable=True),  # JSON array
        sa.Column('color_palette', sa.Text(), nullable=True),  # JSON array
        sa.Column('style_tags', sa.Text(), nullable=True),  # JSON array
        sa.Column('thumbnail_url', sa.String(500), nullable=True),
        sa.Column('is_public', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create indexes for mood_boards
    op.create_index('idx_moodboards_user', 'mood_boards', ['user_id'])
    op.create_index('idx_moodboards_org', 'mood_boards', ['organization_id'])
    op.create_index('idx_moodboards_status', 'mood_boards', ['status'])


def downgrade() -> None:
    # Drop mood_boards indexes and table
    op.drop_index('idx_moodboards_status', table_name='mood_boards')
    op.drop_index('idx_moodboards_org', table_name='mood_boards')
    op.drop_index('idx_moodboards_user', table_name='mood_boards')
    op.drop_table('mood_boards')

    # Drop research_results indexes and table
    op.drop_index('idx_research_starred', table_name='research_results')
    op.drop_index('idx_research_status', table_name='research_results')
    op.drop_index('idx_research_org', table_name='research_results')
    op.drop_index('idx_research_user', table_name='research_results')
    op.drop_table('research_results')

    # Drop reports indexes and table
    op.drop_index('idx_reports_starred', table_name='reports')
    op.drop_index('idx_reports_status', table_name='reports')
    op.drop_index('idx_reports_org', table_name='reports')
    op.drop_index('idx_reports_user', table_name='reports')
    op.drop_table('reports')
