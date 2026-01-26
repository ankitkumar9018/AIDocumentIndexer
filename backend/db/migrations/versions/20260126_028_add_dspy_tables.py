"""Add DSPy training examples and optimization jobs tables

Revision ID: 028_add_dspy_tables
Revises: 027_add_agent_publishing
Create Date: 2026-01-26

Adds tables for DSPy prompt optimization (Phase 93):
- dspy_training_examples: Training data for DSPy signatures
- dspy_optimization_jobs: Optimization run tracking
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '028_add_dspy_tables'
down_revision = '027_add_agent_publishing'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create dspy_training_examples table
    op.create_table(
        'dspy_training_examples',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('signature_name', sa.String(100), nullable=False),
        sa.Column('inputs', sa.JSON(), nullable=True),
        sa.Column('outputs', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(50), server_default='manual'),
        sa.Column('source_id', sa.String(255), nullable=True),
        sa.Column('quality_score', sa.Float(), server_default='1.0'),
        sa.Column('is_active', sa.Boolean(), server_default='1'),
    )
    op.create_index('idx_dspy_example_signature', 'dspy_training_examples', ['signature_name'])
    op.create_index('idx_dspy_example_active', 'dspy_training_examples', ['is_active'])
    op.create_index('idx_dspy_example_source', 'dspy_training_examples', ['source'])

    # Create dspy_optimization_jobs table
    op.create_table(
        'dspy_optimization_jobs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('signature_name', sa.String(100), nullable=False),
        sa.Column('optimizer_type', sa.String(50), server_default='bootstrap_few_shot'),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('num_train_examples', sa.Integer(), server_default='0'),
        sa.Column('num_dev_examples', sa.Integer(), server_default='0'),
        sa.Column('baseline_score', sa.Float(), nullable=True),
        sa.Column('optimized_score', sa.Float(), nullable=True),
        sa.Column('improvement_pct', sa.Float(), nullable=True),
        sa.Column('compiled_state', sa.JSON(), nullable=True),
        sa.Column('prompt_version_id', sa.String(36),
                  sa.ForeignKey('agent_prompt_versions.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('agent_id', sa.String(36),
                  sa.ForeignKey('agent_definitions.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
    )
    op.create_index('idx_dspy_job_signature', 'dspy_optimization_jobs', ['signature_name'])
    op.create_index('idx_dspy_job_status', 'dspy_optimization_jobs', ['status'])


def downgrade() -> None:
    op.drop_table('dspy_optimization_jobs')
    op.drop_table('dspy_training_examples')
