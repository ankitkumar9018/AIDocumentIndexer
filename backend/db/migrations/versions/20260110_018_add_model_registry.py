"""Add model registry for configurable model metadata.

Revision ID: 018
Revises: 017
Create Date: 2026-01-10

This migration adds:
1. Model registry table for storing model quality/cost/latency scores
2. Makes model access tiers fully configurable through the database
3. Allows admins to add new models and configure their properties
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '018'
down_revision = '017'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # =========================================================================
    # 1. Create Model Registry Table
    # =========================================================================
    op.create_table(
        'model_registry',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),

        # Model identification
        sa.Column('provider_type', sa.String(50), nullable=False,
                  comment='Provider type (openai, anthropic, google, ollama, etc.)'),
        sa.Column('model_name', sa.String(100), nullable=False,
                  comment='Model name (gpt-4o, claude-3-5-sonnet, etc.)'),
        sa.Column('display_name', sa.String(200), nullable=True,
                  comment='Human-readable display name'),
        sa.Column('description', sa.Text(), nullable=True,
                  comment='Model description and capabilities'),

        # Scores (all configurable by admin)
        sa.Column('quality_score', sa.Integer(), default=50, nullable=False,
                  comment='Quality score 0-100 (higher = better quality)'),
        sa.Column('cost_per_million_tokens', sa.Float(), default=1.0, nullable=False,
                  comment='Cost per 1M tokens in USD (input+output average)'),
        sa.Column('latency_score', sa.Integer(), default=3, nullable=False,
                  comment='Latency score 1-5 (lower = faster)'),

        # Capabilities
        sa.Column('max_context_tokens', sa.Integer(), nullable=True,
                  comment='Maximum context window size'),
        sa.Column('max_output_tokens', sa.Integer(), nullable=True,
                  comment='Maximum output tokens'),
        sa.Column('supports_vision', sa.Boolean(), default=False, nullable=False),
        sa.Column('supports_function_calling', sa.Boolean(), default=False, nullable=False),
        sa.Column('supports_streaming', sa.Boolean(), default=True, nullable=False),
        sa.Column('supports_json_mode', sa.Boolean(), default=False, nullable=False),

        # Categories/tags for filtering
        sa.Column('model_family', sa.String(50), nullable=True,
                  comment='Model family (gpt-4, claude-3, llama-3, etc.)'),
        sa.Column('tier', sa.String(50), nullable=True,
                  comment='Tier classification (basic, standard, advanced, enterprise)'),
        sa.Column('tags', sa.Text() if not is_postgresql else postgresql.JSONB(),
                  nullable=True,
                  comment='Additional tags for categorization'),

        # Minimum access level required (redundant with access groups but useful for quick filtering)
        sa.Column('min_access_level', sa.Integer(), default=1, nullable=False,
                  comment='Minimum user access level required'),

        # Organization scope (null = global/system-wide)
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'),
                  nullable=True,
                  comment='Org-specific model (null = available to all)'),

        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_deprecated', sa.Boolean(), default=False, nullable=False,
                  comment='Mark model as deprecated (still usable but not recommended)'),
        sa.Column('deprecated_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    # Indexes
    op.create_index('idx_model_registry_provider', 'model_registry', ['provider_type'])
    op.create_index('idx_model_registry_model', 'model_registry', ['model_name'])
    op.create_index('idx_model_registry_quality', 'model_registry', ['quality_score'])
    op.create_index('idx_model_registry_cost', 'model_registry', ['cost_per_million_tokens'])
    op.create_index('idx_model_registry_tier', 'model_registry', ['tier'])
    op.create_index('idx_model_registry_org', 'model_registry', ['organization_id'])
    op.create_unique_constraint('uq_model_registry_provider_model_org', 'model_registry',
                                ['provider_type', 'model_name', 'organization_id'])

    # =========================================================================
    # 2. Seed Default Models
    # =========================================================================
    if is_postgresql:
        op.execute("""
            INSERT INTO model_registry (
                id, provider_type, model_name, display_name, quality_score,
                cost_per_million_tokens, latency_score, max_context_tokens, max_output_tokens,
                supports_vision, supports_function_calling, supports_json_mode,
                model_family, tier, min_access_level, is_active
            )
            VALUES
            -- OpenAI Models
            (gen_random_uuid(), 'openai', 'gpt-4o', 'GPT-4o', 95, 7.50, 2, 128000, 16384, true, true, true, 'gpt-4', 'standard', 5, true),
            (gen_random_uuid(), 'openai', 'gpt-4o-mini', 'GPT-4o Mini', 85, 0.375, 1, 128000, 16384, true, true, true, 'gpt-4', 'basic', 1, true),
            (gen_random_uuid(), 'openai', 'gpt-4-turbo', 'GPT-4 Turbo', 93, 20.0, 3, 128000, 4096, true, true, true, 'gpt-4', 'advanced', 10, true),
            (gen_random_uuid(), 'openai', 'gpt-4', 'GPT-4', 90, 45.0, 4, 8192, 4096, false, true, true, 'gpt-4', 'advanced', 10, true),
            (gen_random_uuid(), 'openai', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 75, 1.0, 1, 16385, 4096, false, true, true, 'gpt-3.5', 'basic', 1, true),
            (gen_random_uuid(), 'openai', 'o1', 'o1 (Reasoning)', 98, 15.0, 5, 200000, 100000, true, false, false, 'o1', 'enterprise', 50, true),
            (gen_random_uuid(), 'openai', 'o1-mini', 'o1 Mini (Reasoning)', 92, 3.0, 3, 128000, 65536, false, false, false, 'o1', 'advanced', 10, true),

            -- Anthropic Models
            (gen_random_uuid(), 'anthropic', 'claude-3-5-sonnet-latest', 'Claude 3.5 Sonnet', 92, 9.0, 2, 200000, 8192, true, true, true, 'claude-3.5', 'standard', 5, true),
            (gen_random_uuid(), 'anthropic', 'claude-3-5-haiku-latest', 'Claude 3.5 Haiku', 82, 1.25, 1, 200000, 8192, true, true, true, 'claude-3.5', 'basic', 1, true),
            (gen_random_uuid(), 'anthropic', 'claude-3-opus-latest', 'Claude 3 Opus', 95, 37.5, 5, 200000, 4096, true, true, true, 'claude-3', 'enterprise', 50, true),
            (gen_random_uuid(), 'anthropic', 'claude-3-sonnet-20240229', 'Claude 3 Sonnet', 88, 9.0, 3, 200000, 4096, true, true, true, 'claude-3', 'standard', 5, true),
            (gen_random_uuid(), 'anthropic', 'claude-3-haiku-20240307', 'Claude 3 Haiku', 80, 0.625, 1, 200000, 4096, true, true, true, 'claude-3', 'basic', 1, true),

            -- Google Models
            (gen_random_uuid(), 'google', 'gemini-1.5-pro', 'Gemini 1.5 Pro', 88, 5.0, 3, 2097152, 8192, true, true, true, 'gemini-1.5', 'advanced', 10, true),
            (gen_random_uuid(), 'google', 'gemini-1.5-flash', 'Gemini 1.5 Flash', 80, 0.3, 1, 1048576, 8192, true, true, true, 'gemini-1.5', 'basic', 1, true),
            (gen_random_uuid(), 'google', 'gemini-2.0-flash-exp', 'Gemini 2.0 Flash (Exp)', 85, 0.5, 1, 1048576, 8192, true, true, true, 'gemini-2.0', 'standard', 5, true),

            -- Meta/Llama Models (via Ollama, Together, Groq, etc.)
            (gen_random_uuid(), 'ollama', 'llama3.3:70b', 'Llama 3.3 70B', 85, 0.9, 3, 131072, 4096, false, true, true, 'llama-3.3', 'standard', 5, true),
            (gen_random_uuid(), 'ollama', 'llama3.2:8b', 'Llama 3.2 8B', 70, 0.1, 2, 131072, 4096, false, true, true, 'llama-3.2', 'basic', 1, true),
            (gen_random_uuid(), 'ollama', 'llama3.2:3b', 'Llama 3.2 3B', 60, 0.05, 1, 131072, 4096, false, true, true, 'llama-3.2', 'basic', 1, true),
            (gen_random_uuid(), 'groq', 'llama-3.3-70b-versatile', 'Llama 3.3 70B (Groq)', 85, 0.6, 1, 131072, 32768, false, true, true, 'llama-3.3', 'standard', 5, true),

            -- Mistral Models
            (gen_random_uuid(), 'mistral', 'mistral-large-latest', 'Mistral Large', 88, 8.0, 2, 128000, 8192, false, true, true, 'mistral', 'advanced', 10, true),
            (gen_random_uuid(), 'mistral', 'mistral-small-latest', 'Mistral Small', 75, 0.6, 1, 128000, 8192, false, true, true, 'mistral', 'basic', 1, true),
            (gen_random_uuid(), 'ollama', 'mixtral:8x7b', 'Mixtral 8x7B', 78, 0.6, 2, 32768, 4096, false, true, true, 'mixtral', 'basic', 1, true),

            -- Cohere Models
            (gen_random_uuid(), 'cohere', 'command-r-plus', 'Command R+', 85, 3.0, 2, 128000, 4096, false, true, true, 'command-r', 'standard', 5, true),
            (gen_random_uuid(), 'cohere', 'command-r', 'Command R', 78, 0.5, 1, 128000, 4096, false, true, true, 'command-r', 'basic', 1, true),

            -- Embedding Models
            (gen_random_uuid(), 'openai', 'text-embedding-3-large', 'OpenAI Embeddings Large', 95, 0.13, 1, 8191, null, false, false, false, 'embeddings', 'standard', 1, true),
            (gen_random_uuid(), 'openai', 'text-embedding-3-small', 'OpenAI Embeddings Small', 85, 0.02, 1, 8191, null, false, false, false, 'embeddings', 'basic', 1, true),
            (gen_random_uuid(), 'cohere', 'embed-english-v3.0', 'Cohere Embed English v3', 88, 0.10, 1, 512, null, false, false, false, 'embeddings', 'standard', 1, true)

            ON CONFLICT (provider_type, model_name, organization_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                quality_score = EXCLUDED.quality_score,
                cost_per_million_tokens = EXCLUDED.cost_per_million_tokens,
                latency_score = EXCLUDED.latency_score,
                updated_at = now()
        """)
    else:
        # SQLite version (simplified)
        op.execute("""
            INSERT OR REPLACE INTO model_registry (
                id, provider_type, model_name, display_name, quality_score,
                cost_per_million_tokens, latency_score, model_family, tier, min_access_level, is_active,
                supports_vision, supports_function_calling, supports_streaming, supports_json_mode
            )
            VALUES
            (lower(hex(randomblob(16))), 'openai', 'gpt-4o', 'GPT-4o', 95, 7.50, 2, 'gpt-4', 'standard', 5, 1, 1, 1, 1, 1),
            (lower(hex(randomblob(16))), 'openai', 'gpt-4o-mini', 'GPT-4o Mini', 85, 0.375, 1, 'gpt-4', 'basic', 1, 1, 1, 1, 1, 1),
            (lower(hex(randomblob(16))), 'openai', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 75, 1.0, 1, 'gpt-3.5', 'basic', 1, 1, 0, 1, 1, 1),
            (lower(hex(randomblob(16))), 'anthropic', 'claude-3-5-sonnet-latest', 'Claude 3.5 Sonnet', 92, 9.0, 2, 'claude-3.5', 'standard', 5, 1, 1, 1, 1, 1),
            (lower(hex(randomblob(16))), 'anthropic', 'claude-3-5-haiku-latest', 'Claude 3.5 Haiku', 82, 1.25, 1, 'claude-3.5', 'basic', 1, 1, 1, 1, 1, 1)
        """)

    # =========================================================================
    # 3. Enable RLS on model_registry (PostgreSQL only)
    # =========================================================================
    if is_postgresql:
        op.execute("ALTER TABLE model_registry ENABLE ROW LEVEL SECURITY")

        op.execute("""
            CREATE POLICY model_registry_org ON model_registry
            FOR ALL
            USING (
                organization_id IS NULL
                OR organization_id = current_setting('app.current_org_id', true)::uuid
            )
        """)


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # Drop RLS
    if is_postgresql:
        op.execute("DROP POLICY IF EXISTS model_registry_org ON model_registry")
        op.execute("ALTER TABLE model_registry DISABLE ROW LEVEL SECURITY")

    # Drop table
    op.drop_constraint('uq_model_registry_provider_model_org', 'model_registry', type_='unique')
    op.drop_index('idx_model_registry_org', 'model_registry')
    op.drop_index('idx_model_registry_tier', 'model_registry')
    op.drop_index('idx_model_registry_cost', 'model_registry')
    op.drop_index('idx_model_registry_quality', 'model_registry')
    op.drop_index('idx_model_registry_model', 'model_registry')
    op.drop_index('idx_model_registry_provider', 'model_registry')
    op.drop_table('model_registry')
