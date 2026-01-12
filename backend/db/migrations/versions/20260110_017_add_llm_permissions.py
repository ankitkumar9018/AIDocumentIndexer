"""Add LLM model permissions and service-level routing.

Revision ID: 017
Revises: 016
Create Date: 2026-01-10

This migration adds:
1. Model access groups for organizing models by permission level
2. User/role-based model access permissions
3. Service-level LLM configuration (each service can have its own LLM)
4. Model routing rules for intelligent model selection

Based on best practices from:
- LiteLLM RBAC and model access controls
- OpenRouter provider routing
- NVIDIA LLM Router Blueprint
- Microsoft Azure Model Router
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '017'
down_revision = '016b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # =========================================================================
    # 1. Create Model Access Groups Table
    # =========================================================================
    op.create_table(
        'model_access_groups',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),
        sa.Column('name', sa.String(100), unique=True, nullable=False,
                  comment='Group name (e.g., "basic", "advanced", "enterprise")'),
        sa.Column('description', sa.Text(), nullable=True),

        # Access level (higher = more access)
        sa.Column('access_level', sa.Integer(), nullable=False, default=1,
                  comment='Access level (1=basic, 10=enterprise, 100=admin)'),

        # Models in this group (wildcard patterns supported)
        sa.Column('model_patterns', sa.Text() if not is_postgresql else postgresql.JSONB(),
                  nullable=False,
                  comment='List of model patterns (e.g., ["gpt-3.5-*", "gpt-4o-mini"])'),

        # Restrictions
        sa.Column('max_tokens_per_request', sa.Integer(), nullable=True,
                  comment='Max tokens per request for models in this group'),
        sa.Column('max_requests_per_minute', sa.Integer(), nullable=True,
                  comment='Rate limit for models in this group'),
        sa.Column('max_cost_per_day_usd', sa.Float(), nullable=True,
                  comment='Max daily cost for models in this group'),

        # Organization scope
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'),
                  nullable=True,
                  comment='Org-specific group (null = global)'),

        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    op.create_index('idx_model_access_groups_name', 'model_access_groups', ['name'])
    op.create_index('idx_model_access_groups_level', 'model_access_groups', ['access_level'])
    op.create_index('idx_model_access_groups_org', 'model_access_groups', ['organization_id'])

    # =========================================================================
    # 2. Create User Model Access Table (which groups a user can access)
    # =========================================================================
    op.create_table(
        'user_model_access',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),
        sa.Column('user_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('access_group_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('model_access_groups.id', ondelete='CASCADE'),
                  nullable=False),

        # Override settings for this user
        sa.Column('custom_max_tokens', sa.Integer(), nullable=True),
        sa.Column('custom_rate_limit', sa.Integer(), nullable=True),
        sa.Column('custom_daily_budget_usd', sa.Float(), nullable=True),

        sa.Column('granted_by_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('granted_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When access expires (null = never)'),
    )

    op.create_index('idx_user_model_access_user', 'user_model_access', ['user_id'])
    op.create_index('idx_user_model_access_group', 'user_model_access', ['access_group_id'])
    op.create_unique_constraint('uq_user_model_access_user_group', 'user_model_access',
                                ['user_id', 'access_group_id'])

    # =========================================================================
    # 3. Create Service LLM Configuration Table
    # =========================================================================
    op.create_table(
        'service_llm_configs',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),

        # Service identifier
        sa.Column('service_name', sa.String(100), nullable=False,
                  comment='Service name (e.g., "workflow", "audio_overview", "rag", "chat")'),

        # Operation within the service (for fine-grained control)
        sa.Column('operation_name', sa.String(100), nullable=True,
                  comment='Specific operation (e.g., "script_generation", "summarization")'),

        # Organization scope
        sa.Column('organization_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'),
                  nullable=True,
                  comment='Org-specific config (null = global default)'),

        # LLM Configuration
        sa.Column('provider_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('llm_providers.id', ondelete='SET NULL'),
                  nullable=True,
                  comment='Primary provider for this service'),
        sa.Column('model_name', sa.String(100), nullable=True,
                  comment='Specific model (null = use provider default)'),

        # Model parameters
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('max_tokens', sa.Integer(), nullable=True),
        sa.Column('top_p', sa.Float(), nullable=True),
        sa.Column('frequency_penalty', sa.Float(), nullable=True),
        sa.Column('presence_penalty', sa.Float(), nullable=True),

        # Fallback configuration
        sa.Column('fallback_provider_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('llm_providers.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('fallback_model_name', sa.String(100), nullable=True),

        # Routing strategy
        sa.Column('routing_strategy', sa.String(50), default='default', nullable=False,
                  comment='Routing strategy: default, cost_optimized, quality_optimized, latency_optimized'),

        # User override settings
        sa.Column('allow_user_override', sa.Boolean(), default=True, nullable=False,
                  comment='Whether users can override this config'),
        sa.Column('allowed_override_models', sa.Text() if not is_postgresql else postgresql.JSONB(),
                  nullable=True,
                  comment='Models users can switch to (null = any allowed by their access)'),

        # Minimum access level required
        sa.Column('min_access_level', sa.Integer(), default=1, nullable=False,
                  comment='Minimum access level to use this service'),

        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    op.create_index('idx_service_llm_configs_service', 'service_llm_configs', ['service_name'])
    op.create_index('idx_service_llm_configs_org', 'service_llm_configs', ['organization_id'])
    op.create_unique_constraint('uq_service_llm_configs_service_op_org', 'service_llm_configs',
                                ['service_name', 'operation_name', 'organization_id'])

    # =========================================================================
    # 4. Create User Service LLM Overrides Table
    # =========================================================================
    op.create_table(
        'user_service_llm_overrides',
        sa.Column('id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  primary_key=True,
                  server_default=sa.text("gen_random_uuid()") if is_postgresql else None),
        sa.Column('user_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'),
                  nullable=False),

        # Service to override
        sa.Column('service_name', sa.String(100), nullable=False),
        sa.Column('operation_name', sa.String(100), nullable=True),

        # User's choice
        sa.Column('provider_id', sa.String(36) if not is_postgresql else postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('llm_providers.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('model_name', sa.String(100), nullable=True),

        # Custom parameters (if allowed)
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('max_tokens', sa.Integer(), nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    op.create_index('idx_user_service_llm_overrides_user', 'user_service_llm_overrides', ['user_id'])
    op.create_index('idx_user_service_llm_overrides_service', 'user_service_llm_overrides', ['service_name'])
    op.create_unique_constraint('uq_user_service_llm_overrides', 'user_service_llm_overrides',
                                ['user_id', 'service_name', 'operation_name'])

    # =========================================================================
    # 5. Seed Default Model Access Groups
    # =========================================================================
    if is_postgresql:
        op.execute("""
            INSERT INTO model_access_groups (id, name, description, access_level, model_patterns, is_active)
            VALUES
            (
                gen_random_uuid(),
                'basic',
                'Basic tier - Access to cost-effective models',
                1,
                '["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-haiku-*", "llama-*-8b-*", "gemini-*-flash"]',
                true
            ),
            (
                gen_random_uuid(),
                'standard',
                'Standard tier - Access to most production models',
                5,
                '["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-sonnet-*", "claude-3-5-haiku-*", "llama-*", "gemini-*", "mixtral-*"]',
                true
            ),
            (
                gen_random_uuid(),
                'advanced',
                'Advanced tier - Access to all standard models plus specialized ones',
                10,
                '["gpt-4*", "claude-*", "llama-*", "gemini-*", "mixtral-*", "command-r*"]',
                true
            ),
            (
                gen_random_uuid(),
                'enterprise',
                'Enterprise tier - Access to all models including GPT-4 Turbo and Claude Opus',
                50,
                '["*"]',
                true
            ),
            (
                gen_random_uuid(),
                'admin',
                'Admin tier - Unrestricted access to all models',
                100,
                '["*"]',
                true
            )
            ON CONFLICT (name) DO NOTHING
        """)
    else:
        # SQLite version
        op.execute("""
            INSERT OR IGNORE INTO model_access_groups (id, name, description, access_level, model_patterns, is_active)
            VALUES
            (
                lower(hex(randomblob(16))),
                'basic',
                'Basic tier - Access to cost-effective models',
                1,
                '["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-haiku-*", "llama-*-8b-*", "gemini-*-flash"]',
                1
            ),
            (
                lower(hex(randomblob(16))),
                'standard',
                'Standard tier - Access to most production models',
                5,
                '["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-sonnet-*", "claude-3-5-haiku-*", "llama-*", "gemini-*", "mixtral-*"]',
                1
            ),
            (
                lower(hex(randomblob(16))),
                'advanced',
                'Advanced tier - Access to all standard models plus specialized ones',
                10,
                '["gpt-4*", "claude-*", "llama-*", "gemini-*", "mixtral-*", "command-r*"]',
                1
            ),
            (
                lower(hex(randomblob(16))),
                'enterprise',
                'Enterprise tier - Access to all models including GPT-4 Turbo and Claude Opus',
                50,
                '["*"]',
                1
            ),
            (
                lower(hex(randomblob(16))),
                'admin',
                'Admin tier - Unrestricted access to all models',
                100,
                '["*"]',
                1
            )
        """)

    # =========================================================================
    # 6. Seed Default Service LLM Configurations
    # =========================================================================
    if is_postgresql:
        op.execute("""
            INSERT INTO service_llm_configs (id, service_name, operation_name, routing_strategy, allow_user_override, min_access_level, is_active)
            VALUES
            -- Chat service
            (gen_random_uuid(), 'chat', 'default', 'default', true, 1, true),
            (gen_random_uuid(), 'chat', 'general', 'default', true, 1, true),
            (gen_random_uuid(), 'chat', 'agent', 'quality_optimized', true, 5, true),

            -- RAG service
            (gen_random_uuid(), 'rag', 'query', 'default', true, 1, true),
            (gen_random_uuid(), 'rag', 'rerank', 'latency_optimized', false, 1, true),

            -- Document processing
            (gen_random_uuid(), 'document', 'summarization', 'default', true, 1, true),
            (gen_random_uuid(), 'document', 'extraction', 'default', true, 1, true),
            (gen_random_uuid(), 'document', 'ocr', 'default', false, 1, true),

            -- Audio overview
            (gen_random_uuid(), 'audio_overview', 'script_generation', 'quality_optimized', true, 5, true),

            -- Workflow
            (gen_random_uuid(), 'workflow', 'default', 'default', true, 5, true),
            (gen_random_uuid(), 'workflow', 'code_execution', 'quality_optimized', true, 10, true),

            -- Document generation
            (gen_random_uuid(), 'generation', 'pptx', 'default', true, 5, true),
            (gen_random_uuid(), 'generation', 'docx', 'default', true, 5, true),
            (gen_random_uuid(), 'generation', 'pdf', 'default', true, 5, true),

            -- Image generation
            (gen_random_uuid(), 'image', 'generate', 'default', true, 5, true),

            -- Embeddings
            (gen_random_uuid(), 'embeddings', 'default', 'latency_optimized', false, 1, true)
            ON CONFLICT (service_name, operation_name, organization_id) DO NOTHING
        """)

    # =========================================================================
    # 7. Enable RLS on new tables (PostgreSQL only)
    # =========================================================================
    if is_postgresql:
        # Enable RLS
        op.execute("ALTER TABLE model_access_groups ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE user_model_access ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE service_llm_configs ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE user_service_llm_overrides ENABLE ROW LEVEL SECURITY")

        # Create policies
        op.execute("""
            CREATE POLICY model_access_groups_org ON model_access_groups
            FOR ALL
            USING (
                organization_id IS NULL
                OR organization_id = current_setting('app.current_org_id', true)::uuid
            )
        """)

        op.execute("""
            CREATE POLICY user_model_access_own ON user_model_access
            FOR ALL
            USING (user_id = current_setting('app.current_user_id', true)::uuid)
        """)

        op.execute("""
            CREATE POLICY service_llm_configs_org ON service_llm_configs
            FOR ALL
            USING (
                organization_id IS NULL
                OR organization_id = current_setting('app.current_org_id', true)::uuid
            )
        """)

        op.execute("""
            CREATE POLICY user_service_llm_overrides_own ON user_service_llm_overrides
            FOR ALL
            USING (user_id = current_setting('app.current_user_id', true)::uuid)
        """)


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_postgresql = dialect == 'postgresql'

    # Drop RLS policies
    if is_postgresql:
        op.execute("DROP POLICY IF EXISTS user_service_llm_overrides_own ON user_service_llm_overrides")
        op.execute("DROP POLICY IF EXISTS service_llm_configs_org ON service_llm_configs")
        op.execute("DROP POLICY IF EXISTS user_model_access_own ON user_model_access")
        op.execute("DROP POLICY IF EXISTS model_access_groups_org ON model_access_groups")

        op.execute("ALTER TABLE user_service_llm_overrides DISABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE service_llm_configs DISABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE user_model_access DISABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE model_access_groups DISABLE ROW LEVEL SECURITY")

    # Drop tables in reverse order
    op.drop_constraint('uq_user_service_llm_overrides', 'user_service_llm_overrides', type_='unique')
    op.drop_index('idx_user_service_llm_overrides_service', 'user_service_llm_overrides')
    op.drop_index('idx_user_service_llm_overrides_user', 'user_service_llm_overrides')
    op.drop_table('user_service_llm_overrides')

    op.drop_constraint('uq_service_llm_configs_service_op_org', 'service_llm_configs', type_='unique')
    op.drop_index('idx_service_llm_configs_org', 'service_llm_configs')
    op.drop_index('idx_service_llm_configs_service', 'service_llm_configs')
    op.drop_table('service_llm_configs')

    op.drop_constraint('uq_user_model_access_user_group', 'user_model_access', type_='unique')
    op.drop_index('idx_user_model_access_group', 'user_model_access')
    op.drop_index('idx_user_model_access_user', 'user_model_access')
    op.drop_table('user_model_access')

    op.drop_index('idx_model_access_groups_org', 'model_access_groups')
    op.drop_index('idx_model_access_groups_level', 'model_access_groups')
    op.drop_index('idx_model_access_groups_name', 'model_access_groups')
    op.drop_table('model_access_groups')
