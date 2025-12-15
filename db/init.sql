-- =============================================================================
-- AIDocumentIndexer - Database Initialization
-- =============================================================================
-- This script sets up:
-- 1. pgvector extension for vector similarity search
-- 2. Core tables (documents, chunks, users, roles, etc.)
-- 3. Row-Level Security (RLS) policies for permission enforcement
-- 4. Indexes for optimal query performance
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE processing_mode AS ENUM ('full', 'smart', 'text_only');
CREATE TYPE storage_mode AS ENUM ('rag', 'query_only');

-- =============================================================================
-- TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Access Tiers (Dynamic, Admin-Configurable)
-- -----------------------------------------------------------------------------
CREATE TABLE access_tiers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    level INTEGER NOT NULL CHECK (level >= 1 AND level <= 100),
    description TEXT,
    color VARCHAR(7) DEFAULT '#6B7280',  -- For UI display
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert default tiers
INSERT INTO access_tiers (name, level, description, color) VALUES
    ('Intern', 10, 'Access to public documents only', '#10B981'),
    ('Staff', 30, 'Access to internal and public documents', '#3B82F6'),
    ('Manager', 50, 'Access to confidential, internal, and public documents', '#8B5CF6'),
    ('Executive', 80, 'Access to all documents', '#F59E0B'),
    ('Admin', 100, 'Full system access including administration', '#EF4444');

-- -----------------------------------------------------------------------------
-- Users
-- -----------------------------------------------------------------------------
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    access_tier_id UUID NOT NULL REFERENCES access_tiers(id),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES users(id)
);

-- -----------------------------------------------------------------------------
-- Documents
-- -----------------------------------------------------------------------------
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash for deduplication
    filename VARCHAR(500) NOT NULL,
    original_filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),

    -- Access control
    access_tier_id UUID NOT NULL REFERENCES access_tiers(id),

    -- Processing info
    processing_status processing_status DEFAULT 'pending',
    processing_mode processing_mode DEFAULT 'smart',
    storage_mode storage_mode DEFAULT 'rag',
    processing_error TEXT,
    processed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    title VARCHAR(500),
    description TEXT,
    language VARCHAR(10) DEFAULT 'en',
    page_count INTEGER,
    word_count INTEGER,
    tags TEXT[],

    -- Source info (for auto-indexed files)
    source_path VARCHAR(1000),
    is_auto_indexed BOOLEAN DEFAULT false,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    uploaded_by UUID REFERENCES users(id),

    -- Unique constraint to prevent duplicate files
    CONSTRAINT unique_file_hash UNIQUE (file_hash)
);

-- -----------------------------------------------------------------------------
-- Document Chunks (with Embeddings)
-- -----------------------------------------------------------------------------
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,  -- For deduplication

    -- Embedding (1536 dimensions for OpenAI, adjust if using different model)
    embedding vector(1536),

    -- Position info
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section_title VARCHAR(500),

    -- Access control (inherited from document but stored for RLS performance)
    access_tier_id UUID NOT NULL REFERENCES access_tiers(id),

    -- Metadata
    token_count INTEGER,
    char_count INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Scraped Web Content
-- -----------------------------------------------------------------------------
CREATE TABLE scraped_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url VARCHAR(2000) NOT NULL,
    url_hash VARCHAR(64) NOT NULL,  -- SHA-256 of URL
    title VARCHAR(500),
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,

    -- Embedding
    embedding vector(1536),

    -- Storage preference
    stored_permanently BOOLEAN DEFAULT false,

    -- Access control
    access_tier_id UUID REFERENCES access_tiers(id),
    scraped_by UUID REFERENCES users(id),

    -- Timestamps
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE  -- For temporary storage
);

-- -----------------------------------------------------------------------------
-- Chat Sessions
-- -----------------------------------------------------------------------------
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    title VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Chat Messages
-- -----------------------------------------------------------------------------
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,

    -- Source citations
    source_document_ids UUID[],  -- Array of document IDs used
    source_chunks TEXT,  -- JSON array of chunk references

    -- Feedback
    is_helpful BOOLEAN,
    feedback TEXT,

    -- LLM info
    model_used VARCHAR(100),
    tokens_used INTEGER,
    latency_ms INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Audit Log
-- -----------------------------------------------------------------------------
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),  -- 'document', 'user', 'tier', etc.
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- Processing Queue
-- -----------------------------------------------------------------------------
CREATE TABLE processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    priority INTEGER DEFAULT 0,
    status processing_status DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Documents indexes
CREATE INDEX idx_documents_file_hash ON documents(file_hash);
CREATE INDEX idx_documents_access_tier ON documents(access_tier_id);
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_documents_file_type ON documents(file_type);

-- Chunks indexes
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_access_tier ON chunks(access_tier_id);

-- Vector similarity search index (HNSW for best performance)
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX idx_chunks_content_fts ON chunks USING gin(to_tsvector('english', content));

-- Scraped content indexes
CREATE INDEX idx_scraped_url_hash ON scraped_content(url_hash);
CREATE INDEX idx_scraped_embedding ON scraped_content USING hnsw (embedding vector_cosine_ops);

-- Chat indexes
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);

-- Audit log indexes
CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_action ON audit_log(action);
CREATE INDEX idx_audit_created_at ON audit_log(created_at DESC);

-- Processing queue indexes
CREATE INDEX idx_queue_status ON processing_queue(status);
CREATE INDEX idx_queue_priority ON processing_queue(priority DESC, created_at ASC);

-- =============================================================================
-- ROW-LEVEL SECURITY (RLS)
-- =============================================================================

-- Enable RLS on sensitive tables
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE scraped_content ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- Create application role for RLS
-- Note: In production, create separate roles for different access levels
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'aidoc_app') THEN
        CREATE ROLE aidoc_app WITH LOGIN PASSWORD 'aidoc_app_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO aidoc_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO aidoc_app;

-- -----------------------------------------------------------------------------
-- RLS Policies for Documents
-- -----------------------------------------------------------------------------

-- Users can only see documents at or below their access tier level
CREATE POLICY documents_select_policy ON documents
    FOR SELECT
    USING (
        access_tier_id IN (
            SELECT at.id FROM access_tiers at
            WHERE at.level <= (
                SELECT at2.level FROM users u
                JOIN access_tiers at2 ON u.access_tier_id = at2.id
                WHERE u.id = current_setting('app.current_user_id', true)::uuid
            )
        )
    );

-- Users can insert documents with tier at or below their level
CREATE POLICY documents_insert_policy ON documents
    FOR INSERT
    WITH CHECK (
        access_tier_id IN (
            SELECT at.id FROM access_tiers at
            WHERE at.level <= (
                SELECT at2.level FROM users u
                JOIN access_tiers at2 ON u.access_tier_id = at2.id
                WHERE u.id = current_setting('app.current_user_id', true)::uuid
            )
        )
    );

-- -----------------------------------------------------------------------------
-- RLS Policies for Chunks
-- -----------------------------------------------------------------------------

-- Users can only see chunks from documents they have access to
CREATE POLICY chunks_select_policy ON chunks
    FOR SELECT
    USING (
        access_tier_id IN (
            SELECT at.id FROM access_tiers at
            WHERE at.level <= (
                SELECT at2.level FROM users u
                JOIN access_tiers at2 ON u.access_tier_id = at2.id
                WHERE u.id = current_setting('app.current_user_id', true)::uuid
            )
        )
    );

-- -----------------------------------------------------------------------------
-- RLS Policies for Chat Sessions
-- -----------------------------------------------------------------------------

-- Users can only see their own chat sessions
CREATE POLICY chat_sessions_select_policy ON chat_sessions
    FOR SELECT
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

CREATE POLICY chat_sessions_insert_policy ON chat_sessions
    FOR INSERT
    WITH CHECK (user_id = current_setting('app.current_user_id', true)::uuid);

-- -----------------------------------------------------------------------------
-- RLS Policies for Chat Messages
-- -----------------------------------------------------------------------------

-- Users can only see messages from their sessions
CREATE POLICY chat_messages_select_policy ON chat_messages
    FOR SELECT
    USING (
        session_id IN (
            SELECT id FROM chat_sessions
            WHERE user_id = current_setting('app.current_user_id', true)::uuid
        )
    );

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_access_tiers_updated_at BEFORE UPDATE ON access_tiers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_sessions_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for vector similarity search with RLS
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10,
    user_tier_level int DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity float,
    page_number int,
    filename VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) as similarity,
        c.page_number,
        d.filename
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    JOIN access_tiers at ON c.access_tier_id = at.id
    WHERE at.level <= user_tier_level
      AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function for hybrid search (vector + full-text)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    match_count int DEFAULT 10,
    user_tier_level int DEFAULT 10,
    vector_weight float DEFAULT 0.5
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score float,
    vector_score float,
    text_score float,
    page_number int,
    filename VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.content,
        (vector_weight * (1 - (c.embedding <=> query_embedding))) +
        ((1 - vector_weight) * ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', query_text))) as combined_score,
        1 - (c.embedding <=> query_embedding) as vector_score,
        ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', query_text)) as text_score,
        c.page_number,
        d.filename
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    JOIN access_tiers at ON c.access_tier_id = at.id
    WHERE at.level <= user_tier_level
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Note: Admin user will be created by the application on first startup
-- using credentials from environment variables

COMMENT ON DATABASE aidocindexer IS 'AIDocumentIndexer - Intelligent Document Archive with RAG';
