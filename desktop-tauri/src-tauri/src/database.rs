//! Local SQLite database for LOCAL MODE
//!
//! Stores documents, chunks, and embeddings locally.

use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::{DatabaseStats, DocumentInfo, SearchResult};

pub struct LocalDatabase {
    conn: Mutex<Connection>,
}

impl LocalDatabase {
    pub fn new() -> Result<Self> {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("aidocindexer");

        std::fs::create_dir_all(&data_dir)?;
        let db_path = data_dir.join("local.db");

        let conn = Connection::open(db_path)?;

        // Create tables
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
            "#,
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Insert a new document
    pub fn insert_document(&self, name: &str, path: &str) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO documents (name, path, updated_at) VALUES (?1, ?2, CURRENT_TIMESTAMP)",
            params![name, path],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Insert a chunk with embedding
    pub fn insert_chunk(
        &self,
        document_id: i64,
        chunk_index: i32,
        content: &str,
        embedding: &[f32],
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();

        // Serialize embedding as bytes
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES (?1, ?2, ?3, ?4)",
            params![document_id, chunk_index, content, embedding_bytes],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Search for similar chunks using cosine similarity
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<ChunkResult>> {
        let conn = self.conn.lock().unwrap();

        // Simple keyword search (for now)
        // TODO: Implement vector similarity search when query embedding is provided
        let mut stmt = conn.prepare(
            r#"
            SELECT c.id, c.document_id, c.chunk_index, c.content, d.name
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.content LIKE ?1
            LIMIT ?2
            "#,
        )?;

        let query_pattern = format!("%{}%", query);
        let results = stmt
            .query_map(params![query_pattern, top_k as i64], |row| {
                Ok(ChunkResult {
                    id: row.get(0)?,
                    document_id: row.get(1)?,
                    chunk_index: row.get(2)?,
                    content: row.get(3)?,
                    document_name: row.get(4)?,
                    score: 0.5, // Placeholder score for keyword search
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(results)
    }

    /// Search using vector similarity
    pub fn search_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<ChunkResult>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            r#"
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding, d.name
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            "#,
        )?;

        let mut results: Vec<ChunkResult> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let document_id: i64 = row.get(1)?;
                let chunk_index: i32 = row.get(2)?;
                let content: String = row.get(3)?;
                let embedding_bytes: Vec<u8> = row.get(4)?;
                let document_name: String = row.get(5)?;

                // Deserialize embedding
                let embedding: Vec<f32> = embedding_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                // Calculate cosine similarity
                let score = cosine_similarity(query_embedding, &embedding);

                Ok(ChunkResult {
                    id,
                    document_id,
                    chunk_index,
                    content,
                    document_name,
                    score,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Sort by score (highest first) and take top_k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    /// List all documents
    pub fn list_documents(&self) -> Result<Vec<DocumentInfo>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            r#"
            SELECT d.id, d.name, d.path, d.created_at, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            "#,
        )?;

        let documents = stmt
            .query_map([], |row| {
                Ok(DocumentInfo {
                    id: row.get::<_, i64>(0)?.to_string(),
                    name: row.get(1)?,
                    path: row.get(2)?,
                    created_at: row.get(3)?,
                    chunk_count: row.get(4)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(documents)
    }

    /// Find document by path
    pub fn find_document_by_path(&self, path: &str) -> Result<Option<DocumentRecord>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare("SELECT id, name, path FROM documents WHERE path = ?1")?;

        let result = stmt
            .query_row(params![path], |row| {
                Ok(DocumentRecord {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    path: row.get(2)?,
                })
            })
            .ok();

        Ok(result)
    }

    /// Delete document and its chunks
    pub fn delete_document(&self, document_id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM chunks WHERE document_id = ?1", params![document_id])?;
        conn.execute("DELETE FROM documents WHERE id = ?1", params![document_id])?;
        Ok(())
    }

    /// Delete chunks for a document (for re-indexing)
    pub fn delete_chunks_for_document(&self, document_id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM chunks WHERE document_id = ?1", params![document_id])?;
        Ok(())
    }

    /// Get database statistics
    pub fn get_stats(&self) -> Result<DatabaseStats> {
        let conn = self.conn.lock().unwrap();

        let document_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        let chunk_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        let total_size_bytes: i64 = conn.query_row(
            "SELECT COALESCE(SUM(LENGTH(content) + LENGTH(embedding)), 0) FROM chunks",
            [],
            |row| row.get(0),
        )?;

        Ok(DatabaseStats {
            document_count,
            chunk_count,
            total_size_bytes,
        })
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub id: i64,
    pub document_id: i64,
    pub chunk_index: i32,
    pub content: String,
    pub document_name: String,
    pub score: f32,
}

impl From<ChunkResult> for SearchResult {
    fn from(chunk: ChunkResult) -> Self {
        SearchResult {
            id: chunk.id.to_string(),
            document_id: chunk.document_id.to_string(),
            document_name: chunk.document_name,
            content: chunk.content,
            score: chunk.score,
            chunk_index: chunk.chunk_index,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DocumentRecord {
    pub id: i64,
    pub name: String,
    pub path: String,
}
