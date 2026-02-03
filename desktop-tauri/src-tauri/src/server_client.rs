//! Server client for SERVER MODE
//!
//! Connects to AIDocumentIndexer backend server.

use std::path::PathBuf;

use anyhow::Result;
use reqwest::{multipart, Client};
use serde::{Deserialize, Serialize};

use crate::{ChatResponse, IndexResult, SearchResult};

pub struct ServerClient {
    client: Client,
    default_url: String,
}

impl ServerClient {
    pub fn new(default_url: &str) -> Self {
        Self {
            client: Client::new(),
            default_url: default_url.trim_end_matches('/').to_string(),
        }
    }

    fn get_headers(&self, api_key: &Option<String>) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(key) = api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", key).parse().unwrap(),
            );
        }
        headers
    }

    /// Check server health
    pub async fn health_check(&self, server_url: &str) -> Result<()> {
        let url = format!("{}/api/v1/health", server_url);
        self.client
            .get(&url)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Chat with documents
    pub async fn chat(
        &self,
        server_url: &str,
        api_key: &Option<String>,
        message: &str,
        collection_id: Option<String>,
    ) -> Result<ChatResponse> {
        let url = format!("{}/api/v1/chat/query", server_url);

        let request = ServerChatRequest {
            query: message.to_string(),
            collection_id,
            top_k: Some(5),
            include_sources: Some(true),
        };

        let response: ServerChatResponse = self
            .client
            .post(&url)
            .headers(self.get_headers(api_key))
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        Ok(ChatResponse {
            content: response.answer,
            sources: response
                .sources
                .into_iter()
                .map(|s| SearchResult {
                    id: s.chunk_id,
                    document_id: s.document_id,
                    document_name: s.document_title.unwrap_or_default(),
                    content: s.content,
                    score: s.score,
                    chunk_index: s.chunk_index.unwrap_or(0),
                })
                .collect(),
        })
    }

    /// Search documents
    pub async fn search(
        &self,
        server_url: &str,
        api_key: &Option<String>,
        query: &str,
        collection_id: Option<String>,
        top_k: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let url = format!("{}/api/v1/search", server_url);

        let request = ServerSearchRequest {
            query: query.to_string(),
            collection_id,
            top_k: top_k.map(|k| k as i32),
        };

        let response: ServerSearchResponse = self
            .client
            .post(&url)
            .headers(self.get_headers(api_key))
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        Ok(response
            .results
            .into_iter()
            .map(|s| SearchResult {
                id: s.chunk_id,
                document_id: s.document_id,
                document_name: s.document_title.unwrap_or_default(),
                content: s.content,
                score: s.score,
                chunk_index: s.chunk_index.unwrap_or(0),
            })
            .collect())
    }

    /// Upload file to server
    pub async fn upload_file(
        &self,
        server_url: &str,
        api_key: &Option<String>,
        path: PathBuf,
        collection_id: Option<String>,
    ) -> Result<IndexResult> {
        let url = format!("{}/api/v1/upload/single", server_url);

        // Read file
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_bytes = std::fs::read(&path)?;

        // Create multipart form
        let part = multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")?;

        let mut form = multipart::Form::new().part("file", part);

        if let Some(coll_id) = collection_id {
            form = form.text("collection_id", coll_id);
        }

        let mut request = self.client.post(&url).multipart(form);

        if let Some(key) = api_key {
            request = request.header(reqwest::header::AUTHORIZATION, format!("Bearer {}", key));
        }

        let response: ServerUploadResponse = request.send().await?.json().await?;

        Ok(IndexResult {
            document_id: response.document_id,
            chunks_created: response.chunk_count.unwrap_or(0) as usize,
        })
    }

    /// Get collections
    pub async fn get_collections(
        &self,
        server_url: &str,
        api_key: &Option<String>,
    ) -> Result<Vec<Collection>> {
        let url = format!("{}/api/v1/collections", server_url);

        let response: Vec<Collection> = self
            .client
            .get(&url)
            .headers(self.get_headers(api_key))
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }

    /// Get documents
    pub async fn get_documents(
        &self,
        server_url: &str,
        api_key: &Option<String>,
        collection_id: Option<&str>,
    ) -> Result<Vec<Document>> {
        let mut url = format!("{}/api/v1/documents", server_url);
        if let Some(coll) = collection_id {
            url = format!("{}?collection_id={}", url, coll);
        }

        let response: DocumentsResponse = self
            .client
            .get(&url)
            .headers(self.get_headers(api_key))
            .send()
            .await?
            .json()
            .await?;

        Ok(response.documents)
    }
}

// =============================================================================
// API Types
// =============================================================================

#[derive(Debug, Serialize)]
struct ServerChatRequest {
    query: String,
    collection_id: Option<String>,
    top_k: Option<i32>,
    include_sources: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ServerChatResponse {
    answer: String,
    sources: Vec<ServerSource>,
}

#[derive(Debug, Serialize)]
struct ServerSearchRequest {
    query: String,
    collection_id: Option<String>,
    top_k: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct ServerSearchResponse {
    results: Vec<ServerSource>,
}

#[derive(Debug, Deserialize)]
struct ServerSource {
    chunk_id: String,
    document_id: String,
    document_title: Option<String>,
    content: String,
    score: f32,
    chunk_index: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct ServerUploadResponse {
    document_id: String,
    chunk_count: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub document_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub name: String,
    pub file_type: Option<String>,
    pub chunk_count: i32,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
struct DocumentsResponse {
    documents: Vec<Document>,
}
