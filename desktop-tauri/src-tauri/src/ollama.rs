//! Ollama client for LOCAL MODE
//!
//! Provides chat, embedding, and model management.

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OllamaClient {
    client: Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Check if Ollama is running
    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}/api/tags", self.base_url);
        self.client.get(&url).send().await?.error_for_status()?;
        Ok(())
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        let response: ModelsResponse = self.client.get(&url).send().await?.json().await?;

        Ok(response.models.into_iter().map(|m| m.name).collect())
    }

    /// Generate text completion
    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.base_url);

        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
            options: Some(GenerateOptions {
                temperature: Some(0.7),
                num_predict: Some(2048),
                ..Default::default()
            }),
        };

        let response: GenerateResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.response)
    }

    /// Generate text completion with streaming
    pub async fn generate_stream<F>(
        &self,
        model: &str,
        prompt: &str,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let url = format!("{}/api/generate", self.base_url);

        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: true,
            options: Some(GenerateOptions {
                temperature: Some(0.7),
                num_predict: Some(2048),
                ..Default::default()
            }),
        };

        let response = self.client.post(&url).json(&request).send().await?;

        let mut full_response = String::new();
        let mut stream = response.bytes_stream();

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);

            for line in text.lines() {
                if let Ok(partial) = serde_json::from_str::<GenerateResponse>(line) {
                    full_response.push_str(&partial.response);
                    callback(&partial.response);
                }
            }
        }

        Ok(full_response)
    }

    /// Chat with context
    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            options: Some(GenerateOptions {
                temperature: Some(0.7),
                ..Default::default()
            }),
        };

        let response: ChatResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.message.content)
    }

    /// Generate embeddings for multiple texts
    pub async fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);

        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let request = EmbedRequest {
                model: model.to_string(),
                input: text.clone(),
            };

            let response: EmbedResponse = self
                .client
                .post(&url)
                .json(&request)
                .send()
                .await?
                .json()
                .await?;

            embeddings.push(response.embeddings.into_iter().next().unwrap_or_default());
        }

        Ok(embeddings)
    }

    /// Generate single embedding
    pub async fn embed_single(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.base_url);

        let request = EmbedRequest {
            model: model.to_string(),
            input: text.to_string(),
        };

        let response: EmbedResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.embeddings.into_iter().next().unwrap_or_default())
    }

    /// Pull a model
    pub async fn pull_model(&self, model: &str) -> Result<()> {
        let url = format!("{}/api/pull", self.base_url);

        let request = PullRequest {
            name: model.to_string(),
            stream: false,
        };

        self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Debug, Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<GenerateOptions>,
}

#[derive(Debug, Default, Serialize)]
struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
    #[allow(dead_code)]
    done: bool,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<GenerateOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
}

#[derive(Debug, Serialize)]
struct PullRequest {
    name: String,
    stream: bool,
}
