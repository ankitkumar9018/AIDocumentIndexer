//! Local embedding generation
//!
//! Provides embedding generation using fastembed or Ollama.

use anyhow::Result;

/// Embedding model type
#[derive(Debug, Clone)]
pub enum EmbeddingModel {
    /// Use Ollama for embeddings (default)
    Ollama(String),
    /// Use fastembed local model (optional feature)
    #[cfg(feature = "local-embeddings")]
    FastEmbed(String),
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        EmbeddingModel::Ollama("nomic-embed-text".to_string())
    }
}

/// Generate embeddings for texts
#[cfg(feature = "local-embeddings")]
pub async fn embed_texts_local(
    texts: &[String],
    _model: &str,
) -> Result<Vec<Vec<f32>>> {
    use fastembed::{EmbeddingModel, TextEmbedding};

    let model = TextEmbedding::try_new(
        fastembed::InitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_show_download_progress(true)
    )?;

    let embeddings = model.embed(
        texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        None,
    )?;

    Ok(embeddings)
}

/// Fallback when local-embeddings feature is not enabled
#[cfg(not(feature = "local-embeddings"))]
pub async fn embed_texts_local(
    _texts: &[String],
    _model: &str,
) -> Result<Vec<Vec<f32>>> {
    Err(anyhow::anyhow!(
        "Local embeddings not enabled. Use Ollama or enable 'local-embeddings' feature."
    ))
}

/// Calculate cosine similarity between two embedding vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Find top-k most similar embeddings
pub fn find_similar(
    query: &[f32],
    embeddings: &[Vec<f32>],
    top_k: usize,
) -> Vec<(usize, f32)> {
    let mut similarities: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(query, emb)))
        .collect();

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(top_k);
    similarities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_find_similar() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0],    // similarity = 1.0
            vec![0.0, 1.0],    // similarity = 0.0
            vec![0.707, 0.707], // similarity ≈ 0.707
        ];

        let results = find_similar(&query, &embeddings, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Most similar
        assert_eq!(results[1].0, 2); // Second most similar
    }
}
