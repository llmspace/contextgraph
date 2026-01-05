//! Embedding provider trait for text-to-vector conversion.
//!
//! This trait defines the interface for embedding generation that MCP handlers
//! use. Implementations include:
//! - `StubEmbeddingProvider`: Deterministic test embeddings (hash-based)
//! - Real Candle-based provider (M06-T04)
//!
//! # Performance Requirements (constitution.yaml:108-122)
//! - Single embed: <10ms
//! - Batch embed (64): <50ms
//!
//! # FAIL FAST: No fallbacks. Errors propagate immediately.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::{CoreError, CoreResult};

/// Result of embedding generation with metadata.
///
/// Contains the embedding vector plus diagnostic information
/// for performance monitoring and debugging.
///
/// # Example
///
/// ```rust
/// use context_graph_core::traits::EmbeddingOutput;
/// use std::time::Duration;
///
/// let output = EmbeddingOutput::new(
///     vec![0.1, 0.2, 0.3],
///     "stub-embedding-v1",
///     Duration::from_micros(500),
/// ).unwrap();
///
/// assert_eq!(output.dimensions, 3);
/// assert_eq!(output.model_id, "stub-embedding-v1");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    /// The embedding vector (1536 dimensions by default).
    pub vector: Vec<f32>,
    /// Model identifier that generated this embedding.
    pub model_id: String,
    /// Actual vector dimensions.
    pub dimensions: usize,
    /// Time taken to generate embedding.
    #[serde(with = "duration_serde")]
    pub latency: Duration,
}

/// Custom serde implementation for Duration.
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_nanos().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = u128::deserialize(deserializer)?;
        Ok(Duration::from_nanos(nanos as u64))
    }
}

impl EmbeddingOutput {
    /// Create new embedding output with validation.
    ///
    /// # Arguments
    ///
    /// * `vector` - The embedding vector (must not be empty)
    /// * `model_id` - Identifier of the model that generated this embedding
    /// * `latency` - Time taken to generate the embedding
    ///
    /// # Returns
    ///
    /// `EmbeddingOutput` on success, or `CoreError::Embedding` if vector is empty.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::Embedding` if vector is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_core::traits::EmbeddingOutput;
    /// use std::time::Duration;
    ///
    /// // Valid embedding
    /// let result = EmbeddingOutput::new(
    ///     vec![0.1, 0.2, 0.3],
    ///     "test-model",
    ///     Duration::from_millis(5),
    /// );
    /// assert!(result.is_ok());
    ///
    /// // Empty vector fails
    /// let result = EmbeddingOutput::new(
    ///     vec![],
    ///     "test-model",
    ///     Duration::from_millis(5),
    /// );
    /// assert!(result.is_err());
    /// ```
    pub fn new(
        vector: Vec<f32>,
        model_id: impl Into<String>,
        latency: Duration,
    ) -> CoreResult<Self> {
        if vector.is_empty() {
            return Err(CoreError::Embedding("Empty embedding vector".into()));
        }
        let dimensions = vector.len();
        Ok(Self {
            vector,
            model_id: model_id.into(),
            dimensions,
            latency,
        })
    }

    /// Get the magnitude (L2 norm) of the embedding vector.
    ///
    /// Used to verify normalization.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_core::traits::EmbeddingOutput;
    /// use std::time::Duration;
    ///
    /// let output = EmbeddingOutput::new(
    ///     vec![0.6, 0.8],  // 3-4-5 triangle, magnitude = 1.0
    ///     "test",
    ///     Duration::from_millis(1),
    /// ).unwrap();
    ///
    /// assert!((output.magnitude() - 1.0).abs() < 0.001);
    /// ```
    pub fn magnitude(&self) -> f32 {
        self.vector.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

/// Trait for embedding generation.
///
/// Provides async interface for converting text to dense vector representations.
/// All implementations must be thread-safe (Send + Sync) for use in async handlers.
///
/// # Error Handling
///
/// FAIL FAST: Errors propagate immediately. No fallbacks to fake embeddings.
/// If embedding generation fails, the caller receives the error.
///
/// # Performance
///
/// Implementations must meet these budgets (constitution.yaml:115-116):
/// - Single embed: <10ms
/// - Batch embed (64): <50ms
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_core::traits::EmbeddingProvider;
/// use context_graph_core::stubs::StubEmbeddingProvider;
///
/// let provider = StubEmbeddingProvider::new();
/// let embedding = provider.embed("Some text content").await?;
/// assert_eq!(embedding.dimensions, 1536);
/// ```
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for single text content.
    ///
    /// # Arguments
    ///
    /// * `content` - Text to embed (max 8192 tokens typical)
    ///
    /// # Returns
    ///
    /// `EmbeddingOutput` with vector and metadata on success.
    ///
    /// # Errors
    ///
    /// - `CoreError::Embedding` if generation fails or content is empty
    ///
    /// # Performance
    ///
    /// Target: <10ms (constitution.yaml:115)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output = provider.embed("Hello world").await?;
    /// assert_eq!(output.dimensions, provider.dimensions());
    /// ```
    async fn embed(&self, content: &str) -> CoreResult<EmbeddingOutput>;

    /// Generate embeddings for batch of texts.
    ///
    /// More efficient than calling `embed` in a loop.
    /// Implementations should process in parallel where possible.
    ///
    /// # Arguments
    ///
    /// * `contents` - Slice of texts to embed
    ///
    /// # Returns
    ///
    /// Vector of `EmbeddingOutput` in same order as input.
    ///
    /// # Errors
    ///
    /// - `CoreError::Embedding` if any generation fails
    ///
    /// # Performance
    ///
    /// Target: <50ms for 64 items (constitution.yaml:116)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let contents = vec!["first".to_string(), "second".to_string()];
    /// let outputs = provider.embed_batch(&contents).await?;
    /// assert_eq!(outputs.len(), 2);
    /// ```
    async fn embed_batch(&self, contents: &[String]) -> CoreResult<Vec<EmbeddingOutput>>;

    /// Get the output dimension for embeddings.
    ///
    /// Default is 1536 (OpenAI ada-002 compatible, FuseMoE output).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dim = provider.dimensions();
    /// assert_eq!(dim, 1536);
    /// ```
    fn dimensions(&self) -> usize;

    /// Get the model identifier string.
    ///
    /// Used for logging, debugging, and `EmbeddingOutput.model_id`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let id = provider.model_id();
    /// assert!(id.contains("embedding"));
    /// ```
    fn model_id(&self) -> &str;

    /// Check if provider is ready to generate embeddings.
    ///
    /// Returns false if model needs initialization (weight loading, GPU warm-up).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if provider.is_ready() {
    ///     let output = provider.embed("text").await?;
    /// }
    /// ```
    fn is_ready(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_output_new_success() {
        let result = EmbeddingOutput::new(
            vec![0.1, 0.2, 0.3],
            "test-model",
            Duration::from_millis(5),
        );
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dimensions, 3);
        assert_eq!(output.model_id, "test-model");
        assert_eq!(output.vector.len(), 3);
    }

    #[test]
    fn test_embedding_output_empty_vector_fails() {
        let result = EmbeddingOutput::new(vec![], "test-model", Duration::from_millis(5));
        assert!(result.is_err());
        match result {
            Err(CoreError::Embedding(msg)) => {
                assert!(msg.contains("Empty"));
            }
            _ => panic!("Expected CoreError::Embedding"),
        }
    }

    #[test]
    fn test_embedding_output_magnitude() {
        // 3-4-5 triangle scaled: 0.6^2 + 0.8^2 = 0.36 + 0.64 = 1.0
        let output =
            EmbeddingOutput::new(vec![0.6, 0.8], "test", Duration::from_millis(1)).unwrap();
        assert!((output.magnitude() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_output_serialization() {
        let output =
            EmbeddingOutput::new(vec![0.1, 0.2], "test-model", Duration::from_micros(500)).unwrap();
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("test-model"));

        let deserialized: EmbeddingOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_id, output.model_id);
        assert_eq!(deserialized.dimensions, output.dimensions);
    }
}
