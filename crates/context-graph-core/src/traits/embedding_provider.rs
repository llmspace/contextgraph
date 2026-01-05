//! Embedding provider trait for vector generation.
//!
//! **DEPRECATED**: This module contains the legacy [`EmbeddingProvider`] trait and
//! [`EmbeddingOutput`] struct which produce only a single 1536D vector.
//!
//! For new code, use [`MultiArrayEmbeddingProvider`] from the
//! `multi_array_embedding_provider` module, which generates complete 13-embedding
//! SemanticFingerprints with all semantic dimensions.
//!
//! # Migration Guide
//!
//! ```ignore
//! // OLD (deprecated):
//! use context_graph_core::traits::{EmbeddingProvider, EmbeddingOutput};
//!
//! // NEW (recommended):
//! use context_graph_core::traits::{MultiArrayEmbeddingProvider, MultiArrayEmbeddingOutput};
//! ```
//!
//! # Performance Requirements (constitution.yaml)
//!
//! - Single embed: <10ms latency
//! - Batch embed (64 items): <50ms latency
//! - Default dimensions: 1536 (OpenAI text-embedding-3-small compatible)
//!
//! # Example (Legacy)
//!
//! ```ignore
//! use context_graph_core::traits::{EmbeddingProvider, EmbeddingOutput};
//!
//! async fn generate_embedding<P: EmbeddingProvider>(provider: &P, content: &str) {
//!     let output = provider.embed(content).await.unwrap();
//!     assert_eq!(output.dimensions, provider.dimensions());
//!     assert_eq!(output.vector.len(), output.dimensions);
//! }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::CoreResult;

/// Output from an embedding generation operation.
///
/// Contains the generated embedding vector along with metadata about
/// the generation process including model identification and timing.
///
/// # Fields
///
/// - `vector`: The 1536-dimensional embedding vector (default dimension)
/// - `model_id`: Identifier of the model used for generation
/// - `dimensions`: Length of the embedding vector (typically 1536)
/// - `latency`: Time taken to generate the embedding
///
/// # Performance Constraints (constitution.yaml)
///
/// - Single embed latency: <10ms
/// - Batch embed (64) latency: <50ms
///
/// # Example
///
/// ```
/// use context_graph_core::traits::EmbeddingOutput;
/// use std::time::Duration;
///
/// let output = EmbeddingOutput {
///     vector: vec![0.1; 1536],
///     model_id: "text-embedding-3-small".to_string(),
///     dimensions: 1536,
///     latency: Duration::from_millis(5),
/// };
///
/// assert_eq!(output.vector.len(), output.dimensions);
/// assert!(output.latency < Duration::from_millis(10));
/// ```
#[deprecated(
    since = "0.2.0",
    note = "Use MultiArrayEmbeddingOutput for complete 13-embedding metrics"
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    /// The embedding vector (1536D by default).
    ///
    /// Each component is a 32-bit float representing a semantic dimension.
    /// The vector is L2-normalized for cosine similarity computation.
    pub vector: Vec<f32>,

    /// Model identifier used for embedding generation.
    ///
    /// Examples: "text-embedding-3-small", "text-embedding-ada-002"
    pub model_id: String,

    /// Dimensionality of the embedding vector.
    ///
    /// Standard dimension is 1536 for OpenAI models.
    /// Must equal `vector.len()`.
    pub dimensions: usize,

    /// Time taken to generate this embedding.
    ///
    /// Performance target: <10ms for single embeddings.
    #[serde(with = "duration_serde")]
    pub latency: Duration,
}

/// Serde support for Duration serialization.
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
    /// Create a new EmbeddingOutput with validation.
    ///
    /// # Arguments
    ///
    /// * `vector` - The embedding vector
    /// * `model_id` - Model identifier
    /// * `latency` - Generation time
    ///
    /// # Returns
    ///
    /// A new `EmbeddingOutput` with dimensions automatically set from vector length.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::EmbeddingOutput;
    /// use std::time::Duration;
    ///
    /// let output = EmbeddingOutput::new(
    ///     vec![0.1; 1536],
    ///     "text-embedding-3-small".to_string(),
    ///     Duration::from_millis(5),
    /// );
    /// assert_eq!(output.dimensions, 1536);
    /// ```
    pub fn new(vector: Vec<f32>, model_id: String, latency: Duration) -> Self {
        let dimensions = vector.len();
        Self {
            vector,
            model_id,
            dimensions,
            latency,
        }
    }
}

/// Trait for embedding generation providers.
///
/// Implementations provide semantic embedding generation capabilities,
/// transforming text content into dense vector representations suitable
/// for similarity search and retrieval operations.
///
/// # Performance Requirements (constitution.yaml)
///
/// Implementations MUST meet these performance targets:
/// - Single embed (`embed`): <10ms latency
/// - Batch embed of 64 items (`embed_batch`): <50ms latency
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to support concurrent access
/// in multi-threaded environments.
///
/// # Example Implementation
///
/// ```ignore
/// use async_trait::async_trait;
/// use context_graph_core::traits::{EmbeddingProvider, EmbeddingOutput};
/// use context_graph_core::error::CoreResult;
///
/// struct OpenAIEmbedder {
///     model: String,
///     dimensions: usize,
/// }
///
/// #[async_trait]
/// impl EmbeddingProvider for OpenAIEmbedder {
///     async fn embed(&self, content: &str) -> CoreResult<EmbeddingOutput> {
///         // Implementation that calls OpenAI API
///         todo!()
///     }
///
///     async fn embed_batch(&self, contents: &[String]) -> CoreResult<Vec<EmbeddingOutput>> {
///         // Batch implementation for efficiency
///         todo!()
///     }
///
///     fn dimensions(&self) -> usize {
///         self.dimensions
///     }
///
///     fn model_id(&self) -> &str {
///         &self.model
///     }
///
///     fn is_ready(&self) -> bool {
///         true
///     }
/// }
/// ```
#[deprecated(
    since = "0.2.0",
    note = "Use MultiArrayEmbeddingProvider for 13-embedding SemanticFingerprint generation. EmbeddingProvider produces only a single 1536D vector."
)]
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for a single piece of content.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to embed
    ///
    /// # Returns
    ///
    /// An [`EmbeddingOutput`] containing the embedding vector and metadata.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::Embedding` if:
    /// - Content is empty
    /// - Provider is not ready (`is_ready()` returns false)
    /// - Embedding generation fails
    /// - Timeout exceeded (>10ms target)
    ///
    /// # Performance
    ///
    /// Target latency: <10ms (constitution.yaml requirement)
    async fn embed(&self, content: &str) -> CoreResult<EmbeddingOutput>;

    /// Generate embeddings for multiple pieces of content in batch.
    ///
    /// Batch processing is more efficient than individual calls for
    /// multiple items, amortizing API overhead across the batch.
    ///
    /// # Arguments
    ///
    /// * `contents` - Slice of text content to embed
    ///
    /// # Returns
    ///
    /// A vector of [`EmbeddingOutput`] in the same order as input contents.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::Embedding` if:
    /// - Any content is empty
    /// - Provider is not ready
    /// - Batch processing fails
    /// - Timeout exceeded (>50ms for 64 items)
    ///
    /// # Performance
    ///
    /// Target latency for 64 items: <50ms (constitution.yaml requirement)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let contents = vec![
    ///     "First document".to_string(),
    ///     "Second document".to_string(),
    /// ];
    /// let embeddings = provider.embed_batch(&contents).await?;
    /// assert_eq!(embeddings.len(), 2);
    /// ```
    async fn embed_batch(&self, contents: &[String]) -> CoreResult<Vec<EmbeddingOutput>>;

    /// Get the dimensionality of embeddings produced by this provider.
    ///
    /// Standard dimension is 1536 for OpenAI text-embedding models.
    /// All embeddings from a provider will have this dimension.
    ///
    /// # Returns
    ///
    /// The number of dimensions in the embedding vectors (typically 1536).
    fn dimensions(&self) -> usize;

    /// Get the model identifier for this provider.
    ///
    /// Used for tracking which model generated embeddings and ensuring
    /// compatibility when comparing embeddings.
    ///
    /// # Returns
    ///
    /// The model ID string (e.g., "text-embedding-3-small").
    fn model_id(&self) -> &str;

    /// Check if the provider is ready to generate embeddings.
    ///
    /// Providers may require initialization (loading models, establishing
    /// connections) before they can generate embeddings.
    ///
    /// # Returns
    ///
    /// `true` if the provider is ready to accept embedding requests,
    /// `false` if initialization is incomplete or provider is in error state.
    fn is_ready(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_output_new() {
        let vector = vec![0.1, 0.2, 0.3];
        let output = EmbeddingOutput::new(
            vector.clone(),
            "test-model".to_string(),
            Duration::from_millis(5),
        );

        assert_eq!(output.vector, vector);
        assert_eq!(output.model_id, "test-model");
        assert_eq!(output.dimensions, 3);
        assert_eq!(output.latency, Duration::from_millis(5));
    }

    #[test]
    fn test_embedding_output_serialization() {
        let output = EmbeddingOutput::new(
            vec![0.1, 0.2, 0.3],
            "test-model".to_string(),
            Duration::from_millis(5),
        );

        let json = serde_json::to_string(&output).unwrap();
        let deserialized: EmbeddingOutput = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.vector, output.vector);
        assert_eq!(deserialized.model_id, output.model_id);
        assert_eq!(deserialized.dimensions, output.dimensions);
        assert_eq!(deserialized.latency, output.latency);
    }

    #[test]
    fn test_embedding_output_1536_dimensions() {
        let vector = vec![0.0; 1536];
        let output = EmbeddingOutput::new(
            vector,
            "text-embedding-3-small".to_string(),
            Duration::from_millis(8),
        );

        assert_eq!(output.dimensions, 1536);
        assert_eq!(output.vector.len(), 1536);
    }
}
