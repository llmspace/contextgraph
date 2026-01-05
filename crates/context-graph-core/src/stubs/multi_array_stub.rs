//! Stub implementation of MultiArrayEmbeddingProvider for testing.
//!
//! Provides a deterministic, fast stub that generates reproducible embeddings
//! based on content hash. This is useful for unit tests and development
//! before real embedding models are integrated.
//!
//! # Determinism
//!
//! The stub generates embeddings based on the content's byte sum, ensuring:
//! - Same content always produces the same embedding
//! - Different content produces different embeddings
//! - No external dependencies or randomness
//!
//! # Performance
//!
//! Simulates 5ms latency per embedder (65ms total) for realistic test timing.

use async_trait::async_trait;
use std::time::Duration;

use crate::error::CoreResult;
use crate::traits::{MultiArrayEmbeddingOutput, MultiArrayEmbeddingProvider};
use crate::types::fingerprint::{SemanticFingerprint, SparseVector, NUM_EMBEDDERS};

/// Stub implementation of MultiArrayEmbeddingProvider for testing.
///
/// Generates deterministic embeddings based on content hash.
/// No external model dependencies - pure computation based on input bytes.
///
/// # Example
///
/// ```
/// use context_graph_core::stubs::StubMultiArrayProvider;
/// use context_graph_core::traits::MultiArrayEmbeddingProvider;
///
/// let provider = StubMultiArrayProvider::new();
/// assert!(provider.is_ready());
///
/// // Same content produces same embedding
/// // let output1 = provider.embed_all("test").await.unwrap();
/// // let output2 = provider.embed_all("test").await.unwrap();
/// // assert_eq!(output1.fingerprint, output2.fingerprint);
/// ```
#[derive(Debug, Clone, Default)]
pub struct StubMultiArrayProvider;

impl StubMultiArrayProvider {
    /// Create a new StubMultiArrayProvider.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Generate a deterministic base value from content.
    ///
    /// Uses byte sum modulo 256 to create a value in [0, 1].
    #[inline]
    fn content_hash(content: &str) -> f32 {
        let sum: u32 = content.bytes().map(u32::from).sum();
        (sum % 256) as f32 / 255.0
    }

    /// Generate a deterministic value for a specific dimension.
    ///
    /// Combines content hash with dimension index for variety.
    #[inline]
    fn deterministic_value(base: f32, dim_idx: usize) -> f32 {
        // Create variation across dimensions while remaining deterministic
        let offset = ((dim_idx as f32 * 0.0173) % 1.0) - 0.5;
        (base + offset).clamp(0.0, 1.0)
    }

    /// Fill a dense embedding vector deterministically.
    fn fill_dense_embedding(content: &str, dim: usize) -> Vec<f32> {
        let base = Self::content_hash(content);
        (0..dim)
            .map(|i| Self::deterministic_value(base, i))
            .collect()
    }

    /// Generate a deterministic sparse vector.
    fn generate_sparse_vector(content: &str) -> SparseVector {
        let base_sum: u32 = content.bytes().map(u32::from).sum();
        // Generate 10 sparse entries based on content
        let num_entries = 10.min(((base_sum % 20) + 5) as usize);

        let mut indices: Vec<u16> = (0..num_entries)
            .map(|i| ((base_sum as usize * 17 + i * 2003) % 30000) as u16)
            .collect();
        indices.sort();
        indices.dedup();

        let values: Vec<f32> = indices
            .iter()
            .enumerate()
            .map(|(i, _)| Self::deterministic_value(Self::content_hash(content), i))
            .collect();

        SparseVector::new(indices, values).unwrap_or_else(|_| SparseVector::empty())
    }

    /// Generate deterministic token embeddings.
    fn generate_token_embeddings(content: &str) -> Vec<Vec<f32>> {
        let base = Self::content_hash(content);
        // Generate ~1 token per 5 characters, minimum 1
        let num_tokens = ((content.len() / 5).max(1)).min(64);

        (0..num_tokens)
            .map(|token_idx| {
                (0..128) // E12 has 128D per token
                    .map(|dim| Self::deterministic_value(base, dim + token_idx * 128))
                    .collect()
            })
            .collect()
    }
}

#[async_trait]
impl MultiArrayEmbeddingProvider for StubMultiArrayProvider {
    async fn embed_all(&self, content: &str) -> CoreResult<MultiArrayEmbeddingOutput> {
        // Generate deterministic fingerprint
        let mut fingerprint = SemanticFingerprint::zeroed();

        // Fill dense embeddings
        fingerprint.e1_semantic = Self::fill_dense_embedding(content, 1024);
        fingerprint.e2_temporal_recent = Self::fill_dense_embedding(content, 512);
        fingerprint.e3_temporal_periodic = Self::fill_dense_embedding(content, 512);
        fingerprint.e4_temporal_positional = Self::fill_dense_embedding(content, 512);
        fingerprint.e5_causal = Self::fill_dense_embedding(content, 768);
        fingerprint.e7_code = Self::fill_dense_embedding(content, 256);
        fingerprint.e8_graph = Self::fill_dense_embedding(content, 384);
        fingerprint.e9_hdc = Self::fill_dense_embedding(content, 10000);
        fingerprint.e10_multimodal = Self::fill_dense_embedding(content, 768);
        fingerprint.e11_entity = Self::fill_dense_embedding(content, 384);

        // Fill sparse embeddings
        fingerprint.e6_sparse = Self::generate_sparse_vector(content);
        fingerprint.e13_splade = Self::generate_sparse_vector(content);

        // Fill token-level embeddings
        fingerprint.e12_late_interaction = Self::generate_token_embeddings(content);

        // Simulated latencies: 5ms per embedder
        let per_embedder_latency = [Duration::from_millis(5); NUM_EMBEDDERS];
        let total_latency = Duration::from_millis(5 * NUM_EMBEDDERS as u64);

        Ok(MultiArrayEmbeddingOutput {
            fingerprint,
            total_latency,
            per_embedder_latency,
            model_ids: core::array::from_fn(|i| format!("stub-e{}", i + 1)),
        })
    }

    async fn embed_batch_all(
        &self,
        contents: &[String],
    ) -> CoreResult<Vec<MultiArrayEmbeddingOutput>> {
        let mut results = Vec::with_capacity(contents.len());
        for content in contents {
            results.push(self.embed_all(content).await?);
        }
        Ok(results)
    }

    fn model_ids(&self) -> [&str; NUM_EMBEDDERS] {
        [
            "stub-e1",
            "stub-e2",
            "stub-e3",
            "stub-e4",
            "stub-e5",
            "stub-e6",
            "stub-e7",
            "stub-e8",
            "stub-e9",
            "stub-e10",
            "stub-e11",
            "stub-e12",
            "stub-e13",
        ]
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn health_status(&self) -> [bool; NUM_EMBEDDERS] {
        [true; NUM_EMBEDDERS]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the provider is always ready.
    #[test]
    fn test_is_ready() {
        let provider = StubMultiArrayProvider::new();
        assert!(provider.is_ready());
    }

    /// Test that health_status returns all true.
    #[test]
    fn test_health_status_all_true() {
        let provider = StubMultiArrayProvider::new();
        let health = provider.health_status();
        assert_eq!(health.len(), NUM_EMBEDDERS);
        assert!(health.iter().all(|&h| h));
    }

    /// Test that model_ids returns correct format.
    #[test]
    fn test_model_ids_format() {
        let provider = StubMultiArrayProvider::new();
        let ids = provider.model_ids();
        assert_eq!(ids.len(), NUM_EMBEDDERS);
        assert_eq!(ids[0], "stub-e1");
        assert_eq!(ids[12], "stub-e13");
    }

    /// Test deterministic output: same content produces same fingerprint.
    #[tokio::test]
    async fn test_deterministic_same_content() {
        let provider = StubMultiArrayProvider::new();

        let output1 = provider.embed_all("test content").await.unwrap();
        let output2 = provider.embed_all("test content").await.unwrap();

        // Fingerprints should be identical for same content
        assert_eq!(output1.fingerprint, output2.fingerprint);
    }

    /// Test deterministic output: different content produces different fingerprint.
    #[tokio::test]
    async fn test_deterministic_different_content() {
        let provider = StubMultiArrayProvider::new();

        let output1 = provider.embed_all("content A").await.unwrap();
        let output2 = provider.embed_all("content B").await.unwrap();

        // Fingerprints should differ for different content
        assert_ne!(output1.fingerprint, output2.fingerprint);
    }

    /// Test that simulated latency is 5ms per embedder.
    #[tokio::test]
    async fn test_latency_simulation() {
        let provider = StubMultiArrayProvider::new();
        let output = provider.embed_all("test").await.unwrap();

        // Total latency should be 5ms * 13 = 65ms
        assert_eq!(output.total_latency, Duration::from_millis(65));

        // Each embedder should have 5ms latency
        for latency in output.per_embedder_latency.iter() {
            assert_eq!(*latency, Duration::from_millis(5));
        }
    }

    /// Test that embed_all produces correct embedding dimensions.
    #[tokio::test]
    async fn test_embedding_dimensions() {
        let provider = StubMultiArrayProvider::new();
        let output = provider.embed_all("test content").await.unwrap();

        let fp = &output.fingerprint;
        assert_eq!(fp.e1_semantic.len(), 1024);
        assert_eq!(fp.e2_temporal_recent.len(), 512);
        assert_eq!(fp.e3_temporal_periodic.len(), 512);
        assert_eq!(fp.e4_temporal_positional.len(), 512);
        assert_eq!(fp.e5_causal.len(), 768);
        assert_eq!(fp.e7_code.len(), 256);
        assert_eq!(fp.e8_graph.len(), 384);
        assert_eq!(fp.e9_hdc.len(), 10000);
        assert_eq!(fp.e10_multimodal.len(), 768);
        assert_eq!(fp.e11_entity.len(), 384);
    }

    /// Test that sparse vectors are generated correctly.
    #[tokio::test]
    async fn test_sparse_vectors() {
        let provider = StubMultiArrayProvider::new();
        let output = provider.embed_all("test content for sparse").await.unwrap();

        // Sparse vectors should have entries
        assert!(!output.fingerprint.e6_sparse.is_empty());
        assert!(!output.fingerprint.e13_splade.is_empty());

        // Check that indices are sorted (validation happens in SparseVector::new)
        let e6 = &output.fingerprint.e6_sparse;
        for i in 1..e6.indices.len() {
            assert!(e6.indices[i] > e6.indices[i - 1]);
        }
    }

    /// Test that token embeddings are generated correctly.
    #[tokio::test]
    async fn test_token_embeddings() {
        let provider = StubMultiArrayProvider::new();
        let output = provider.embed_all("this is a longer test content for tokens")
            .await
            .unwrap();

        // Should have multiple tokens
        assert!(!output.fingerprint.e12_late_interaction.is_empty());

        // Each token should have 128D
        for token_embed in &output.fingerprint.e12_late_interaction {
            assert_eq!(token_embed.len(), 128);
        }
    }

    /// Test batch embedding.
    #[tokio::test]
    async fn test_batch_embedding() {
        let provider = StubMultiArrayProvider::new();
        let contents = vec![
            "first content".to_string(),
            "second content".to_string(),
            "third content".to_string(),
        ];

        let outputs = provider.embed_batch_all(&contents).await.unwrap();
        assert_eq!(outputs.len(), 3);

        // Each output should have valid fingerprint
        for output in &outputs {
            assert_eq!(output.fingerprint.e1_semantic.len(), 1024);
        }
    }

    /// Test that default dimensions method works.
    #[test]
    fn test_dimensions_default() {
        let provider = StubMultiArrayProvider::new();
        let dims = provider.dimensions();

        assert_eq!(dims[0], 1024); // E1
        assert_eq!(dims[1], 512); // E2
        assert_eq!(dims[5], 0); // E6 sparse
        assert_eq!(dims[8], 10000); // E9 HDC
        assert_eq!(dims[12], 0); // E13 sparse
    }

    /// Test empty content handling.
    #[tokio::test]
    async fn test_empty_content() {
        let provider = StubMultiArrayProvider::new();
        let output = provider.embed_all("").await.unwrap();

        // Should produce valid fingerprint even for empty content
        assert_eq!(output.fingerprint.e1_semantic.len(), 1024);
    }

    /// Test that content_hash is deterministic.
    #[test]
    fn test_content_hash_determinism() {
        let hash1 = StubMultiArrayProvider::content_hash("test");
        let hash2 = StubMultiArrayProvider::content_hash("test");
        assert_eq!(hash1, hash2);

        let hash3 = StubMultiArrayProvider::content_hash("different");
        assert_ne!(hash1, hash3);
    }

    /// Test model_ids in output matches trait method.
    #[tokio::test]
    async fn test_output_model_ids_match() {
        let provider = StubMultiArrayProvider::new();
        let output = provider.embed_all("test").await.unwrap();

        let trait_ids = provider.model_ids();
        for (i, output_id) in output.model_ids.iter().enumerate() {
            assert_eq!(output_id, trait_ids[i]);
        }
    }
}
