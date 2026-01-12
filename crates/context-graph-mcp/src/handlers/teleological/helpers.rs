//! Helper functions for Teleological handlers.
//!
//! Contains shared helper methods used by multiple teleological handlers.

use super::utils::{compute_alignments_from_embeddings, extract_embeddings_from_fingerprint};
use crate::handlers::Handlers;
use context_graph_core::teleological::{
    groups::GroupAlignments, services::FusionEngine, TeleologicalVector,
};
use context_graph_core::types::fingerprint::SemanticFingerprint;
use tracing::error;
use uuid::Uuid;

impl Handlers {
    /// Helper: Retrieve TeleologicalVector from store by ID.
    pub(super) async fn retrieve_vector_from_store(
        &self,
        vector_id: &str,
    ) -> Result<(TeleologicalVector, Option<SemanticFingerprint>), String> {
        let uuid = Uuid::parse_str(vector_id).map_err(|e| {
            format!(
                "FAIL FAST: Invalid query_vector_id '{}': {}",
                vector_id, e
            )
        })?;

        // Retrieve fingerprint from store
        match self.teleological_store.retrieve(uuid).await {
            Ok(Some(fingerprint)) => {
                // Keep semantic fingerprint for embedder score computation
                let semantic_fp = fingerprint.semantic.clone();

                // Create a basic TeleologicalVector from the stored fingerprint's purpose_vector
                let purpose_vector = fingerprint.purpose_vector.clone();

                // For cross-correlations, compute from the fingerprint's semantic embeddings
                // This is a simplified version - full computation would use all embedders
                let cross_correlations = vec![0.0f32; 78]; // 13*(13-1)/2 = 78 pairs

                // Group alignments from purpose vector components
                let group_alignments = GroupAlignments::new(
                    purpose_vector.alignments[0], // factual
                    purpose_vector.alignments[1], // temporal
                    purpose_vector.alignments[2], // causal
                    purpose_vector.alignments[3], // relational
                    purpose_vector.alignments[4], // qualitative
                    purpose_vector.alignments[5], // implementation
                );

                let tv = TeleologicalVector::with_all(
                    purpose_vector,
                    cross_correlations,
                    group_alignments,
                    1.0, // confidence
                );
                Ok((tv, Some(semantic_fp)))
            }
            Ok(None) => Err(format!(
                "FAIL FAST: query_vector_id '{}' not found in store",
                vector_id
            )),
            Err(e) => {
                error!("Failed to retrieve vector {}: {}", vector_id, e);
                Err(format!(
                    "FAIL FAST: Failed to retrieve query_vector_id '{}': {}",
                    vector_id, e
                ))
            }
        }
    }

    /// Helper: Compute TeleologicalVector and SemanticFingerprint from content string.
    ///
    /// Returns both the fused TeleologicalVector and the SemanticFingerprint.
    /// The SemanticFingerprint is needed for computing per-embedder similarity scores
    /// in search_purpose() - without it, embedder_scores would be all zeros.
    ///
    /// Shared logic between search_teleological and compute_teleological_vector.
    pub(super) async fn compute_query_vector_from_content(
        &self,
        content: &str,
    ) -> Result<(TeleologicalVector, SemanticFingerprint), String> {
        // Use multi_array_provider to get all 13 embeddings
        let embedding_result = match self.multi_array_provider.embed_all(content).await {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to compute embeddings for query: {}", e);
                return Err(format!(
                    "FAIL FAST: Embedding computation failed for query_content: {}. \
                     Check embedding provider connection and configuration.",
                    e
                ));
            }
        };

        // Keep the semantic fingerprint for embedder score computation
        // Note: MultiArrayEmbeddingOutput.fingerprint IS the SemanticFingerprint
        let semantic_fingerprint = embedding_result.fingerprint.clone();

        // CONSTITUTION COMPLIANT: Extract embeddings using helper
        let embeddings = extract_embeddings_from_fingerprint(&embedding_result.fingerprint)
            .map_err(|e| e.to_error_string())?;

        // Compute alignments and fuse
        let alignments = compute_alignments_from_embeddings(&embeddings);
        let fusion_engine = FusionEngine::new();
        let fusion_result = fusion_engine.fuse_from_alignments(&alignments);

        Ok((fusion_result.vector, semantic_fingerprint))
    }
}
