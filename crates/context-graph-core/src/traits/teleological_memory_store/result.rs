//! Search result types for teleological memory queries.

use serde::{Deserialize, Serialize};

use crate::types::fingerprint::TeleologicalFingerprint;

/// Search result from teleological memory queries.
///
/// Contains the matched fingerprint along with scoring metadata
/// for ranking and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalSearchResult {
    /// The matched teleological fingerprint.
    pub fingerprint: TeleologicalFingerprint,

    /// Overall similarity score [0.0, 1.0].
    /// Computed differently depending on search type.
    pub similarity: f32,

    /// Per-embedder similarity scores (13 values for E1-E13).
    /// Sparse embeddings (E6, E13) use sparse dot product.
    pub embedder_scores: [f32; 13],

    /// Purpose alignment score (cosine similarity of purpose vectors).
    pub purpose_alignment: f32,

    /// Stage scores from the 5-stage retrieval pipeline.
    /// [sparse_recall, semantic_ann, precision, rerank, teleological]
    pub stage_scores: [f32; 5],

    /// Original content text (if requested and available).
    ///
    /// This field is `None` when:
    /// - `include_content=false` in search options (default)
    /// - Content was never stored for this fingerprint
    /// - Backend doesn't support content storage
    ///
    /// TASK-CONTENT-004: Added for content hydration in search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl TeleologicalSearchResult {
    /// Create a new search result with computed scores.
    pub fn new(
        fingerprint: TeleologicalFingerprint,
        similarity: f32,
        embedder_scores: [f32; 13],
        purpose_alignment: f32,
    ) -> Self {
        Self {
            fingerprint,
            similarity,
            embedder_scores,
            purpose_alignment,
            stage_scores: [0.0; 5], // Populated by pipeline stages
            content: None,          // Populated by content hydration
        }
    }

    /// Get the dominant embedder (highest score).
    pub fn dominant_embedder(&self) -> usize {
        self.embedder_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::fingerprint::{JohariFingerprint, PurposeVector, SemanticFingerprint};

    #[test]
    fn test_search_result_dominant_embedder() {
        let mut scores = [0.1; 13];
        scores[5] = 0.9; // E6 is dominant

        let result = TeleologicalSearchResult {
            fingerprint: TeleologicalFingerprint::new(
                SemanticFingerprint::zeroed(),
                PurposeVector::default(),
                JohariFingerprint::zeroed(),
                [0u8; 32],
            ),
            similarity: 0.8,
            embedder_scores: scores,
            purpose_alignment: 0.7,
            stage_scores: [0.0; 5],
            content: None,
        };

        assert_eq!(result.dominant_embedder(), 5);
    }
}
