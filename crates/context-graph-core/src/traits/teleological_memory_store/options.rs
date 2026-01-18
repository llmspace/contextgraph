//! Search options for teleological memory queries.

use serde::{Deserialize, Serialize};

use crate::types::fingerprint::SemanticFingerprint;

/// Search options for teleological memory queries.
///
/// Controls filtering, pagination, and result formatting for
/// semantic and purpose-based searches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalSearchOptions {
    /// Maximum number of results to return.
    /// Default: 10, Max: 1000
    pub top_k: usize,

    /// Minimum similarity threshold [0.0, 1.0].
    /// Results below this threshold are filtered out.
    /// Default: 0.0 (no filtering)
    pub min_similarity: f32,

    /// Include soft-deleted items in results.
    /// Default: false
    pub include_deleted: bool,

    /// Filter by minimum alignment to Strategic goals.
    /// None = no filtering.
    pub min_alignment: Option<f32>,

    /// Embedder indices to use for search (0-12).
    /// Empty = use all embedders.
    pub embedder_indices: Vec<usize>,

    /// Optional semantic fingerprint for computing per-embedder scores.
    /// When provided in search_purpose(), enables computation of actual
    /// cosine similarity scores for each embedder instead of returning zeros.
    /// This is essential for search_teleological to return meaningful embedder_scores.
    #[serde(skip)]
    pub semantic_query: Option<SemanticFingerprint>,

    /// Whether to include original content text in search results.
    ///
    /// When `true`, the `content` field of `TeleologicalSearchResult` will be
    /// populated with the original text (if available). When `false` (default),
    /// the `content` field will be `None` for better performance.
    ///
    /// Default: `false` (opt-in for performance reasons)
    ///
    /// TASK-CONTENT-005: Added for content hydration in search results.
    #[serde(default)]
    pub include_content: bool,
}

impl Default for TeleologicalSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_deleted: false,
            min_alignment: None,
            embedder_indices: Vec::new(),
            semantic_query: None,
            include_content: false, // TASK-CONTENT-005: Opt-in for performance
        }
    }
}

impl TeleologicalSearchOptions {
    /// Create options for a quick top-k search.
    #[inline]
    pub fn quick(top_k: usize) -> Self {
        Self {
            top_k,
            ..Default::default()
        }
    }

    /// Create options with minimum similarity threshold.
    #[inline]
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold;
        self
    }

    /// Create options with alignment filter.
    #[inline]
    pub fn with_min_alignment(mut self, threshold: f32) -> Self {
        self.min_alignment = Some(threshold);
        self
    }

    /// Create options filtering by specific embedders.
    #[inline]
    pub fn with_embedders(mut self, indices: Vec<usize>) -> Self {
        self.embedder_indices = indices;
        self
    }

    /// Attach semantic fingerprint for computing per-embedder similarity scores.
    /// When provided, search_purpose() will compute actual cosine similarities
    /// between query and stored semantic fingerprints instead of returning zeros.
    #[inline]
    pub fn with_semantic_query(mut self, semantic: SemanticFingerprint) -> Self {
        self.semantic_query = Some(semantic);
        self
    }

    /// Set whether to include original content text in search results.
    ///
    /// When `true`, content will be fetched and included in results.
    /// Default is `false` for better performance.
    ///
    /// TASK-CONTENT-005: Builder method for content inclusion.
    #[inline]
    pub fn with_include_content(mut self, include: bool) -> Self {
        self.include_content = include;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_default() {
        let opts = TeleologicalSearchOptions::default();
        assert_eq!(opts.top_k, 10);
        assert_eq!(opts.min_similarity, 0.0);
        assert!(!opts.include_deleted);
        assert!(opts.min_alignment.is_none());
        assert!(opts.embedder_indices.is_empty());
    }

    #[test]
    fn test_search_options_quick() {
        let opts = TeleologicalSearchOptions::quick(50);
        assert_eq!(opts.top_k, 50);
    }

    #[test]
    fn test_search_options_builder() {
        let opts = TeleologicalSearchOptions::quick(20)
            .with_min_similarity(0.5)
            .with_min_alignment(0.75)
            .with_embedders(vec![0, 1, 2]);

        assert_eq!(opts.top_k, 20);
        assert_eq!(opts.min_similarity, 0.5);
        assert_eq!(opts.min_alignment, Some(0.75));
        assert_eq!(opts.embedder_indices, vec![0, 1, 2]);
    }
}
