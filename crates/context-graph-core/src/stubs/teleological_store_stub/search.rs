//! Search operations for the in-memory teleological store.
//!
//! This module implements semantic, purpose, text, and sparse search operations.
//! All operations are O(n) full table scans - suitable for testing only.

use std::collections::HashSet;

use tracing::{debug, error, warn};
use uuid::Uuid;

use super::similarity::compute_semantic_scores;
use super::InMemoryTeleologicalStore;
use crate::error::{CoreError, CoreResult};
use crate::traits::{TeleologicalSearchOptions, TeleologicalSearchResult};
use crate::types::fingerprint::{PurposeVector, SemanticFingerprint, SparseVector, NUM_EMBEDDERS};

impl InMemoryTeleologicalStore {
    /// Search by semantic fingerprint similarity.
    pub async fn search_semantic_impl(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        debug!(
            "Semantic search with top_k={}, min_similarity={}",
            options.top_k, options.min_similarity
        );

        let mut results: Vec<TeleologicalSearchResult> = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            let id = *entry.key();
            let fp = entry.value();

            if !options.include_deleted && deleted_ids.contains(&id) {
                continue;
            }

            let embedder_scores = compute_semantic_scores(query, &fp.semantic);

            let active_scores: Vec<f32> = if options.embedder_indices.is_empty() {
                embedder_scores.to_vec()
            } else {
                options
                    .embedder_indices
                    .iter()
                    .filter_map(|&i| embedder_scores.get(i).copied())
                    .collect()
            };

            let similarity = if active_scores.is_empty() {
                0.0
            } else {
                active_scores.iter().sum::<f32>() / active_scores.len() as f32
            };

            if similarity < options.min_similarity {
                continue;
            }

            if let Some(min_align) = options.min_alignment {
                if fp.alignment_score < min_align {
                    continue;
                }
            }

            let purpose_alignment = fp.purpose_vector.similarity(&PurposeVector::default());

            results.push(TeleologicalSearchResult::new(
                fp.clone(),
                similarity,
                embedder_scores,
                purpose_alignment,
            ));
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(options.top_k);
        debug!("Semantic search returned {} results", results.len());
        Ok(results)
    }

    /// Search by purpose vector similarity.
    pub async fn search_purpose_impl(
        &self,
        query: &PurposeVector,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        debug!(
            "Purpose search with top_k={}, min_similarity={}",
            options.top_k, options.min_similarity
        );

        let mut results: Vec<TeleologicalSearchResult> = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            let id = *entry.key();
            let fp = entry.value();

            if !options.include_deleted && deleted_ids.contains(&id) {
                continue;
            }

            let purpose_alignment = query.similarity(&fp.purpose_vector);

            if purpose_alignment < options.min_similarity {
                continue;
            }

            if let Some(min_align) = options.min_alignment {
                if fp.alignment_score < min_align {
                    continue;
                }
            }

            let embedder_scores = match &options.semantic_query {
                Some(query_semantic) => compute_semantic_scores(query_semantic, &fp.semantic),
                None => {
                    warn!(
                        "search_purpose: No semantic_query provided - embedder_scores will be uniform."
                    );
                    [purpose_alignment; NUM_EMBEDDERS]
                }
            };

            results.push(TeleologicalSearchResult::new(
                fp.clone(),
                purpose_alignment,
                embedder_scores,
                purpose_alignment,
            ));
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(options.top_k);
        debug!("Purpose search returned {} results", results.len());
        Ok(results)
    }

    /// Search by text query (requires embedding provider).
    pub async fn search_text_impl(
        &self,
        _text: &str,
        _options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        error!(
            "search_text not supported in InMemoryTeleologicalStore (requires embedding provider)"
        );
        Err(CoreError::FeatureDisabled {
            feature: "text_search".to_string(),
        })
    }

    /// Search by sparse vector (SPLADE-style).
    pub async fn search_sparse_impl(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>> {
        debug!(
            "Sparse search with top_k={}, query nnz={}",
            top_k,
            sparse_query.nnz()
        );

        let mut results: Vec<(Uuid, f32)> = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            let id = *entry.key();
            let fp = entry.value();

            if deleted_ids.contains(&id) {
                continue;
            }

            let score = sparse_query.dot(&fp.semantic.e13_splade);

            if score > 0.0 {
                results.push((id, score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        debug!("Sparse search returned {} results", results.len());
        Ok(results)
    }
}
