//! GWT computation helper functions.
//!
//! TASK-UTL-P1-001: Extracted computation logic for delta S and delta C.

use serde_json::json;
use tracing::warn;

use context_graph_core::johari::NUM_EMBEDDERS;
use context_graph_core::teleological::Embedder;
use context_graph_core::types::fingerprint::{EmbeddingRef, TeleologicalFingerprint};
use context_graph_utl::coherence::{
    compute_cluster_fit, ClusterContext, ClusterFitConfig, ClusterFitResult, CoherenceTracker,
};
use context_graph_utl::config::{CoherenceConfig, SurpriseConfig};
use context_graph_utl::surprise::embedder_entropy::EmbedderEntropyFactory;

use super::constants::{ALPHA, BETA, EMBEDDER_NAMES, GAMMA};
use super::helpers::{create_divergent_cluster, mean_pool_tokens, sparse_to_dense_truncated};

/// Result of delta S computation per embedder.
pub(crate) struct DeltaSResult {
    pub per_embedder: [f32; NUM_EMBEDDERS],
    pub aggregate: f32,
    pub diagnostics: Vec<serde_json::Value>,
}

/// Result of delta C computation.
pub(crate) struct DeltaCResult {
    pub delta_c: f32,
    pub connectivity: f32,
    pub cluster_fit: f32,
    pub consistency: f32,
    pub cluster_fit_result: ClusterFitResult,
    pub similarity_weight: f32,
    pub consistency_weight: f32,
}

/// Compute delta S (entropy change) for each embedder.
pub(crate) fn compute_delta_s(
    old_fp: &TeleologicalFingerprint,
    new_fp: &TeleologicalFingerprint,
    include_diagnostics: bool,
) -> DeltaSResult {
    let surprise_config = SurpriseConfig::default();
    let mut delta_s_per_embedder = [0.0f32; NUM_EMBEDDERS];
    let mut diagnostics_per_embedder: Vec<serde_json::Value> = Vec::new();

    for embedder in Embedder::all() {
        let idx = embedder.index();
        let calculator = EmbedderEntropyFactory::create(embedder, &surprise_config);

        // Get embeddings from fingerprints
        let old_embedding = old_fp.semantic.get(embedder);
        let new_embedding = new_fp.semantic.get(embedder);

        // Extract dense vectors (sparse/token-level handled differently)
        let (old_vec, new_vec) = match (old_embedding, new_embedding) {
            (EmbeddingRef::Dense(old), EmbeddingRef::Dense(new)) => (old.to_vec(), new.to_vec()),
            (EmbeddingRef::Sparse(old_sparse), EmbeddingRef::Sparse(new_sparse)) => {
                // For sparse embeddings, convert to dense for delta S computation
                let max_dim = 1024; // Use a reasonable dimension
                let old_dense = sparse_to_dense_truncated(old_sparse, max_dim);
                let new_dense = sparse_to_dense_truncated(new_sparse, max_dim);
                (old_dense, new_dense)
            }
            (EmbeddingRef::TokenLevel(old_tokens), EmbeddingRef::TokenLevel(new_tokens)) => {
                // For token-level embeddings, use mean pooling
                let old_pooled = mean_pool_tokens(old_tokens);
                let new_pooled = mean_pool_tokens(new_tokens);
                (old_pooled, new_pooled)
            }
            _ => {
                // Mismatched types - should not happen with valid fingerprints
                warn!(
                    "gwt/compute_delta_sc: mismatched embedding types for {:?}",
                    embedder
                );
                delta_s_per_embedder[idx] = 1.0; // Max surprise for error
                continue;
            }
        };

        // Compute delta S
        let history = vec![old_vec.clone()];
        let delta_s = match calculator.compute_delta_s(&new_vec, &history, 5) {
            Ok(ds) => ds.clamp(0.0, 1.0),
            Err(e) => {
                warn!(
                    "gwt/compute_delta_sc: delta S computation failed for {:?}: {}",
                    embedder, e
                );
                1.0 // Max surprise on error
            }
        };

        // Check for NaN/Inf per AP-10
        let delta_s = if delta_s.is_nan() || delta_s.is_infinite() {
            warn!(
                "gwt/compute_delta_sc: delta S for {:?} was NaN/Inf, clamping to 1.0",
                embedder
            );
            1.0
        } else {
            delta_s
        };

        delta_s_per_embedder[idx] = delta_s;

        if include_diagnostics {
            diagnostics_per_embedder.push(json!({
                "embedder": EMBEDDER_NAMES[idx],
                "embedder_index": idx,
                "delta_s": delta_s,
                "old_embedding_dim": old_vec.len(),
                "new_embedding_dim": new_vec.len(),
            }));
        }
    }

    // Compute aggregate delta S (equal weights)
    let delta_s_aggregate: f32 =
        delta_s_per_embedder.iter().sum::<f32>() / NUM_EMBEDDERS as f32;
    let delta_s_aggregate = delta_s_aggregate.clamp(0.0, 1.0);

    DeltaSResult {
        per_embedder: delta_s_per_embedder,
        aggregate: delta_s_aggregate,
        diagnostics: diagnostics_per_embedder,
    }
}

/// Compute delta C (coherence change) using three-component formula.
///
/// Per constitution.yaml line 166:
/// delta C = 0.4*Connectivity + 0.4*ClusterFit + 0.2*Consistency
pub(crate) fn compute_delta_c(
    old_fp: &TeleologicalFingerprint,
    new_fp: &TeleologicalFingerprint,
) -> DeltaCResult {
    let coherence_config = CoherenceConfig::default();
    let tracker = CoherenceTracker::new(&coherence_config);

    // Use semantic embedding (E1) for coherence computation
    let old_semantic = &old_fp.semantic.e1_semantic;
    let new_semantic = &new_fp.semantic.e1_semantic;
    let history = vec![old_semantic.clone()];

    // 1. Connectivity component: similarity between old and new embeddings
    let connectivity = tracker
        .compute_coherence_legacy(new_semantic, &history)
        .clamp(0.0, 1.0);

    // 2. ClusterFit component: silhouette-based cluster fit
    let cluster_fit_config = ClusterFitConfig::default();

    // Create cluster context: same_cluster = old embedding, nearest = orthogonal
    let same_cluster = vec![old_semantic.clone()];

    // For nearest_cluster, use an embedding that represents "different" content
    let nearest_cluster = create_divergent_cluster(old_semantic, new_semantic);

    let cluster_context = ClusterContext::new(same_cluster, nearest_cluster);
    let cluster_fit_result = compute_cluster_fit(new_semantic, &cluster_context, &cluster_fit_config);
    let cluster_fit = cluster_fit_result.score;

    // 3. Consistency component: from CoherenceTracker's window variance
    let mut temp_tracker = CoherenceTracker::new(&coherence_config);
    temp_tracker.update(old_semantic);
    let consistency_raw = temp_tracker.update_and_compute(new_semantic);
    let consistency = consistency_raw.clamp(0.0, 1.0);

    // Combine components using constitution weights
    let delta_c_raw = ALPHA * connectivity + BETA * cluster_fit + GAMMA * consistency;
    let delta_c = delta_c_raw.clamp(0.0, 1.0);

    // Check for NaN/Inf per AP-10
    let delta_c = if delta_c.is_nan() || delta_c.is_infinite() {
        warn!("gwt/compute_delta_sc: delta C was NaN/Inf, clamping to 0.5");
        0.5
    } else {
        delta_c
    };

    DeltaCResult {
        delta_c,
        connectivity,
        cluster_fit,
        consistency,
        cluster_fit_result,
        similarity_weight: coherence_config.similarity_weight,
        consistency_weight: coherence_config.consistency_weight,
    }
}
