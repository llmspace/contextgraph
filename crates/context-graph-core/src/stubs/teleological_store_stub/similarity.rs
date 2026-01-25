//! Similarity calculation functions for the in-memory teleological store.
//!
//! This module contains cosine similarity and semantic score computation functions
//! used by the search operations.

use crate::types::fingerprint::{SemanticFingerprint, NUM_EMBEDDERS};

/// Compute cosine similarity between two dense vectors.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Compute semantic similarity across all embedders.
pub(crate) fn compute_semantic_scores(
    query: &SemanticFingerprint,
    target: &SemanticFingerprint,
) -> [f32; NUM_EMBEDDERS] {
    let mut scores = [0.0_f32; NUM_EMBEDDERS];

    // E1: Semantic
    scores[0] = cosine_similarity(&query.e1_semantic, &target.e1_semantic);

    // E2: Temporal Recent
    scores[1] = cosine_similarity(&query.e2_temporal_recent, &target.e2_temporal_recent);

    // E3: Temporal Periodic
    scores[2] = cosine_similarity(&query.e3_temporal_periodic, &target.e3_temporal_periodic);

    // E4: Temporal Positional
    scores[3] = cosine_similarity(
        &query.e4_temporal_positional,
        &target.e4_temporal_positional,
    );

    // E5: Causal
    scores[4] = cosine_similarity(&query.e5_causal, &target.e5_causal);

    // E6: Sparse (use sparse dot product normalized)
    scores[5] = query.e6_sparse.cosine_similarity(&target.e6_sparse);

    // E7: Code
    scores[6] = cosine_similarity(&query.e7_code, &target.e7_code);

    // E8: Graph - use active vector (supports both legacy and dual format)
    scores[7] = cosine_similarity(query.e8_active_vector(), target.e8_active_vector());

    // E9: HDC
    scores[8] = cosine_similarity(&query.e9_hdc, &target.e9_hdc);

    // E10: Multimodal - use active vector (supports both legacy and dual format)
    scores[9] = cosine_similarity(query.e10_active_vector(), target.e10_active_vector());

    // E11: Entity
    scores[10] = cosine_similarity(&query.e11_entity, &target.e11_entity);

    // E12: Late Interaction (simplified: average token similarities)
    scores[11] =
        compute_late_interaction_score(&query.e12_late_interaction, &target.e12_late_interaction);

    // E13: SPLADE
    scores[12] = query.e13_splade.cosine_similarity(&target.e13_splade);

    scores
}

/// Compute ColBERT-style late interaction score (MaxSim).
pub(crate) fn compute_late_interaction_score(
    query_tokens: &[Vec<f32>],
    target_tokens: &[Vec<f32>],
) -> f32 {
    if query_tokens.is_empty() || target_tokens.is_empty() {
        return 0.0;
    }

    // MaxSim: for each query token, find max similarity to any target token
    let mut total = 0.0_f32;
    for q_tok in query_tokens {
        let max_sim = target_tokens
            .iter()
            .map(|t_tok| cosine_similarity(q_tok, t_tok))
            .fold(f32::NEG_INFINITY, f32::max);
        if max_sim.is_finite() {
            total += max_sim;
        }
    }

    total / query_tokens.len() as f32
}
