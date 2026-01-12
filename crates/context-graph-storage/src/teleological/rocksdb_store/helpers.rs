//! Helper functions for RocksDbTeleologicalStore.
//!
//! Contains utility functions for computing similarity, purpose alignment,
//! and Johari quadrant analysis.

use context_graph_core::types::fingerprint::{JohariFingerprint, PurposeVector, NUM_EMBEDDERS};

/// Get the aggregate dominant quadrant across all 13 embedders.
///
/// Aggregates quadrant weights across all embedders and returns the index
/// of the dominant quadrant (0=Open, 1=Hidden, 2=Blind, 3=Unknown).
pub fn get_aggregate_dominant_quadrant(johari: &JohariFingerprint) -> usize {
    let mut totals = [0.0_f32; 4];
    for quadrant in johari.quadrants.iter().take(NUM_EMBEDDERS) {
        for (total, &q_val) in totals.iter_mut().zip(quadrant.iter()) {
            *total += q_val;
        }
    }

    // Find dominant quadrant
    totals
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(3) // Default to Unknown
}

/// Compute cosine similarity between two dense vectors.
pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Compute purpose alignment for a fingerprint.
pub fn query_purpose_alignment(pv: &PurposeVector) -> f32 {
    pv.aggregate_alignment()
}
