//! Helper functions for UTL computation.
//!
//! TASK-UTL-P1-001: Embedding manipulation and Johari classification.

use context_graph_core::types::fingerprint::SparseVector;
use context_graph_core::types::JohariQuadrant;

/// Create a synthetic "nearest cluster" for ClusterFit computation.
///
/// Generates an embedding that represents "different" content from the query
/// to measure distinctiveness. Uses the difference vector direction to create
/// a divergent representation (opposite direction of change).
pub(super) fn create_divergent_cluster(old: &[f32], new: &[f32]) -> Vec<Vec<f32>> {
    if old.is_empty() || new.is_empty() || old.len() != new.len() {
        return vec![vec![0.5; old.len().max(128)]];
    }

    // Compute the difference vector: represents the change direction
    let diff: Vec<f32> = old.iter().zip(new.iter()).map(|(o, n)| n - o).collect();

    // Compute magnitude for normalization
    let diff_mag: f32 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();

    if diff_mag < 1e-10 {
        // If embeddings are identical, use a perpendicular approximation
        // Shift the vector to create distinctiveness
        let perpendicular: Vec<f32> = old
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v + 0.1 } else { v - 0.1 })
            .collect();
        return vec![perpendicular];
    }

    // Create the "opposite" direction: old - normalized_diff
    // This represents content that diverges from the change direction
    let opposite: Vec<f32> = old
        .iter()
        .zip(diff.iter())
        .map(|(o, d)| o - (d / diff_mag) * 0.5)
        .collect();

    vec![opposite]
}

/// Classify a (ΔS, ΔC) pair into a JohariQuadrant.
///
/// Per constitution.yaml johari mapping:
/// - Open: ΔS < threshold, ΔC > threshold (low surprise, high coherence)
/// - Blind: ΔS > threshold, ΔC < threshold (high surprise, low coherence)
/// - Hidden: ΔS < threshold, ΔC < threshold (low surprise, low coherence)
/// - Unknown: ΔS > threshold, ΔC > threshold (high surprise, high coherence)
pub(super) fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant {
    match (delta_s < threshold, delta_c > threshold) {
        (true, true) => JohariQuadrant::Open, // Low surprise, high coherence
        (false, false) => JohariQuadrant::Blind, // High surprise, low coherence
        (true, false) => JohariQuadrant::Hidden, // Low surprise, low coherence
        (false, true) => JohariQuadrant::Unknown, // High surprise, high coherence
    }
}

/// Convert sparse vector to truncated dense representation.
pub(super) fn sparse_to_dense_truncated(sparse: &SparseVector, max_dim: usize) -> Vec<f32> {
    let mut dense = vec![0.0f32; max_dim];
    for (&idx, &val) in sparse.indices.iter().zip(sparse.values.iter()) {
        let idx = idx as usize;
        if idx < max_dim {
            dense[idx] = val;
        }
    }
    dense
}

/// Mean pool token-level embeddings.
pub(super) fn mean_pool_tokens(tokens: &[Vec<f32>]) -> Vec<f32> {
    if tokens.is_empty() {
        return vec![0.0f32; 128]; // ColBERT token dim
    }

    let dim = tokens[0].len();
    let mut pooled = vec![0.0f32; dim];
    let n = tokens.len() as f32;

    for token in tokens {
        for (i, &val) in token.iter().enumerate() {
            if i < pooled.len() {
                pooled[i] += val / n;
            }
        }
    }

    pooled
}
