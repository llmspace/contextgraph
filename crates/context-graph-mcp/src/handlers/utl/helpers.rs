//! Helper functions for UTL computation.
//!
//! TASK-UTL-P1-001: Embedding manipulation.
//! Note: Uses embedder categories per PRD v6.

use context_graph_core::types::fingerprint::SparseVector;

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
