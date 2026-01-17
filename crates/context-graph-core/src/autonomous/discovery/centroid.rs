//! Centroid computation for teleological arrays.
//!
//! This module provides functions for computing centroids of clusters
//! by averaging vectors across all 13 embedders.

use std::collections::HashMap;

use crate::types::fingerprint::{
    SemanticFingerprint, SparseVector, TeleologicalArray, E10_DIM, E11_DIM, E12_TOKEN_DIM, E1_DIM,
    E2_DIM, E3_DIM, E4_DIM, E5_DIM, E7_DIM, E8_DIM, E9_DIM,
};

/// Compute centroid for a cluster of teleological arrays.
///
/// Each embedder's vectors are averaged separately.
/// Result is a valid TeleologicalArray.
///
/// # Panics
///
/// Panics if members is empty (FAIL FAST policy).
pub fn compute_centroid(members: &[&TeleologicalArray]) -> TeleologicalArray {
    assert!(
        !members.is_empty(),
        "FAIL FAST: Cannot compute centroid of empty cluster"
    );

    let n = members.len() as f32;

    // Average each dense embedding separately
    let e1_semantic = average_dense_vectors(members.iter().map(|m| &m.e1_semantic), E1_DIM);
    let e2_temporal_recent =
        average_dense_vectors(members.iter().map(|m| &m.e2_temporal_recent), E2_DIM);
    let e3_temporal_periodic =
        average_dense_vectors(members.iter().map(|m| &m.e3_temporal_periodic), E3_DIM);
    let e4_temporal_positional =
        average_dense_vectors(members.iter().map(|m| &m.e4_temporal_positional), E4_DIM);
    let e5_causal = average_dense_vectors(members.iter().map(|m| &m.e5_causal), E5_DIM);
    let e7_code = average_dense_vectors(members.iter().map(|m| &m.e7_code), E7_DIM);
    let e8_graph = average_dense_vectors(members.iter().map(|m| &m.e8_graph), E8_DIM);
    let e9_hdc = average_dense_vectors(members.iter().map(|m| &m.e9_hdc), E9_DIM);
    let e10_multimodal = average_dense_vectors(members.iter().map(|m| &m.e10_multimodal), E10_DIM);
    let e11_entity = average_dense_vectors(members.iter().map(|m| &m.e11_entity), E11_DIM);

    // Average sparse vectors (E6, E13)
    let e6_sparse = average_sparse_vectors(members.iter().map(|m| &m.e6_sparse), n);
    let e13_splade = average_sparse_vectors(members.iter().map(|m| &m.e13_splade), n);

    // Average token-level vectors (E12)
    let e12_late_interaction =
        average_token_vectors(members.iter().map(|m| &m.e12_late_interaction));

    SemanticFingerprint {
        e1_semantic,
        e2_temporal_recent,
        e3_temporal_periodic,
        e4_temporal_positional,
        e5_causal,
        e6_sparse,
        e7_code,
        e8_graph,
        e9_hdc,
        e10_multimodal,
        e11_entity,
        e12_late_interaction,
        e13_splade,
    }
}

/// Average dense vectors element-wise.
pub fn average_dense_vectors<'a, I>(vectors: I, dim: usize) -> Vec<f32>
where
    I: Iterator<Item = &'a Vec<f32>>,
{
    let mut sum = vec![0.0_f32; dim];
    let mut count = 0;

    for vec in vectors {
        if vec.len() == dim {
            for (i, &val) in vec.iter().enumerate() {
                sum[i] += val;
            }
            count += 1;
        }
    }

    if count > 0 {
        for val in &mut sum {
            *val /= count as f32;
        }
    }

    sum
}

/// Average sparse vectors by combining indices and averaging values.
pub fn average_sparse_vectors<'a, I>(vectors: I, n: f32) -> SparseVector
where
    I: Iterator<Item = &'a SparseVector>,
{
    // Collect all (index, value) pairs and sum values per index
    let mut index_sums: HashMap<u16, f32> = HashMap::new();

    for sparse in vectors {
        for (idx, val) in sparse.indices.iter().zip(sparse.values.iter()) {
            *index_sums.entry(*idx).or_insert(0.0) += *val;
        }
    }

    // Average and build result
    let mut pairs: Vec<(u16, f32)> = index_sums
        .into_iter()
        .map(|(idx, sum)| (idx, sum / n))
        .collect();

    // Sort by index for SparseVector construction
    pairs.sort_by_key(|(idx, _)| *idx);

    let indices: Vec<u16> = pairs.iter().map(|(idx, _)| *idx).collect();
    let values: Vec<f32> = pairs.iter().map(|(_, val)| *val).collect();

    SparseVector::new(indices, values).unwrap_or_else(|e| {
        eprintln!(
            "[GoalDiscoveryPipeline] Warning: Failed to construct sparse centroid: {:?}",
            e
        );
        SparseVector::empty()
    })
}

/// Average token-level vectors.
pub fn average_token_vectors<'a, I>(vectors: I) -> Vec<Vec<f32>>
where
    I: Iterator<Item = &'a Vec<Vec<f32>>>,
{
    let all_tokens: Vec<&Vec<Vec<f32>>> = vectors.collect();

    if all_tokens.is_empty() {
        return Vec::new();
    }

    // Find average token count
    let total_tokens: usize = all_tokens.iter().map(|t| t.len()).sum();
    let avg_token_count = (total_tokens / all_tokens.len()).max(1);

    // Collect all tokens and average by position
    let mut result = Vec::with_capacity(avg_token_count);

    for pos in 0..avg_token_count {
        let mut sum = vec![0.0_f32; E12_TOKEN_DIM];
        let mut count = 0;

        for tokens in &all_tokens {
            if let Some(token) = tokens.get(pos) {
                if token.len() == E12_TOKEN_DIM {
                    for (i, &val) in token.iter().enumerate() {
                        sum[i] += val;
                    }
                    count += 1;
                }
            }
        }

        if count > 0 {
            for val in &mut sum {
                *val /= count as f32;
            }
        }

        result.push(sum);
    }

    result
}

/// Compute L2 norm of a vector.
pub fn l2_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Create a zeroed fingerprint for empty cluster fallback.
#[cfg(any(test, feature = "test-utils"))]
pub fn create_zeroed_fingerprint() -> SemanticFingerprint {
    SemanticFingerprint::zeroed()
}

#[cfg(not(any(test, feature = "test-utils")))]
pub fn create_zeroed_fingerprint() -> SemanticFingerprint {
    // In production, create a minimal valid fingerprint
    SemanticFingerprint {
        e1_semantic: vec![0.0; E1_DIM],
        e2_temporal_recent: vec![0.0; E2_DIM],
        e3_temporal_periodic: vec![0.0; E3_DIM],
        e4_temporal_positional: vec![0.0; E4_DIM],
        e5_causal: vec![0.0; E5_DIM],
        e6_sparse: SparseVector::empty(),
        e7_code: vec![0.0; E7_DIM],
        e8_graph: vec![0.0; E8_DIM],
        e9_hdc: vec![0.0; E9_DIM],
        e10_multimodal: vec![0.0; E10_DIM],
        e11_entity: vec![0.0; E11_DIM],
        e12_late_interaction: Vec::new(),
        e13_splade: SparseVector::empty(),
    }
}
