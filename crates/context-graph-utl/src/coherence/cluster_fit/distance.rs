//! Distance computation functions for ClusterFit.
//!
//! Provides various distance metrics for computing silhouette coefficient.

use super::types::DistanceMetric;

/// Compute vector magnitude (L2 norm).
pub(crate) fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute cosine distance between two vectors.
///
/// Cosine distance = 1 - cosine_similarity
/// Range: [0, 2] but typically [0, 1] for normalized vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine distance clamped to [0, 2]. Returns 0.0 for edge cases.
pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);

    // Handle zero magnitude vectors gracefully
    if mag_a < 1e-10 || mag_b < 1e-10 {
        return 0.0;
    }

    let cosine_sim = dot / (mag_a * mag_b);

    // Cosine distance = 1 - similarity
    // Clamp similarity first to handle floating point errors
    let clamped_sim = cosine_sim.clamp(-1.0, 1.0);
    let distance = 1.0 - clamped_sim;

    // Handle potential NaN/Infinity per AP-10
    if distance.is_nan() || distance.is_infinite() {
        0.0
    } else {
        distance.clamp(0.0, 2.0)
    }
}

/// Compute Euclidean (L2) distance between two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Euclidean distance (non-negative). Returns 0.0 for edge cases.
pub(crate) fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();

    let distance = sum_sq.sqrt();

    // Handle potential NaN/Infinity per AP-10
    if distance.is_nan() || distance.is_infinite() {
        0.0
    } else {
        distance
    }
}

/// Compute Manhattan (L1) distance between two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Manhattan distance (non-negative). Returns 0.0 for edge cases.
pub(crate) fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let distance: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();

    // Handle potential NaN/Infinity per AP-10
    if distance.is_nan() || distance.is_infinite() {
        0.0
    } else {
        distance
    }
}

/// Compute distance between two vectors using the specified metric.
pub(crate) fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::Manhattan => manhattan_distance(a, b),
    }
}

/// Compute mean distance from a query vector to a set of cluster members.
///
/// # Arguments
/// * `query` - The query embedding vector
/// * `cluster` - The cluster member embeddings
/// * `metric` - Distance metric to use
/// * `max_sample` - Maximum number of members to sample
///
/// # Returns
/// Mean distance, or None if cluster is empty or has no valid members.
pub(crate) fn mean_distance_to_cluster(
    query: &[f32],
    cluster: &[Vec<f32>],
    metric: DistanceMetric,
    max_sample: usize,
) -> Option<f32> {
    if cluster.is_empty() || query.is_empty() {
        return None;
    }

    // Sample if cluster is too large
    let members: Vec<&Vec<f32>> = if cluster.len() > max_sample {
        // Simple deterministic sampling: take evenly spaced members
        let step = cluster.len() / max_sample;
        cluster
            .iter()
            .step_by(step.max(1))
            .take(max_sample)
            .collect()
    } else {
        cluster.iter().collect()
    };

    let mut sum = 0.0f64;
    let mut count = 0usize;

    for member in members {
        // Skip members with mismatched dimensions
        if member.len() != query.len() {
            continue;
        }
        // Skip zero-magnitude members for cosine distance
        if metric == DistanceMetric::Cosine && magnitude(member) < 1e-10 {
            continue;
        }

        let dist = compute_distance(query, member, metric);
        sum += dist as f64;
        count += 1;
    }

    if count == 0 {
        None
    } else {
        let mean = (sum / count as f64) as f32;
        // Ensure no NaN/Infinity per AP-10
        if mean.is_nan() || mean.is_infinite() {
            None
        } else {
            Some(mean)
        }
    }
}
