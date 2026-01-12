//! Mathematical utility functions for UTL computation.
//!
//! Contains low-level math operations used in the UTL processor:
//! - Sigmoid activation
//! - Cosine similarity
//! - Euclidean distance
//! - KNN distance computation
//! - Delta S/C computations

/// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute cosine similarity between two vectors.
///
/// Returns value in [-1, 1]. Returns 0.0 for empty or zero-magnitude vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
        return 0.0;
    }

    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

/// Compute Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute k-th nearest neighbor distance.
///
/// Returns the distance to the k-th nearest reference embedding.
/// If fewer than k references exist, returns average distance.
pub fn compute_knn_distance(input: &[f32], references: &[Vec<f32>], k: usize) -> Option<f32> {
    if references.is_empty() || input.is_empty() {
        return None;
    }

    let mut distances: Vec<f32> = references
        .iter()
        .filter(|r| r.len() == input.len())
        .map(|r| euclidean_distance(input, r))
        .filter(|d| d.is_finite())
        .collect();

    if distances.is_empty() {
        return None;
    }

    distances.sort_by(|a, b| a.partial_cmp(b).expect("distances should be finite"));

    // Return k-th distance (0-indexed), or last if fewer than k
    let idx = (k.saturating_sub(1)).min(distances.len() - 1);
    Some(distances[idx])
}

/// Compute ΔS (surprise/entropy) using KNN distance.
///
/// Per constitution: ΔS_knn = σ((d_k - μ_corpus) / σ_corpus)
///
/// Returns value in [0, 1] where:
/// - High ΔS = input is far from known embeddings (novel)
/// - Low ΔS = input is close to known embeddings (familiar)
pub fn compute_delta_s_from_embeddings(
    input: &[f32],
    references: &[Vec<f32>],
    mean_dist: f32,
    std_dist: f32,
    k: usize,
) -> f32 {
    let d_k = match compute_knn_distance(input, references, k) {
        Some(d) => d,
        None => return 0.5, // Default to medium surprise if no references
    };

    // Avoid division by zero
    let std_dist = std_dist.max(f32::EPSILON);

    // Normalized z-score
    let z = (d_k - mean_dist) / std_dist;

    // Apply sigmoid to map to [0, 1]
    sigmoid(z)
}

/// Compute ΔC (coherence change) using connectivity measure.
///
/// Per constitution:
/// ΔC = α × Connectivity + β × ClusterFit + γ × Consistency
/// Simplified: ΔC = Connectivity = |{neighbors: sim(e, n) > θ_edge}| / max_edges
///
/// Returns value in [0, 1] where:
/// - High ΔC = input integrates well with existing knowledge
/// - Low ΔC = input is disconnected from existing knowledge
pub fn compute_delta_c_from_embeddings(
    input: &[f32],
    references: &[Vec<f32>],
    edge_threshold: f32,
    max_edges: usize,
) -> f32 {
    if references.is_empty() || input.is_empty() {
        return 0.0; // No coherence with empty corpus
    }

    // Count neighbors above similarity threshold
    let neighbor_count = references
        .iter()
        .filter(|r| r.len() == input.len())
        .map(|r| cosine_similarity(input, r))
        .filter(|sim| *sim > edge_threshold)
        .count();

    // Normalize by max_edges
    let max_edges = max_edges.max(1);
    (neighbor_count as f32 / max_edges as f32).clamp(0.0, 1.0)
}

/// Fallback: Generate a deterministic value from input hash.
///
/// Used when embeddings are not available. Still produces consistent
/// values but is NOT a real UTL computation.
#[allow(dead_code)]
pub fn hash_to_float(input: &str, seed: u64) -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    seed.hash(&mut hasher);
    let hash = hasher.finish();
    // Map to [0.0, 1.0]
    (hash as f64 / u64::MAX as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.0001);

        // Opposite vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 0.0001);

        // Different lengths returns 0
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Empty vectors return 0
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        // Same point
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(euclidean_distance(&a, &b).abs() < 0.0001);

        // Unit distance
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 0.0001);

        // Different lengths returns MAX
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        assert_eq!(euclidean_distance(&a, &b), f32::MAX);
    }

    #[test]
    fn test_sigmoid() {
        // sigmoid(0) = 0.5
        assert!((sigmoid(0.0) - 0.5).abs() < 0.0001);

        // sigmoid(large) -> 1.0
        assert!(sigmoid(10.0) > 0.99);

        // sigmoid(-large) -> 0.0
        assert!(sigmoid(-10.0) < 0.01);

        // sigmoid is monotonic
        assert!(sigmoid(1.0) > sigmoid(0.0));
        assert!(sigmoid(0.0) > sigmoid(-1.0));
    }

    #[test]
    fn test_knn_distance_computation() {
        let input = vec![0.0, 0.0, 0.0];
        let references = vec![
            vec![1.0, 0.0, 0.0], // distance 1.0
            vec![2.0, 0.0, 0.0], // distance 2.0
            vec![3.0, 0.0, 0.0], // distance 3.0
        ];

        // k=1 should return 1.0 (closest)
        let d1 = compute_knn_distance(&input, &references, 1).unwrap();
        assert!((d1 - 1.0).abs() < 0.0001);

        // k=2 should return 2.0 (second closest)
        let d2 = compute_knn_distance(&input, &references, 2).unwrap();
        assert!((d2 - 2.0).abs() < 0.0001);

        // k=3 should return 3.0 (third closest)
        let d3 = compute_knn_distance(&input, &references, 3).unwrap();
        assert!((d3 - 3.0).abs() < 0.0001);

        // k > n should return last distance
        let d5 = compute_knn_distance(&input, &references, 5).unwrap();
        assert!((d5 - 3.0).abs() < 0.0001);

        // Empty references returns None
        let empty: Vec<Vec<f32>> = vec![];
        assert!(compute_knn_distance(&input, &empty, 1).is_none());
    }
}
