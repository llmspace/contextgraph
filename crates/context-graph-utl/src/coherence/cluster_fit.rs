//! ClusterFit types for silhouette-based coherence component.
//!
//! # Constitution Reference
//!
//! Per constitution.yaml line 166:
//! ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
//!
//! ClusterFit measures how well a vertex fits within its semantic cluster
//! using the silhouette coefficient: s = (b - a) / max(a, b)
//!
//! # Output Range
//!
//! All outputs are clamped per AP-10 (no NaN/Infinity):
//! - `ClusterFitResult.score`: [0, 1]
//! - `ClusterFitResult.silhouette`: [-1, 1]

use serde::{Deserialize, Serialize};

/// Configuration for ClusterFit calculation.
///
/// # Constitution Reference
/// - Line 166: ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
/// - ClusterFit uses silhouette score: (b - a) / max(a, b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterFitConfig {
    /// Minimum cluster size for valid calculation.
    /// Default: 2 (silhouette requires at least 2 members)
    pub min_cluster_size: usize,

    /// Distance metric to use.
    /// Default: Cosine (matches semantic embedding space)
    pub distance_metric: DistanceMetric,

    /// Fallback value when cluster fit cannot be computed.
    /// Default: 0.5 (neutral - per AP-10 no NaN allowed)
    pub fallback_value: f32,

    /// Maximum cluster members to sample for performance.
    /// Default: 1000 (prevents O(n²) explosion)
    pub max_sample_size: usize,
}

impl Default for ClusterFitConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 2,
            distance_metric: DistanceMetric::default(),
            fallback_value: 0.5,
            max_sample_size: 1000,
        }
    }
}

/// Distance metric options for cluster distance calculation.
///
/// Used to compute intra-cluster distance (a) and inter-cluster distance (b)
/// for silhouette coefficient: s = (b - a) / max(a, b)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cosine_similarity.
    /// Best for normalized embeddings (most common).
    #[default]
    Cosine,
    /// Euclidean (L2) distance.
    /// Use for non-normalized embeddings.
    Euclidean,
    /// Manhattan (L1) distance.
    /// More robust to outliers.
    Manhattan,
}

/// Cluster context providing embeddings for cluster fit calculation.
///
/// Contains the data needed to compute silhouette coefficient for a vertex.
#[derive(Debug, Clone)]
pub struct ClusterContext {
    /// Embeddings of vertices in the same cluster (excluding the query vertex).
    /// Must have at least `min_cluster_size - 1` members for valid computation.
    pub same_cluster: Vec<Vec<f32>>,

    /// Embeddings of vertices in the nearest other cluster.
    /// Used to compute inter-cluster distance (b).
    pub nearest_cluster: Vec<Vec<f32>>,

    /// Optional precomputed cluster centroids for efficiency.
    /// Index corresponds to cluster ID.
    pub centroids: Option<Vec<Vec<f32>>>,
}

impl ClusterContext {
    /// Create new cluster context.
    pub fn new(same_cluster: Vec<Vec<f32>>, nearest_cluster: Vec<Vec<f32>>) -> Self {
        Self {
            same_cluster,
            nearest_cluster,
            centroids: None,
        }
    }

    /// Create with precomputed centroids.
    pub fn with_centroids(
        same_cluster: Vec<Vec<f32>>,
        nearest_cluster: Vec<Vec<f32>>,
        centroids: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            same_cluster,
            nearest_cluster,
            centroids: Some(centroids),
        }
    }
}

/// Result of ClusterFit calculation with diagnostics.
///
/// # Output Range
/// - `score`: [0, 1] normalized for UTL formula
/// - `silhouette`: [-1, 1] raw coefficient
#[derive(Debug, Clone)]
pub struct ClusterFitResult {
    /// Normalized cluster fit score [0, 1].
    /// Derived from silhouette: (silhouette + 1) / 2
    pub score: f32,

    /// Raw silhouette coefficient [-1, 1].
    /// -1 = wrong cluster, 0 = boundary, +1 = well-clustered
    pub silhouette: f32,

    /// Mean intra-cluster distance (a).
    /// Average distance to same-cluster members.
    pub intra_distance: f32,

    /// Mean nearest-cluster distance (b).
    /// Average distance to nearest other cluster.
    pub inter_distance: f32,
}

impl ClusterFitResult {
    /// Create result from raw silhouette and distances.
    ///
    /// Automatically computes normalized score.
    pub fn new(silhouette: f32, intra_distance: f32, inter_distance: f32) -> Self {
        // Normalize silhouette from [-1, 1] to [0, 1]
        let score = (silhouette + 1.0) / 2.0;
        Self {
            score: score.clamp(0.0, 1.0),
            silhouette: silhouette.clamp(-1.0, 1.0),
            intra_distance,
            inter_distance,
        }
    }

    /// Create a fallback result when computation is not possible.
    pub fn fallback(value: f32) -> Self {
        Self {
            score: value.clamp(0.0, 1.0),
            silhouette: 0.0, // Neutral
            intra_distance: 0.0,
            inter_distance: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_fit_config_default() {
        let config = ClusterFitConfig::default();
        assert_eq!(config.min_cluster_size, 2);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert_eq!(config.fallback_value, 0.5);
        assert_eq!(config.max_sample_size, 1000);
    }

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_cluster_context_new() {
        let same = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let nearest = vec![vec![0.5, 0.6]];
        let ctx = ClusterContext::new(same.clone(), nearest.clone());

        assert_eq!(ctx.same_cluster.len(), 2);
        assert_eq!(ctx.nearest_cluster.len(), 1);
        assert!(ctx.centroids.is_none());
    }

    #[test]
    fn test_cluster_context_with_centroids() {
        let same = vec![vec![0.1, 0.2]];
        let nearest = vec![vec![0.5, 0.6]];
        let centroids = vec![vec![0.2, 0.3], vec![0.6, 0.7]];

        let ctx = ClusterContext::with_centroids(same, nearest, centroids);
        assert!(ctx.centroids.is_some());
        assert_eq!(ctx.centroids.unwrap().len(), 2);
    }

    #[test]
    fn test_cluster_fit_result_new() {
        // Perfect clustering: silhouette = 1.0
        let result = ClusterFitResult::new(1.0, 0.1, 0.9);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.silhouette, 1.0);

        // Worst clustering: silhouette = -1.0
        let result = ClusterFitResult::new(-1.0, 0.9, 0.1);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.silhouette, -1.0);

        // Boundary: silhouette = 0.0
        let result = ClusterFitResult::new(0.0, 0.5, 0.5);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.silhouette, 0.0);
    }

    #[test]
    fn test_cluster_fit_result_clamps_output() {
        // Test clamping for out-of-range values
        let result = ClusterFitResult::new(1.5, 0.1, 0.9);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.silhouette, 1.0);

        let result = ClusterFitResult::new(-1.5, 0.9, 0.1);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.silhouette, -1.0);
    }

    #[test]
    fn test_cluster_fit_result_fallback() {
        let result = ClusterFitResult::fallback(0.5);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.silhouette, 0.0);
        assert_eq!(result.intra_distance, 0.0);
        assert_eq!(result.inter_distance, 0.0);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = ClusterFitConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: ClusterFitConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.min_cluster_size, restored.min_cluster_size);
        assert_eq!(config.distance_metric, restored.distance_metric);
        assert_eq!(config.fallback_value, restored.fallback_value);
    }

    #[test]
    fn test_distance_metric_serialization() {
        let metric = DistanceMetric::Euclidean;
        let json = serde_json::to_string(&metric).unwrap();
        assert!(json.contains("Euclidean"));

        let restored: DistanceMetric = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, DistanceMetric::Euclidean);
    }
}
