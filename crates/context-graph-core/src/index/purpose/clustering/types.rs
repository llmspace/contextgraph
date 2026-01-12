//! Type definitions for k-means clustering.
//!
//! Contains the core types representing clusters and clustering results.

use uuid::Uuid;

use crate::index::config::PURPOSE_VECTOR_DIM;

use super::super::entry::GoalId;

/// A cluster of memories with similar purpose vectors.
///
/// Represents the result of k-means clustering on purpose vectors,
/// containing the centroid, members, and quality metrics.
#[derive(Clone, Debug)]
pub struct PurposeCluster {
    /// Cluster centroid (13D purpose vector).
    ///
    /// Computed as the mean of all member purpose vectors.
    pub centroid: [f32; PURPOSE_VECTOR_DIM],

    /// Memory IDs belonging to this cluster.
    ///
    /// All memories whose purpose vectors are closest to this centroid.
    pub members: Vec<Uuid>,

    /// Intra-cluster coherence score [0.0, 1.0].
    ///
    /// Higher values indicate more tightly clustered members.
    /// Computed as 1.0 - (mean distance to centroid / max possible distance).
    pub coherence: f32,

    /// Dominant goal for this cluster.
    ///
    /// The most frequent primary goal among cluster members.
    /// None if no metadata is available.
    pub dominant_goal: Option<GoalId>,
}

impl PurposeCluster {
    /// Create a new cluster with computed metrics.
    ///
    /// # Arguments
    ///
    /// * `centroid` - The 13D centroid of the cluster
    /// * `members` - UUIDs of memories in this cluster
    /// * `coherence` - Intra-cluster coherence score
    /// * `dominant_goal` - Most common goal in the cluster
    pub fn new(
        centroid: [f32; PURPOSE_VECTOR_DIM],
        members: Vec<Uuid>,
        coherence: f32,
        dominant_goal: Option<GoalId>,
    ) -> Self {
        Self {
            centroid,
            members,
            coherence,
            dominant_goal,
        }
    }

    /// Check if the cluster is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Get the number of members in this cluster.
    #[inline]
    pub fn len(&self) -> usize {
        self.members.len()
    }
}

/// Result of k-means clustering operation.
///
/// Contains the final clusters and convergence information.
#[derive(Clone, Debug)]
pub struct ClusteringResult {
    /// The clusters found by k-means.
    ///
    /// Length equals k from the configuration.
    pub clusters: Vec<PurposeCluster>,

    /// Number of iterations to converge.
    ///
    /// If converged is false, this equals max_iterations.
    pub iterations: usize,

    /// Whether convergence was achieved.
    ///
    /// True if max centroid movement fell below threshold.
    pub converged: bool,

    /// Total within-cluster sum of squares (WCSS).
    ///
    /// Lower values indicate better clustering.
    /// Sum of squared distances from each point to its centroid.
    pub wcss: f32,
}

impl ClusteringResult {
    /// Create a new clustering result.
    pub fn new(
        clusters: Vec<PurposeCluster>,
        iterations: usize,
        converged: bool,
        wcss: f32,
    ) -> Self {
        Self {
            clusters,
            iterations,
            converged,
            wcss,
        }
    }

    /// Get the number of clusters.
    #[inline]
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Get the total number of points across all clusters.
    pub fn total_points(&self) -> usize {
        self.clusters.iter().map(|c| c.len()).sum()
    }

    /// Get the average cluster size.
    pub fn avg_cluster_size(&self) -> f32 {
        if self.clusters.is_empty() {
            0.0
        } else {
            self.total_points() as f32 / self.clusters.len() as f32
        }
    }

    /// Get the average coherence across all clusters.
    pub fn avg_coherence(&self) -> f32 {
        if self.clusters.is_empty() {
            0.0
        } else {
            let sum: f32 = self.clusters.iter().map(|c| c.coherence).sum();
            sum / self.clusters.len() as f32
        }
    }
}
