//! K-means clustering implementation.
//!
//! Provides the StandardKMeans implementation using k-means++ initialization.

use crate::index::config::PURPOSE_VECTOR_DIM;

use super::super::entry::PurposeIndexEntry;
use super::super::error::{PurposeIndexError, PurposeIndexResult};
use super::algorithms::{build_clusters, compute_centroids, compute_wcss, kmeans_plus_plus_init};
use super::config::KMeansConfig;
use super::metrics::euclidean_distance_squared;
use super::types::ClusteringResult;

/// Trait for k-means clustering on purpose vectors.
///
/// Implementors provide k-means clustering functionality for
/// collections of purpose index entries.
pub trait KMeansPurposeClustering {
    /// Cluster purpose vectors using k-means algorithm.
    ///
    /// # Arguments
    ///
    /// * `entries` - The purpose index entries to cluster
    /// * `config` - K-means configuration
    ///
    /// # Errors
    ///
    /// Returns `PurposeIndexError::ClusteringError` if:
    /// - entries is empty
    /// - k > entries.len()
    /// - Algorithm fails to converge within max_iterations (still returns partial result)
    ///
    /// # Fail-Fast
    ///
    /// Invalid inputs cause immediate errors. No fallbacks.
    fn cluster_purposes(
        &self,
        entries: &[PurposeIndexEntry],
        config: &KMeansConfig,
    ) -> PurposeIndexResult<ClusteringResult>;
}

/// Standard k-means++ implementation for purpose vectors.
///
/// Uses k-means++ initialization for better initial centroids
/// and standard Lloyd's algorithm for iteration.
#[derive(Clone, Debug, Default)]
pub struct StandardKMeans;

impl StandardKMeans {
    /// Create a new StandardKMeans clusterer.
    pub fn new() -> Self {
        Self
    }
}

impl KMeansPurposeClustering for StandardKMeans {
    fn cluster_purposes(
        &self,
        entries: &[PurposeIndexEntry],
        config: &KMeansConfig,
    ) -> PurposeIndexResult<ClusteringResult> {
        // FAIL FAST: Validate inputs
        if entries.is_empty() {
            return Err(PurposeIndexError::clustering("entries must not be empty"));
        }
        if config.k > entries.len() {
            return Err(PurposeIndexError::clustering(format!(
                "k ({}) must be <= entries.len() ({})",
                config.k,
                entries.len()
            )));
        }

        println!(
            "[CLUSTERING] Starting k-means: k={}, n={}, max_iter={}",
            config.k,
            entries.len(),
            config.max_iterations
        );

        // Extract vectors for clustering
        let vectors: Vec<[f32; PURPOSE_VECTOR_DIM]> = entries
            .iter()
            .map(|e| e.purpose_vector.alignments)
            .collect();

        // Initialize centroids using k-means++
        let mut centroids = kmeans_plus_plus_init(&vectors, config.k);

        println!(
            "[CLUSTERING] Initialized {} centroids using k-means++",
            centroids.len()
        );

        // Main k-means loop
        let mut assignments = vec![0usize; entries.len()];
        let mut iterations = 0;
        let mut converged = false;

        for iter in 0..config.max_iterations {
            iterations = iter + 1;

            // Assignment step: assign each point to nearest centroid
            for (i, vector) in vectors.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance_squared(vector, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }

                assignments[i] = best_cluster;
            }

            // Update step: recompute centroids
            let new_centroids = compute_centroids(&vectors, &assignments, config.k);

            // Check convergence: max centroid movement
            let max_movement = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_distance_squared(old, new).sqrt())
                .fold(0.0f32, |a, b| a.max(b));

            centroids = new_centroids;

            if max_movement < config.convergence_threshold {
                converged = true;
                println!(
                    "[CLUSTERING] Converged at iteration {} (movement={:.2e})",
                    iterations, max_movement
                );
                break;
            }
        }

        if !converged {
            println!(
                "[CLUSTERING] Did not converge after {} iterations",
                iterations
            );
        }

        // Build clusters with metadata
        let clusters = build_clusters(entries, &assignments, &centroids, config.k);

        // Compute WCSS (within-cluster sum of squares)
        let wcss = compute_wcss(&vectors, &assignments, &centroids);

        println!(
            "[CLUSTERING] Completed: {} clusters, {} iterations, WCSS={:.4}",
            clusters.len(),
            iterations,
            wcss
        );

        Ok(ClusteringResult::new(clusters, iterations, converged, wcss))
    }
}
