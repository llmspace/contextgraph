//! K-means clustering for 13D purpose vectors.
//!
//! # CRITICAL: NO FALLBACKS
//!
//! Clustering failures are fatal. No partial results returned.
//! If clustering cannot complete, error propagates immediately.
//!
//! # Overview
//!
//! This module implements k-means++ clustering for purpose vectors,
//! enabling discovery of natural groupings in teleological alignment space.
//!
//! # Algorithm
//!
//! 1. Initialize k centroids using k-means++ (smart initialization)
//! 2. Assign each vector to nearest centroid (Euclidean distance)
//! 3. Recompute centroids as mean of assigned vectors
//! 4. Repeat until convergence or max iterations
//!
//! # Fail-Fast Validation
//!
//! - k must be > 0 and <= entries.len()
//! - max_iterations must be > 0
//! - convergence_threshold must be > 0.0
//! - entries must not be empty

mod algorithms;
mod clusterer;
mod config;
mod metrics;
#[cfg(test)]
mod tests;
mod types;

// Re-export public API for backwards compatibility
pub use clusterer::{KMeansPurposeClustering, StandardKMeans};
pub use config::KMeansConfig;
pub use types::{ClusteringResult, PurposeCluster};
