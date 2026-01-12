//! Configuration for k-means clustering.
//!
//! Provides validated configuration for clustering parameters.

use super::super::error::{PurposeIndexError, PurposeIndexResult};

/// Configuration for k-means clustering.
///
/// # Validation
///
/// All parameters are validated at construction time.
/// Invalid configurations result in immediate errors.
#[derive(Clone, Debug)]
pub struct KMeansConfig {
    /// Number of clusters (k).
    ///
    /// Must be > 0 and <= number of data points.
    pub k: usize,

    /// Maximum iterations before stopping.
    ///
    /// Must be > 0. Typical values: 50-300.
    pub max_iterations: usize,

    /// Convergence threshold for centroid movement.
    ///
    /// Iteration stops when max centroid movement is below this.
    /// Must be > 0.0. Typical value: 1e-6.
    pub convergence_threshold: f32,
}

impl KMeansConfig {
    /// Create a new configuration with validation.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of clusters (must be > 0)
    /// * `max_iterations` - Maximum iterations (must be > 0)
    /// * `convergence_threshold` - Convergence threshold (must be > 0.0)
    ///
    /// # Errors
    ///
    /// Returns `PurposeIndexError::ClusteringError` if any parameter is invalid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = KMeansConfig::new(5, 100, 1e-6)?;
    /// ```
    pub fn new(
        k: usize,
        max_iterations: usize,
        convergence_threshold: f32,
    ) -> PurposeIndexResult<Self> {
        if k == 0 {
            return Err(PurposeIndexError::clustering("k must be > 0"));
        }
        if max_iterations == 0 {
            return Err(PurposeIndexError::clustering("max_iterations must be > 0"));
        }
        if convergence_threshold <= 0.0 {
            return Err(PurposeIndexError::clustering(
                "convergence_threshold must be > 0.0",
            ));
        }
        if convergence_threshold.is_nan() || convergence_threshold.is_infinite() {
            return Err(PurposeIndexError::clustering(
                "convergence_threshold must be a finite positive number",
            ));
        }

        Ok(Self {
            k,
            max_iterations,
            convergence_threshold,
        })
    }

    /// Create a default configuration for the given number of clusters.
    ///
    /// Uses max_iterations=100 and convergence_threshold=1e-6.
    ///
    /// # Errors
    ///
    /// Returns error if k is 0.
    pub fn with_k(k: usize) -> PurposeIndexResult<Self> {
        Self::new(k, 100, 1e-6)
    }
}

impl Default for KMeansConfig {
    /// Default configuration: k=3, max_iterations=100, convergence_threshold=1e-6.
    fn default() -> Self {
        Self {
            k: 3,
            max_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }
}
