//! Discovery metadata for autonomously discovered goals.

use super::error::GoalNodeError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// How a goal was discovered.
///
/// Goals are discovered AUTONOMOUSLY from memory patterns.
/// Manual goal creation is forbidden per ARCH-03.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    /// Discovered via k-means or HDBSCAN clustering of fingerprints.
    Clustering,
    /// Discovered via pattern recognition in purpose vectors.
    PatternRecognition,
    /// Created by decomposing a parent goal into sub-goals.
    Decomposition,
    /// Bootstrapped from initial memory analysis (first Strategic goal).
    Bootstrap,
}

/// Metadata about how a goal was autonomously discovered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalDiscoveryMetadata {
    /// How this goal was discovered.
    pub method: DiscoveryMethod,
    /// Confidence score [0.0, 1.0].
    pub confidence: f32,
    /// Number of memories in the cluster that formed this goal.
    pub cluster_size: usize,
    /// Coherence score of the cluster [0.0, 1.0].
    pub coherence: f32,
    /// Timestamp when discovery occurred.
    pub discovered_at: DateTime<Utc>,
}

impl GoalDiscoveryMetadata {
    /// Create new discovery metadata with validation.
    ///
    /// # Errors
    ///
    /// Returns `GoalNodeError` if:
    /// - Confidence is not in [0.0, 1.0]
    /// - Coherence is not in [0.0, 1.0]
    /// - Cluster size is 0 for non-Bootstrap methods
    pub fn new(
        method: DiscoveryMethod,
        confidence: f32,
        cluster_size: usize,
        coherence: f32,
    ) -> Result<Self, GoalNodeError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(GoalNodeError::InvalidConfidence(confidence));
        }
        if !(0.0..=1.0).contains(&coherence) {
            return Err(GoalNodeError::InvalidCoherence(coherence));
        }
        if cluster_size == 0 && method != DiscoveryMethod::Bootstrap {
            return Err(GoalNodeError::EmptyCluster);
        }
        Ok(Self {
            method,
            confidence,
            cluster_size,
            coherence,
            discovered_at: Utc::now(),
        })
    }

    /// Create bootstrap metadata (for initial Strategic goal).
    ///
    /// Bootstrap goals start with zero confidence and coherence,
    /// which will be computed after more data is available.
    pub fn bootstrap() -> Self {
        Self {
            method: DiscoveryMethod::Bootstrap,
            confidence: 0.0,
            cluster_size: 0,
            coherence: 0.0,
            discovered_at: Utc::now(),
        }
    }
}
