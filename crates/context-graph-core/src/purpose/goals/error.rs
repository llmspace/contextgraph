//! Error types for goal hierarchy operations.
//!
//! Goals emerge autonomously from data patterns.

use crate::types::fingerprint::ValidationError;
use thiserror::Error;
use uuid::Uuid;

/// Error when creating or validating a GoalNode.
#[derive(Debug, Clone, Error)]
pub enum GoalNodeError {
    /// The teleological array failed validation.
    #[error("Invalid teleological array: {0}")]
    InvalidArray(#[from] ValidationError),

    /// Discovery confidence is out of range [0.0, 1.0].
    #[error("Discovery confidence must be in [0.0, 1.0], got {0}")]
    InvalidConfidence(f32),

    /// Discovery coherence is out of range [0.0, 1.0].
    #[error("Discovery coherence must be in [0.0, 1.0], got {0}")]
    InvalidCoherence(f32),

    /// Cluster size must be > 0 for discovered goals.
    #[error("Cluster size must be > 0 for discovered goals")]
    EmptyCluster,
}

/// Errors for goal hierarchy operations.
#[derive(Debug, Error)]
pub enum GoalHierarchyError {
    /// Referenced parent goal does not exist.
    #[error("Parent goal not found: {0}")]
    ParentNotFound(Uuid),

    /// Referenced goal does not exist.
    #[error("Goal not found: {0}")]
    GoalNotFound(Uuid),
}
