//! Error types for drift detection.
//!
//! All errors are fatal per FAIL FAST principle.

use crate::teleological::Embedder;

/// Error types for drift detection. All errors are fatal per FAIL FAST.
#[derive(Debug, Clone)]
pub enum DriftError {
    /// No memories provided for analysis
    EmptyMemories,
    /// Goal has invalid embeddings (NaN, Inf, or missing)
    InvalidGoal { reason: String },
    /// Comparison failed for a specific embedder
    ComparisonFailed { embedder: Embedder, reason: String },
    /// Invalid threshold configuration
    InvalidThresholds { reason: String },
    /// Comparison validation error
    ComparisonValidationFailed { reason: String },
}

impl std::fmt::Display for DriftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriftError::EmptyMemories => write!(f, "No memories provided for drift analysis"),
            DriftError::InvalidGoal { reason } => write!(f, "Invalid goal: {}", reason),
            DriftError::ComparisonFailed { embedder, reason } => {
                write!(f, "Comparison failed for {:?}: {}", embedder, reason)
            }
            DriftError::InvalidThresholds { reason } => {
                write!(f, "Invalid thresholds: {}", reason)
            }
            DriftError::ComparisonValidationFailed { reason } => {
                write!(f, "Comparison validation failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for DriftError {}
