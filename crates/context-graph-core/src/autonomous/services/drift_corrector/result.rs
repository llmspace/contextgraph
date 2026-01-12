//! Correction result types for drift correction.

use serde::{Deserialize, Serialize};

use super::CorrectionStrategy;

/// Result of applying a correction strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectionResult {
    /// The strategy that was applied
    pub strategy_applied: CorrectionStrategy,

    /// Alignment before correction
    pub alignment_before: f32,

    /// Alignment after correction
    pub alignment_after: f32,

    /// Whether the correction was successful
    pub success: bool,
}

impl CorrectionResult {
    /// Create a new correction result
    pub fn new(strategy: CorrectionStrategy, before: f32, after: f32, success: bool) -> Self {
        Self {
            strategy_applied: strategy,
            alignment_before: before,
            alignment_after: after,
            success,
        }
    }

    /// Calculate the improvement achieved
    pub fn improvement(&self) -> f32 {
        self.alignment_after - self.alignment_before
    }
}
