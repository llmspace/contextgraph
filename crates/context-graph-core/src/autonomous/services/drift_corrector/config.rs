//! Configuration types for drift correction.

use serde::{Deserialize, Serialize};

/// Configuration for the drift corrector
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftCorrectorConfig {
    /// Threshold adjustment delta for moderate drift
    pub moderate_threshold_delta: f32,

    /// Threshold adjustment delta for severe drift
    pub severe_threshold_delta: f32,

    /// Goal reinforcement factor for moderate drift
    pub moderate_reinforcement: f32,

    /// Goal reinforcement factor for severe drift
    pub severe_reinforcement: f32,

    /// Minimum improvement required for success
    pub min_improvement: f32,

    /// Maximum weight adjustment allowed
    pub max_weight_adjustment: f32,
}

impl Default for DriftCorrectorConfig {
    fn default() -> Self {
        Self {
            moderate_threshold_delta: 0.02,
            severe_threshold_delta: 0.05,
            moderate_reinforcement: 1.2,
            severe_reinforcement: 1.5,
            min_improvement: 0.01,
            max_weight_adjustment: 0.2,
        }
    }
}
