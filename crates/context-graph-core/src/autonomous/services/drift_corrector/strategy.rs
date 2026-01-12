//! Correction strategy types for drift correction.

use serde::{Deserialize, Serialize};

/// Correction strategy to apply based on drift analysis
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CorrectionStrategy {
    /// No correction needed
    NoAction,

    /// Adjust alignment thresholds by delta
    ThresholdAdjustment {
        /// Delta to apply to thresholds (positive = tighten, negative = loosen)
        delta: f32,
    },

    /// Rebalance section weights
    WeightRebalance {
        /// Vector of (index, adjustment) pairs for weight changes
        adjustments: Vec<(usize, f32)>,
    },

    /// Reinforce goal alignment with emphasis factor
    GoalReinforcement {
        /// Factor to emphasize goal alignment (1.0 = normal, >1.0 = increased)
        emphasis_factor: f32,
    },

    /// Emergency intervention required
    EmergencyIntervention {
        /// Reason for requiring intervention
        reason: String,
    },
}
