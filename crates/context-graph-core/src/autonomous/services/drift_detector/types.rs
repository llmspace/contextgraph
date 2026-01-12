//! NORTH-010: DriftDetector Types
//!
//! Data types for drift detection including observations, recommendations, and state.

use crate::autonomous::drift::{DriftSeverity, DriftTrend};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Minimum samples required for reliable statistics
pub const MIN_SAMPLES_DEFAULT: usize = 10;

/// A single data point for drift detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorDataPoint {
    /// Timestamp of observation (Unix epoch millis or monotonic counter)
    pub timestamp: u64,
    /// Alignment score at this point
    pub alignment: f32,
    /// Delta from the rolling mean at time of observation
    pub delta_from_mean: f32,
}

/// Recommendation for handling detected drift
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftRecommendation {
    /// No action needed - alignment is stable
    NoAction,
    /// Continue monitoring - minor drift detected
    Monitor,
    /// Review recent memories - moderate drift may indicate quality issues
    ReviewMemories,
    /// Adjust thresholds - drift may indicate threshold miscalibration
    AdjustThresholds,
    /// Recalibrate baseline - severe drift requires re-establishing baseline
    RecalibrateBaseline,
    /// User intervention required - critical drift detected
    UserIntervention,
}

/// Internal state for drift detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorState {
    /// Current alignment (most recent observation)
    pub current_alignment: f32,
    /// Rolling mean of alignment scores
    pub rolling_mean: f32,
    /// Rolling variance of alignment scores
    pub rolling_variance: f32,
    /// Current trend direction
    pub trend: DriftTrend,
    /// Current severity level
    pub severity: DriftSeverity,
    /// Historical data points within the window
    pub data_points: VecDeque<DetectorDataPoint>,
    /// Baseline alignment to compare against
    pub baseline: f32,
}

impl Default for DetectorState {
    fn default() -> Self {
        Self {
            current_alignment: 0.75,
            rolling_mean: 0.75,
            rolling_variance: 0.0,
            trend: DriftTrend::Stable,
            severity: DriftSeverity::None,
            data_points: VecDeque::with_capacity(256),
            baseline: 0.75,
        }
    }
}
