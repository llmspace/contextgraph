//! Core types for teleological drift detection.
//!
//! Contains both legacy types for NORTH-010/011 services and new types for TASK-LOGIC-010.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::teleological::Embedder;

use super::error::DriftError;

// ============================================
// CONSTANTS
// ============================================

/// Number of embedders in the system.
pub const NUM_EMBEDDERS: usize = 13;

/// Minimum samples required for trend analysis.
pub const MIN_TREND_SAMPLES: usize = 3;

/// Maximum most-drifted embedders to return.
pub const MAX_MOST_DRIFTED: usize = 5;

// ============================================
// BACKWARD COMPATIBILITY TYPES
// (Preserved from original drift.rs for NORTH-010/011 services)
// ============================================

/// Drift detection configuration (legacy, for NORTH-010/011 services).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Enable continuous monitoring
    pub monitoring: DriftMonitoring,

    /// Alert threshold (alignment drop to trigger alert)
    pub alert_threshold: f32, // default: 0.05

    /// Enable auto-correction
    pub auto_correct: bool,

    /// Severe drift threshold (requires user intervention)
    pub severe_threshold: f32, // default: 0.10

    /// Rolling window size in days
    pub window_days: u32, // default: 7
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            monitoring: DriftMonitoring::Continuous,
            alert_threshold: 0.05,
            auto_correct: true,
            severe_threshold: 0.10,
            window_days: 7,
        }
    }
}

/// Monitoring mode for drift detection (legacy).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DriftMonitoring {
    /// Continuous real-time monitoring
    Continuous,
    /// Periodic checks at specified interval
    Periodic { interval_hours: u32 },
    /// Manual checks only
    Manual,
}

/// Drift severity levels (legacy, 4 levels for NORTH-010/011 services).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DriftSeverity {
    /// No significant drift detected
    None,
    /// Mild drift (< alert_threshold)
    Mild,
    /// Moderate drift (>= alert_threshold, < severe_threshold)
    Moderate,
    /// Severe drift (>= severe_threshold)
    Severe,
}

/// A single drift data point for history (legacy).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftDataPoint {
    /// Mean alignment score at this point
    pub alignment_mean: f32,
    /// Number of new memories added
    pub new_memories_count: u32,
    /// Timestamp of this data point
    pub timestamp: DateTime<Utc>,
}

/// Current drift state (legacy, for NORTH-010/011 services).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftState {
    /// Rolling mean alignment (window-based)
    pub rolling_mean: f32,

    /// Baseline alignment (established mean)
    pub baseline: f32,

    /// Current drift magnitude (baseline - rolling_mean)
    pub drift: f32,

    /// Drift severity
    pub severity: DriftSeverity,

    /// Trend direction
    pub trend: DriftTrend,

    /// Last check timestamp
    pub checked_at: DateTime<Utc>,

    /// Historical data points
    pub history: VecDeque<DriftDataPoint>,
}

impl Default for DriftState {
    fn default() -> Self {
        Self {
            rolling_mean: 0.75,
            baseline: 0.75,
            drift: 0.0,
            severity: DriftSeverity::None,
            trend: DriftTrend::Stable,
            checked_at: Utc::now(),
            history: VecDeque::with_capacity(168), // 7 days * 24 hours
        }
    }
}

impl DriftState {
    /// Create a new DriftState with a specific baseline
    pub fn with_baseline(baseline: f32) -> Self {
        Self {
            rolling_mean: baseline,
            baseline,
            drift: 0.0,
            severity: DriftSeverity::None,
            trend: DriftTrend::Stable,
            checked_at: Utc::now(),
            history: VecDeque::with_capacity(168),
        }
    }

    /// Add a new data point and update rolling statistics
    pub fn add_data_point(&mut self, mean_alignment: f32, new_memories: u32, config: &DriftConfig) {
        let point = DriftDataPoint {
            alignment_mean: mean_alignment,
            new_memories_count: new_memories,
            timestamp: Utc::now(),
        };

        self.history.push_back(point);

        // Keep only window_days worth of data (assuming hourly points)
        let max_points = (config.window_days * 24) as usize;
        while self.history.len() > max_points {
            self.history.pop_front();
        }

        self.update_rolling_mean();
        self.update_severity(config);
        self.checked_at = Utc::now();
    }

    /// Update the rolling mean from history
    fn update_rolling_mean(&mut self) {
        if self.history.is_empty() {
            return;
        }

        let sum: f32 = self.history.iter().map(|p| p.alignment_mean).sum();
        let old_mean = self.rolling_mean;
        self.rolling_mean = sum / self.history.len() as f32;
        self.drift = self.baseline - self.rolling_mean;

        // Determine trend based on change in rolling mean
        let delta = self.rolling_mean - old_mean;
        self.trend = if delta.abs() < 0.01 {
            DriftTrend::Stable
        } else if delta > 0.0 {
            DriftTrend::Improving
        } else {
            DriftTrend::Declining
        };
    }

    /// Update severity classification based on drift magnitude
    fn update_severity(&mut self, config: &DriftConfig) {
        self.severity = if self.drift.abs() >= config.severe_threshold {
            DriftSeverity::Severe
        } else if self.drift.abs() >= config.alert_threshold {
            DriftSeverity::Moderate
        } else if self.drift.abs() > 0.01 {
            DriftSeverity::Mild
        } else {
            DriftSeverity::None
        };
    }

    /// Reset baseline to current rolling mean
    pub fn reset_baseline(&mut self) {
        self.baseline = self.rolling_mean;
        self.drift = 0.0;
        self.severity = DriftSeverity::None;
    }

    /// Check if drift requires attention (moderate or severe)
    pub fn requires_attention(&self) -> bool {
        matches!(
            self.severity,
            DriftSeverity::Moderate | DriftSeverity::Severe
        )
    }

    /// Check if drift is severe enough to require user intervention
    pub fn requires_intervention(&self) -> bool {
        matches!(self.severity, DriftSeverity::Severe)
    }

    /// Get the number of data points in history
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get total new memories across all data points in history
    pub fn total_new_memories(&self) -> u64 {
        self.history
            .iter()
            .map(|p| p.new_memories_count as u64)
            .sum()
    }
}

// ============================================
// TELEOLOGICAL DRIFT TYPES (TASK-LOGIC-010)
// ============================================

/// Drift severity levels (5 levels, ordered worst-to-best for Ord).
///
/// Uses ordering where Critical < High < Medium < Low < None, so that
/// sorting in ascending order puts worst drift first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriftLevel {
    /// Critical drift: similarity < 0.40
    Critical,
    /// High drift: similarity >= 0.40, < 0.55
    High,
    /// Medium drift: similarity >= 0.55, < 0.70
    Medium,
    /// Low drift: similarity >= 0.70, < 0.85
    Low,
    /// No significant drift: similarity >= 0.85
    None,
}

impl DriftLevel {
    /// Classify a similarity score to a drift level using the given thresholds.
    #[inline]
    pub fn from_similarity(similarity: f32, thresholds: &DriftThresholds) -> Self {
        if similarity >= thresholds.none_min {
            DriftLevel::None
        } else if similarity >= thresholds.low_min {
            DriftLevel::Low
        } else if similarity >= thresholds.medium_min {
            DriftLevel::Medium
        } else if similarity >= thresholds.high_min {
            DriftLevel::High
        } else {
            DriftLevel::Critical
        }
    }

    /// Check if this level indicates drift occurred.
    #[inline]
    pub fn has_drifted(self) -> bool {
        self != DriftLevel::None
    }

    /// Check if this level requires recommendations.
    #[inline]
    pub fn needs_recommendation(self) -> bool {
        matches!(
            self,
            DriftLevel::Critical | DriftLevel::High | DriftLevel::Medium
        )
    }
}

/// Drift trend direction computed from history via linear regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftTrend {
    /// Alignment is improving (positive slope in similarity)
    Improving,
    /// Alignment is stable (|slope| < 0.01)
    Stable,
    /// Alignment is worsening (negative slope in similarity) - new name for TASK-LOGIC-010
    Worsening,
    /// Alignment is declining (legacy name, same as Worsening)
    Declining,
}

/// Configuration for drift thresholds.
///
/// Thresholds define similarity score boundaries for each drift level.
/// Must satisfy: none_min > low_min > medium_min > high_min
#[derive(Debug, Clone)]
pub struct DriftThresholds {
    /// Minimum similarity for DriftLevel::None (default: 0.85)
    pub none_min: f32,
    /// Minimum similarity for DriftLevel::Low (default: 0.70)
    pub low_min: f32,
    /// Minimum similarity for DriftLevel::Medium (default: 0.55)
    pub medium_min: f32,
    /// Minimum similarity for DriftLevel::High (default: 0.40)
    pub high_min: f32,
    // Below high_min = DriftLevel::Critical
}

impl Default for DriftThresholds {
    fn default() -> Self {
        Self {
            none_min: 0.85,
            low_min: 0.70,
            medium_min: 0.55,
            high_min: 0.40,
        }
    }
}

impl DriftThresholds {
    /// Validate that thresholds are in proper order.
    ///
    /// Returns an error if thresholds are not strictly decreasing.
    pub fn validate(&self) -> Result<(), DriftError> {
        if self.none_min <= self.low_min
            || self.low_min <= self.medium_min
            || self.medium_min <= self.high_min
            || self.high_min <= 0.0
            || self.none_min > 1.0
        {
            return Err(DriftError::InvalidThresholds {
                reason: format!(
                    "Thresholds must be: 1.0 >= none ({}) > low ({}) > medium ({}) > high ({}) > 0.0",
                    self.none_min, self.low_min, self.medium_min, self.high_min
                ),
            });
        }
        Ok(())
    }
}

// ============================================
// RESULT TYPES
// ============================================

/// Result of drift analysis.
#[derive(Debug)]
pub struct DriftResult {
    /// Overall drift assessment
    pub overall_drift: OverallDrift,
    /// Per-embedder drift breakdown for all 13 embedders
    pub per_embedder_drift: PerEmbedderDrift,
    /// Most drifted embedders, sorted worst first (max 5)
    pub most_drifted_embedders: Vec<EmbedderDriftInfo>,
    /// Recommendations for addressing drift (only for Medium+ drift)
    pub recommendations: Vec<super::recommendations::DriftRecommendation>,
    /// Trend analysis if history available
    pub trend: Option<super::history::TrendAnalysis>,
    /// Number of memories analyzed
    pub analyzed_count: usize,
    /// Timestamp of analysis
    pub timestamp: DateTime<Utc>,
}

/// Overall drift assessment.
#[derive(Debug)]
pub struct OverallDrift {
    /// Whether any drift was detected (drift_level > None)
    pub has_drifted: bool,
    /// Drift score: 1.0 - similarity (0.0 = no drift, 1.0 = total drift)
    pub drift_score: f32,
    /// Classified drift severity
    pub drift_level: DriftLevel,
    /// Raw similarity score (0.0 to 1.0)
    pub similarity: f32,
}

/// Per-embedder drift breakdown for all 13 embedders.
#[derive(Debug)]
pub struct PerEmbedderDrift {
    /// Drift info for each embedder, indexed by Embedder::index()
    pub embedder_drift: [EmbedderDriftInfo; NUM_EMBEDDERS],
}

/// Drift info for a single embedder.
#[derive(Debug, Clone)]
pub struct EmbedderDriftInfo {
    /// The embedder this info is for
    pub embedder: Embedder,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Classified drift level
    pub drift_level: DriftLevel,
    /// Drift score: 1.0 - similarity
    pub drift_score: f32,
}

impl EmbedderDriftInfo {
    /// Create a new embedder drift info.
    pub fn new(embedder: Embedder, similarity: f32, thresholds: &DriftThresholds) -> Self {
        let clamped = similarity.clamp(0.0, 1.0);
        Self {
            embedder,
            similarity: clamped,
            drift_level: DriftLevel::from_similarity(clamped, thresholds),
            drift_score: 1.0 - clamped,
        }
    }
}
