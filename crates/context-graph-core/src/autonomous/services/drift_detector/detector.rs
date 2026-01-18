//! DriftDetector Implementation
//!
//! Core drift detection service for monitoring alignment.

use crate::autonomous::drift::{DriftConfig, DriftMonitoring, DriftSeverity, DriftTrend};

use super::types::{DetectorDataPoint, DetectorState, DriftRecommendation, MIN_SAMPLES_DEFAULT};

/// Service for detecting alignment drift
#[derive(Clone, Debug)]
pub struct DriftDetector {
    config: DriftConfig,
    state: DetectorState,
    min_samples: usize,
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DriftDetector {
    /// Create a new DriftDetector with default configuration
    pub fn new() -> Self {
        Self {
            config: DriftConfig::default(),
            state: DetectorState::default(),
            min_samples: MIN_SAMPLES_DEFAULT,
        }
    }

    /// Create a new DriftDetector with custom configuration
    pub fn with_config(config: DriftConfig) -> Self {
        Self {
            config,
            state: DetectorState::default(),
            min_samples: MIN_SAMPLES_DEFAULT,
        }
    }

    /// Set custom minimum samples requirement
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        if min_samples == 0 {
            panic!("FAIL FAST: min_samples must be > 0");
        }
        self.min_samples = min_samples;
        self
    }

    /// Set initial baseline alignment
    pub fn with_baseline(mut self, baseline: f32) -> Self {
        if !(0.0..=1.0).contains(&baseline) {
            panic!(
                "FAIL FAST: baseline must be in range [0.0, 1.0], got {}",
                baseline
            );
        }
        self.state.baseline = baseline;
        self.state.rolling_mean = baseline;
        self.state.current_alignment = baseline;
        self
    }

    /// Add a new alignment observation
    ///
    /// # Panics
    /// Panics if alignment is not in range [0.0, 1.0]
    pub fn add_observation(&mut self, alignment: f32, timestamp: u64) {
        if !(0.0..=1.0).contains(&alignment) {
            panic!(
                "FAIL FAST: alignment must be in range [0.0, 1.0], got {}",
                alignment
            );
        }

        // Compute delta from current rolling mean before updating
        let delta_from_mean = alignment - self.state.rolling_mean;

        // Create and store data point
        let point = DetectorDataPoint {
            timestamp,
            alignment,
            delta_from_mean,
        };
        self.state.data_points.push_back(point);

        // Update current alignment
        self.state.current_alignment = alignment;

        // Trim old data points beyond window
        self.trim_window(timestamp);

        // Recompute rolling statistics
        self.compute_rolling_stats();
    }

    /// Trim data points outside the rolling window
    fn trim_window(&mut self, current_timestamp: u64) {
        // Window in milliseconds (assuming timestamp is millis)
        let window_ms = self.config.window_days as u64 * 24 * 60 * 60 * 1000;
        let cutoff = current_timestamp.saturating_sub(window_ms);

        while let Some(front) = self.state.data_points.front() {
            if front.timestamp < cutoff {
                self.state.data_points.pop_front();
            } else {
                break;
            }
        }
    }

    /// Compute rolling mean and variance from data points
    pub fn compute_rolling_stats(&mut self) {
        let n = self.state.data_points.len();
        if n == 0 {
            return;
        }

        // Compute mean
        let sum: f32 = self.state.data_points.iter().map(|p| p.alignment).sum();
        self.state.rolling_mean = sum / n as f32;

        // Compute variance using Welford's online algorithm for numerical stability
        if n < 2 {
            self.state.rolling_variance = 0.0;
        } else {
            let variance_sum: f32 = self
                .state
                .data_points
                .iter()
                .map(|p| {
                    let diff = p.alignment - self.state.rolling_mean;
                    diff * diff
                })
                .sum();
            self.state.rolling_variance = variance_sum / (n - 1) as f32;
        }

        // Update severity and trend
        self.update_severity();
        self.update_trend();
    }

    /// Update severity classification based on drift from baseline
    fn update_severity(&mut self) {
        let drift = (self.state.baseline - self.state.rolling_mean).abs();

        self.state.severity = if drift >= self.config.severe_threshold {
            DriftSeverity::Severe
        } else if drift >= self.config.alert_threshold {
            DriftSeverity::Moderate
        } else if drift > 0.01 {
            DriftSeverity::Mild
        } else {
            DriftSeverity::None
        };
    }

    /// Update trend based on recent data points
    fn update_trend(&mut self) {
        let n = self.state.data_points.len();
        if n < 3 {
            self.state.trend = DriftTrend::Stable;
            return;
        }

        // Use linear regression slope on recent points to determine trend
        // We use the last min(n, 10) points for trend detection
        let trend_window = n.min(10);
        let recent: Vec<f32> = self
            .state
            .data_points
            .iter()
            .skip(n - trend_window)
            .map(|p| p.alignment)
            .collect();

        // Compute slope using least squares
        let slope = self.compute_slope(&recent);

        // Classify trend based on slope magnitude
        const SLOPE_THRESHOLD: f32 = 0.005;
        self.state.trend = if slope > SLOPE_THRESHOLD {
            DriftTrend::Improving
        } else if slope < -SLOPE_THRESHOLD {
            DriftTrend::Declining
        } else {
            DriftTrend::Stable
        };
    }

    /// Compute slope using least squares regression
    pub(crate) fn compute_slope(&self, values: &[f32]) -> f32 {
        let n = values.len();
        if n < 2 {
            return 0.0;
        }

        let n_f = n as f32;

        // x values are indices 0, 1, 2, ...
        // sum_x = n*(n-1)/2
        let sum_x: f32 = (0..n).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..n).map(|i| (i * i) as f32).sum();

        let denominator = n_f * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return 0.0;
        }

        (n_f * sum_xy - sum_x * sum_y) / denominator
    }

    /// Detect current drift severity
    pub fn detect_drift(&self) -> DriftSeverity {
        self.state.severity.clone()
    }

    /// Compute current trend direction
    pub fn compute_trend(&self) -> DriftTrend {
        self.state.trend
    }

    /// Get the current drift score (distance from baseline)
    pub fn get_drift_score(&self) -> f32 {
        (self.state.baseline - self.state.rolling_mean).abs()
    }

    /// Check if drift requires attention (moderate or severe)
    pub fn requires_attention(&self) -> bool {
        matches!(
            self.state.severity,
            DriftSeverity::Moderate | DriftSeverity::Severe
        )
    }

    /// Check if drift requires user intervention (severe only)
    pub fn requires_intervention(&self) -> bool {
        matches!(self.state.severity, DriftSeverity::Severe)
    }

    /// Get recommendation based on current drift state
    pub fn get_recommendation(&self) -> DriftRecommendation {
        // Not enough samples for reliable analysis
        if self.state.data_points.len() < self.min_samples {
            return DriftRecommendation::Monitor;
        }

        match (&self.state.severity, &self.state.trend) {
            // No drift - no action
            (DriftSeverity::None, _) => DriftRecommendation::NoAction,

            // Mild drift - just monitor
            (DriftSeverity::Mild, DriftTrend::Improving) => DriftRecommendation::NoAction,
            (DriftSeverity::Mild, _) => DriftRecommendation::Monitor,

            // Moderate drift - depends on trend
            (DriftSeverity::Moderate, DriftTrend::Improving) => DriftRecommendation::Monitor,
            (DriftSeverity::Moderate, DriftTrend::Stable) => DriftRecommendation::ReviewMemories,
            (DriftSeverity::Moderate, DriftTrend::Declining | DriftTrend::Worsening) => {
                DriftRecommendation::AdjustThresholds
            }

            // Severe drift - serious action needed
            (DriftSeverity::Severe, DriftTrend::Improving) => {
                DriftRecommendation::RecalibrateBaseline
            }
            (DriftSeverity::Severe, DriftTrend::Stable) => DriftRecommendation::RecalibrateBaseline,
            (DriftSeverity::Severe, DriftTrend::Declining | DriftTrend::Worsening) => {
                DriftRecommendation::UserIntervention
            }
        }
    }

    /// Get the current rolling mean
    pub fn rolling_mean(&self) -> f32 {
        self.state.rolling_mean
    }

    /// Get the current rolling variance
    pub fn rolling_variance(&self) -> f32 {
        self.state.rolling_variance
    }

    /// Get the current baseline
    pub fn baseline(&self) -> f32 {
        self.state.baseline
    }

    /// Get number of data points in the window
    pub fn data_point_count(&self) -> usize {
        self.state.data_points.len()
    }

    /// Reset the baseline to the current rolling mean
    pub fn reset_baseline(&mut self) {
        self.state.baseline = self.state.rolling_mean;
        self.update_severity();
    }

    /// Check if continuous monitoring is enabled
    pub fn is_continuous_monitoring(&self) -> bool {
        matches!(self.config.monitoring, DriftMonitoring::Continuous)
    }

    /// Get a reference to the internal state (for testing/inspection)
    pub fn state(&self) -> &DetectorState {
        &self.state
    }

    /// Get a reference to the config
    pub fn config(&self) -> &DriftConfig {
        &self.config
    }
}
