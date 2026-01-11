//! Meta-Cognitive Core Implementation
//!
//! Implements self-correction and adaptive learning through meta-cognitive monitoring
//! as specified in Constitution v4.0.0 Section gwt.meta_cognitive (lines 410-417).
//!
//! ## Formula
//!
//! MetaScore = σ(2 × (L_predicted - L_actual))
//!
//! Where:
//! - L_predicted: Meta-UTL predicted learning score
//! - L_actual: Actual observed learning score
//! - σ: Sigmoid function
//!
//! ## Self-Correction Protocol
//!
//! - Low MetaScore (<0.5) for 5+ consecutive operations → increase Acetylcholine, trigger dream
//! - High MetaScore (>0.9) → reduce meta-monitoring frequency

use crate::error::CoreResult;
use chrono::Utc;
use std::collections::VecDeque;

use super::types::{
    FrequencyAdjustment, MetaCognitiveLoop, MetaCognitiveState, ScoreTrend, ACH_BASELINE, ACH_MAX,
    ACH_DECAY_RATE,
};

impl MetaCognitiveLoop {
    /// Create a new meta-cognitive loop
    pub fn new() -> Self {
        Self {
            recent_scores: VecDeque::new(),
            max_history: 20,
            consecutive_low_scores: 0,
            consecutive_high_scores: 0,
            acetylcholine_level: ACH_BASELINE, // Default learning rate (baseline)
            monitoring_frequency: 1.0,         // 1 Hz by default
            last_update: Utc::now(),
        }
    }

    /// Evaluate meta-cognitive score
    ///
    /// # Arguments
    /// - `predicted_learning`: L_predicted (0 to 1)
    /// - `actual_learning`: L_actual (0 to 1)
    ///
    /// # Returns
    /// MetaScore = σ(2 × (L_predicted - L_actual))
    pub async fn evaluate(
        &mut self,
        predicted_learning: f32,
        actual_learning: f32,
    ) -> CoreResult<MetaCognitiveState> {
        // Clamp inputs to valid range
        let pred = predicted_learning.clamp(0.0, 1.0);
        let actual = actual_learning.clamp(0.0, 1.0);

        // Compute prediction error
        let error = pred - actual;

        // Meta-score = sigmoid(2 × error)
        let meta_score = self.sigmoid(2.0 * error);

        // Track score history
        self.recent_scores.push_back(meta_score);
        if self.recent_scores.len() > self.max_history {
            self.recent_scores.pop_front();
        }

        // Update consecutive counters
        if meta_score < 0.5 {
            self.consecutive_low_scores += 1;
            self.consecutive_high_scores = 0;
        } else if meta_score > 0.9 {
            self.consecutive_high_scores += 1;
            self.consecutive_low_scores = 0;
        } else {
            self.consecutive_low_scores = 0;
            self.consecutive_high_scores = 0;
        }

        // Calculate average
        let avg_meta_score = if self.recent_scores.is_empty() {
            0.5
        } else {
            self.recent_scores.iter().sum::<f32>() / self.recent_scores.len() as f32
        };

        // Determine trend
        let trend = self.calculate_trend();

        // Check for dream trigger (low MetaScore for 5+ consecutive operations)
        let dream_triggered = self.consecutive_low_scores >= 5;
        if dream_triggered {
            // Increase ACh on dream trigger (learning rate boost)
            self.acetylcholine_level =
                (self.acetylcholine_level * 1.5).clamp(ACH_BASELINE, ACH_MAX);
            self.consecutive_low_scores = 0; // Reset after triggering
        } else {
            // Decay ACh toward baseline when not triggered (homeostatic regulation)
            // Per constitution spec: neuromodulators must decay toward baseline
            self.acetylcholine_level =
                self.decay_toward(self.acetylcholine_level, ACH_BASELINE, ACH_DECAY_RATE);
        }

        // Check for frequency adjustment
        let frequency_adjustment = if self.consecutive_high_scores >= 5 {
            FrequencyAdjustment::Increase // Reduce monitoring frequency
        } else if self.consecutive_low_scores >= 3 {
            FrequencyAdjustment::Decrease // Increase monitoring frequency
        } else {
            FrequencyAdjustment::None
        };

        // Apply frequency adjustment
        match frequency_adjustment {
            FrequencyAdjustment::Increase => {
                self.monitoring_frequency = (self.monitoring_frequency * 0.8).max(0.1);
            }
            FrequencyAdjustment::Decrease => {
                self.monitoring_frequency = (self.monitoring_frequency * 1.5).min(10.0);
            }
            FrequencyAdjustment::None => {}
        }

        self.last_update = Utc::now();

        Ok(MetaCognitiveState {
            meta_score,
            avg_meta_score,
            trend,
            acetylcholine: self.acetylcholine_level,
            dream_triggered,
            frequency_adjustment,
        })
    }

    /// Get the current Acetylcholine level (learning rate)
    pub fn acetylcholine(&self) -> f32 {
        self.acetylcholine_level
    }

    /// Get the current monitoring frequency
    pub fn monitoring_frequency(&self) -> f32 {
        self.monitoring_frequency
    }

    /// Get recent meta-scores as a vector (for plotting/debugging)
    pub fn get_recent_scores(&self) -> Vec<f32> {
        self.recent_scores.iter().copied().collect()
    }

    /// Logistic sigmoid: σ(x) = 1 / (1 + e^(-x))
    pub(crate) fn sigmoid(&self, x: f32) -> f32 {
        (1.0 / (1.0 + (-x).exp())).clamp(0.0, 1.0)
    }

    /// Decay a value toward a target baseline
    ///
    /// Implements exponential decay: value = value + (target - value) * rate
    /// This ensures smooth convergence toward baseline without overshooting.
    ///
    /// # Arguments
    /// - `current`: Current value
    /// - `target`: Target baseline to decay toward
    /// - `rate`: Decay rate (0.0 to 1.0, higher = faster decay)
    fn decay_toward(&self, current: f32, target: f32, rate: f32) -> f32 {
        let rate = rate.clamp(0.0, 1.0);
        current + (target - current) * rate
    }

    /// Determine score trend from recent history
    fn calculate_trend(&self) -> ScoreTrend {
        if self.recent_scores.len() < 3 {
            return ScoreTrend::Stable;
        }

        let len = self.recent_scores.len();
        let first_half: f32 =
            self.recent_scores.iter().take(len / 2).sum::<f32>() / (len / 2) as f32;
        let second_half: f32 =
            self.recent_scores.iter().skip(len / 2).sum::<f32>() / (len - len / 2) as f32;

        let delta = second_half - first_half;

        if delta > 0.1 {
            ScoreTrend::Increasing
        } else if delta < -0.1 {
            ScoreTrend::Decreasing
        } else {
            ScoreTrend::Stable
        }
    }
}

impl Default for MetaCognitiveLoop {
    fn default() -> Self {
        Self::new()
    }
}
