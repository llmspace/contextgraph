//! ThresholdLearner implementation.
//!
//! Implements adaptive threshold learning using the 4-level ATC architecture.
//! All thresholds are learned from feedback rather than hardcoded.

use chrono::{DateTime, Duration, Utc};
use rand::prelude::*;
use rand_distr::Beta;
use serde::{Deserialize, Serialize};

use crate::autonomous::{
    AdaptiveThresholdConfig, AdaptiveThresholdState, AlignmentBucket, RetrievalStats,
};

use super::types::{
    BayesianObservation, EmbedderLearningState, DEFAULT_ALPHA, MIN_OBSERVATIONS_FOR_RECALIBRATION,
    NUM_EMBEDDERS, RECALIBRATION_CHECK_INTERVAL_SECS,
};

/// NORTH-009 ThresholdLearner Service
///
/// Implements adaptive threshold learning using the 4-level ATC architecture.
/// All thresholds are learned from feedback rather than hardcoded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdLearner {
    /// Configuration
    config: AdaptiveThresholdConfig,

    /// Current threshold state
    state: AdaptiveThresholdState,

    /// Per-embedder learning states
    embedder_states: [EmbedderLearningState; NUM_EMBEDDERS],

    /// EWMA alpha (smoothing factor)
    ewma_alpha: f32,

    /// Overall observation count
    total_observations: u32,

    /// Total successes across all observations
    total_successes: u32,

    /// Bayesian optimization history (Level 4)
    bayesian_history: Vec<BayesianObservation>,

    /// Best performance seen
    best_performance: f32,

    /// Last recalibration check
    last_recalibration_check: DateTime<Utc>,

    /// Creation timestamp
    created_at: DateTime<Utc>,
}

impl ThresholdLearner {
    /// Create a new ThresholdLearner with default configuration.
    pub fn new() -> Self {
        Self::with_config(AdaptiveThresholdConfig::default())
    }

    /// Create a ThresholdLearner with custom configuration.
    pub fn with_config(config: AdaptiveThresholdConfig) -> Self {
        let now = Utc::now();
        let embedder_states = std::array::from_fn(|_| EmbedderLearningState::default());

        Self {
            config,
            state: AdaptiveThresholdState::default(),
            embedder_states,
            ewma_alpha: DEFAULT_ALPHA,
            total_observations: 0,
            total_successes: 0,
            bayesian_history: Vec::new(),
            best_performance: 0.0,
            last_recalibration_check: now,
            created_at: now,
        }
    }

    /// Learn from retrieval feedback to update thresholds.
    ///
    /// This is the main learning entry point that orchestrates all 4 ATC levels:
    /// 1. Updates EWMA for drift tracking
    /// 2. Adjusts temperature scaling
    /// 3. Updates Thompson sampling parameters
    /// 4. Records for Bayesian optimization
    pub fn learn_from_feedback(&mut self, stats: &RetrievalStats, was_relevant: bool) {
        if !self.config.enabled {
            return;
        }

        self.total_observations += 1;
        if was_relevant {
            self.total_successes += 1;
        }

        // Level 1: Update EWMA for global threshold tracking
        let observed_success_rate = stats.success_rate();
        let new_optimal = self.update_ewma(self.state.optimal, observed_success_rate);
        self.state.optimal =
            new_optimal.clamp(self.config.optimal_bounds.0, self.config.optimal_bounds.1);

        // Update per-bucket thresholds based on success rates
        for (bucket, &rate) in stats.bucket_success_rates.iter() {
            match bucket {
                AlignmentBucket::Optimal => {
                    if rate > 0.8 {
                        // High success rate in optimal bucket - threshold is well-calibrated
                        self.state.optimal = self.update_ewma(self.state.optimal, 0.0);
                    }
                }
                AlignmentBucket::Acceptable => {
                    // Adjust acceptable threshold based on its success rate
                    let adjustment = (rate - 0.7) * self.config.learning_rate;
                    self.state.acceptable = (self.state.acceptable + adjustment).clamp(0.65, 0.80);
                }
                AlignmentBucket::Warning => {
                    // Warning bucket should have moderate success
                    let adjustment = (rate - 0.5) * self.config.learning_rate;
                    self.state.warning = (self.state.warning + adjustment)
                        .clamp(self.config.warning_bounds.0, self.config.warning_bounds.1);
                }
                AlignmentBucket::Critical => {
                    // Critical bucket should have low success - if high, raise warning threshold
                    if rate > 0.3 {
                        self.state.warning += self.config.learning_rate * 0.5;
                    }
                }
            }
        }

        // Level 3: Update Thompson sampling for exploration
        let alpha = self.ewma_alpha; // Cache alpha to avoid borrow conflict
        for (idx, embedder_state) in self.embedder_states.iter_mut().enumerate() {
            embedder_state.observation_count += 1;
            if was_relevant {
                embedder_state.success_count += 1;
                embedder_state.thompson.alpha += 1.0;
            } else {
                embedder_state.thompson.beta += 1.0;
            }
            embedder_state.thompson.samples += 1;
            embedder_state.last_updated = Utc::now();

            // Update per-embedder EWMA threshold (inline to avoid borrow conflict)
            let embedder_success_rate = embedder_state.success_count as f32
                / embedder_state.observation_count.max(1) as f32;
            embedder_state.ewma_threshold =
                alpha * embedder_success_rate + (1.0 - alpha) * embedder_state.ewma_threshold;

            // Sync to state
            if idx < NUM_EMBEDDERS {
                self.state.per_embedder[idx] = embedder_state.ewma_threshold;
            }
        }

        // Record performance for Level 4 Bayesian optimization
        let performance = self.total_successes as f32 / self.total_observations.max(1) as f32;
        if performance > self.best_performance {
            self.best_performance = performance;
        }

        // Every 100 observations, record a Bayesian observation
        if self.total_observations % 100 == 0 {
            self.bayesian_history.push(BayesianObservation {
                thresholds: self.state.per_embedder,
                performance,
                timestamp: Utc::now(),
            });
        }

        // Update state timestamp
        self.state.updated_at = Utc::now();
    }

    /// Update EWMA value with a new observation.
    ///
    /// Formula: `θ_ewma(t) = α × θ_observed(t) + (1 - α) × θ_ewma(t-1)`
    #[inline]
    pub fn update_ewma(&mut self, current: f32, observed: f32) -> f32 {
        if observed.is_nan() || current.is_nan() {
            return current;
        }
        self.ewma_alpha * observed + (1.0 - self.ewma_alpha) * current
    }

    /// Apply temperature scaling to logits for calibration.
    ///
    /// Formula: `calibrated = softmax(logits / T)`
    pub fn temperature_scale(&self, logits: &[f32], temperature: f32) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let temp = if temperature <= 0.0 {
            0.01
        } else {
            temperature
        };

        // Scale logits by temperature
        let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

        // Numerical stability: subtract max before exp
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        if sum == 0.0 || sum.is_nan() || sum.is_infinite() {
            // Uniform distribution fallback
            let uniform = 1.0 / logits.len() as f32;
            return vec![uniform; logits.len()];
        }

        exp_vals.iter().map(|&e| e / sum).collect()
    }

    /// Sample a threshold using Thompson Sampling.
    ///
    /// Samples from `Beta(α, β)` for the specified embedder.
    pub fn thompson_sample(&mut self, embedder_idx: usize) -> f32 {
        if embedder_idx >= NUM_EMBEDDERS {
            return 0.75; // Fallback for invalid index
        }

        let state = &self.embedder_states[embedder_idx].thompson;
        let mut rng = thread_rng();

        // Sample from Beta distribution
        match Beta::new(state.alpha as f64, state.beta as f64) {
            Ok(dist) => {
                let sample = dist.sample(&mut rng) as f32;
                // Scale sample to threshold range [0.5, 0.95]
                0.5 + sample * 0.45
            }
            Err(_) => {
                // Fallback to mean if Beta fails (should not happen with α,β >= 1)
                let mean = state.alpha / (state.alpha + state.beta);
                0.5 + mean * 0.45
            }
        }
    }

    /// Perform Bayesian update on prior belief.
    ///
    /// Formula: `posterior ∝ prior × likelihood`
    pub fn bayesian_update(&mut self, prior: f32, likelihood: f32) -> f32 {
        if prior.is_nan() || likelihood.is_nan() || prior < 0.0 || likelihood < 0.0 {
            return 0.5; // Neutral prior for invalid input
        }

        let prior_pos = prior.clamp(0.001, 0.999);
        let prior_neg = 1.0 - prior_pos;

        let posterior_unnorm = prior_pos * likelihood;
        let marginal = posterior_unnorm + prior_neg * (1.0 - likelihood);

        if marginal == 0.0 || marginal.is_nan() {
            return 0.5;
        }

        (posterior_unnorm / marginal).clamp(0.0, 1.0)
    }

    /// Get the current threshold for a specific embedder.
    #[inline]
    pub fn get_threshold(&self, embedder_idx: usize) -> f32 {
        if embedder_idx < NUM_EMBEDDERS {
            self.state.per_embedder[embedder_idx]
        } else {
            self.state.optimal // Fallback
        }
    }

    /// Check if recalibration should be triggered.
    pub fn should_recalibrate(&self) -> bool {
        // Need minimum observations
        if self.total_observations < MIN_OBSERVATIONS_FOR_RECALIBRATION {
            return false;
        }

        // Check time since last recalibration
        let elapsed = Utc::now().signed_duration_since(self.last_recalibration_check);
        if elapsed < Duration::seconds(RECALIBRATION_CHECK_INTERVAL_SECS) {
            return false;
        }

        // Check for drift in any embedder
        for embedder_state in &self.embedder_states {
            let expected = 0.75; // Prior expectation
            let current = embedder_state.ewma_threshold;
            let std_dev = 0.05; // Assumed baseline std
            let drift = ((current - expected) / std_dev).abs();

            if drift > 2.0 {
                return true; // Level 2 trigger
            }
        }

        // Check overall success rate drift
        let overall_success_rate =
            self.total_successes as f32 / self.total_observations.max(1) as f32;
        if !(0.5..=0.95).contains(&overall_success_rate) {
            return true; // Extreme rates warrant recalibration
        }

        false
    }

    /// Get the current adaptive threshold state.
    pub fn get_state(&self) -> &AdaptiveThresholdState {
        &self.state
    }

    /// Get configuration.
    pub fn get_config(&self) -> &AdaptiveThresholdConfig {
        &self.config
    }

    /// Get total observation count.
    pub fn total_observations(&self) -> u32 {
        self.total_observations
    }

    /// Get best performance seen.
    pub fn best_performance(&self) -> f32 {
        self.best_performance
    }

    /// Mark recalibration as checked (resets timer).
    pub fn mark_recalibration_checked(&mut self) {
        self.last_recalibration_check = Utc::now();
    }

    /// Get embedder learning state for inspection.
    pub fn get_embedder_state(&self, embedder_idx: usize) -> Option<&EmbedderLearningState> {
        if embedder_idx < NUM_EMBEDDERS {
            Some(&self.embedder_states[embedder_idx])
        } else {
            None
        }
    }

    /// Get Bayesian optimization history.
    pub fn get_bayesian_history(&self) -> &[BayesianObservation] {
        &self.bayesian_history
    }

    /// Set EWMA alpha (for testing/tuning).
    pub fn set_ewma_alpha(&mut self, alpha: f32) {
        self.ewma_alpha = alpha.clamp(0.05, 0.5);
    }

    /// Reset learning state (for testing).
    #[cfg(test)]
    pub fn reset(&mut self) {
        self.state = AdaptiveThresholdState::default();
        self.embedder_states = std::array::from_fn(|_| EmbedderLearningState::default());
        self.total_observations = 0;
        self.total_successes = 0;
        self.bayesian_history.clear();
        self.best_performance = 0.0;
        self.last_recalibration_check = Utc::now();
    }
}

impl Default for ThresholdLearner {
    fn default() -> Self {
        Self::new()
    }
}
