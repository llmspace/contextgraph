//! Drift corrector service implementation.

use crate::autonomous::{DriftConfig, DriftSeverity, DriftState, DriftTrend};

use super::{CorrectionResult, CorrectionStrategy, DriftCorrectorConfig};

/// Drift corrector service for applying correction strategies
#[derive(Clone, Debug)]
pub struct DriftCorrector {
    /// Configuration for correction behavior
    pub(crate) config: DriftCorrectorConfig,

    /// Current threshold adjustment state
    threshold_adjustment: f32,

    /// Current weight adjustments (index -> adjustment)
    weight_adjustments: Vec<(usize, f32)>,

    /// Current goal emphasis factor
    goal_emphasis: f32,

    /// Number of corrections applied
    corrections_applied: u64,

    /// Number of successful corrections
    successful_corrections: u64,
}

impl Default for DriftCorrector {
    fn default() -> Self {
        Self::new()
    }
}

impl DriftCorrector {
    /// Create a new drift corrector with default configuration
    pub fn new() -> Self {
        Self {
            config: DriftCorrectorConfig::default(),
            threshold_adjustment: 0.0,
            weight_adjustments: Vec::new(),
            goal_emphasis: 1.0,
            corrections_applied: 0,
            successful_corrections: 0,
        }
    }

    /// Create a new drift corrector with custom configuration
    pub fn with_config(config: DriftCorrectorConfig) -> Self {
        Self {
            config,
            threshold_adjustment: 0.0,
            weight_adjustments: Vec::new(),
            goal_emphasis: 1.0,
            corrections_applied: 0,
            successful_corrections: 0,
        }
    }

    /// Select appropriate correction strategy based on drift state
    pub fn select_strategy(&self, state: &DriftState) -> CorrectionStrategy {
        match state.severity {
            DriftSeverity::None => CorrectionStrategy::NoAction,

            DriftSeverity::Mild => {
                // Mild drift: slight goal reinforcement if declining/worsening
                if matches!(state.trend, DriftTrend::Declining | DriftTrend::Worsening) {
                    CorrectionStrategy::GoalReinforcement {
                        emphasis_factor: 1.1,
                    }
                } else {
                    CorrectionStrategy::NoAction
                }
            }

            DriftSeverity::Moderate => {
                // Moderate drift: threshold adjustment or reinforcement based on trend
                match state.trend {
                    DriftTrend::Declining | DriftTrend::Worsening => {
                        CorrectionStrategy::ThresholdAdjustment {
                            delta: self.config.moderate_threshold_delta,
                        }
                    }
                    DriftTrend::Stable => CorrectionStrategy::GoalReinforcement {
                        emphasis_factor: self.config.moderate_reinforcement,
                    },
                    DriftTrend::Improving => CorrectionStrategy::NoAction,
                }
            }

            DriftSeverity::Severe => {
                // Severe drift: aggressive correction or intervention
                match state.trend {
                    DriftTrend::Declining | DriftTrend::Worsening => {
                        // Critical: requires human intervention
                        CorrectionStrategy::EmergencyIntervention {
                            reason: format!(
                                "Severe drift ({:.3}) with declining trend. Manual review required.",
                                state.drift
                            ),
                        }
                    }
                    DriftTrend::Stable => {
                        // Severe but stable: aggressive threshold adjustment
                        CorrectionStrategy::ThresholdAdjustment {
                            delta: self.config.severe_threshold_delta,
                        }
                    }
                    DriftTrend::Improving => {
                        // Severe but improving: reinforcement to accelerate recovery
                        CorrectionStrategy::GoalReinforcement {
                            emphasis_factor: self.config.severe_reinforcement,
                        }
                    }
                }
            }
        }
    }

    /// Apply a correction strategy to the drift state
    pub fn apply_correction(
        &mut self,
        state: &mut DriftState,
        strategy: &CorrectionStrategy,
    ) -> CorrectionResult {
        let alignment_before = state.rolling_mean;

        match strategy {
            CorrectionStrategy::NoAction => {
                CorrectionResult::new(strategy.clone(), alignment_before, alignment_before, true)
            }

            CorrectionStrategy::ThresholdAdjustment { delta } => {
                self.adjust_thresholds(*delta);

                // Simulate alignment improvement from threshold adjustment
                // In practice, this would be measured after subsequent operations
                let improvement = delta * 0.5; // Conservative estimate
                let alignment_after = (alignment_before + improvement).clamp(0.0, 1.0);

                self.corrections_applied += 1;
                let success = self.evaluate_correction(alignment_before, alignment_after);
                if success {
                    self.successful_corrections += 1;
                }

                CorrectionResult::new(strategy.clone(), alignment_before, alignment_after, success)
            }

            CorrectionStrategy::WeightRebalance { adjustments } => {
                self.rebalance_weights(adjustments);

                // Weight rebalancing typically has moderate impact
                let total_adjustment: f32 = adjustments.iter().map(|(_, adj)| adj.abs()).sum();
                let improvement = (total_adjustment * 0.3).min(0.05);
                let alignment_after = (alignment_before + improvement).clamp(0.0, 1.0);

                self.corrections_applied += 1;
                let success = self.evaluate_correction(alignment_before, alignment_after);
                if success {
                    self.successful_corrections += 1;
                }

                CorrectionResult::new(strategy.clone(), alignment_before, alignment_after, success)
            }

            CorrectionStrategy::GoalReinforcement { emphasis_factor } => {
                self.reinforce_goal(*emphasis_factor);

                // Goal reinforcement has gradual effect
                let improvement = (emphasis_factor - 1.0) * 0.1;
                let alignment_after = (alignment_before + improvement).clamp(0.0, 1.0);

                self.corrections_applied += 1;
                let success = self.evaluate_correction(alignment_before, alignment_after);
                if success {
                    self.successful_corrections += 1;
                }

                CorrectionResult::new(strategy.clone(), alignment_before, alignment_after, success)
            }

            CorrectionStrategy::EmergencyIntervention { .. } => {
                // Emergency intervention doesn't automatically improve alignment
                // It requires human action
                self.corrections_applied += 1;

                CorrectionResult::new(strategy.clone(), alignment_before, alignment_before, false)
            }
        }
    }

    /// Adjust thresholds by the specified delta
    pub fn adjust_thresholds(&mut self, delta: f32) {
        self.threshold_adjustment += delta;
        // Clamp to reasonable bounds
        self.threshold_adjustment = self.threshold_adjustment.clamp(-0.2, 0.2);
    }

    /// Rebalance weights with the specified adjustments
    pub fn rebalance_weights(&mut self, adjustments: &[(usize, f32)]) {
        for (idx, adj) in adjustments {
            // Clamp adjustment to max allowed
            let clamped_adj = adj.clamp(
                -self.config.max_weight_adjustment,
                self.config.max_weight_adjustment,
            );

            // Update or insert adjustment for this index
            if let Some(existing) = self.weight_adjustments.iter_mut().find(|(i, _)| i == idx) {
                existing.1 = (existing.1 + clamped_adj).clamp(-0.5, 0.5);
            } else {
                self.weight_adjustments.push((*idx, clamped_adj));
            }
        }
    }

    /// Reinforce goal with the specified emphasis factor
    pub fn reinforce_goal(&mut self, emphasis: f32) {
        // Combine emphasis factors multiplicatively but clamp to reasonable range
        self.goal_emphasis = (self.goal_emphasis * emphasis).clamp(0.5, 2.0);
    }

    /// Evaluate whether a correction was successful
    pub fn evaluate_correction(&self, before: f32, after: f32) -> bool {
        let improvement = after - before;
        improvement >= self.config.min_improvement
    }

    /// Get current threshold adjustment
    pub fn current_threshold_adjustment(&self) -> f32 {
        self.threshold_adjustment
    }

    /// Get current weight adjustments
    pub fn current_weight_adjustments(&self) -> &[(usize, f32)] {
        &self.weight_adjustments
    }

    /// Get current goal emphasis
    pub fn current_goal_emphasis(&self) -> f32 {
        self.goal_emphasis
    }

    /// Get correction statistics
    pub fn correction_stats(&self) -> (u64, u64, f32) {
        let success_rate = if self.corrections_applied > 0 {
            self.successful_corrections as f32 / self.corrections_applied as f32
        } else {
            0.0
        };
        (
            self.corrections_applied,
            self.successful_corrections,
            success_rate,
        )
    }

    /// Reset corrector state
    pub fn reset(&mut self) {
        self.threshold_adjustment = 0.0;
        self.weight_adjustments.clear();
        self.goal_emphasis = 1.0;
        self.corrections_applied = 0;
        self.successful_corrections = 0;
    }

    /// Auto-select and apply correction for given state
    pub fn auto_correct(
        &mut self,
        state: &mut DriftState,
        _config: &DriftConfig,
    ) -> CorrectionResult {
        let strategy = self.select_strategy(state);
        self.apply_correction(state, &strategy)
    }
}
