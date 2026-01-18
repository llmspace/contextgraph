//! Default implementation of GoalAlignmentCalculator.

mod alignment;
mod misalignment;
mod patterns;

use std::time::Instant;

use async_trait::async_trait;
use tracing::{debug, error};

use super::result::AlignmentResult;
use super::trait_def::GoalAlignmentCalculator;
use super::weights::TeleologicalWeights;
use crate::alignment::config::AlignmentConfig;
use crate::alignment::error::AlignmentError;
use crate::alignment::misalignment::MisalignmentFlags;
use crate::alignment::pattern::EmbedderBreakdown;
use crate::alignment::score::GoalAlignmentScore;
use crate::purpose::GoalLevel;
use crate::types::fingerprint::TeleologicalFingerprint;

/// Default implementation of GoalAlignmentCalculator.
///
/// Uses cosine similarity for embedding comparison and
/// supports all features from the alignment module.
///
/// # Multi-Space Alignment
///
/// This calculator computes alignment across ALL 13 embedding spaces
/// as required by constitution.yaml. The alignment formula is:
///
/// ```text
/// A_multi = SUM_i(w_i * A(E_i, V)) where SUM(w_i) = 1
/// ```
#[derive(Debug, Clone, Default)]
pub struct DefaultAlignmentCalculator {
    /// Teleological weights for multi-space alignment.
    teleological_weights: TeleologicalWeights,
}

impl DefaultAlignmentCalculator {
    /// Create a new calculator with default (uniform) teleological weights.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a calculator with custom teleological weights.
    ///
    /// # Arguments
    /// * `weights` - Custom weights for the 13 embedding spaces (must sum to 1.0)
    pub fn with_weights(weights: TeleologicalWeights) -> Self {
        weights.validate().expect("Invalid teleological weights");
        Self {
            teleological_weights: weights,
        }
    }

    /// Get the current teleological weights.
    pub fn teleological_weights(&self) -> &TeleologicalWeights {
        &self.teleological_weights
    }

    /// Get propagation weight for a goal level.
    ///
    /// From constitution.yaml (after TASK-P0-001):
    /// - Strategic: 1.0 (top-level, emergent)
    /// - Tactical: 0.6
    /// - Immediate: 0.3
    #[inline]
    pub(crate) fn get_propagation_weight(level: GoalLevel) -> f32 {
        // Strategic is top-level in 3-level hierarchy (per PRD v6)
        match level {
            GoalLevel::Strategic => 1.0,
            GoalLevel::Tactical => 0.6,
            GoalLevel::Immediate => 0.3,
        }
    }

    /// Compute embedder breakdown from purpose vector.
    fn compute_embedder_breakdown(
        &self,
        fingerprint: &TeleologicalFingerprint,
    ) -> EmbedderBreakdown {
        EmbedderBreakdown::from_purpose_vector(&fingerprint.purpose_vector)
    }
}

#[async_trait]
impl GoalAlignmentCalculator for DefaultAlignmentCalculator {
    async fn compute_alignment(
        &self,
        fingerprint: &TeleologicalFingerprint,
        config: &AlignmentConfig,
    ) -> Result<AlignmentResult, AlignmentError> {
        let start = Instant::now();

        // Validate config if enabled
        if config.validate_hierarchy {
            config.validate().map_err(AlignmentError::InvalidConfig)?;
        }

        // Topics emerge autonomously from clustering (per PRD v6)
        if config.hierarchy.is_empty() {
            return Err(AlignmentError::NoTopLevelGoals);
        }

        if !config.hierarchy.has_top_level_goals() {
            return Err(AlignmentError::NoTopLevelGoals);
        }

        // Compute alignment for each goal
        let mut goal_scores = Vec::with_capacity(config.hierarchy.len());

        for goal in config.hierarchy.iter() {
            // Check timeout
            let elapsed_ms = start.elapsed().as_millis() as u64;
            if elapsed_ms > config.timeout_ms {
                error!(
                    elapsed_ms = elapsed_ms,
                    limit_ms = config.timeout_ms,
                    goals_processed = goal_scores.len(),
                    "Alignment computation timeout"
                );
                return Err(AlignmentError::Timeout {
                    elapsed_ms,
                    limit_ms: config.timeout_ms,
                });
            }

            let score =
                self.compute_goal_alignment(&fingerprint.semantic, goal, &config.level_weights);

            // Apply minimum alignment threshold
            if score.alignment >= config.min_alignment {
                goal_scores.push(score);
            }
        }

        // Compute composite score
        let score = GoalAlignmentScore::compute(goal_scores, config.level_weights);

        // Detect misalignment flags
        let mut flags = self.detect_misalignment_flags(
            &score,
            &config.misalignment_thresholds,
            &config.hierarchy,
        );

        // Compute embedder breakdown if enabled
        let embedder_breakdown = if config.include_embedder_breakdown {
            let breakdown = self.compute_embedder_breakdown(fingerprint);
            self.check_inconsistent_alignment(
                &mut flags,
                &breakdown,
                &config.misalignment_thresholds,
            );
            Some(breakdown)
        } else {
            None
        };

        // Detect patterns if enabled
        let patterns = if config.detect_patterns {
            self.detect_patterns(&score, &flags, config)
        } else {
            Vec::new()
        };

        let computation_time_us = start.elapsed().as_micros() as u64;

        debug!(
            composite_score = score.composite_score,
            goal_count = score.goal_count(),
            misaligned_count = score.misaligned_count,
            pattern_count = patterns.len(),
            time_us = computation_time_us,
            "Alignment computation complete"
        );

        Ok(AlignmentResult {
            score,
            flags,
            patterns,
            embedder_breakdown,
            computation_time_us,
        })
    }

    async fn compute_alignment_batch(
        &self,
        fingerprints: &[&TeleologicalFingerprint],
        config: &AlignmentConfig,
    ) -> Vec<Result<AlignmentResult, AlignmentError>> {
        let mut results = Vec::with_capacity(fingerprints.len());

        for fingerprint in fingerprints {
            results.push(self.compute_alignment(fingerprint, config).await);
        }

        results
    }

    fn detect_patterns(
        &self,
        score: &GoalAlignmentScore,
        flags: &MisalignmentFlags,
        config: &AlignmentConfig,
    ) -> Vec<crate::alignment::pattern::AlignmentPattern> {
        patterns::detect_patterns(self, score, flags, config)
    }
}
