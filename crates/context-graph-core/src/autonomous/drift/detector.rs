//! Teleological drift detector implementation.
//!
//! Uses TeleologicalComparator for per-embedder array comparison.

use chrono::Utc;

use crate::teleological::{Embedder, MatrixSearchConfig, SearchStrategy, TeleologicalComparator};
use crate::types::SemanticFingerprint;

use super::error::DriftError;
use super::history::{compute_trend, DriftHistory, DriftHistoryEntry, TrendAnalysis};
use super::recommendations::{DriftRecommendation, RecommendationPriority};
use super::types::{
    DriftLevel, DriftResult, DriftThresholds, EmbedderDriftInfo, OverallDrift, PerEmbedderDrift,
    MAX_MOST_DRIFTED, NUM_EMBEDDERS,
};

// ============================================
// TELEOLOGICAL DRIFT DETECTOR
// ============================================

/// Teleological drift detector using per-embedder array comparison.
///
/// Uses TeleologicalComparator from TASK-LOGIC-004 for apples-to-apples
/// comparison across all 13 embedders.
#[derive(Debug)]
pub struct TeleologicalDriftDetector {
    /// The comparator for fingerprint comparison (reserved for future per-embedder comparisons)
    #[allow(dead_code)]
    comparator: TeleologicalComparator,
    /// History for trend analysis
    history: DriftHistory,
    /// Thresholds for drift classification
    thresholds: DriftThresholds,
}

impl TeleologicalDriftDetector {
    /// Create a new detector with default thresholds.
    pub fn new(comparator: TeleologicalComparator) -> Self {
        Self {
            comparator,
            history: DriftHistory::new(100),
            thresholds: DriftThresholds::default(),
        }
    }

    /// Create a detector with custom thresholds.
    ///
    /// # Errors
    ///
    /// Returns error if thresholds are invalid.
    pub fn with_thresholds(
        comparator: TeleologicalComparator,
        thresholds: DriftThresholds,
    ) -> Result<Self, DriftError> {
        thresholds.validate()?;
        Ok(Self {
            comparator,
            history: DriftHistory::new(100),
            thresholds,
        })
    }

    /// Check drift of memories against a goal (stateless).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `memories` is empty (FAIL FAST)
    /// - `goal` has invalid embeddings (NaN, Inf)
    /// - Comparison fails
    pub fn check_drift(
        &self,
        memories: &[SemanticFingerprint],
        goal: &SemanticFingerprint,
        strategy: SearchStrategy,
    ) -> Result<DriftResult, DriftError> {
        // FAIL FAST: Empty memories
        if memories.is_empty() {
            return Err(DriftError::EmptyMemories);
        }

        // FAIL FAST: Validate goal
        self.validate_fingerprint(goal)?;

        // Compare all memories against the goal and aggregate
        let per_embedder_sims = self.compute_per_embedder_similarities(memories, goal, strategy)?;

        // Build result
        self.build_result(per_embedder_sims, memories.len(), None)
    }

    /// Check drift and update history for trend analysis (stateful).
    ///
    /// # Errors
    ///
    /// Same as `check_drift`.
    pub fn check_drift_with_history(
        &mut self,
        memories: &[SemanticFingerprint],
        goal: &SemanticFingerprint,
        goal_id: &str,
        strategy: SearchStrategy,
    ) -> Result<DriftResult, DriftError> {
        // FAIL FAST: Empty memories
        if memories.is_empty() {
            return Err(DriftError::EmptyMemories);
        }

        // FAIL FAST: Validate goal
        self.validate_fingerprint(goal)?;

        // Compare all memories against the goal and aggregate
        let per_embedder_sims = self.compute_per_embedder_similarities(memories, goal, strategy)?;

        // Record history
        let overall_sim = per_embedder_sims.iter().sum::<f32>() / NUM_EMBEDDERS as f32;
        let entry = DriftHistoryEntry {
            timestamp: Utc::now(),
            overall_similarity: overall_sim,
            per_embedder: per_embedder_sims,
            memories_analyzed: memories.len(),
        };
        self.history.add(goal_id, entry);

        // Compute trend
        let trend = self.get_trend(goal_id);

        // Build result
        self.build_result(per_embedder_sims, memories.len(), trend)
    }

    /// Get drift trend for a goal from history.
    ///
    /// Returns None if fewer than 3 history samples.
    pub fn get_trend(&self, goal_id: &str) -> Option<TrendAnalysis> {
        let entries = self.history.get(goal_id)?;
        compute_trend(entries, &self.thresholds)
    }

    /// Validate that a fingerprint has valid embeddings (no NaN/Inf).
    fn validate_fingerprint(&self, fp: &SemanticFingerprint) -> Result<(), DriftError> {
        for embedder in Embedder::all() {
            if let Some(slice) = fp.get_embedding(embedder.index()) {
                match slice {
                    crate::types::EmbeddingSlice::Dense(values) => {
                        for (i, &v) in values.iter().enumerate() {
                            if v.is_nan() {
                                return Err(DriftError::InvalidGoal {
                                    reason: format!("NaN at index {} in {:?}", i, embedder),
                                });
                            }
                            if v.is_infinite() {
                                return Err(DriftError::InvalidGoal {
                                    reason: format!("Infinity at index {} in {:?}", i, embedder),
                                });
                            }
                        }
                    }
                    crate::types::EmbeddingSlice::Sparse(sv) => {
                        for &v in &sv.values {
                            if v.is_nan() || v.is_infinite() {
                                return Err(DriftError::InvalidGoal {
                                    reason: format!(
                                        "Invalid value in sparse vector for {:?}",
                                        embedder
                                    ),
                                });
                            }
                        }
                    }
                    crate::types::EmbeddingSlice::TokenLevel(tokens) => {
                        for (t_idx, token) in tokens.iter().enumerate() {
                            for (i, &v) in token.iter().enumerate() {
                                if v.is_nan() || v.is_infinite() {
                                    return Err(DriftError::InvalidGoal {
                                        reason: format!(
                                            "Invalid value at token {} index {} in {:?}",
                                            t_idx, i, embedder
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute per-embedder similarities by averaging across all memories.
    fn compute_per_embedder_similarities(
        &self,
        memories: &[SemanticFingerprint],
        goal: &SemanticFingerprint,
        strategy: SearchStrategy,
    ) -> Result<[f32; NUM_EMBEDDERS], DriftError> {
        // Accumulate per-embedder scores
        let mut sums = [0.0f32; NUM_EMBEDDERS];
        let mut counts = [0usize; NUM_EMBEDDERS];

        for memory in memories {
            // Use comparator with the specified strategy
            let config = MatrixSearchConfig {
                strategy,
                ..MatrixSearchConfig::default()
            };
            let comparator = TeleologicalComparator::with_config(config);

            let result = comparator.compare(memory, goal).map_err(|e| {
                DriftError::ComparisonValidationFailed {
                    reason: format!("{:?}", e),
                }
            })?;

            // Aggregate per-embedder scores
            for (idx, score) in result.per_embedder.iter().enumerate() {
                if let Some(s) = score {
                    sums[idx] += s;
                    counts[idx] += 1;
                }
            }
        }

        // Compute averages
        let mut averages = [0.0f32; NUM_EMBEDDERS];
        for idx in 0..NUM_EMBEDDERS {
            if counts[idx] > 0 {
                averages[idx] = sums[idx] / counts[idx] as f32;
            }
        }

        Ok(averages)
    }

    /// Build the drift result from per-embedder similarities.
    fn build_result(
        &self,
        per_embedder_sims: [f32; NUM_EMBEDDERS],
        analyzed_count: usize,
        trend: Option<TrendAnalysis>,
    ) -> Result<DriftResult, DriftError> {
        // Build per-embedder drift info
        let embedder_drift: [EmbedderDriftInfo; NUM_EMBEDDERS] = std::array::from_fn(|idx| {
            let embedder = Embedder::from_index(idx).expect("valid index");
            EmbedderDriftInfo::new(embedder, per_embedder_sims[idx], &self.thresholds)
        });

        // Compute overall similarity (average)
        let overall_similarity = per_embedder_sims.iter().sum::<f32>() / NUM_EMBEDDERS as f32;
        let overall_drift_level = DriftLevel::from_similarity(overall_similarity, &self.thresholds);

        // Find most drifted embedders (sorted worst first)
        let mut sorted: Vec<EmbedderDriftInfo> = embedder_drift.to_vec();
        sorted.sort_by(|a, b| a.drift_level.cmp(&b.drift_level));
        let most_drifted: Vec<EmbedderDriftInfo> = sorted
            .into_iter()
            .filter(|info| info.drift_level.has_drifted())
            .take(MAX_MOST_DRIFTED)
            .collect();

        // Generate recommendations for Medium+ drift
        let recommendations = self.generate_recommendations(&embedder_drift);

        Ok(DriftResult {
            overall_drift: OverallDrift {
                has_drifted: overall_drift_level.has_drifted(),
                drift_score: 1.0 - overall_similarity,
                drift_level: overall_drift_level,
                similarity: overall_similarity,
            },
            per_embedder_drift: PerEmbedderDrift { embedder_drift },
            most_drifted_embedders: most_drifted,
            recommendations,
            trend,
            analyzed_count,
            timestamp: Utc::now(),
        })
    }

    /// Generate recommendations based on drift analysis.
    fn generate_recommendations(
        &self,
        embedder_drift: &[EmbedderDriftInfo; NUM_EMBEDDERS],
    ) -> Vec<DriftRecommendation> {
        let mut recommendations = Vec::new();

        for info in embedder_drift {
            // Only generate recommendations for Medium or worse drift
            if !info.drift_level.needs_recommendation() {
                continue;
            }

            let (issue, suggestion) = get_embedder_recommendation(info);

            recommendations.push(DriftRecommendation {
                embedder: info.embedder,
                issue,
                suggestion,
                priority: RecommendationPriority::from(info.drift_level),
            });
        }

        // Sort by priority (Critical first)
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        recommendations
    }
}

/// Get embedder-specific recommendation text.
fn get_embedder_recommendation(info: &EmbedderDriftInfo) -> (String, String) {
    let severity = match info.drift_level {
        DriftLevel::Critical => "critical",
        DriftLevel::High => "high",
        DriftLevel::Medium => "moderate",
        _ => "minor",
    };

    let (issue, suggestion) = match info.embedder {
        Embedder::Semantic => (
            format!(
                "Semantic meaning drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Review core semantic content alignment with goals".to_string(),
        ),
        Embedder::TemporalRecent => (
            format!(
                "Recent temporal context drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Ensure recent memories are being captured appropriately".to_string(),
        ),
        Embedder::TemporalPeriodic => (
            format!(
                "Periodic pattern drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Check cyclical patterns are being maintained".to_string(),
        ),
        Embedder::TemporalPositional => (
            format!(
                "Positional temporal drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Review sequence ordering and positional context".to_string(),
        ),
        Embedder::Causal => (
            format!(
                "Causal reasoning drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Strengthen cause-effect relationship tracking".to_string(),
        ),
        Embedder::Sparse => (
            format!(
                "Lexical/keyword drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Review keyword relevance and lexical alignment".to_string(),
        ),
        Embedder::Code => (
            format!(
                "Code structure drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Ensure code-related memories align with technical goals".to_string(),
        ),
        Embedder::Emotional => (
            format!(
                "Emotional/connectivity drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Review connectivity and emotional alignment patterns".to_string(),
        ),
        Embedder::Hdc => (
            format!(
                "Hyperdimensional pattern drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Check holographic pattern consistency".to_string(),
        ),
        Embedder::Multimodal => (
            format!(
                "Multimodal drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Review cross-modal content alignment".to_string(),
        ),
        Embedder::Entity => (
            format!(
                "Entity recognition drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Ensure named entities are consistently identified".to_string(),
        ),
        Embedder::LateInteraction => (
            format!(
                "Token-level precision drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Review fine-grained token matching patterns".to_string(),
        ),
        Embedder::KeywordSplade => (
            format!(
                "Keyword expansion drift at {} level (sim: {:.2})",
                severity, info.similarity
            ),
            "Check learned keyword expansion coverage".to_string(),
        ),
    };

    (issue, suggestion)
}
