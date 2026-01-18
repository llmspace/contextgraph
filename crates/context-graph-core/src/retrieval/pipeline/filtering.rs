//! Stage 4 teleological filtering implementation.
//!
//! This module provides the core filtering logic for the teleological
//! retrieval pipeline, including:
//! - Purpose alignment computation
//! - Goal hierarchy alignment
//! - Filtering by thresholds

use tracing::{debug, instrument, warn};

use crate::alignment::{AlignmentConfig, GoalAlignmentCalculator};
use crate::error::CoreResult;
use crate::traits::TeleologicalMemoryStore;
use crate::types::fingerprint::TeleologicalFingerprint;

use super::super::teleological_query::TeleologicalQuery;
use super::super::teleological_result::ScoredMemory;
use super::super::{AggregatedMatch, MultiEmbeddingQueryExecutor};
use super::DefaultTeleologicalPipeline;

impl<E, A, S> DefaultTeleologicalPipeline<E, A, S>
where
    E: MultiEmbeddingQueryExecutor,
    A: GoalAlignmentCalculator,
    S: TeleologicalMemoryStore,
{
    /// Apply Stage 4 teleological filtering to candidates.
    ///
    /// This is the core teleological filtering that:
    /// 1. Computes purpose alignment for each candidate
    /// 2. Computes goal hierarchy alignment
    /// 3. Filters by minimum alignment threshold
    #[instrument(skip(self, candidates, query), fields(candidate_count = candidates.len()))]
    pub(crate) async fn apply_stage4_filtering(
        &self,
        candidates: &[(&TeleologicalFingerprint, &AggregatedMatch)],
        query: &TeleologicalQuery,
    ) -> CoreResult<(Vec<ScoredMemory>, usize, f32)> {
        let config = query.effective_config();
        let min_alignment = config.min_alignment_threshold;

        let mut results = Vec::with_capacity(candidates.len());
        let mut filtered_count = 0;
        let mut filtered_alignments = Vec::new();

        // Build alignment config
        let alignment_config = AlignmentConfig::with_hierarchy((*self.goal_hierarchy).clone())
            .with_min_alignment(min_alignment);

        for (fingerprint, aggregated) in candidates {
            // Compute goal alignment
            let alignment_result = self
                .alignment_calculator
                .compute_alignment(fingerprint, &alignment_config)
                .await;

            let (goal_alignment, is_misaligned) = match alignment_result {
                Ok(result) => (
                    result.score.composite_score,
                    result.flags.needs_intervention(),
                ),
                Err(e) => {
                    warn!(
                        memory_id = %fingerprint.id,
                        error = %e,
                        "Alignment computation failed, using default"
                    );
                    // FAIL FAST alternative: return error
                    // For graceful degradation, we use default score
                    (0.0, true)
                }
            };

            // Get purpose alignment from fingerprint's purpose vector
            let purpose_alignment = fingerprint.alignment_score;

            // Check if filtered by alignment threshold
            if goal_alignment < min_alignment {
                filtered_count += 1;
                filtered_alignments.push(goal_alignment);
                debug!(
                    memory_id = %fingerprint.id,
                    goal_alignment = goal_alignment,
                    threshold = min_alignment,
                    "Filtered by alignment threshold"
                );
                continue;
            }

            // Create scored memory
            let scored = ScoredMemory::new(
                fingerprint.id,
                aggregated.aggregate_score,
                self.compute_avg_similarity(aggregated),
                purpose_alignment,
                goal_alignment,
                aggregated.space_count,
            )
            .with_misalignment(is_misaligned);

            results.push(scored);
        }

        // Compute average alignment of filtered candidates
        let filtered_avg = if filtered_alignments.is_empty() {
            0.0
        } else {
            filtered_alignments.iter().sum::<f32>() / filtered_alignments.len() as f32
        };

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to teleological_limit
        let limit = config.teleological_limit;
        if results.len() > limit {
            results.truncate(limit);
        }

        debug!(
            input_count = candidates.len(),
            output_count = results.len(),
            filtered = filtered_count,
            avg_filtered_alignment = filtered_avg,
            "Stage 4 filtering complete"
        );

        Ok((results, filtered_count, filtered_avg))
    }

    /// Compute average content similarity from space contributions.
    pub(crate) fn compute_avg_similarity(&self, aggregated: &AggregatedMatch) -> f32 {
        if aggregated.space_contributions.is_empty() {
            return aggregated.aggregate_score;
        }

        let sum: f32 = aggregated
            .space_contributions
            .iter()
            .map(|c| c.similarity)
            .sum();
        sum / aggregated.space_contributions.len() as f32
    }
}
