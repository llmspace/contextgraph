//! Consolidation service implementation.
//!
//! This module contains the main `ConsolidationService` which handles:
//! - Finding consolidation candidates based on similarity thresholds
//! - Computing cosine similarity between memory embeddings
//! - Merging multiple memories into one
//! - Respecting daily merge limits

use crate::autonomous::curation::{ConsolidationConfig, ConsolidationReport, MemoryId};

use super::types::{MemoryContent, MemoryPair, ServiceConsolidationCandidate};

/// Service for consolidating similar memories
#[derive(Clone, Debug)]
pub struct ConsolidationService {
    /// Configuration
    config: ConsolidationConfig,
    /// Daily merge counter
    daily_merges: u32,
}

impl Default for ConsolidationService {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolidationService {
    /// Create a new consolidation service with default config
    pub fn new() -> Self {
        Self {
            config: ConsolidationConfig::default(),
            daily_merges: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ConsolidationConfig) -> Self {
        Self {
            config,
            daily_merges: 0,
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }

    /// Get the daily merge count
    pub fn daily_merges(&self) -> u32 {
        self.daily_merges
    }

    /// Reset daily merge counter (called at start of new day)
    pub fn reset_daily_counter(&mut self) {
        self.daily_merges = 0;
    }

    /// Find consolidation candidates from memory pairs
    ///
    /// Evaluates each pair and returns candidates that meet the threshold criteria.
    pub fn find_consolidation_candidates(
        &self,
        memories: &[MemoryPair],
    ) -> Vec<ServiceConsolidationCandidate> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut candidates = Vec::new();

        for pair in memories {
            let similarity = self.compute_similarity(&pair.a, &pair.b);
            let alignment_diff = pair.alignment_diff();

            if self.should_consolidate(similarity, alignment_diff) {
                let combined_alignment =
                    self.compute_combined_alignment(&[pair.a.alignment, pair.b.alignment]);
                let target_id = MemoryId::new();

                candidates.push(ServiceConsolidationCandidate::new(
                    vec![pair.a.id.clone(), pair.b.id.clone()],
                    target_id,
                    similarity,
                    combined_alignment,
                ));
            }
        }

        candidates
    }

    /// Compute cosine similarity between two memory contents
    ///
    /// Returns a value in [0, 1] where 1 means identical.
    /// Fails fast if embeddings have different dimensions.
    pub fn compute_similarity(&self, a: &MemoryContent, b: &MemoryContent) -> f32 {
        if a.embedding.len() != b.embedding.len() {
            return 0.0; // Fail fast: incompatible dimensions
        }

        if a.embedding.is_empty() {
            return 0.0; // Fail fast: empty embeddings
        }

        // Compute dot product
        let dot: f32 = a
            .embedding
            .iter()
            .zip(b.embedding.iter())
            .map(|(x, y)| x * y)
            .sum();

        // Compute magnitudes
        let mag_a: f32 = a.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
            return 0.0; // Fail fast: zero magnitude
        }

        let similarity = dot / (mag_a * mag_b);

        // Clamp to [0, 1] (cosine can be negative but we only care about positive similarity)
        similarity.clamp(0.0, 1.0)
    }

    /// Determine if two memories should be consolidated
    ///
    /// Returns true if similarity exceeds threshold AND alignment difference is within tolerance.
    pub fn should_consolidate(&self, similarity: f32, alignment_diff: f32) -> bool {
        similarity >= self.config.similarity_threshold
            && alignment_diff <= self.config.theta_diff_threshold
    }

    /// Perform consolidation on a set of candidates
    ///
    /// Respects daily merge limits and returns a report of actions taken.
    pub fn consolidate(
        &mut self,
        candidates: &[ServiceConsolidationCandidate],
    ) -> ConsolidationReport {
        let mut report = ConsolidationReport {
            candidates_found: candidates.len(),
            merged: 0,
            skipped: 0,
            daily_limit_reached: false,
        };

        if !self.config.enabled {
            report.skipped = candidates.len();
            return report;
        }

        for _candidate in candidates {
            if self.daily_merges >= self.config.max_daily_merges {
                report.daily_limit_reached = true;
                report.skipped += 1;
                continue;
            }

            // Perform merge (in real implementation, this would update storage)
            self.daily_merges += 1;
            report.merged += 1;
        }

        report
    }

    /// Merge multiple memory contents into one
    ///
    /// Combines text and averages embeddings weighted by access count.
    /// The merged memory gets a new ID.
    pub fn merge_memories(&self, sources: &[MemoryContent]) -> MemoryContent {
        if sources.is_empty() {
            return MemoryContent::new(MemoryId::new(), Vec::new(), String::new(), 0.0);
        }

        if sources.len() == 1 {
            return sources[0].clone();
        }

        // Determine embedding dimension
        let dim = sources.iter().map(|s| s.dimension()).max().unwrap_or(0);

        if dim == 0 {
            // No valid embeddings, just merge text
            let combined_text = sources
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" | ");

            let alignments: Vec<f32> = sources.iter().map(|s| s.alignment).collect();
            let combined_alignment = self.compute_combined_alignment(&alignments);
            let total_access: u32 = sources.iter().map(|s| s.access_count).sum();

            return MemoryContent::new(
                MemoryId::new(),
                Vec::new(),
                combined_text,
                combined_alignment,
            )
            .with_access_count(total_access);
        }

        // Compute weighted average embedding
        let total_weight: f32 = sources
            .iter()
            .map(|s| s.access_count as f32 + 1.0) // +1 to avoid zero weight
            .sum();

        let mut merged_embedding = vec![0.0f32; dim];

        for source in sources {
            if source.dimension() != dim {
                continue; // Skip incompatible dimensions
            }

            let weight = (source.access_count as f32 + 1.0) / total_weight;
            for (i, val) in source.embedding.iter().enumerate() {
                merged_embedding[i] += val * weight;
            }
        }

        // Normalize the merged embedding
        let magnitude: f32 = merged_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > f32::EPSILON {
            for val in &mut merged_embedding {
                *val /= magnitude;
            }
        }

        // Combine text content
        let combined_text = sources
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ");

        // Compute combined alignment
        let alignments: Vec<f32> = sources.iter().map(|s| s.alignment).collect();
        let combined_alignment = self.compute_combined_alignment(&alignments);

        // Sum access counts
        let total_access: u32 = sources.iter().map(|s| s.access_count).sum();

        MemoryContent::new(
            MemoryId::new(),
            merged_embedding,
            combined_text,
            combined_alignment,
        )
        .with_access_count(total_access)
    }

    /// Compute combined alignment from multiple alignment scores
    ///
    /// Uses weighted average favoring higher alignments.
    pub fn compute_combined_alignment(&self, alignments: &[f32]) -> f32 {
        if alignments.is_empty() {
            return 0.0;
        }

        if alignments.len() == 1 {
            return alignments[0];
        }

        // Weight by alignment^2 to favor higher alignments
        let weights: Vec<f32> = alignments.iter().map(|a| a * a).collect();
        let total_weight: f32 = weights.iter().sum();

        if total_weight < f32::EPSILON {
            // All alignments near zero, use simple average
            return alignments.iter().sum::<f32>() / alignments.len() as f32;
        }

        let weighted_sum: f32 = alignments
            .iter()
            .zip(weights.iter())
            .map(|(a, w)| a * w)
            .sum();

        weighted_sum / total_weight
    }
}
