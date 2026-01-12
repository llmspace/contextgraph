//! Scoring logic for goal candidates
//!
//! Provides methods for calculating scores based on position,
//! density, length, keywords, and other metrics.

use std::collections::HashSet;

use super::constants::{GOAL_KEYWORDS, PURPOSE_STARTERS};
use super::types::{BootstrapServiceConfig, GoalCandidate};

/// Scoring helper trait for BootstrapService
pub trait CandidateScoring {
    /// Get the service configuration
    fn config(&self) -> &BootstrapServiceConfig;

    /// Score a goal candidate based on section weights and content analysis
    ///
    /// # Arguments
    ///
    /// * `candidate` - The goal candidate to score
    ///
    /// # Returns
    ///
    /// A score between 0.0 and 1.0 (higher is better)
    fn score_candidate(&self, candidate: &GoalCandidate) -> f32 {
        let weights = &self.config().bootstrap_config.section_weights;

        // Position score: favor first and last sections (U-shaped curve)
        let position_score = calculate_position_score(candidate.position, weights.position_weight);

        // Density score: favor sentences with higher keyword density
        let density_score = candidate.density * weights.density_weight;

        // Length score: favor moderately-sized sentences (not too short, not too long)
        let length_score = calculate_length_score(&candidate.text);

        // Keyword boost: more keywords = higher score
        let keyword_score = (candidate.keyword_count as f32 / 5.0).min(1.0);

        // Purpose starter bonus
        let purpose_bonus = if has_purpose_starter(&candidate.text) {
            0.2
        } else {
            0.0
        };

        // IDF adjustment (if enabled)
        let idf_multiplier = if weights.apply_idf {
            calculate_idf_boost(candidate)
        } else {
            1.0
        };

        // Combine scores
        let raw_score = (position_score * 0.25)
            + (density_score * 0.25)
            + (length_score * 0.15)
            + (keyword_score * 0.25)
            + (purpose_bonus * 0.10);

        (raw_score * idf_multiplier).clamp(0.0, 1.0)
    }
}

/// Calculate position score with U-shaped curve
pub fn calculate_position_score(position: f32, weight: f32) -> f32 {
    // U-shaped curve: favor positions near 0 or 1
    let distance_from_middle = (position - 0.5).abs() * 2.0;
    distance_from_middle * weight / 2.0
}

/// Calculate length score (prefer 50-200 character sentences)
pub fn calculate_length_score(text: &str) -> f32 {
    let len = text.len();
    if len < 30 {
        0.3
    } else if len < 50 {
        0.5
    } else if len <= 200 {
        1.0
    } else if len <= 400 {
        0.7
    } else {
        0.4
    }
}

/// Check if text starts with a purpose statement
pub fn has_purpose_starter(text: &str) -> bool {
    let lower = text.to_lowercase();
    PURPOSE_STARTERS
        .iter()
        .any(|&starter| lower.starts_with(starter))
}

/// Calculate IDF-like boost for rare keywords
pub fn calculate_idf_boost(candidate: &GoalCandidate) -> f32 {
    // Simple heuristic: boost candidates with less common keyword combinations
    let unique_keywords: HashSet<&str> = GOAL_KEYWORDS
        .iter()
        .copied()
        .filter(|&kw| candidate.text.to_lowercase().contains(kw))
        .collect();

    if unique_keywords.len() <= 1 {
        1.0
    } else if unique_keywords.len() <= 3 {
        1.1
    } else {
        1.2
    }
}
