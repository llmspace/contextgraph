//! Adaptive threshold types for alignment optimization
//!
//! This module defines types for adaptive threshold learning that optimizes
//! alignment thresholds based on retrieval success rates.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Adaptive threshold configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveThresholdConfig {
    /// Enable adaptive threshold learning
    pub enabled: bool,

    /// EWMA learning rate
    pub learning_rate: f32, // default: 0.01

    /// Bounds for optimal threshold
    pub optimal_bounds: (f32, f32), // default: (0.70, 0.85)

    /// Bounds for warning threshold
    pub warning_bounds: (f32, f32), // default: (0.50, 0.65)
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.01,
            optimal_bounds: (0.70, 0.85),
            warning_bounds: (0.50, 0.65),
        }
    }
}

/// Alignment bucket for categorizing retrieval outcomes
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AlignmentBucket {
    Optimal,    // >= optimal threshold
    Acceptable, // >= acceptable, < optimal
    Warning,    // >= warning, < acceptable
    Critical,   // < warning
}

/// Retrieval statistics for threshold learning
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RetrievalStats {
    /// Success rate per alignment bucket
    pub bucket_success_rates: HashMap<AlignmentBucket, f32>,
    /// Total retrievals
    pub total_retrievals: u64,
    /// Successful retrievals (user engaged with result)
    pub successful_retrievals: u64,
}

impl RetrievalStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn success_rate(&self) -> f32 {
        if self.total_retrievals == 0 {
            return 0.0;
        }
        self.successful_retrievals as f32 / self.total_retrievals as f32
    }

    pub fn record_retrieval(&mut self, bucket: AlignmentBucket, was_successful: bool) {
        self.total_retrievals += 1;
        if was_successful {
            self.successful_retrievals += 1;
        }

        let rate = self.bucket_success_rates.entry(bucket).or_insert(0.0);
        // EWMA update with alpha = 0.1
        let alpha = 0.1;
        let outcome = if was_successful { 1.0 } else { 0.0 };
        *rate = alpha * outcome + (1.0 - alpha) * *rate;
    }
}

/// Current adaptive threshold state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveThresholdState {
    /// Per-embedder thresholds (13 embedders)
    pub per_embedder: [f32; 13],

    /// Overall thresholds
    pub optimal: f32,
    pub acceptable: f32,
    pub warning: f32,
    pub critical: f32,

    /// Statistics used for learning
    pub retrieval_stats: RetrievalStats,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl Default for AdaptiveThresholdState {
    fn default() -> Self {
        Self {
            per_embedder: [0.75; 13], // Default optimal threshold for all
            optimal: 0.75,
            acceptable: 0.70,
            warning: 0.55,
            critical: 0.40,
            retrieval_stats: RetrievalStats::default(),
            updated_at: Utc::now(),
        }
    }
}

impl AdaptiveThresholdState {
    /// Classify an alignment score into a bucket
    pub fn classify(&self, alignment: f32) -> AlignmentBucket {
        if alignment >= self.optimal {
            AlignmentBucket::Optimal
        } else if alignment >= self.acceptable {
            AlignmentBucket::Acceptable
        } else if alignment >= self.warning {
            AlignmentBucket::Warning
        } else {
            AlignmentBucket::Critical
        }
    }

    /// Check if alignment meets minimum acceptable threshold
    pub fn is_acceptable(&self, alignment: f32) -> bool {
        alignment >= self.warning
    }

    /// Get threshold for specific embedder
    pub fn get_embedder_threshold(&self, embedder_idx: usize) -> f32 {
        if embedder_idx < 13 {
            self.per_embedder[embedder_idx]
        } else {
            self.optimal // Fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold_config_default() {
        let config = AdaptiveThresholdConfig::default();
        assert!(config.enabled);
        assert!((config.learning_rate - 0.01).abs() < f32::EPSILON);
        assert!((config.optimal_bounds.0 - 0.70).abs() < f32::EPSILON);
        assert!((config.optimal_bounds.1 - 0.85).abs() < f32::EPSILON);
        assert!((config.warning_bounds.0 - 0.50).abs() < f32::EPSILON);
        assert!((config.warning_bounds.1 - 0.65).abs() < f32::EPSILON);
    }

    #[test]
    fn test_alignment_bucket_equality() {
        assert_eq!(AlignmentBucket::Optimal, AlignmentBucket::Optimal);
        assert_ne!(AlignmentBucket::Optimal, AlignmentBucket::Warning);
    }

    #[test]
    fn test_alignment_bucket_hash() {
        let mut map: HashMap<AlignmentBucket, i32> = HashMap::new();
        map.insert(AlignmentBucket::Optimal, 1);
        map.insert(AlignmentBucket::Warning, 2);
        assert_eq!(map.get(&AlignmentBucket::Optimal), Some(&1));
        assert_eq!(map.get(&AlignmentBucket::Warning), Some(&2));
    }

    #[test]
    fn test_retrieval_stats_new() {
        let stats = RetrievalStats::new();
        assert_eq!(stats.total_retrievals, 0);
        assert_eq!(stats.successful_retrievals, 0);
        assert!(stats.bucket_success_rates.is_empty());
    }

    #[test]
    fn test_retrieval_stats_success_rate_empty() {
        let stats = RetrievalStats::new();
        assert!((stats.success_rate() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_retrieval_stats_success_rate() {
        let mut stats = RetrievalStats::new();
        stats.total_retrievals = 10;
        stats.successful_retrievals = 7;
        assert!((stats.success_rate() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_retrieval_stats_record_retrieval() {
        let mut stats = RetrievalStats::new();

        // Record a successful retrieval
        stats.record_retrieval(AlignmentBucket::Optimal, true);
        assert_eq!(stats.total_retrievals, 1);
        assert_eq!(stats.successful_retrievals, 1);

        // Check EWMA update: 0.1 * 1.0 + 0.9 * 0.0 = 0.1
        let rate = stats.bucket_success_rates.get(&AlignmentBucket::Optimal);
        assert!(rate.is_some());
        assert!((rate.unwrap() - 0.1).abs() < f32::EPSILON);

        // Record a failed retrieval
        stats.record_retrieval(AlignmentBucket::Optimal, false);
        assert_eq!(stats.total_retrievals, 2);
        assert_eq!(stats.successful_retrievals, 1);

        // Check EWMA update: 0.1 * 0.0 + 0.9 * 0.1 = 0.09
        let rate = stats.bucket_success_rates.get(&AlignmentBucket::Optimal);
        assert!((rate.unwrap() - 0.09).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adaptive_threshold_state_default() {
        let state = AdaptiveThresholdState::default();
        assert_eq!(state.per_embedder.len(), 13);
        for threshold in state.per_embedder.iter() {
            assert!((*threshold - 0.75).abs() < f32::EPSILON);
        }
        assert!((state.optimal - 0.75).abs() < f32::EPSILON);
        assert!((state.acceptable - 0.70).abs() < f32::EPSILON);
        assert!((state.warning - 0.55).abs() < f32::EPSILON);
        assert!((state.critical - 0.40).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adaptive_threshold_state_classify() {
        let state = AdaptiveThresholdState::default();

        // Optimal: >= 0.75
        assert_eq!(state.classify(0.80), AlignmentBucket::Optimal);
        assert_eq!(state.classify(0.75), AlignmentBucket::Optimal);

        // Acceptable: >= 0.70, < 0.75
        assert_eq!(state.classify(0.72), AlignmentBucket::Acceptable);
        assert_eq!(state.classify(0.70), AlignmentBucket::Acceptable);

        // Warning: >= 0.55, < 0.70
        assert_eq!(state.classify(0.60), AlignmentBucket::Warning);
        assert_eq!(state.classify(0.55), AlignmentBucket::Warning);

        // Critical: < 0.55
        assert_eq!(state.classify(0.50), AlignmentBucket::Critical);
        assert_eq!(state.classify(0.30), AlignmentBucket::Critical);
    }

    #[test]
    fn test_adaptive_threshold_state_is_acceptable() {
        let state = AdaptiveThresholdState::default();

        // Acceptable threshold is at warning level (0.55)
        assert!(state.is_acceptable(0.80));
        assert!(state.is_acceptable(0.55));
        assert!(!state.is_acceptable(0.54));
        assert!(!state.is_acceptable(0.30));
    }

    #[test]
    fn test_adaptive_threshold_state_get_embedder_threshold() {
        let mut state = AdaptiveThresholdState::default();
        state.per_embedder[0] = 0.80;
        state.per_embedder[12] = 0.65;

        assert!((state.get_embedder_threshold(0) - 0.80).abs() < f32::EPSILON);
        assert!((state.get_embedder_threshold(12) - 0.65).abs() < f32::EPSILON);
        assert!((state.get_embedder_threshold(5) - 0.75).abs() < f32::EPSILON);

        // Out of bounds returns optimal
        assert!((state.get_embedder_threshold(13) - state.optimal).abs() < f32::EPSILON);
        assert!((state.get_embedder_threshold(100) - state.optimal).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = AdaptiveThresholdConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AdaptiveThresholdConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.enabled);
        assert!((deserialized.learning_rate - 0.01).abs() < f32::EPSILON);

        let state = AdaptiveThresholdState::default();
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: AdaptiveThresholdState = serde_json::from_str(&json).unwrap();
        assert!((deserialized.optimal - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_alignment_bucket_serialization() {
        let bucket = AlignmentBucket::Optimal;
        let json = serde_json::to_string(&bucket).unwrap();
        assert_eq!(json, "\"Optimal\"");

        let deserialized: AlignmentBucket = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, AlignmentBucket::Optimal);
    }
}
