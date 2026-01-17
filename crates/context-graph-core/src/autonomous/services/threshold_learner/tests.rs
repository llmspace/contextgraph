//! Tests for the ThresholdLearner service.
//!
//! This module contains basic unit tests for the core functionality.

use crate::autonomous::{AlignmentBucket, RetrievalStats};

use super::learner::ThresholdLearner;
use super::types::NUM_EMBEDDERS;
use crate::autonomous::AdaptiveThresholdConfig;

#[test]
fn test_threshold_learner_new() {
    let learner = ThresholdLearner::new();

    assert!(learner.get_config().enabled);
    assert_eq!(learner.total_observations(), 0);
    assert!((learner.get_state().optimal - 0.75).abs() < f32::EPSILON);

    println!("[PASS] test_threshold_learner_new: Created with default config");
}

#[test]
fn test_threshold_learner_with_config() {
    let config = AdaptiveThresholdConfig {
        enabled: true,
        learning_rate: 0.02,
        optimal_bounds: (0.72, 0.88),
        warning_bounds: (0.48, 0.62),
    };

    let learner = ThresholdLearner::with_config(config);

    assert!((learner.get_config().learning_rate - 0.02).abs() < f32::EPSILON);
    assert!((learner.get_config().optimal_bounds.0 - 0.72).abs() < f32::EPSILON);

    println!("[PASS] test_threshold_learner_with_config: Custom config applied");
}

#[test]
fn test_update_ewma_basic() {
    let mut learner = ThresholdLearner::new();
    learner.set_ewma_alpha(0.2);

    // EWMA = 0.2 * 0.80 + 0.8 * 0.75 = 0.16 + 0.60 = 0.76
    let result = learner.update_ewma(0.75, 0.80);
    assert!((result - 0.76).abs() < 0.001);

    // EWMA = 0.2 * 0.60 + 0.8 * 0.75 = 0.12 + 0.60 = 0.72
    let result2 = learner.update_ewma(0.75, 0.60);
    assert!((result2 - 0.72).abs() < 0.001);

    println!("[PASS] test_update_ewma_basic: EWMA formula verified");
}

#[test]
fn test_update_ewma_nan_handling() {
    let mut learner = ThresholdLearner::new();

    // NaN observed should return current
    let result = learner.update_ewma(0.75, f32::NAN);
    assert!((result - 0.75).abs() < f32::EPSILON);

    // NaN current should return current (NaN)
    let result2 = learner.update_ewma(f32::NAN, 0.80);
    assert!(result2.is_nan());

    println!("[PASS] test_update_ewma_nan_handling: NaN handled gracefully");
}

#[test]
fn test_temperature_scale_basic() {
    let learner = ThresholdLearner::new();

    let logits = vec![2.0, 1.0, 0.1];
    let scaled = learner.temperature_scale(&logits, 1.0);

    // Should sum to 1.0
    let sum: f32 = scaled.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // Higher logit should have higher probability
    assert!(scaled[0] > scaled[1]);
    assert!(scaled[1] > scaled[2]);

    println!("[PASS] test_temperature_scale_basic: Softmax normalization works");
}

#[test]
fn test_temperature_scale_high_temp() {
    let learner = ThresholdLearner::new();

    let logits = vec![2.0, 1.0, 0.1];
    let scaled_high = learner.temperature_scale(&logits, 2.0);
    let scaled_low = learner.temperature_scale(&logits, 0.5);

    // High temp should produce more uniform distribution
    let variance_high: f32 = scaled_high
        .iter()
        .map(|p| (p - 1.0 / 3.0).powi(2))
        .sum::<f32>()
        / 3.0;
    let variance_low: f32 = scaled_low
        .iter()
        .map(|p| (p - 1.0 / 3.0).powi(2))
        .sum::<f32>()
        / 3.0;

    assert!(variance_high < variance_low);

    println!("[PASS] test_temperature_scale_high_temp: Temperature scaling verified");
}

#[test]
fn test_temperature_scale_empty() {
    let learner = ThresholdLearner::new();
    let result = learner.temperature_scale(&[], 1.0);
    assert!(result.is_empty());

    println!("[PASS] test_temperature_scale_empty: Empty input handled");
}

#[test]
fn test_temperature_scale_zero_temp() {
    let learner = ThresholdLearner::new();
    let logits = vec![1.0, 0.5, 0.0];

    // Zero temp should not panic, uses 0.01 instead
    let result = learner.temperature_scale(&logits, 0.0);
    assert!(!result.is_empty());
    assert!(result.iter().all(|&p| !p.is_nan()));

    println!("[PASS] test_temperature_scale_zero_temp: Zero temp handled");
}

#[test]
fn test_thompson_sample_range() {
    let mut learner = ThresholdLearner::new();

    // Sample multiple times to check range
    for _ in 0..100 {
        let sample = learner.thompson_sample(0);
        assert!(
            (0.5..=0.95).contains(&sample),
            "Sample {} out of expected range [0.5, 0.95]",
            sample
        );
    }

    println!("[PASS] test_thompson_sample_range: Samples within expected range");
}

#[test]
fn test_thompson_sample_invalid_idx() {
    let mut learner = ThresholdLearner::new();

    // Invalid index should return fallback
    let sample = learner.thompson_sample(100);
    assert!((sample - 0.75).abs() < f32::EPSILON);

    println!("[PASS] test_thompson_sample_invalid_idx: Invalid index returns fallback");
}

#[test]
fn test_thompson_sample_updates_with_feedback() {
    let mut learner = ThresholdLearner::new();

    // Get initial Thompson state
    let initial_alpha = learner.get_embedder_state(0).unwrap().thompson.alpha;
    let initial_beta = learner.get_embedder_state(0).unwrap().thompson.beta;

    // Learn from positive feedback
    let stats = RetrievalStats::new();
    learner.learn_from_feedback(&stats, true);

    // Alpha should increase (success)
    assert!(learner.get_embedder_state(0).unwrap().thompson.alpha > initial_alpha);
    assert!(
        (learner.get_embedder_state(0).unwrap().thompson.beta - initial_beta).abs() < f32::EPSILON
    );

    // Learn from negative feedback
    learner.learn_from_feedback(&stats, false);

    // Beta should increase (failure)
    assert!(learner.get_embedder_state(0).unwrap().thompson.beta > initial_beta);

    println!("[PASS] test_thompson_sample_updates_with_feedback: Thompson params update correctly");
}

#[test]
fn test_bayesian_update_basic() {
    let mut learner = ThresholdLearner::new();

    // Strong evidence should move posterior toward likelihood
    let posterior = learner.bayesian_update(0.5, 0.9);
    assert!(
        posterior > 0.5,
        "Posterior {} should be > 0.5 with strong likelihood",
        posterior
    );

    // Weak evidence with high prior - posterior should be influenced by both
    // With Bayes rule: P(H|E) = P(E|H)*P(H) / P(E)
    // When prior=0.8, likelihood=0.5, the posterior remains relatively high
    let posterior2 = learner.bayesian_update(0.8, 0.5);
    // Posterior should be between prior and likelihood, or close to 0.5 (neutral likelihood)
    assert!(
        (0.4..=0.85).contains(&posterior2),
        "Posterior {} should be in reasonable range with neutral likelihood",
        posterior2
    );

    println!("[PASS] test_bayesian_update_basic: Bayes rule applied correctly");
}

#[test]
fn test_bayesian_update_edge_cases() {
    let mut learner = ThresholdLearner::new();

    // NaN should return neutral
    let result = learner.bayesian_update(f32::NAN, 0.5);
    assert!((result - 0.5).abs() < f32::EPSILON);

    // Negative should return neutral
    let result2 = learner.bayesian_update(-0.5, 0.5);
    assert!((result2 - 0.5).abs() < f32::EPSILON);

    println!("[PASS] test_bayesian_update_edge_cases: Edge cases handled");
}

#[test]
fn test_get_threshold_valid_indices() {
    let learner = ThresholdLearner::new();

    // All valid indices should return a threshold
    for idx in 0..NUM_EMBEDDERS {
        let threshold = learner.get_threshold(idx);
        assert!((0.0..=1.0).contains(&threshold));
    }

    println!("[PASS] test_get_threshold_valid_indices: All embedder thresholds accessible");
}

#[test]
fn test_get_threshold_invalid_index() {
    let learner = ThresholdLearner::new();

    // Invalid index should return optimal as fallback
    let threshold = learner.get_threshold(100);
    assert!((threshold - learner.get_state().optimal).abs() < f32::EPSILON);

    println!("[PASS] test_get_threshold_invalid_index: Invalid index returns fallback");
}

#[test]
fn test_should_recalibrate_no_observations() {
    let learner = ThresholdLearner::new();

    // No observations - should not recalibrate
    assert!(!learner.should_recalibrate());

    println!("[PASS] test_should_recalibrate_no_observations: No recal with no data");
}

#[test]
fn test_learn_from_feedback_updates_state() {
    let mut learner = ThresholdLearner::new();

    let initial_obs = learner.total_observations();

    let mut stats = RetrievalStats::new();
    stats.record_retrieval(AlignmentBucket::Optimal, true);

    learner.learn_from_feedback(&stats, true);

    assert_eq!(learner.total_observations(), initial_obs + 1);

    println!("[PASS] test_learn_from_feedback_updates_state: State updated on feedback");
}

#[test]
fn test_learn_from_feedback_disabled() {
    let config = AdaptiveThresholdConfig {
        enabled: false,
        ..Default::default()
    };
    let mut learner = ThresholdLearner::with_config(config);

    let stats = RetrievalStats::new();
    learner.learn_from_feedback(&stats, true);

    // Should not update when disabled
    assert_eq!(learner.total_observations(), 0);

    println!("[PASS] test_learn_from_feedback_disabled: Disabled config prevents learning");
}

#[test]
fn test_get_embedder_state() {
    let learner = ThresholdLearner::new();

    // Valid index
    let state = learner.get_embedder_state(0);
    assert!(state.is_some());
    assert_eq!(state.unwrap().observation_count, 0);

    // Invalid index
    let state_invalid = learner.get_embedder_state(100);
    assert!(state_invalid.is_none());

    println!("[PASS] test_get_embedder_state: Embedder state accessible");
}

#[test]
fn test_set_ewma_alpha_clamping() {
    let mut learner = ThresholdLearner::new();

    learner.set_ewma_alpha(0.01);
    // Cannot directly check private field, so verify indirectly via EWMA calculation
    let result = learner.update_ewma(0.75, 1.0);
    // With alpha=0.05 (clamped): 0.05 * 1.0 + 0.95 * 0.75 = 0.05 + 0.7125 = 0.7625
    assert!((result - 0.7625).abs() < 0.01);

    learner.set_ewma_alpha(0.9);
    let result2 = learner.update_ewma(0.75, 1.0);
    // With alpha=0.5 (clamped): 0.5 * 1.0 + 0.5 * 0.75 = 0.5 + 0.375 = 0.875
    assert!((result2 - 0.875).abs() < 0.01);

    learner.set_ewma_alpha(0.3);
    let result3 = learner.update_ewma(0.75, 1.0);
    // With alpha=0.3: 0.3 * 1.0 + 0.7 * 0.75 = 0.3 + 0.525 = 0.825
    assert!((result3 - 0.825).abs() < 0.01);

    println!("[PASS] test_set_ewma_alpha_clamping: Alpha clamped to valid range");
}

#[test]
fn test_reset() {
    let mut learner = ThresholdLearner::new();

    // Add some data
    let stats = RetrievalStats::new();
    for _ in 0..50 {
        learner.learn_from_feedback(&stats, true);
    }

    assert!(learner.total_observations() > 0);

    // Reset
    learner.reset();

    assert_eq!(learner.total_observations(), 0);
    assert!(learner.get_bayesian_history().is_empty());
    assert!((learner.best_performance() - 0.0).abs() < f32::EPSILON);

    println!("[PASS] test_reset: State reset to initial values");
}

#[test]
fn test_default_impl() {
    let learner1 = ThresholdLearner::default();
    let learner2 = ThresholdLearner::new();

    assert_eq!(learner1.total_observations(), learner2.total_observations());
    assert!((learner1.get_state().optimal - learner2.get_state().optimal).abs() < f32::EPSILON);

    println!("[PASS] test_default_impl: Default and new produce same result");
}
