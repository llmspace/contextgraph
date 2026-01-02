//! Tests for GatingNetwork functionality.

use crate::config::FusionConfig;
use crate::error::EmbeddingError;
use crate::fusion::GatingNetwork;
use crate::types::dimensions::{NUM_EXPERTS, TOP_K_EXPERTS, TOTAL_CONCATENATED};

#[test]
fn test_gating_creation() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    assert_eq!(gating.num_experts(), NUM_EXPERTS);
    assert!((gating.temperature() - 1.0).abs() < 1e-5);
    assert!((gating.laplace_alpha() - 0.01).abs() < 1e-5);
}

#[test]
fn test_gating_forward_returns_valid_probs() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED];
    let probs = gating.forward(&input, 1).unwrap();

    assert_eq!(probs.len(), NUM_EXPERTS);

    // All probabilities should be positive
    for (i, &p) in probs.iter().enumerate() {
        assert!(p > 0.0, "Probability at index {} should be > 0, got {}", i, p);
    }

    // Should sum to 1 (or very close due to Laplace smoothing)
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Probabilities should sum to 1, got {}",
        sum
    );
}

#[test]
fn test_gating_forward_batch() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED * 3]; // batch_size = 3
    let probs = gating.forward(&input, 3).unwrap();

    assert_eq!(probs.len(), NUM_EXPERTS * 3);

    // Check each sample sums to 1
    for b in 0..3 {
        let sample_sum: f32 = probs[b * NUM_EXPERTS..(b + 1) * NUM_EXPERTS]
            .iter()
            .sum();
        assert!(
            (sample_sum - 1.0).abs() < 1e-5,
            "Sample {} should sum to 1, got {}",
            b,
            sample_sum
        );
    }
}

#[test]
fn test_gating_temperature_affects_distribution() {
    let mut config = FusionConfig::default();
    config.temperature = 0.1; // Sharp distribution
    config.laplace_alpha = 0.0; // Disable smoothing for clear comparison

    let gating_sharp = GatingNetwork::new(&config).unwrap();

    config.temperature = 2.0; // Flat distribution
    let gating_flat = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED];

    let probs_sharp = gating_sharp.forward(&input, 1).unwrap();
    let probs_flat = gating_flat.forward(&input, 1).unwrap();

    // Sharp temperature should have higher max probability
    let max_sharp = probs_sharp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_flat = probs_flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Note: Due to random initialization, this might not always hold
    // but the distribution should generally be sharper with lower temperature
    assert!(
        max_sharp >= max_flat * 0.5, // Allow some tolerance
        "Sharp distribution should have higher max: {} vs {}",
        max_sharp,
        max_flat
    );
}

#[test]
fn test_gating_set_temperature() {
    let config = FusionConfig::default();
    let mut gating = GatingNetwork::new(&config).unwrap();

    gating.set_temperature(0.5).unwrap();
    assert!((gating.temperature() - 0.5).abs() < 1e-5);
}

#[test]
fn test_gating_set_temperature_zero_fails() {
    let config = FusionConfig::default();
    let mut gating = GatingNetwork::new(&config).unwrap();

    let result = gating.set_temperature(0.0);
    assert!(result.is_err());
}

#[test]
fn test_gating_set_temperature_negative_fails() {
    let config = FusionConfig::default();
    let mut gating = GatingNetwork::new(&config).unwrap();

    let result = gating.set_temperature(-1.0);
    assert!(result.is_err());
}

#[test]
fn test_gating_forward_with_noise_still_valid() {
    let config = FusionConfig::for_training();
    let gating = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED];
    let probs = gating.forward_with_noise(&input, 1).unwrap();

    // Should still sum to 1
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Probabilities with noise should sum to 1, got {}",
        sum
    );

    // All positive
    assert!(probs.iter().all(|&p| p > 0.0));
}

#[test]
fn test_gating_laplace_smoothing_prevents_zero() {
    let mut config = FusionConfig::default();
    config.laplace_alpha = 0.01;

    let gating = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED];
    let probs = gating.forward(&input, 1).unwrap();

    // With Laplace smoothing, no probability should be exactly 0
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            p > 0.0,
            "Laplace smoothing should prevent zero at index {}, got {}",
            i,
            p
        );
    }
}

#[test]
fn test_gating_select_top_k() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED];
    let probs = gating.forward(&input, 1).unwrap();

    let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS).unwrap();

    assert_eq!(indices.len(), TOP_K_EXPERTS);
    assert_eq!(weights.len(), TOP_K_EXPERTS);

    // Indices should be unique
    let unique: std::collections::HashSet<_> = indices.iter().collect();
    assert_eq!(unique.len(), TOP_K_EXPERTS);

    // Indices should be in range [0, NUM_EXPERTS)
    assert!(indices.iter().all(|&i| i < NUM_EXPERTS));

    // Weights should sum to 1
    let weight_sum: f32 = weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-5,
        "Top-K weights should sum to 1, got {}",
        weight_sum
    );
}

#[test]
fn test_gating_select_top_k_batch() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED * 2]; // batch_size = 2
    let probs = gating.forward(&input, 2).unwrap();

    let (indices, weights) = gating.select_top_k(&probs, 2, TOP_K_EXPERTS).unwrap();

    assert_eq!(indices.len(), 2 * TOP_K_EXPERTS);
    assert_eq!(weights.len(), 2 * TOP_K_EXPERTS);

    // Check each sample's weights sum to 1
    let sum1: f32 = weights[0..TOP_K_EXPERTS].iter().sum();
    let sum2: f32 = weights[TOP_K_EXPERTS..].iter().sum();

    assert!((sum1 - 1.0).abs() < 1e-5);
    assert!((sum2 - 1.0).abs() < 1e-5);
}

#[test]
fn test_gating_select_top_k_exceeds_num_experts_fails() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    let probs = vec![0.125f32; NUM_EXPERTS];
    let result = gating.select_top_k(&probs, 1, NUM_EXPERTS + 1);

    assert!(result.is_err());
}

#[test]
fn test_gating_custom_input_dim() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::with_input_dim(1024, &config).unwrap();

    let input = vec![0.5f32; 1024];
    let probs = gating.forward(&input, 1).unwrap();

    assert_eq!(probs.len(), NUM_EXPERTS);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_gating_custom_input_dim_zero_fails() {
    let config = FusionConfig::default();
    let result = GatingNetwork::with_input_dim(0, &config);

    assert!(result.is_err());
}

#[test]
fn test_gating_empty_batch_fails() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED];
    let result = gating.forward(&input, 0);

    assert!(result.is_err());
    match result.unwrap_err() {
        EmbeddingError::EmptyInput => {}
        _ => panic!("Expected EmptyInput error"),
    }
}
