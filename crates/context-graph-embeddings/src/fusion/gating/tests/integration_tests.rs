//! Integration tests for the complete gating pipeline.

use crate::config::FusionConfig;
use crate::fusion::GatingNetwork;
use crate::types::dimensions::{NUM_EXPERTS, TOP_K_EXPERTS, TOTAL_CONCATENATED};

#[test]
fn test_full_pipeline_integration() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    // Simulate real concatenated embedding
    let input = vec![0.1f32; TOTAL_CONCATENATED];

    // Forward pass
    let probs = gating.forward(&input, 1).unwrap();

    // Select top-K
    let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS).unwrap();

    // Verify
    assert_eq!(probs.len(), NUM_EXPERTS);
    assert_eq!(indices.len(), TOP_K_EXPERTS);
    assert_eq!(weights.len(), TOP_K_EXPERTS);

    let prob_sum: f32 = probs.iter().sum();
    let weight_sum: f32 = weights.iter().sum();

    assert!((prob_sum - 1.0).abs() < 1e-5);
    assert!((weight_sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_training_vs_inference_mode() {
    let inference_config = FusionConfig::for_inference();
    let training_config = FusionConfig::for_training();

    let gating_inference = GatingNetwork::new(&inference_config).unwrap();
    let gating_training = GatingNetwork::new(&training_config).unwrap();

    // Verify configurations
    assert!((gating_inference.noise_std() - 0.0).abs() < 1e-5);
    assert!(gating_training.noise_std() > 0.0);
}

#[test]
fn test_batch_consistency() {
    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();

    // Same input processed as batch vs individual
    let input = vec![0.5f32; TOTAL_CONCATENATED];

    let _single_probs = gating.forward(&input, 1).unwrap();

    let batch_input = vec![0.5f32; TOTAL_CONCATENATED * 2];
    let batch_probs = gating.forward(&batch_input, 2).unwrap();

    // Both samples in batch should be identical (same input)
    for i in 0..NUM_EXPERTS {
        assert!(
            (batch_probs[i] - batch_probs[NUM_EXPERTS + i]).abs() < 1e-5,
            "Batch samples should be identical at index {}",
            i
        );
    }
}
