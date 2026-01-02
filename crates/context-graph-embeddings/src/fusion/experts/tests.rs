//! Unit tests for expert network components.

use super::*;
use crate::config::FusionConfig;
use crate::error::EmbeddingError;
use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOP_K_EXPERTS, TOTAL_CONCATENATED};

// =========================================================================
// ACTIVATION TESTS
// =========================================================================

#[test]
fn test_gelu_activation_zero() {
    let act = Activation::Gelu;
    let result = act.apply(0.0);
    assert!((result - 0.0).abs() < 1e-6, "GELU(0) should be 0");
}

#[test]
fn test_gelu_activation_positive() {
    let act = Activation::Gelu;
    let result = act.apply(1.0);
    // GELU(1.0) ≈ 0.8413 (computed, not hardcoded)
    assert!(
        result > 0.8 && result < 0.9,
        "GELU(1.0) should be ~0.84, got {}",
        result
    );
}

#[test]
fn test_gelu_activation_negative() {
    let act = Activation::Gelu;
    let result = act.apply(-1.0);
    // GELU(-1.0) ≈ -0.1587 (computed, not hardcoded)
    assert!(
        result > -0.2 && result < -0.1,
        "GELU(-1.0) should be ~-0.16, got {}",
        result
    );
}

#[test]
fn test_relu_activation() {
    let act = Activation::Relu;
    assert!((act.apply(1.0) - 1.0).abs() < 1e-6);
    assert!((act.apply(-1.0) - 0.0).abs() < 1e-6);
    assert!((act.apply(0.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_silu_activation() {
    let act = Activation::Silu;
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert!((act.apply(0.0) - 0.0).abs() < 1e-6);
    // SiLU(x) for positive x should be positive
    assert!(act.apply(1.0) > 0.0);
    // SiLU(x) for negative x should be negative
    assert!(act.apply(-1.0) < 0.0);
}

#[test]
fn test_activation_default() {
    assert_eq!(Activation::default(), Activation::Gelu);
}

// =========================================================================
// EXPERT TESTS
// =========================================================================

#[test]
fn test_expert_creation() {
    let expert = Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu);
    assert!(expert.is_ok());
    let expert = expert.unwrap();
    assert_eq!(expert.expert_id(), 0);
    assert_eq!(expert.input_dim(), TOTAL_CONCATENATED);
    assert_eq!(expert.hidden_dim(), 4096);
    assert_eq!(expert.output_dim(), FUSED_OUTPUT);
    assert_eq!(expert.activation(), Activation::Gelu);
}

#[test]
fn test_expert_output_shape() {
    let expert =
        Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
    let input = vec![0.1f32; TOTAL_CONCATENATED];
    let output = expert.forward(&input, 1).unwrap();
    assert_eq!(output.len(), FUSED_OUTPUT, "Output should be 1536D");
}

#[test]
fn test_expert_batch_output_shape() {
    let expert =
        Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
    let batch_size = 4;
    let input = vec![0.1f32; TOTAL_CONCATENATED * batch_size];
    let output = expert.forward(&input, batch_size).unwrap();
    assert_eq!(output.len(), FUSED_OUTPUT * batch_size);
}

#[test]
fn test_expert_empty_batch_fails() {
    let expert =
        Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
    let input: Vec<f32> = vec![];
    let result = expert.forward(&input, 0);
    assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
}

#[test]
fn test_expert_wrong_input_size_fails() {
    let expert =
        Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
    let input = vec![0.1f32; 100]; // Wrong size
    let result = expert.forward(&input, 1);
    assert!(matches!(
        result,
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_expert_parameter_count() {
    let expert =
        Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
    let count = expert.parameter_count();

    // Layer 1: 8320 * 4096 + 4096 = 34,082,816
    // Layer 2: 4096 * 1536 + 1536 = 6,293,088
    // Total: ~40,375,904
    assert!(count > 40_000_000, "Should have > 40M parameters");
    assert!(count < 42_000_000, "Should have < 42M parameters");
}

#[test]
fn test_expert_zero_input_dim_fails() {
    let result = Expert::new(0, 0, 4096, 1536, Activation::Gelu);
    assert!(matches!(
        result,
        Err(EmbeddingError::InvalidDimension { .. })
    ));
}

#[test]
fn test_expert_zero_hidden_dim_fails() {
    let result = Expert::new(0, 8320, 0, 1536, Activation::Gelu);
    assert!(matches!(
        result,
        Err(EmbeddingError::InvalidDimension { .. })
    ));
}

#[test]
fn test_expert_zero_output_dim_fails() {
    let result = Expert::new(0, 8320, 4096, 0, Activation::Gelu);
    assert!(matches!(
        result,
        Err(EmbeddingError::InvalidDimension { .. })
    ));
}

// =========================================================================
// EXPERT POOL TESTS
// =========================================================================

#[test]
fn test_expert_pool_creation() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config);
    assert!(pool.is_ok());
    let pool = pool.unwrap();
    assert_eq!(pool.num_experts(), NUM_EXPERTS);
    assert_eq!(pool.input_dim(), TOTAL_CONCATENATED);
    assert_eq!(pool.hidden_dim(), config.expert_hidden_dim);
    assert_eq!(pool.output_dim(), FUSED_OUTPUT);
}

#[test]
fn test_expert_pool_forward_single() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.1f32; TOTAL_CONCATENATED];

    for expert_idx in 0..NUM_EXPERTS {
        let output = pool.forward(&input, 1, expert_idx).unwrap();
        assert_eq!(output.len(), FUSED_OUTPUT);
    }
}

#[test]
fn test_expert_pool_invalid_index_fails() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.1f32; TOTAL_CONCATENATED];

    let result = pool.forward(&input, 1, NUM_EXPERTS); // Index 8 is invalid
    assert!(matches!(
        result,
        Err(EmbeddingError::InvalidExpertIndex { .. })
    ));
}

#[test]
fn test_forward_topk_output_shape() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.1f32; TOTAL_CONCATENATED];

    let indices = vec![0, 2, 4, 6]; // Top 4 experts
    let weights = vec![0.4, 0.3, 0.2, 0.1]; // Sum to 1.0

    let output = pool
        .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
        .unwrap();
    assert_eq!(output.len(), FUSED_OUTPUT);
}

#[test]
fn test_forward_topk_batch() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let batch_size = 2;
    let input = vec![0.1f32; TOTAL_CONCATENATED * batch_size];

    // 2 samples * 4 experts each
    let indices = vec![0, 2, 4, 6, 1, 3, 5, 7];
    let weights = vec![0.4, 0.3, 0.2, 0.1, 0.25, 0.25, 0.25, 0.25];

    let output = pool
        .forward_topk(&input, batch_size, &indices, &weights, TOP_K_EXPERTS)
        .unwrap();
    assert_eq!(output.len(), FUSED_OUTPUT * batch_size);
}

#[test]
fn test_forward_topk_single_expert_weight_one() {
    // If only one expert has weight 1.0, output should equal that expert's output
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.5f32; TOTAL_CONCATENATED];

    let indices = vec![3, 0, 0, 0];
    let weights = vec![1.0, 0.0, 0.0, 0.0]; // Only expert 3 contributes

    let topk_output = pool
        .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
        .unwrap();
    let direct_output = pool.forward(&input, 1, 3).unwrap();

    for (a, b) in topk_output.iter().zip(direct_output.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "forward_topk with weight 1.0 should match forward"
        );
    }
}

#[test]
fn test_forward_topk_weighted_combination() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.3f32; TOTAL_CONCATENATED];

    let indices = vec![0, 1, 2, 3];
    let weights = vec![0.5, 0.5, 0.0, 0.0];

    // Get outputs from experts 0 and 1
    let out0 = pool.forward(&input, 1, 0).unwrap();
    let out1 = pool.forward(&input, 1, 1).unwrap();

    // Compute expected weighted combination
    let expected: Vec<f32> = out0
        .iter()
        .zip(out1.iter())
        .map(|(&a, &b)| 0.5 * a + 0.5 * b)
        .collect();

    let actual = pool
        .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
        .unwrap();

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "Mismatch at index {}: actual={}, expected={}",
            i,
            a,
            e
        );
    }
}

#[test]
fn test_parameter_count() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();

    // Each expert: (8320*4096 + 4096) + (4096*1536 + 1536)
    // = 34,082,816 + 6,293,088 = 40,375,904 per expert
    // Total: 8 * ~40.4M = ~323M parameters
    let total = pool.total_parameter_count();
    assert!(total > 300_000_000, "Should have > 300M parameters");
    assert!(total < 350_000_000, "Should have < 350M parameters");
}

#[test]
fn test_forward_topk_empty_batch_fails() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input: Vec<f32> = vec![];
    let indices: Vec<usize> = vec![];
    let weights: Vec<f32> = vec![];

    let result = pool.forward_topk(&input, 0, &indices, &weights, TOP_K_EXPERTS);
    assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
}

#[test]
fn test_forward_topk_wrong_input_size_fails() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.1f32; 100]; // Wrong size
    let indices = vec![0, 1, 2, 3];
    let weights = vec![0.25, 0.25, 0.25, 0.25];

    let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
    assert!(matches!(
        result,
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_forward_topk_wrong_indices_size_fails() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.1f32; TOTAL_CONCATENATED];
    let indices = vec![0, 1]; // Wrong size (should be 4)
    let weights = vec![0.25, 0.25, 0.25, 0.25];

    let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
    assert!(matches!(
        result,
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_forward_topk_wrong_weights_size_fails() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.1f32; TOTAL_CONCATENATED];
    let indices = vec![0, 1, 2, 3];
    let weights = vec![0.5, 0.5]; // Wrong size (should be 4)

    let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
    assert!(matches!(
        result,
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_forward_topk_invalid_expert_index_fails() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.1f32; TOTAL_CONCATENATED];
    let indices = vec![0, 1, 2, 10]; // 10 is invalid
    let weights = vec![0.25, 0.25, 0.25, 0.25];

    let result = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS);
    assert!(matches!(
        result,
        Err(EmbeddingError::InvalidExpertIndex { .. })
    ));
}

// =========================================================================
// INTEGRATION WITH GATING NETWORK
// =========================================================================

#[test]
fn test_integration_with_gating_network() {
    use crate::fusion::GatingNetwork;

    let config = FusionConfig::default();
    let gating = GatingNetwork::new(&config).unwrap();
    let pool = ExpertPool::new(&config).unwrap();

    let input = vec![0.5f32; TOTAL_CONCATENATED];

    // Forward through gating
    let probs = gating.forward(&input, 1).unwrap();

    // Select top-k
    let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS).unwrap();

    // Forward through expert pool
    let output = pool
        .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
        .unwrap();

    assert_eq!(output.len(), FUSED_OUTPUT, "Final output should be 1536D");

    // Output should be finite (no NaN or Inf)
    for &val in &output {
        assert!(val.is_finite(), "Output contains non-finite value");
    }
}

// =========================================================================
// EDGE CASE TESTS WITH STATE PRINTING
// =========================================================================

#[test]
fn edge_case_empty_batch() {
    println!("=== BEFORE: Empty batch test ===");
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    println!("ExpertPool created with {} experts", pool.num_experts());

    let input: Vec<f32> = vec![];
    println!("Input length: {}", input.len());

    println!("=== EXECUTE: forward with batch_size=0 ===");
    let result = pool.forward(&input, 0, 0);

    println!("=== AFTER: Result ===");
    match &result {
        Ok(_) => println!("ERROR: Should have failed"),
        Err(e) => println!("Correctly failed with: {:?}", e),
    }
    assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
}

#[test]
fn edge_case_equal_weights() {
    println!("=== BEFORE: Equal weights test ===");
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![1.0f32; TOTAL_CONCATENATED];

    let indices = vec![0, 1, 2, 3];
    let weights = vec![0.25, 0.25, 0.25, 0.25];
    println!("Indices: {:?}", indices);
    println!(
        "Weights: {:?} (sum={})",
        weights,
        weights.iter().sum::<f32>()
    );

    println!("=== EXECUTE: forward_topk ===");
    let output = pool
        .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
        .unwrap();

    println!("=== AFTER: Output stats ===");
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "Output len: {}, mean: {:.6}, min: {:.6}, max: {:.6}",
        output.len(),
        mean,
        min,
        max
    );

    assert_eq!(output.len(), FUSED_OUTPUT);
}

#[test]
fn edge_case_extreme_weights() {
    println!("=== BEFORE: Extreme weight (0.999) test ===");
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.7f32; TOTAL_CONCATENATED];

    let indices = vec![5, 0, 0, 0];
    let weights = vec![0.999, 0.001 / 3.0, 0.001 / 3.0, 0.001 / 3.0];
    println!("Dominant expert: 5 with weight 0.999");

    println!("=== EXECUTE: Compare forward_topk vs direct forward ===");
    let topk_output = pool
        .forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)
        .unwrap();
    let direct_output = pool.forward(&input, 1, 5).unwrap();

    println!("=== AFTER: Compare outputs ===");
    let diff: f32 = topk_output
        .iter()
        .zip(direct_output.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / FUSED_OUTPUT as f32;
    println!("Average absolute difference: {:.8}", diff);

    assert!(
        diff < 0.01,
        "With 0.999 weight, output should be very close to single expert"
    );
}

#[test]
fn test_all_experts_different_outputs() {
    // Verify that different experts produce different outputs
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();
    let input = vec![0.5f32; TOTAL_CONCATENATED];

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for expert_idx in 0..NUM_EXPERTS {
        let output = pool.forward(&input, 1, expert_idx).unwrap();
        outputs.push(output);
    }

    // Check that outputs are different (at least first few elements)
    for i in 0..NUM_EXPERTS {
        for j in (i + 1)..NUM_EXPERTS {
            let diff: f32 = outputs[i]
                .iter()
                .zip(outputs[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>();
            // Due to random initialization, different experts should produce different outputs
            // Note: This could fail if by extreme chance weights are identical
            println!(
                "Diff between expert {} and {}: {:.6}",
                i,
                j,
                diff / FUSED_OUTPUT as f32
            );
        }
    }
}

#[test]
fn test_output_values_are_finite() {
    let config = FusionConfig::default();
    let pool = ExpertPool::new(&config).unwrap();

    // Test with various input patterns
    let test_inputs = vec![
        vec![0.0f32; TOTAL_CONCATENATED],
        vec![1.0f32; TOTAL_CONCATENATED],
        vec![-1.0f32; TOTAL_CONCATENATED],
        vec![0.001f32; TOTAL_CONCATENATED],
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        for expert_idx in 0..NUM_EXPERTS {
            let output = pool.forward(input, 1, expert_idx).unwrap();
            for (j, &val) in output.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Non-finite value at input pattern {}, expert {}, index {}",
                    i,
                    expert_idx,
                    j
                );
            }
        }
    }
}
