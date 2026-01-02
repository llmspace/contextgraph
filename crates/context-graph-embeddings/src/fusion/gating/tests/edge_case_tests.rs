//! Edge case tests for gating module components.

use crate::fusion::{LayerNorm, Linear};

#[test]
fn test_edge_case_very_large_input() {
    let norm = LayerNorm::new(4).unwrap();
    let input = vec![1e10, 1e10, 1e10, 1e10];
    let output = norm.forward(&input, 1).unwrap();

    // All same values -> mean matches input -> normalized to all zeros
    // When all inputs are the same, variance is 0
    // normalized = (x - mean) / sqrt(eps) ~= 0
    assert!(output.iter().all(|&x| x.abs() < 1.0));
}

#[test]
fn test_edge_case_very_small_input() {
    let norm = LayerNorm::new(4).unwrap();
    let input = vec![1e-10, 2e-10, 3e-10, 4e-10];
    let output = norm.forward(&input, 1).unwrap();

    // Should still normalize properly
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_edge_case_all_zeros_input() {
    let norm = LayerNorm::new(4).unwrap();
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let output = norm.forward(&input, 1).unwrap();

    // All zeros -> mean = 0, variance = 0
    // normalized = 0 / sqrt(eps) = 0
    assert!(output.iter().all(|&x| x.abs() < 1e-5));
}

#[test]
fn test_edge_case_negative_input() {
    let norm = LayerNorm::new(4).unwrap();
    let input = vec![-10.0, -5.0, 0.0, 5.0];
    let output = norm.forward(&input, 1).unwrap();

    // Should handle negative inputs fine
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_edge_case_mixed_sign_input() {
    let linear = Linear::new(4, 2).unwrap();
    let input = vec![-1.0, -0.5, 0.5, 1.0];
    let output = linear.forward(&input, 1).unwrap();

    // Just verify it completes without error
    assert_eq!(output.len(), 2);
}
