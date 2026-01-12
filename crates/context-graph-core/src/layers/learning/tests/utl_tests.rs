//! UTL Weight Computer tests - REAL implementations, NO MOCKS.

use crate::layers::learning::{UtlWeightComputer, GRADIENT_CLIP};

#[test]
fn test_utl_weight_computation() {
    let computer = UtlWeightComputer::new(0.0005);
    let result = computer.compute_update(0.8, 0.6).unwrap();

    // eta*(S x C_w) = 0.0005 * (0.8 * 0.6) = 0.0005 * 0.48 = 0.00024
    let expected = 0.0005 * 0.8 * 0.6;
    assert!(
        (result.value - expected).abs() < 1e-9,
        "Expected {}, got {}",
        expected,
        result.value
    );
    assert!(!result.was_clipped);
    println!(
        "[VERIFIED] UTL weight computation: eta*(S x C_w) = {}",
        result.value
    );
}

#[test]
fn test_utl_zero_surprise() {
    let computer = UtlWeightComputer::new(0.0005);
    let result = computer.compute_update(0.0, 0.9).unwrap();

    // Zero surprise = zero delta
    assert!(result.value.abs() < 1e-9);
    println!("[VERIFIED] Zero surprise produces zero delta");
}

#[test]
fn test_utl_zero_coherence() {
    let computer = UtlWeightComputer::new(0.0005);
    let result = computer.compute_update(0.9, 0.0).unwrap();

    // Zero coherence = zero delta
    assert!(result.value.abs() < 1e-9);
    println!("[VERIFIED] Zero coherence produces zero delta");
}

#[test]
fn test_utl_max_values() {
    let computer = UtlWeightComputer::new(0.0005);
    let result = computer.compute_update(1.0, 1.0).unwrap();

    // eta*(1.0*1.0) = 0.0005
    assert!((result.value - 0.0005).abs() < 1e-9);
    println!("[VERIFIED] Max inputs: delta = {}", result.value);
}

#[test]
fn test_utl_gradient_clipping() {
    // Use a very high learning rate to trigger clipping
    let computer = UtlWeightComputer::new(10.0); // Way too high
    let result = computer.compute_update(1.0, 1.0).unwrap();

    // Should be clipped to GRADIENT_CLIP (1.0)
    assert!(
        result.value.abs() <= GRADIENT_CLIP + 1e-6,
        "Should be clipped, got {}",
        result.value
    );
    assert!(result.was_clipped, "Should have been clipped");
    println!("[VERIFIED] Gradient clipping works: {} -> 1.0", 10.0);
}

#[test]
fn test_utl_nan_surprise_rejected() {
    let computer = UtlWeightComputer::new(0.0005);
    let result = computer.compute_update(f32::NAN, 0.5);

    assert!(result.is_err(), "NaN surprise should be rejected");
    println!("[VERIFIED] NaN surprise rejected per AP-009");
}

#[test]
fn test_utl_nan_coherence_rejected() {
    let computer = UtlWeightComputer::new(0.0005);
    let result = computer.compute_update(0.5, f32::NAN);

    assert!(result.is_err(), "NaN coherence should be rejected");
    println!("[VERIFIED] NaN coherence rejected per AP-009");
}

#[test]
fn test_utl_infinity_rejected() {
    let computer = UtlWeightComputer::new(0.0005);

    assert!(computer.compute_update(f32::INFINITY, 0.5).is_err());
    assert!(computer.compute_update(f32::NEG_INFINITY, 0.5).is_err());
    assert!(computer.compute_update(0.5, f32::INFINITY).is_err());
    assert!(computer.compute_update(0.5, f32::NEG_INFINITY).is_err());

    println!("[VERIFIED] Infinity values rejected per AP-009");
}

#[test]
fn test_utl_input_clamping() {
    let computer = UtlWeightComputer::new(0.0005);

    // Values > 1.0 should be clamped
    let result = computer.compute_update(2.0, 1.5).unwrap();
    assert_eq!(result.surprise, 1.0, "Surprise should be clamped to 1.0");
    assert_eq!(
        result.coherence_w, 1.0,
        "Coherence should be clamped to 1.0"
    );

    // Values < 0.0 should be clamped
    let result = computer.compute_update(-0.5, -0.3).unwrap();
    assert_eq!(result.surprise, 0.0, "Surprise should be clamped to 0.0");
    assert_eq!(
        result.coherence_w, 0.0,
        "Coherence should be clamped to 0.0"
    );

    println!("[VERIFIED] Input values clamped to [0, 1]");
}
