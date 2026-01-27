//! Integration tests for E5 embed_dual_with_hint()
//!
//! Test Suite 3 from CAUSAL_HINT_INTEGRATION_TESTING.md
//!
//! Note: Full embedding integration is tested through Test Suite 4 (MCP store_memory).
//! These tests verify the bias logic and CausalHint API.

use context_graph_core::traits::{CausalDirectionHint, CausalHint};

/// Test 3.5: Bias Factors API Verification
///
/// Directly verify bias_factors() returns correct values
#[test]
fn test_3_5_bias_factors_api() {
    println!("\n=== Test 3.5: Bias Factors API Verification ===");

    // Test Cause hint
    let cause_hint = CausalHint::new(true, CausalDirectionHint::Cause, 0.9, vec![]);
    assert!(cause_hint.is_useful(), "Cause hint should be useful");
    let (cb, eb) = cause_hint.bias_factors();
    assert!((cb - 1.3).abs() < 0.01, "Cause bias should be 1.3, got {}", cb);
    assert!((eb - 0.8).abs() < 0.01, "Effect bias should be 0.8, got {}", eb);
    println!("  Cause hint: cause_bias={}, effect_bias={}", cb, eb);

    // Test Effect hint
    let effect_hint = CausalHint::new(true, CausalDirectionHint::Effect, 0.9, vec![]);
    assert!(effect_hint.is_useful(), "Effect hint should be useful");
    let (cb, eb) = effect_hint.bias_factors();
    assert!((cb - 0.8).abs() < 0.01, "Cause bias should be 0.8, got {}", cb);
    assert!((eb - 1.3).abs() < 0.01, "Effect bias should be 1.3, got {}", eb);
    println!("  Effect hint: cause_bias={}, effect_bias={}", cb, eb);

    // Test Neutral hint
    let neutral_hint = CausalHint::new(true, CausalDirectionHint::Neutral, 0.9, vec![]);
    let (cb, eb) = neutral_hint.bias_factors();
    assert!((cb - 1.0).abs() < 0.01, "Cause bias should be 1.0, got {}", cb);
    assert!((eb - 1.0).abs() < 0.01, "Effect bias should be 1.0, got {}", eb);
    println!("  Neutral hint: cause_bias={}, effect_bias={}", cb, eb);

    // Test non-useful hint (low confidence)
    let low_conf_hint = CausalHint::new(true, CausalDirectionHint::Cause, 0.3, vec![]);
    assert!(
        !low_conf_hint.is_useful(),
        "Low confidence hint should not be useful"
    );
    println!("  Low confidence hint is not useful (confidence < 0.5)");

    // Test non-causal hint
    let non_causal_hint = CausalHint::new(false, CausalDirectionHint::Cause, 0.9, vec![]);
    assert!(
        !non_causal_hint.is_useful(),
        "Non-causal hint should not be useful"
    );
    println!("  Non-causal hint is not useful (is_causal = false)");

    println!("[PASS] Test 3.5: All bias factors API verified");
}

/// Test 3.6: Bias Application Simulation
///
/// Verify bias application logic without requiring GPU model
#[test]
fn test_3_6_bias_application_simulation() {
    println!("\n=== Test 3.6: Bias Application Simulation ===");

    // Simulate baseline embedding
    let baseline_cause: Vec<f32> = vec![0.5, 0.3, 0.2, 0.4];
    let baseline_effect: Vec<f32> = vec![0.4, 0.5, 0.3, 0.2];

    // Calculate baseline magnitudes
    let baseline_cause_mag: f32 = baseline_cause.iter().map(|x| x * x).sum::<f32>().sqrt();
    let baseline_effect_mag: f32 = baseline_effect.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("Baseline magnitudes:");
    println!("  Cause: {}", baseline_cause_mag);
    println!("  Effect: {}", baseline_effect_mag);

    // Simulate Cause hint application (1.3x cause, 0.8x effect)
    let cause_hint = CausalHint::new(true, CausalDirectionHint::Cause, 0.9, vec![]);
    let (cause_bias, effect_bias) = cause_hint.bias_factors();

    let biased_cause: Vec<f32> = baseline_cause.iter().map(|x| x * cause_bias).collect();
    let biased_effect: Vec<f32> = baseline_effect.iter().map(|x| x * effect_bias).collect();

    let biased_cause_mag: f32 = biased_cause.iter().map(|x| x * x).sum::<f32>().sqrt();
    let biased_effect_mag: f32 = biased_effect.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\nWith Cause hint (1.3x cause, 0.8x effect):");
    println!("  Biased cause magnitude: {}", biased_cause_mag);
    println!("  Biased effect magnitude: {}", biased_effect_mag);

    let cause_ratio = biased_cause_mag / baseline_cause_mag;
    let effect_ratio = biased_effect_mag / baseline_effect_mag;

    println!("  Ratios: cause={}, effect={}", cause_ratio, effect_ratio);

    assert!(
        (cause_ratio - 1.3).abs() < 0.01,
        "Cause should be 1.3x baseline"
    );
    assert!(
        (effect_ratio - 0.8).abs() < 0.01,
        "Effect should be 0.8x baseline"
    );

    // Simulate Effect hint application (0.8x cause, 1.3x effect)
    let effect_hint = CausalHint::new(true, CausalDirectionHint::Effect, 0.9, vec![]);
    let (cause_bias, effect_bias) = effect_hint.bias_factors();

    let biased_cause: Vec<f32> = baseline_cause.iter().map(|x| x * cause_bias).collect();
    let biased_effect: Vec<f32> = baseline_effect.iter().map(|x| x * effect_bias).collect();

    let biased_cause_mag: f32 = biased_cause.iter().map(|x| x * x).sum::<f32>().sqrt();
    let biased_effect_mag: f32 = biased_effect.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\nWith Effect hint (0.8x cause, 1.3x effect):");
    println!("  Biased cause magnitude: {}", biased_cause_mag);
    println!("  Biased effect magnitude: {}", biased_effect_mag);

    let cause_ratio = biased_cause_mag / baseline_cause_mag;
    let effect_ratio = biased_effect_mag / baseline_effect_mag;

    assert!(
        (cause_ratio - 0.8).abs() < 0.01,
        "Cause should be 0.8x baseline"
    );
    assert!(
        (effect_ratio - 1.3).abs() < 0.01,
        "Effect should be 1.3x baseline"
    );

    println!("[PASS] Test 3.6: Bias application simulation verified");
}

/// Test 3.7: CausalDirectionHint::from_str
///
/// Verify direction hint parsing
#[test]
fn test_3_7_direction_hint_from_str() {
    println!("\n=== Test 3.7: CausalDirectionHint::from_str ===");

    assert_eq!(
        CausalDirectionHint::from_str("cause"),
        CausalDirectionHint::Cause
    );
    assert_eq!(
        CausalDirectionHint::from_str("Cause"),
        CausalDirectionHint::Cause
    );
    assert_eq!(
        CausalDirectionHint::from_str("CAUSE"),
        CausalDirectionHint::Cause
    );

    assert_eq!(
        CausalDirectionHint::from_str("effect"),
        CausalDirectionHint::Effect
    );
    assert_eq!(
        CausalDirectionHint::from_str("Effect"),
        CausalDirectionHint::Effect
    );
    assert_eq!(
        CausalDirectionHint::from_str("EFFECT"),
        CausalDirectionHint::Effect
    );

    assert_eq!(
        CausalDirectionHint::from_str("neutral"),
        CausalDirectionHint::Neutral
    );
    assert_eq!(
        CausalDirectionHint::from_str("unknown"),
        CausalDirectionHint::Neutral
    );
    assert_eq!(
        CausalDirectionHint::from_str("invalid"),
        CausalDirectionHint::Neutral
    );

    println!("[PASS] Test 3.7: All from_str conversions correct");
}

/// Test 3.8: CausalHint Edge Cases
///
/// Verify edge cases in CausalHint
#[test]
fn test_3_8_causal_hint_edge_cases() {
    println!("\n=== Test 3.8: CausalHint Edge Cases ===");

    // Confidence exactly at threshold (0.5)
    let threshold_hint = CausalHint::new(true, CausalDirectionHint::Cause, 0.5, vec![]);
    assert!(
        threshold_hint.is_useful(),
        "Confidence at threshold (0.5) should be useful"
    );
    println!("  Confidence=0.5 is useful: {}", threshold_hint.is_useful());

    // Confidence just below threshold
    let below_hint = CausalHint::new(true, CausalDirectionHint::Cause, 0.49, vec![]);
    assert!(
        !below_hint.is_useful(),
        "Confidence below threshold (0.49) should not be useful"
    );
    println!("  Confidence=0.49 is useful: {}", below_hint.is_useful());

    // Confidence clamping
    let high_conf_hint = CausalHint::new(true, CausalDirectionHint::Cause, 1.5, vec![]);
    assert!(
        high_conf_hint.confidence <= 1.0,
        "Confidence should be clamped to 1.0"
    );
    println!(
        "  Confidence=1.5 clamped to: {}",
        high_conf_hint.confidence
    );

    let negative_conf_hint = CausalHint::new(true, CausalDirectionHint::Cause, -0.5, vec![]);
    assert!(
        negative_conf_hint.confidence >= 0.0,
        "Confidence should be clamped to 0.0"
    );
    println!(
        "  Confidence=-0.5 clamped to: {}",
        negative_conf_hint.confidence
    );

    // Default/neutral hint
    let default_hint = CausalHint::not_causal();
    assert!(!default_hint.is_useful(), "Default hint should not be useful");
    println!("  Default hint is_useful: {}", default_hint.is_useful());

    println!("[PASS] Test 3.8: All edge cases handled correctly");
}
